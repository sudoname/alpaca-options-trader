"""
SQLite episode store.

Records every trading DECISION under a single `decision_id`, and later attaches
the realized OUTCOME to that same id. This is the source of truth for learning
(RL and the supervised model) and the fix for the broken live RL loop, where the
decision was logged under `order['id']` but the outcome was recorded under a
different key (and gross, not net).

One row per decision; outcome columns are nullable until the trade closes. The
schema is intentionally additive-only (add nullable columns later, never rename)
so growing data never needs a migration.

Pure stdlib (`sqlite3`). Fully testable against an in-memory database with no
credentials and no network.
"""

import json
import math
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    decision_id      TEXT PRIMARY KEY,
    as_of            TEXT,
    created_at       TEXT,
    symbol           TEXT,
    underlying       TEXT,
    strat            TEXT,
    feature_version  TEXT,
    features_json    TEXT,
    state_key        TEXT,
    quote_bid        REAL,
    quote_ask        REAL,
    quote_ts         TEXT,
    modeled_cost_json TEXT,
    rule_action      TEXT,
    rule_confidence  REAL,
    gate_action      TEXT,
    gate_json        TEXT,
    risk_json        TEXT,
    chosen_action    TEXT,
    qty              INTEGER,
    mode             TEXT,
    -- outcome (nullable until close)
    fill_price       REAL,
    exit_price       REAL,
    gross_pnl_pct    REAL,
    net_pnl_pct      REAL,
    net_pnl_dollars  REAL,
    hold_days        INTEGER,
    outcome          TEXT,
    closed_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_state ON episodes(state_key);
CREATE INDEX IF NOT EXISTS idx_episodes_strat_asof ON episodes(strat, as_of);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
"""


class EpisodeStore:
    def __init__(self, db_path: str = "episodes.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            try:
                self.conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.Error:
                pass
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ----------------------------------------------------------- decisions
    def log_decision(
        self,
        *,
        symbol: str,
        underlying: str,
        strat: str,
        features: Dict,
        quote: Optional[Dict],
        modeled_cost: Optional[Dict],
        rule_action: str,
        rule_confidence: float,
        gate: Optional[Dict],
        chosen_action: str,
        qty: int,
        mode: str,
        as_of: Optional[str] = None,
        risk: Optional[Dict] = None,
        decision_id: Optional[str] = None,
    ) -> str:
        """Insert a decision row and return its decision_id (uuid4)."""
        did = decision_id or str(uuid.uuid4())
        quote = quote or {}
        self.conn.execute(
            """
            INSERT INTO episodes (
                decision_id, as_of, created_at, symbol, underlying, strat,
                feature_version, features_json, state_key,
                quote_bid, quote_ask, quote_ts, modeled_cost_json,
                rule_action, rule_confidence, gate_action, gate_json, risk_json,
                chosen_action, qty, mode
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                did,
                as_of or features.get("as_of"),
                datetime.now().isoformat(),
                symbol,
                underlying,
                strat,
                features.get("feature_version"),
                json.dumps(features, default=str),
                features.get("state_key"),
                quote.get("bid"),
                quote.get("ask"),
                str(quote.get("ts")) if quote.get("ts") is not None else None,
                json.dumps(modeled_cost, default=str) if modeled_cost else None,
                rule_action,
                rule_confidence,
                (gate or {}).get("rule_action") if gate else None,
                json.dumps(gate, default=str) if gate else None,
                json.dumps(risk, default=str) if risk else None,
                chosen_action,
                int(qty),
                mode,
            ),
        )
        self.conn.commit()
        return did

    def record_outcome(
        self,
        decision_id: str,
        *,
        fill_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        gross_pnl_pct: Optional[float] = None,
        net_pnl_pct: Optional[float] = None,
        net_pnl_dollars: Optional[float] = None,
        hold_days: int = 0,
        outcome: str = "closed",
        closed_at: Optional[str] = None,
    ) -> bool:
        """Attach an outcome to an existing decision. Idempotent (last write wins)."""
        cur = self.conn.execute(
            """
            UPDATE episodes SET
                fill_price=?, exit_price=?, gross_pnl_pct=?, net_pnl_pct=?,
                net_pnl_dollars=?, hold_days=?, outcome=?, closed_at=?
            WHERE decision_id=?
            """,
            (
                fill_price,
                exit_price,
                gross_pnl_pct,
                net_pnl_pct,
                net_pnl_dollars,
                int(hold_days),
                outcome,
                closed_at or datetime.now().isoformat(),
                decision_id,
            ),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------- queries
    def _rows(self, sql: str, params=()) -> List[Dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def open_decisions(self, strat: Optional[str] = None) -> List[Dict]:
        if strat:
            return self._rows(
                "SELECT * FROM episodes WHERE outcome IS NULL AND strat=? ORDER BY as_of",
                (strat,),
            )
        return self._rows("SELECT * FROM episodes WHERE outcome IS NULL ORDER BY as_of")

    def completed(
        self, *, strat: Optional[str] = None, since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[Dict]:
        sql = "SELECT * FROM episodes WHERE outcome IS NOT NULL"
        params: List = []
        if strat:
            sql += " AND strat=?"; params.append(strat)
        if since:
            sql += " AND as_of>=?"; params.append(since)
        if until:
            sql += " AND as_of<=?"; params.append(until)
        sql += " ORDER BY as_of"
        return self._rows(sql, tuple(params))

    def to_rl_experiences(self, strat: Optional[str] = None) -> List[Dict]:
        """
        Adapt completed rows to the shape train_rl.replay_experiences expects:
        {strat, state_key, action, pnl_pct (NET), pdt_remaining, took_day_trade,
         valid, ts}. Only rows with a chosen direction and a net P/L are usable.
        """
        from rl_env import valid_actions, SKIP

        out = []
        for row in self.completed(strat=strat):
            action = (row.get("chosen_action") or "").upper()
            pnl = row.get("net_pnl_pct")
            if pnl is None or action not in ("CALL", "PUT", SKIP):
                continue
            direction = action if action in ("CALL", "PUT") else ""
            out.append(
                {
                    "strat": row.get("strat", "generic"),
                    "state_key": row.get("state_key"),
                    "action": action,
                    "pnl_pct": float(pnl),
                    "pdt_remaining": None,
                    "took_day_trade": (row.get("mode") == "1DTE"),
                    "valid": valid_actions({"direction": direction, "should_trade": True}),
                    "ts": row.get("closed_at") or row.get("as_of") or "",
                }
            )
        out.sort(key=lambda e: e["ts"])
        return out

    def stats(self) -> Dict:
        total = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        completed = self.conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        pnls = [
            r[0]
            for r in self.conn.execute(
                "SELECT net_pnl_pct FROM episodes WHERE net_pnl_pct IS NOT NULL"
            ).fetchall()
        ]
        ess = _effective_sample_size(len(pnls))
        wins = sum(1 for p in pnls if p > 0)
        return {
            "total": total,
            "completed": completed,
            "completion_rate": (completed / total) if total else 0.0,
            "n_with_pnl": len(pnls),
            "win_rate": (wins / len(pnls)) if pnls else 0.0,
            "mean_net_pnl_pct": (sum(pnls) / len(pnls)) if pnls else 0.0,
            "effective_sample_size": ess,
        }


def _effective_sample_size(n: int) -> float:
    """
    Crude ESS placeholder: with no correlation model yet, ESS == n. Kept as a
    hook so a correlation-aware estimate can replace it without changing callers.
    """
    return float(max(0, n))


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    store = EpisodeStore(":memory:")

    features = {
        "feature_version": "1.0.0",
        "as_of": "2026-01-06T16:00:00",
        "state_key": "strat=spy_1dte|change=up_mild|...",
    }
    did = store.log_decision(
        symbol="SPY260107C00475000",
        underlying="SPY",
        strat="spy_1dte",
        features=features,
        quote={"bid": 1.00, "ask": 1.05, "ts": "2026-01-06T15:59:00"},
        modeled_cost={"cost_pct": 7.0},
        rule_action="CALL",
        rule_confidence=80.0,
        gate={"veto": False, "rule_action": "CALL"},
        chosen_action="CALL",
        qty=1,
        mode="1DTE",
    )
    if not did:
        print("FAIL: log_decision returned no id"); ok = False

    if len(store.open_decisions()) != 1:
        print("FAIL: expected one open decision"); ok = False

    attached = store.record_outcome(
        did, exit_price=1.30, gross_pnl_pct=24.0, net_pnl_pct=15.0,
        net_pnl_dollars=15.0, hold_days=1, outcome="take_profit",
    )
    if not attached:
        print("FAIL: record_outcome did not match decision_id"); ok = False

    if store.open_decisions():
        print("FAIL: decision should no longer be open"); ok = False

    exps = store.to_rl_experiences()
    if len(exps) != 1 or abs(exps[0]["pnl_pct"] - 15.0) > 1e-9:
        print("FAIL: to_rl_experiences should carry NET pnl", exps); ok = False
    if exps[0]["action"] != "CALL":
        print("FAIL: experience action wrong"); ok = False

    # idempotent re-record (last write wins, still one row).
    store.record_outcome(did, net_pnl_pct=10.0, outcome="closed")
    if store.stats()["completed"] != 1:
        print("FAIL: double-record should not create a new row"); ok = False

    # unknown id -> no match.
    if store.record_outcome("does-not-exist", net_pnl_pct=1.0):
        print("FAIL: recording an unknown id should return False"); ok = False

    # persistence round-trip via a temp file.
    import os
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), f"ep_selftest_{uuid.uuid4().hex}.db")
    try:
        s2 = EpisodeStore(tmp)
        d2 = s2.log_decision(
            symbol="QQQ", underlying="QQQ", strat="t", features=features,
            quote=None, modeled_cost=None, rule_action="PUT", rule_confidence=70.0,
            gate=None, chosen_action="PUT", qty=1, mode="swing",
        )
        s2.close()
        s3 = EpisodeStore(tmp)
        if len(s3.open_decisions()) != 1:
            print("FAIL: persistence round-trip lost the row"); ok = False
        s3.close()
    finally:
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(tmp + ext)
            except OSError:
                pass

    store.close()
    print("episode_store self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
