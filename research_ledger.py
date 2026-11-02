"""
Versioned decision ledger (research layer) — immutable snapshots + append-only
idempotent lifecycle events.

WHY A SECOND STORE
------------------
``episodes.db`` (episode_store.py) is the live decision log, but its outcome
columns are updated IN PLACE (last-write-wins). Research-grade evidence needs:

  * immutable decision snapshots (what was known at decision time, never edited)
  * append-only lifecycle events (submit/ack/fill/partial/cancel/reject/exit/
    expiry/outcome), each idempotent so restarts and re-reconciliation cannot
    double-count
  * provenance (observed vs estimated vs stale vs missing) per field group
  * strategy/config/schema versions + git code revision, with secrets redacted
  * the prediction definition (horizon, exit policy, target) frozen at
    decision time

This module adds exactly that in ``research_ledger.db`` WITHOUT touching
``episodes.db``. Rows link back to the existing scheme via the SAME
``decision_id`` (uuid4 from episode_store / shadow_recorder) so one id joins
episodes, ledger snapshots, lifecycle events, orders and outcomes.

GUARANTEES
----------
* Snapshots are INSERT-only; there is deliberately no update API.
* Events carry a deterministic ``event_key`` (sha256 of identifying fields)
  with a UNIQUE index + INSERT OR IGNORE: re-recording the same real-world
  event is a no-op (returns inserted=False).
* Additive-only schema policy (same as episode_store): add nullable columns
  later, never rename. ``schema_version`` is stamped on every row.
* Pure stdlib (sqlite3 + hashlib + subprocess for git rev). No network.
  Never opens a broker connection; cannot place orders.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

SCHEMA_VERSION = 1

# Fields whose names suggest secrets are redacted from any config hash input.
_SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS", "WEBHOOK",
                   "CHAT_ID", "USERNAME", "EMAIL", "MFA", "PICKLE")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id      TEXT PRIMARY KEY,
    decision_id      TEXT NOT NULL,
    schema_version   INTEGER NOT NULL,
    kind             TEXT NOT NULL,            -- 'candidate' | 'shadow_candidate'
    accepted         INTEGER NOT NULL,         -- 1 accepted (order attempted), 0 rejected/skipped
    reject_code      TEXT,                     -- machine reason when accepted=0
    event_at_utc     TEXT NOT NULL,            -- when the decision happened (UTC ISO)
    received_at_utc  TEXT NOT NULL,            -- when this row was written (UTC ISO)
    symbol           TEXT,                     -- OCC option symbol
    underlying       TEXT,
    expiration       TEXT,                     -- YYYY-MM-DD (parsed from OCC)
    strike           REAL,
    option_type      TEXT,                     -- 'call' | 'put'
    multiplier       REAL,                     -- contract multiplier (100 unless known otherwise)
    quote_bid        REAL,
    quote_ask        REAL,
    quote_bid_size   REAL,
    quote_ask_size   REAL,
    quote_ts_utc     TEXT,                     -- quote timestamp if known
    quote_age_s      REAL,                     -- age of quote at decision time, if known
    underlying_price REAL,
    iv               REAL,
    delta            REAL,
    gamma            REAL,
    theta            REAL,
    vega             REAL,
    direction        TEXT,                     -- CALL/PUT (would-be direction)
    score            REAL,
    p_win            REAL,                     -- predicted probability, if any
    ev               REAL,                     -- expected value estimate, if any
    signals_json     TEXT,                     -- signal values at decision time
    regime           TEXT,
    cost_json        TEXT,                     -- cost assumptions used
    sizing_json      TEXT,                     -- qty + sizing rationale
    risk_json        TEXT,                     -- risk engine verdict
    prediction_json  TEXT,                     -- horizon / exit policy / target, frozen now
    provenance_json  TEXT,                     -- per-field-group: observed|estimated|stale|missing
    strategy_version TEXT,
    config_hash      TEXT,                     -- redacted config fingerprint
    code_rev         TEXT,                     -- git HEAD at write time (best effort)
    extra_json       TEXT
);
CREATE INDEX IF NOT EXISTS ix_snap_decision ON snapshots(decision_id);
CREATE INDEX IF NOT EXISTS ix_snap_event_at ON snapshots(event_at_utc);
CREATE INDEX IF NOT EXISTS ix_snap_accept   ON snapshots(accepted);

CREATE TABLE IF NOT EXISTS events (
    event_key        TEXT PRIMARY KEY,         -- deterministic hash -> idempotent
    decision_id      TEXT NOT NULL,
    schema_version   INTEGER NOT NULL,
    event_type       TEXT NOT NULL,            -- submitted|ack|fill|partial_fill|cancel|reject|
                                               -- exit|expiry|assignment|outcome|unresolved
    event_at_utc     TEXT,                     -- when it happened at the broker/market (if known)
    received_at_utc  TEXT NOT NULL,            -- when we learned of it
    broker_order_id  TEXT,
    qty              REAL,
    price            REAL,
    fees             REAL,
    status           TEXT,                     -- broker status string, if any
    source           TEXT,                     -- 'live_hook' | 'reconciler' | 'resolver' | 'test'
    payload_json     TEXT
);
CREATE INDEX IF NOT EXISTS ix_ev_decision ON events(decision_id);
CREATE INDEX IF NOT EXISTS ix_ev_type     ON events(event_type);
CREATE INDEX IF NOT EXISTS ix_ev_order    ON events(broker_order_id);
"""

_TERMINAL_EVENTS = ("outcome", "expiry", "assignment")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_occ_symbol(symbol: str) -> Dict:
    """Parse an OCC option symbol into its contract identifiers.

    'SPY261116C00500000' -> {underlying SPY, expiration 2026-11-16,
    option_type call, strike 500.0}. Returns {} when it doesn't look like an
    OCC symbol (never raises) — callers then record the identifiers as missing
    rather than guessed.
    """
    try:
        s = str(symbol or "").strip().upper().replace(" ", "")
        if len(s) < 16:
            return {}
        tail = s[-15:]
        root = s[:-15]
        yy, mm, dd, cp, strike_raw = tail[0:2], tail[2:4], tail[4:6], tail[6], tail[7:]
        if cp not in ("C", "P") or not (yy + mm + dd + strike_raw).isdigit() or not root:
            return {}
        year = 2000 + int(yy)
        expiration = f"{year:04d}-{int(mm):02d}-{int(dd):02d}"
        # Validate the date parts without accepting nonsense like month 13.
        datetime(year, int(mm), int(dd))
        return {
            "underlying": root,
            "expiration": expiration,
            "option_type": "call" if cp == "C" else "put",
            "strike": int(strike_raw) / 1000.0,
        }
    except (ValueError, TypeError):
        return {}


def code_revision() -> Optional[str]:
    """Current git HEAD (best effort; None when git/repo unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        rev = (out.stdout or "").strip()
        return rev if out.returncode == 0 and len(rev) >= 7 else None
    except Exception:
        return None


def redacted_config_hash(config: Optional[Dict]) -> Optional[str]:
    """Deterministic fingerprint of non-secret config.

    Keys whose names contain a secret marker are dropped entirely (name AND
    value) so the hash input never embeds credentials. None -> None.
    """
    if not isinstance(config, dict):
        return None
    safe = {}
    for k, v in config.items():
        ku = str(k).upper()
        if any(m in ku for m in _SECRET_MARKERS):
            continue
        safe[str(k)] = str(v)
    blob = json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def event_key(decision_id, event_type, broker_order_id=None, event_at=None,
              qty=None, price=None, dedupe_tag=None) -> str:
    """Deterministic identity for a lifecycle event.

    The same real-world event recorded twice (live hook + reconciler, or a
    reconciler re-run after restart) hashes to the same key and is ignored on
    the second insert. ``dedupe_tag`` lets callers distinguish genuinely
    distinct events that would otherwise collide (e.g. two partial fills with
    identical qty/price and no timestamps).
    """
    parts = [
        str(decision_id or ""), str(event_type or ""),
        str(broker_order_id or ""), str(event_at or ""),
        "" if qty is None else repr(float(qty)),
        "" if price is None else repr(float(price)),
        str(dedupe_tag or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _dumps(obj) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def derive_provenance(quote: Optional[Dict], greeks: Optional[Dict],
                      underlying_price, quote_age_s=None,
                      stale_after_s: float = 60.0,
                      overrides: Optional[Dict] = None) -> Dict:
    """Classify field groups as observed / estimated / stale / missing.

    Defaults are derived mechanically (two-sided quote present -> observed;
    old quote -> stale; absent -> missing). Callers who KNOW a value was
    estimated (e.g. mid fallback, modeled greeks) pass ``overrides``.
    """
    prov: Dict[str, str] = {}
    q = quote or {}
    bid, ask = q.get("bid"), q.get("ask")
    try:
        two_sided = float(bid or 0) > 0 and float(ask or 0) > 0
    except (TypeError, ValueError):
        two_sided = False
    if not q or (bid is None and ask is None):
        prov["quote"] = "missing"
    elif quote_age_s is not None and quote_age_s > stale_after_s:
        prov["quote"] = "stale"
    elif two_sided:
        prov["quote"] = "observed"
    else:
        prov["quote"] = "one_sided"
    prov["greeks"] = "observed" if greeks else "missing"
    prov["underlying_price"] = "observed" if underlying_price is not None else "missing"
    if overrides:
        for k, v in overrides.items():
            prov[str(k)] = str(v)
    return prov


class ResearchLedger:
    """SQLite-backed immutable snapshot + append-only event store."""

    def __init__(self, path: str = "research_ledger.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # ----------------------------------------------------------- snapshots
    def record_snapshot(
        self, *,
        decision_id: Optional[str] = None,
        kind: str = "candidate",
        accepted: bool,
        reject_code: Optional[str] = None,
        symbol: Optional[str] = None,
        underlying: Optional[str] = None,
        option: Optional[Dict] = None,       # expiration/strike/option_type/multiplier overrides
        quote: Optional[Dict] = None,        # bid/ask/bid_size/ask_size/ts/age_s
        underlying_price=None,
        greeks: Optional[Dict] = None,       # iv/delta/gamma/theta/vega
        direction: Optional[str] = None,
        score=None, p_win=None, ev=None,
        signals: Optional[Dict] = None,
        regime: Optional[str] = None,
        cost: Optional[Dict] = None,
        sizing: Optional[Dict] = None,
        risk: Optional[Dict] = None,
        prediction: Optional[Dict] = None,
        provenance: Optional[Dict] = None,
        strategy_version: Optional[str] = None,
        config: Optional[Dict] = None,       # hashed with redaction; raw config never stored
        code_rev: Optional[str] = None,
        event_at: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """Insert one immutable decision snapshot; returns its decision_id.

        ``decision_id`` should be the episode_store id when one exists so the
        row joins the live record; a fresh uuid4 is minted otherwise.
        """
        did = decision_id or str(uuid.uuid4())
        sid = str(uuid.uuid4())
        occ = parse_occ_symbol(symbol) if symbol else {}
        opt = dict(occ)
        if isinstance(option, dict):
            opt.update({k: v for k, v in option.items() if v is not None})
        q = quote or {}
        g = greeks or {}
        prov = provenance
        if prov is None:
            prov = derive_provenance(q, g, underlying_price, q.get("age_s"))
        now = utc_now_iso()

        def _f(x):
            try:
                return None if x is None else float(x)
            except (TypeError, ValueError):
                return None

        self.conn.execute(
            """INSERT INTO snapshots (
                snapshot_id, decision_id, schema_version, kind, accepted,
                reject_code, event_at_utc, received_at_utc, symbol, underlying,
                expiration, strike, option_type, multiplier,
                quote_bid, quote_ask, quote_bid_size, quote_ask_size,
                quote_ts_utc, quote_age_s, underlying_price,
                iv, delta, gamma, theta, vega,
                direction, score, p_win, ev,
                signals_json, regime, cost_json, sizing_json, risk_json,
                prediction_json, provenance_json, strategy_version,
                config_hash, code_rev, extra_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sid, did, SCHEMA_VERSION, kind, 1 if accepted else 0,
                reject_code, event_at or now, now, symbol,
                underlying or opt.get("underlying"),
                opt.get("expiration"), _f(opt.get("strike")),
                opt.get("option_type"), _f(opt.get("multiplier")) or 100.0,
                _f(q.get("bid")), _f(q.get("ask")),
                _f(q.get("bid_size")), _f(q.get("ask_size")),
                q.get("ts"), _f(q.get("age_s")), _f(underlying_price),
                _f(g.get("iv")), _f(g.get("delta")), _f(g.get("gamma")),
                _f(g.get("theta")), _f(g.get("vega")),
                direction, _f(score), _f(p_win), _f(ev),
                _dumps(signals), regime, _dumps(cost), _dumps(sizing),
                _dumps(risk), _dumps(prediction), _dumps(prov),
                strategy_version,
                redacted_config_hash(config),
                code_rev if code_rev is not None else code_revision(),
                _dumps(extra),
            ),
        )
        self.conn.commit()
        return did

    # -------------------------------------------------------------- events
    def record_event(
        self, *,
        decision_id: str,
        event_type: str,
        broker_order_id: Optional[str] = None,
        qty=None, price=None, fees=None,
        status: Optional[str] = None,
        event_at: Optional[str] = None,
        source: str = "live_hook",
        payload: Optional[Dict] = None,
        dedupe_tag: Optional[str] = None,
    ):
        """Append one lifecycle event. Returns (event_key, inserted).

        Idempotent: the same identifying fields produce the same key, and a
        duplicate insert is ignored (inserted=False).
        """
        key = event_key(decision_id, event_type, broker_order_id, event_at,
                        qty, price, dedupe_tag)
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO events (
                event_key, decision_id, schema_version, event_type,
                event_at_utc, received_at_utc, broker_order_id,
                qty, price, fees, status, source, payload_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key, str(decision_id), SCHEMA_VERSION, str(event_type),
                event_at, utc_now_iso(), broker_order_id,
                None if qty is None else float(qty),
                None if price is None else float(price),
                None if fees is None else float(fees),
                status, source, _dumps(payload),
            ),
        )
        self.conn.commit()
        return key, cur.rowcount > 0

    # ------------------------------------------------------------- queries
    def _rows(self, sql: str, params=()) -> List[Dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_decision(self, decision_id: str) -> Dict:
        """All ledger knowledge about one decision_id."""
        return {
            "decision_id": decision_id,
            "snapshots": self._rows(
                "SELECT * FROM snapshots WHERE decision_id=? ORDER BY received_at_utc",
                (decision_id,)),
            "events": self._rows(
                "SELECT * FROM events WHERE decision_id=? "
                "ORDER BY COALESCE(event_at_utc, received_at_utc), received_at_utc",
                (decision_id,)),
        }

    def snapshots(self, since: Optional[str] = None, until: Optional[str] = None,
                  accepted: Optional[bool] = None, kind: Optional[str] = None) -> List[Dict]:
        sql, params = "SELECT * FROM snapshots WHERE 1=1", []
        if since:
            sql += " AND event_at_utc >= ?"; params.append(since)
        if until:
            sql += " AND event_at_utc <= ?"; params.append(until)
        if accepted is not None:
            sql += " AND accepted = ?"; params.append(1 if accepted else 0)
        if kind:
            sql += " AND kind = ?"; params.append(kind)
        sql += " ORDER BY event_at_utc"
        return self._rows(sql, tuple(params))

    def events_for_order(self, broker_order_id: str) -> List[Dict]:
        return self._rows(
            "SELECT * FROM events WHERE broker_order_id=? ORDER BY received_at_utc",
            (broker_order_id,))

    def decision_ids_with_event(self, event_type: str) -> List[str]:
        return [r["decision_id"] for r in self._rows(
            "SELECT DISTINCT decision_id FROM events WHERE event_type=?",
            (event_type,))]

    def unresolved(self) -> List[Dict]:
        """Accepted snapshots whose decision has no terminal event yet."""
        marks = ",".join("?" for _ in _TERMINAL_EVENTS)
        return self._rows(
            f"""SELECT s.* FROM snapshots s
                WHERE s.accepted=1 AND s.decision_id NOT IN (
                    SELECT DISTINCT decision_id FROM events
                    WHERE event_type IN ({marks}))
                ORDER BY s.event_at_utc""",
            _TERMINAL_EVENTS)

    def stats(self) -> Dict:
        s: Dict = {"schema_version": SCHEMA_VERSION, "path": self.path}
        s["snapshots"] = self.conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        s["accepted"] = self.conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE accepted=1").fetchone()[0]
        s["rejected"] = s["snapshots"] - s["accepted"]
        s["events"] = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        s["by_event_type"] = {r["event_type"]: r["n"] for r in self._rows(
            "SELECT event_type, COUNT(*) AS n FROM events GROUP BY event_type")}
        s["unresolved"] = len(self.unresolved())
        return s


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # --- OCC parsing ------------------------------------------------------ #
    p = parse_occ_symbol("SPY261116C00500000")
    if p != {"underlying": "SPY", "expiration": "2026-11-16",
             "option_type": "call", "strike": 500.0}:
        print("FAIL: OCC parse", p); ok = False
    if parse_occ_symbol("SPY") != {}:
        print("FAIL: short symbol should parse to {}"); ok = False
    if parse_occ_symbol("SPY261345C00500000") != {}:
        print("FAIL: month 13 should parse to {}"); ok = False
    if parse_occ_symbol(None) != {}:
        print("FAIL: None should parse to {}"); ok = False

    # --- config hash redaction -------------------------------------------- #
    h1 = redacted_config_hash({"ALPACA_API_KEY": "sek", "MAX_RISK": "100"})
    h2 = redacted_config_hash({"ALPACA_API_KEY": "other", "MAX_RISK": "100"})
    h3 = redacted_config_hash({"ALPACA_API_KEY": "sek", "MAX_RISK": "200"})
    if h1 != h2:
        print("FAIL: secret value must not affect hash"); ok = False
    if h1 == h3:
        print("FAIL: non-secret value must affect hash"); ok = False
    if redacted_config_hash(None) is not None:
        print("FAIL: None config -> None hash"); ok = False

    # --- provenance -------------------------------------------------------- #
    pr = derive_provenance({"bid": 1.0, "ask": 1.1}, None, 450.0, quote_age_s=5)
    if pr["quote"] != "observed" or pr["greeks"] != "missing" \
            or pr["underlying_price"] != "observed":
        print("FAIL: provenance observed case", pr); ok = False
    pr2 = derive_provenance({"bid": 1.0, "ask": 1.1}, None, None, quote_age_s=300)
    if pr2["quote"] != "stale":
        print("FAIL: old quote should be stale", pr2); ok = False
    pr3 = derive_provenance({"bid": 1.0, "ask": 0}, None, None)
    if pr3["quote"] != "one_sided":
        print("FAIL: one-sided quote class", pr3); ok = False
    pr4 = derive_provenance(None, None, None, overrides={"greeks": "estimated"})
    if pr4["quote"] != "missing" or pr4["greeks"] != "estimated":
        print("FAIL: provenance overrides", pr4); ok = False

    # --- ledger round trip -------------------------------------------------- #
    led = ResearchLedger(":memory:")
    did = led.record_snapshot(
        accepted=True, symbol="SPY261116C00500000",
        quote={"bid": 1.00, "ask": 1.05, "bid_size": 20, "ask_size": 15, "age_s": 2},
        underlying_price=498.5, greeks={"iv": 0.21, "delta": 0.45},
        direction="CALL", score=7.5, p_win=0.56, ev=12.0,
        signals={"momentum": 0.4}, regime="trend_up",
        cost={"slippage": 0.02}, sizing={"qty": 2, "rule": "fixed"},
        risk={"approved": True},
        prediction={"horizon_minutes": 390,
                    "exit_policy": {"stop_pct": 0.25, "tp_pct": 0.5},
                    "target": "net_pnl_pct>0"},
        strategy_version="smart_trader-v1",
        config={"ALPACA_API_KEY": "sek", "MAX_RISK": "100"},
        code_rev="abc1234",
    )
    row = led.get_decision(did)["snapshots"][0]
    if row["expiration"] != "2026-11-16" or row["strike"] != 500.0 \
            or row["option_type"] != "call" or row["multiplier"] != 100.0:
        print("FAIL: snapshot contract fields", row); ok = False
    if row["config_hash"] != redacted_config_hash({"MAX_RISK": "100"}):
        print("FAIL: config hash should ignore redacted keys"); ok = False
    if row["code_rev"] != "abc1234" or row["schema_version"] != SCHEMA_VERSION:
        print("FAIL: version stamps", row["code_rev"], row["schema_version"]); ok = False
    if json.loads(row["prediction_json"])["horizon_minutes"] != 390:
        print("FAIL: prediction frozen at decision time"); ok = False

    # Rejected snapshot with reason code.
    did2 = led.record_snapshot(accepted=False, reject_code="risk_blocked:max_concurrent",
                               symbol="QQQ261116P00400000", direction="PUT")
    if led.get_decision(did2)["snapshots"][0]["reject_code"] != "risk_blocked:max_concurrent":
        print("FAIL: reject code stored"); ok = False

    # --- idempotent events --------------------------------------------------- #
    k1, ins1 = led.record_event(decision_id=did, event_type="submitted",
                                broker_order_id="ord-1", qty=2,
                                event_at="2026-11-02T14:31:00+00:00")
    k2, ins2 = led.record_event(decision_id=did, event_type="submitted",
                                broker_order_id="ord-1", qty=2,
                                event_at="2026-11-02T14:31:00+00:00")
    if not ins1 or ins2 or k1 != k2:
        print("FAIL: duplicate event must be ignored", ins1, ins2); ok = False

    # Distinct partial fills (same qty/price) separated by dedupe_tag.
    _, a = led.record_event(decision_id=did, event_type="partial_fill",
                            broker_order_id="ord-1", qty=1, price=1.06,
                            dedupe_tag="fill-1")
    _, b = led.record_event(decision_id=did, event_type="partial_fill",
                            broker_order_id="ord-1", qty=1, price=1.06,
                            dedupe_tag="fill-2")
    if not (a and b):
        print("FAIL: distinct partial fills must both insert"); ok = False

    # Unresolved: did has no terminal event yet; did2 was rejected (not accepted).
    unres = led.unresolved()
    if [r["decision_id"] for r in unres] != [did]:
        print("FAIL: unresolved should list the open accepted decision", unres); ok = False
    led.record_event(decision_id=did, event_type="outcome", qty=2, price=1.30,
                     payload={"net_pnl_pct": 18.0, "class": "realized"})
    if led.unresolved():
        print("FAIL: outcome event should resolve the decision"); ok = False

    # No update API exists for snapshots (immutability by construction).
    if any(m.startswith("update") for m in dir(led)):
        print("FAIL: ledger must not expose snapshot update methods"); ok = False

    st = led.stats()
    if st["snapshots"] != 2 or st["accepted"] != 1 or st["rejected"] != 1:
        print("FAIL: stats", st); ok = False

    led.close()
    print("research_ledger self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
