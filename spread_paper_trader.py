"""
Phase 6C — paper (simulated) trading for defined-risk spread PROPOSALS.

This module is the SIMULATION-ONLY counterpart to ``spread_builder.py``. It takes
a :class:`spread_builder.SpreadProposal`, "opens" a simulated spread position,
marks it to market from leg bid/ask/mid quotes, computes unrealized P/L, and
"closes" it — writing closed trades to a JSON history file. It NEVER touches the
broker: there is no Alpaca client here, no order submission, and no live
execution path. Long single-leg call/put execution lives elsewhere and is
completely unaffected by this file.

Design (mirrors the rest of the project):
    * Pure / offline: quotes are passed IN by the caller, so the module is fully
      unit-testable with no network and no creds.
    * Config via :class:`config_loader.ConfigLoader` (shell > .env > default).
    * Feature flag ``USE_SPREAD_PAPER_TRADING`` defaults OFF.
    * Fail-safe persistence: a corrupt/missing JSON file behaves as "empty".

Mark convention (per 1-contract structure, ``CONTRACT_MULTIPLIER`` = 100):
    structure_mark = sum(+mid for BUY legs, -mid for SELL legs)
        -> this is the net debit to ESTABLISH the position you hold.
    A debit spread has a positive entry mark (you paid to own it).
    A credit spread has a negative entry mark (you were paid to put it on).
    unrealized P/L  = (current_mark - entry_mark) * 100
    pnl_percent     = pnl / max_loss * 100
This is sign-correct for both credit and debit structures (verified in tests:
a credit spread that narrows shows a gain; a debit spread that appreciates shows
a gain; a structure pinned at full width shows exactly -max_loss).
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Union

try:  # normal package-relative / flat-layout import
    from spread_builder import CONTRACT_MULTIPLIER, NO_TRADE, SpreadProposal
except Exception:  # pragma: no cover - defensive, keeps the module importable
    CONTRACT_MULTIPLIER = 100.0
    NO_TRADE = "no_trade"
    SpreadProposal = object  # type: ignore

from config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Result/“reason” constants returned by open_position (for callers + tests).
REASON_OPENED = "opened"
REASON_DISABLED = "disabled"
REASON_NO_TRADE = "no_trade"
REASON_INVALID_MAX_LOSS = "invalid_max_loss"
REASON_LOW_ORACLE_SCORE = "low_oracle_score"
REASON_DUPLICATE_POSITION = "duplicate_position"

STATUS_OPEN = "open"
STATUS_CLOSED = "closed"

# Phase 8B: analytics fields every CLOSED trade record is guaranteed to carry.
# Values are captured (best-effort) at open/close time; missing inputs store
# None so the schema is stable for the analytics layer. This is metadata
# capture only — it never alters open/mark/close math or any execution path.
ANALYTICS_FIELDS = (
    "symbol", "date", "strategy", "oracle_score", "volatility_edge",
    "expected_move", "market_expected_move", "actual_move", "pnl",
    "pnl_percent", "max_profit", "max_loss", "exit_reason", "dte", "iv_rank",
)

# Analytics fields supplied (optionally) at OPEN time, via ``context`` or as
# attributes on the proposal. All default to None when unavailable.
_OPEN_CONTEXT_FIELDS = (
    "volatility_edge", "expected_move", "market_expected_move",
    "dte", "iv_rank", "entry_underlying_price",
)


def _coerce_number(value):
    """Best-effort float coercion; returns None for missing/bad values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SpreadPaperConfig:
    """Configuration for the simulated spread paper trader."""

    enabled: bool = False
    min_oracle_score: float = 70.0
    positions_file: str = "spread_paper_positions.json"
    trades_file: str = "spread_paper_trades.json"

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "SpreadPaperConfig":
        cfg = loader if loader is not None else ConfigLoader(path=path)
        return SpreadPaperConfig(
            enabled=cfg.get_bool("USE_SPREAD_PAPER_TRADING", False),
            min_oracle_score=cfg.get_float("SPREAD_MIN_ORACLE_SCORE", 70.0),
            positions_file=cfg.get_str("SPREAD_PAPER_POSITIONS_FILE",
                                       "spread_paper_positions.json"),
            trades_file=cfg.get_str("SPREAD_PAPER_TRADES_FILE",
                                    "spread_paper_trades.json"),
        )


# --------------------------------------------------------------------------- #
# Quote / mark helpers (pure)
# --------------------------------------------------------------------------- #
def _mid_from_bid_ask(bid: Optional[float],
                      ask: Optional[float]) -> Optional[float]:
    """Mid price from a bid/ask pair, tolerant of partial/None data."""
    b = bid if isinstance(bid, (int, float)) and bid > 0 else None
    a = ask if isinstance(ask, (int, float)) and ask > 0 else None
    if b is not None and a is not None:
        return (b + a) / 2.0
    if b is not None:
        return b
    if a is not None:
        return a
    return None


def _leg_key(leg: Mapping) -> str:
    """Stable key for matching a leg to an externally-supplied quote.

    Prefers the OCC option symbol; falls back to action:type:strike so tests can
    quote by structural identity without a real symbol.
    """
    sym = leg.get("symbol") or ""
    if sym:
        return sym
    return f"{leg.get('action')}:{leg.get('type')}:{leg.get('strike')}"


def _leg_mid(leg: Mapping,
             quotes: Optional[Mapping[str, Union[float, Mapping]]]) -> Optional[float]:
    """Resolve a single leg's mid price.

    ``quotes`` (optional) maps a leg key -> either a float mid or a mapping with
    ``bid``/``ask``. When a leg isn't present in ``quotes`` (or ``quotes`` is
    None), fall back to the bid/ask stored on the leg itself.
    """
    if quotes:
        q = quotes.get(_leg_key(leg))
        if q is None:
            q = quotes.get(leg.get("symbol") or "")
        if isinstance(q, (int, float)):
            return float(q) if q > 0 else None
        if isinstance(q, Mapping):
            return _mid_from_bid_ask(q.get("bid"), q.get("ask"))
    return _mid_from_bid_ask(leg.get("bid"), leg.get("ask"))


def compute_mark(legs: List[Mapping],
                 quotes: Optional[Mapping[str, Union[float, Mapping]]] = None) -> float:
    """Signed structure mark = sum(+mid BUY, -mid SELL). Unpriceable legs -> 0."""
    total = 0.0
    for leg in legs:
        mid = _leg_mid(leg, quotes)
        if mid is None:
            continue
        action = str(leg.get("action", "")).lower()
        if action == "buy":
            total += mid
        elif action == "sell":
            total -= mid
    return round(total, 4)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Paper trader
# --------------------------------------------------------------------------- #
class SpreadPaperTrader:
    """Simulated open/mark/close for defined-risk spreads. No broker calls."""

    def __init__(self, config: Optional[SpreadPaperConfig] = None):
        self.config = config or SpreadPaperConfig.from_env()

    # -- persistence (fail-safe) ----------------------------------------- #
    @staticmethod
    def _load_json_list(path: str) -> List[dict]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except Exception as exc:  # corrupt file behaves as empty
            logger.warning("paper trader read failed (%s): %s", path, exc)
        return []

    @staticmethod
    def _save_json_list(path: str, rows: List[dict]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)
        except Exception as exc:
            logger.warning("paper trader write failed (%s): %s", path, exc)

    def load_positions(self) -> List[dict]:
        return self._load_json_list(self.config.positions_file)

    def save_positions(self, rows: List[dict]) -> None:
        self._save_json_list(self.config.positions_file, rows)

    def load_trades(self) -> List[dict]:
        return self._load_json_list(self.config.trades_file)

    def save_trades(self, rows: List[dict]) -> None:
        self._save_json_list(self.config.trades_file, rows)

    def get_open_positions(self) -> List[dict]:
        return [p for p in self.load_positions()
                if p.get("status") == STATUS_OPEN]

    def find_position(self, position_id: str) -> Optional[dict]:
        for p in self.load_positions():
            if p.get("id") == position_id:
                return p
        return None

    # -- open ------------------------------------------------------------- #
    @staticmethod
    def _open_context(proposal: "SpreadProposal",
                      context: Optional[Mapping]) -> Dict[str, Optional[float]]:
        """Resolve the optional analytics fields captured at open time.

        Looks first in the explicit ``context`` mapping, then falls back to a
        same-named attribute on the proposal, else None. Purely additive
        metadata — does not influence any open/reject decision.
        """
        ctx = context or {}
        out: Dict[str, Optional[float]] = {}
        for field_name in _OPEN_CONTEXT_FIELDS:
            if field_name in ctx:
                out[field_name] = _coerce_number(ctx.get(field_name))
            else:
                out[field_name] = _coerce_number(getattr(proposal, field_name, None))
        return out

    def open_position(
        self,
        proposal: "SpreadProposal",
        quotes: Optional[Mapping[str, Union[float, Mapping]]] = None,
        context: Optional[Mapping] = None,
    ) -> dict:
        """Open a simulated spread position from a proposal.

        Returns a result dict: ``{'allowed': bool, 'reason': str,
        'position': dict|None}``. Performs ONLY simulation — no Alpaca order is
        ever submitted. All safety rejections (Req 4) happen here.

        ``context`` (optional, Phase 8B) may carry analytics metadata to capture
        on the record: ``volatility_edge``, ``expected_move``,
        ``market_expected_move``, ``dte``, ``iv_rank``, ``entry_underlying_price``.
        It NEVER affects whether/how a position opens.
        """
        if not self.config.enabled:
            return self._reject(REASON_DISABLED)

        strategy = getattr(proposal, "strategy_name", NO_TRADE)
        symbol = getattr(proposal, "symbol", "") or ""
        oracle_score = float(getattr(proposal, "oracle_score", 0.0) or 0.0)
        max_loss = getattr(proposal, "max_loss", None)

        # Req 4: reject a non-tradeable proposal.
        if strategy == NO_TRADE or not strategy:
            return self._reject(REASON_NO_TRADE, symbol)
        # Req 4: reject if max_loss is missing or non-positive.
        if not isinstance(max_loss, (int, float)) or max_loss <= 0:
            return self._reject(REASON_INVALID_MAX_LOSS, symbol)
        # Req 4: reject below the oracle-score floor.
        if oracle_score < self.config.min_oracle_score:
            return self._reject(REASON_LOW_ORACLE_SCORE, symbol)
        # Req 4: reject a duplicate OPEN spread on the same symbol.
        if any(p.get("symbol") == symbol for p in self.get_open_positions()):
            return self._reject(REASON_DUPLICATE_POSITION, symbol)

        legs = [l.as_dict() for l in getattr(proposal, "legs", [])]
        entry_mark = compute_mark(legs, quotes)
        analytics = self._open_context(proposal, context)
        position = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": _now_iso(),
            "symbol": symbol,
            "strategy": strategy,
            "oracle_score": round(oracle_score, 2),
            "legs": legs,
            "net_credit_or_debit": round(
                float(getattr(proposal, "net_credit_or_debit", 0.0) or 0.0), 4),
            "max_profit": round(float(getattr(proposal, "max_profit", 0.0) or 0.0), 2),
            "max_loss": round(float(max_loss), 2),
            "breakeven": getattr(proposal, "breakeven", None),
            "status": STATUS_OPEN,
            "entry_mark": entry_mark,
            "current_mark": entry_mark,
            "pnl": 0.0,
            "pnl_percent": 0.0,
            "exit_reason": None,
            # Phase 8B analytics metadata (captured at open; may be None).
            "volatility_edge": analytics["volatility_edge"],
            "expected_move": analytics["expected_move"],
            "market_expected_move": analytics["market_expected_move"],
            "dte": analytics["dte"],
            "iv_rank": analytics["iv_rank"],
            "entry_underlying_price": analytics["entry_underlying_price"],
            "actual_move": None,   # finalized at close
        }

        rows = self.load_positions()
        rows.append(position)
        self.save_positions(rows)

        print(f"[SPREAD_PAPER_OPEN] SIMULATED id={position['id']} "
              f"sym={symbol} strategy={strategy} "
              f"oracle={position['oracle_score']:.1f} "
              f"net={position['net_credit_or_debit']:+.2f} "
              f"entry_mark={entry_mark:+.4f} max_loss={position['max_loss']:.2f} "
              f"(paper only, no broker order)")

        # Phase 9C: capture the advisory recommendation at OPEN time (before the
        # outcome is known). Pure observer — fail-open, never affects the trade.
        try:
            import advisory_attribution
            advisory_attribution.record_open(position)
        except Exception as exc:
            print(f"[ADVISORY_ATTRIBUTION] open hook ignored: {exc}")

        return {"allowed": True, "reason": REASON_OPENED, "position": position}

    @staticmethod
    def _reject(reason: str, symbol: str = "") -> dict:
        print(f"[SPREAD_PAPER_OPEN] REJECTED sym={symbol or '?'} reason={reason} "
              f"(paper only, no broker order)")
        return {"allowed": False, "reason": reason, "position": None}

    # -- mark-to-market --------------------------------------------------- #
    def mark_to_market(
        self,
        position: Union[dict, str],
        quotes: Optional[Mapping[str, Union[float, Mapping]]] = None,
        persist: bool = True,
    ) -> Optional[dict]:
        """Recompute current_mark / pnl / pnl_percent for an OPEN position.

        ``position`` may be a position dict or a position id. Returns the updated
        position dict (also persisted when ``persist`` and the position is found
        in the store). Pure math — no broker calls.
        """
        pos = self.find_position(position) if isinstance(position, str) else dict(position)
        if pos is None:
            return None

        current_mark = compute_mark(pos.get("legs", []), quotes)
        entry_mark = float(pos.get("entry_mark", 0.0) or 0.0)
        max_loss = float(pos.get("max_loss", 0.0) or 0.0)
        pnl = round((current_mark - entry_mark) * CONTRACT_MULTIPLIER, 2)
        pnl_percent = round((pnl / max_loss) * 100.0, 2) if max_loss > 0 else 0.0

        pos["current_mark"] = current_mark
        pos["pnl"] = pnl
        pos["pnl_percent"] = pnl_percent

        if persist:
            rows = self.load_positions()
            for i, row in enumerate(rows):
                if row.get("id") == pos.get("id"):
                    rows[i] = pos
                    self.save_positions(rows)
                    break

        print(f"[SPREAD_PAPER_MTM] SIMULATED id={pos.get('id')} "
              f"sym={pos.get('symbol')} current_mark={current_mark:+.4f} "
              f"pnl={pnl:+.2f} ({pnl_percent:+.1f}% of max_loss)")
        return pos

    # -- close ------------------------------------------------------------ #
    @staticmethod
    def _finalize_analytics(record: dict, context: Optional[Mapping]) -> None:
        """Stamp ``date`` + ``actual_move`` and guarantee the analytics schema.

        ``actual_move`` is taken from ``context['actual_move']`` if given, else
        derived from ``context['exit_underlying_price']`` minus the position's
        ``entry_underlying_price`` when both are known, else left None. Any
        :data:`ANALYTICS_FIELDS` not already present are filled with None so the
        closed-trade schema is stable for the analytics layer. Metadata only.
        """
        ctx = context or {}
        # date: prefer the close timestamp, fall back to open timestamp.
        stamp = record.get("closed_at") or record.get("timestamp") or _now_iso()
        record["date"] = str(stamp)[:10]

        actual_move = _coerce_number(ctx.get("actual_move"))
        if actual_move is None:
            exit_px = _coerce_number(ctx.get("exit_underlying_price"))
            entry_px = _coerce_number(record.get("entry_underlying_price"))
            if exit_px is not None and entry_px is not None:
                actual_move = round(exit_px - entry_px, 4)
        if actual_move is not None:
            record["actual_move"] = actual_move

        for field_name in ANALYTICS_FIELDS:
            record.setdefault(field_name, None)

    def close_position(
        self,
        position_id: str,
        quotes: Optional[Mapping[str, Union[float, Mapping]]] = None,
        exit_reason: str = "manual_close",
        context: Optional[Mapping] = None,
    ) -> Optional[dict]:
        """Close a simulated position: final MTM, move to trade history.

        Removes the position from the open store and appends the finalized record
        to the trades file. Returns the closed trade dict, or None if not found /
        already closed. No broker calls.

        ``context`` (optional, Phase 8B) may carry ``actual_move`` or
        ``exit_underlying_price`` so the closed record captures the realized
        underlying move for prediction-accuracy analytics. Metadata only.
        """
        rows = self.load_positions()
        target = None
        remaining = []
        for row in rows:
            if (row.get("id") == position_id
                    and row.get("status") == STATUS_OPEN
                    and target is None):
                target = row
            else:
                remaining.append(row)
        if target is None:
            print(f"[SPREAD_PAPER_CLOSE] not found / not open id={position_id} "
                  f"(paper only)")
            return None

        # Final mark-to-market without re-persisting to the (now-pruned) store.
        marked = self.mark_to_market(target, quotes, persist=False)
        marked["status"] = STATUS_CLOSED
        marked["exit_reason"] = exit_reason
        marked["closed_at"] = _now_iso()
        self._finalize_analytics(marked, context)

        self.save_positions(remaining)
        trades = self.load_trades()
        trades.append(marked)
        self.save_trades(trades)

        print(f"[SPREAD_PAPER_CLOSE] SIMULATED id={marked['id']} "
              f"sym={marked.get('symbol')} exit={exit_reason} "
              f"pnl={marked.get('pnl'):+.2f} "
              f"({marked.get('pnl_percent'):+.1f}% of max_loss) "
              f"(paper only, no broker order)")

        # Phase 9C: append the realized outcome to the entry-time advisory
        # snapshot (advisory fields are NOT recomputed). Fail-open observer.
        try:
            import advisory_attribution
            advisory_attribution.record_close(marked)
        except Exception as exc:
            print(f"[ADVISORY_ATTRIBUTION] close hook ignored: {exc}")

        return marked


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses temp files + a synthetic proposal)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    from spread_builder import SpreadLeg, SpreadProposal as _SP

    ok = True
    d = tempfile.mkdtemp()
    cfg = SpreadPaperConfig(
        enabled=True, min_oracle_score=70.0,
        positions_file=os.path.join(d, "pos.json"),
        trades_file=os.path.join(d, "trades.json"),
    )
    trader = SpreadPaperTrader(cfg)

    # Bull put credit spread: SELL 100 put @2.00, BUY 95 put @1.25 -> credit 0.75.
    legs = [
        SpreadLeg("sell", "put", 100, bid=1.95, ask=2.05),
        SpreadLeg("buy", "put", 95, bid=1.20, ask=1.30),
    ]
    proposal = _SP(strategy_name="bull_put_credit_spread", symbol="SPY",
                   legs=legs, net_credit_or_debit=0.75, max_profit=75.0,
                   max_loss=425.0, breakeven=99.25, width=5.0,
                   oracle_score=80.0)

    res = trader.open_position(proposal)
    if not res["allowed"]:
        print("FAIL: valid proposal should open", res); ok = False
    pos = res["position"]
    # entry_mark = +1.25 (buy) - 2.00 (sell) = -0.75 (credit -> negative).
    if pos and abs(pos["entry_mark"] - (-0.75)) > 1e-6:
        print("FAIL: entry_mark", pos["entry_mark"]); ok = False

    # Duplicate symbol rejected.
    if trader.open_position(proposal)["reason"] != REASON_DUPLICATE_POSITION:
        print("FAIL: duplicate should reject"); ok = False

    # MTM after spread narrows (good for credit seller): SELL 1.00 / BUY 0.50.
    quotes = {
        "sell:put:100": {"bid": 0.95, "ask": 1.05},
        "buy:put:95": {"bid": 0.45, "ask": 0.55},
    }
    marked = trader.mark_to_market(pos["id"], quotes)
    # current_mark = +0.50 - 1.00 = -0.50; pnl = (-0.50 - -0.75)*100 = +25.
    if marked and abs(marked["pnl"] - 25.0) > 1e-6:
        print("FAIL: credit MTM pnl", marked["pnl"]); ok = False

    closed = trader.close_position(pos["id"], quotes, exit_reason="take_profit")
    if not closed or closed["status"] != STATUS_CLOSED:
        print("FAIL: close should finalize", closed); ok = False
    if len(trader.load_trades()) != 1 or trader.get_open_positions():
        print("FAIL: trade history / open pruning"); ok = False

    # no_trade and low-score rejections.
    nt = _SP(strategy_name=NO_TRADE, symbol="QQQ", max_loss=100.0,
             oracle_score=90.0)
    if trader.open_position(nt)["reason"] != REASON_NO_TRADE:
        print("FAIL: no_trade should reject"); ok = False
    low = _SP(strategy_name="debit_call_spread", symbol="QQQ", legs=[],
              net_credit_or_debit=-1.0, max_profit=300.0, max_loss=200.0,
              oracle_score=50.0)
    if trader.open_position(low)["reason"] != REASON_LOW_ORACLE_SCORE:
        print("FAIL: low score should reject"); ok = False

    # Disabled config rejects.
    disabled = SpreadPaperTrader(SpreadPaperConfig(
        enabled=False, positions_file=os.path.join(d, "p2.json"),
        trades_file=os.path.join(d, "t2.json")))
    if disabled.open_position(proposal)["reason"] != REASON_DISABLED:
        print("FAIL: disabled should reject"); ok = False

    print("spread_paper_trader self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
