"""
Fail-open capture hooks wiring the live decision path to the research ledger.

smart_trader calls these tiny wrappers at its existing instrumentation sites
(the same places shadow_recorder hooks in). Rules, in order of importance:

  1. NEVER raise into the trading path. Every public method is try/except'd
     and returns None on any failure; a logging failure cannot block an order,
     a risk check, or position management.
  2. Default OFF. ``ENABLE_DECISION_LEDGER`` (ConfigLoader semantics: shell >
     .env > default; truthy {1,true,yes,on}) gates everything. Disabled means
     every hook is a no-op that touches no file and opens no database.
  3. Account for drops. When enabled but a write fails, the failure is counted
     in ``dropped`` (by hook name) so missing evidence is measurable instead
     of silent.

The hooks only WRITE research_ledger.db. They never read trading state, never
mutate the objects they are given, and never talk to a broker.
"""

from typing import Dict, Optional

FLAG = "ENABLE_DECISION_LEDGER"
LEDGER_PATH_KEY = "DECISION_LEDGER_PATH"
DEFAULT_LEDGER_PATH = "research_ledger.db"


def _utc_from_local(ts_iso: Optional[str]) -> Optional[str]:
    """Best-effort pass-through; the ledger stamps receipt time itself."""
    return ts_iso


def build_prediction(dynamic_levels: Optional[Dict],
                     horizon_minutes: float = 390.0) -> Dict:
    """Freeze the decision-time prediction definition.

    The exit policy is the dynamic stop/take/trailing levels the bot committed
    to at entry; the target is the NET-of-cost sign of the trade over at most
    ``horizon_minutes`` of market time. Stored on the snapshot so later
    evaluation cannot quietly move the goalposts.
    """
    dl = dynamic_levels or {}
    return {
        "horizon_minutes": float(horizon_minutes),
        "target": "net_pnl_pct>0",
        "exit_policy": {
            "stop_loss_pct": dl.get("stop_loss_percent"),
            "take_profit_pct": dl.get("take_profit_percent"),
            "trailing_stop_distance": dl.get("trailing_stop_distance"),
        },
    }


class ResearchCapture:
    """Flag-gated, fail-open facade over ResearchLedger for the live path."""

    def __init__(self, ledger=None, enabled: Optional[bool] = None,
                 ledger_path: Optional[str] = None, config: Optional[Dict] = None):
        self.dropped: Dict[str, int] = {}
        self._config = config
        if enabled is None:
            try:
                from config_loader import ConfigLoader
                cfg = ConfigLoader()
                enabled = cfg.get_bool(FLAG, False)
                if ledger_path is None:
                    ledger_path = cfg.get_str(LEDGER_PATH_KEY, DEFAULT_LEDGER_PATH)
                if config is None:
                    # Fingerprint input only — hashed with secret redaction by
                    # the ledger; raw values are never stored.
                    self._config = dict(cfg._file)
            except Exception:
                enabled = False
        self.enabled = bool(enabled)
        self._ledger = None
        if self.enabled:
            try:
                if ledger is not None:
                    self._ledger = ledger
                else:
                    from research_ledger import ResearchLedger
                    self._ledger = ResearchLedger(ledger_path or DEFAULT_LEDGER_PATH)
            except Exception:
                self._ledger = None
                self._drop("init")

    # ------------------------------------------------------------- plumbing
    def _drop(self, hook: str):
        self.dropped[hook] = self.dropped.get(hook, 0) + 1

    def _ready(self) -> bool:
        return self.enabled and self._ledger is not None

    # ---------------------------------------------------------------- hooks
    def on_candidate(
        self, *,
        decision_id: Optional[str],
        accepted: bool,
        reject_code: Optional[str] = None,
        symbol: Optional[str] = None,
        underlying: Optional[str] = None,
        quote: Optional[Dict] = None,
        underlying_price=None,
        greeks: Optional[Dict] = None,
        direction: Optional[str] = None,
        score=None, p_win=None, ev=None,
        signals: Optional[Dict] = None,
        regime: Optional[str] = None,
        cost: Optional[Dict] = None,
        sizing: Optional[Dict] = None,
        risk: Optional[Dict] = None,
        dynamic_levels: Optional[Dict] = None,
        provenance: Optional[Dict] = None,
        strategy_version: str = "smart_trader",
        extra: Optional[Dict] = None,
    ) -> Optional[str]:
        """Immutable snapshot of one evaluated candidate (accepted or not)."""
        if not self._ready():
            return None
        try:
            return self._ledger.record_snapshot(
                decision_id=decision_id,
                accepted=accepted,
                reject_code=reject_code,
                symbol=symbol,
                underlying=underlying,
                quote=quote,
                underlying_price=underlying_price,
                greeks=greeks,
                direction=direction,
                score=score, p_win=p_win, ev=ev,
                signals=signals, regime=regime,
                cost=cost, sizing=sizing, risk=risk,
                prediction=build_prediction(dynamic_levels),
                provenance=provenance,
                strategy_version=strategy_version,
                config=self._config,
                extra=extra,
            )
        except Exception:
            self._drop("on_candidate")
            return None

    def on_order_submitted(self, decision_id, order_id, *, qty=None,
                           limit_price=None, event_at=None) -> None:
        if not self._ready() or not decision_id:
            return None
        try:
            self._ledger.record_event(
                decision_id=decision_id, event_type="submitted",
                broker_order_id=order_id, qty=qty, price=limit_price,
                event_at=event_at, source="live_hook")
        except Exception:
            self._drop("on_order_submitted")
        return None

    def on_fill(self, decision_id, order_id, *, qty=None, price=None,
                status=None, partial=False, event_at=None,
                dedupe_tag=None) -> None:
        if not self._ready() or not decision_id:
            return None
        try:
            self._ledger.record_event(
                decision_id=decision_id,
                event_type="partial_fill" if partial else "fill",
                broker_order_id=order_id, qty=qty, price=price,
                status=status, event_at=event_at, source="live_hook",
                dedupe_tag=dedupe_tag)
        except Exception:
            self._drop("on_fill")
        return None

    def on_broker_reject(self, decision_id, order_id=None, *, status=None,
                         reason=None, event_at=None) -> None:
        """Broker-side rejection/cancel/expiry-with-no-fill of a submission."""
        if not self._ready() or not decision_id:
            return None
        try:
            self._ledger.record_event(
                decision_id=decision_id, event_type="reject",
                broker_order_id=order_id, status=status,
                event_at=event_at, source="live_hook",
                payload={"reason": reason} if reason else None)
        except Exception:
            self._drop("on_broker_reject")
        return None

    def on_exit(self, decision_id, *, order_id=None, qty=None, exit_price=None,
                reason=None, gross_pnl_pct=None, net_pnl_pct=None,
                net_pnl_dollars=None, event_at=None) -> None:
        """Exit + realized outcome, as two append-only events."""
        if not self._ready() or not decision_id:
            return None
        try:
            self._ledger.record_event(
                decision_id=decision_id, event_type="exit",
                broker_order_id=order_id, qty=qty, price=exit_price,
                event_at=event_at, source="live_hook",
                payload={"reason": reason} if reason else None)
            self._ledger.record_event(
                decision_id=decision_id, event_type="outcome",
                broker_order_id=order_id, qty=qty, price=exit_price,
                event_at=event_at, source="live_hook",
                payload={
                    "class": "realized",
                    "reason": reason,
                    "gross_pnl_pct": gross_pnl_pct,
                    "net_pnl_pct": net_pnl_pct,
                    "net_pnl_dollars": net_pnl_dollars,
                })
        except Exception:
            self._drop("on_exit")
        return None


_singleton: Optional[ResearchCapture] = None


def get_capture() -> ResearchCapture:
    """Process-wide capture instance (constructed once, flag read at startup)."""
    global _singleton
    if _singleton is None:
        try:
            _singleton = ResearchCapture()
        except Exception:
            _singleton = ResearchCapture(enabled=False)
    return _singleton


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from research_ledger import ResearchLedger

    ok = True

    # Disabled -> every hook is a no-op returning None, no ledger touched.
    cap = ResearchCapture(enabled=False)
    if cap.enabled or cap._ledger is not None:
        print("FAIL: disabled capture must not build a ledger"); ok = False
    if cap.on_candidate(decision_id="x", accepted=True) is not None:
        print("FAIL: disabled on_candidate must return None"); ok = False
    cap.on_order_submitted("x", "ord-1")
    cap.on_fill("x", "ord-1", qty=1, price=1.0)
    cap.on_exit("x", exit_price=1.2)
    if cap.dropped:
        print("FAIL: disabled hooks must not count drops", cap.dropped); ok = False

    # Enabled, in-memory ledger: full lifecycle lands.
    led = ResearchLedger(":memory:")
    cap = ResearchCapture(ledger=led, enabled=True, config={"API_KEY": "s", "X": "1"})
    did = cap.on_candidate(
        decision_id="dec-1", accepted=True, symbol="SPY261116C00500000",
        quote={"bid": 1.0, "ask": 1.05}, underlying_price=498.0,
        direction="CALL", score=7.0,
        dynamic_levels={"stop_loss_percent": 0.25, "take_profit_percent": 0.5,
                        "trailing_stop_distance": 0.1},
        risk={"approved": True}, sizing={"qty": 2},
    )
    if did != "dec-1":
        print("FAIL: snapshot should reuse the provided decision_id", did); ok = False
    cap.on_order_submitted("dec-1", "ord-9", qty=2, limit_price=1.05)
    cap.on_fill("dec-1", "ord-9", qty=2, price=1.06, status="filled")
    cap.on_exit("dec-1", order_id="ord-10", qty=2, exit_price=1.30,
                reason="take_profit", gross_pnl_pct=22.6, net_pnl_pct=18.0)
    rec = led.get_decision("dec-1")
    types = [e["event_type"] for e in rec["events"]]
    for t in ("submitted", "fill", "exit", "outcome"):
        if t not in types:
            print("FAIL: missing lifecycle event", t, types); ok = False
    pred = rec["snapshots"][0]["prediction_json"]
    if '"stop_loss_pct": 0.25' not in pred:
        print("FAIL: exit policy frozen in prediction", pred); ok = False

    # A rejected candidate records the reason code.
    cap.on_candidate(decision_id="dec-2", accepted=False,
                     reject_code="risk_blocked:daily_loss",
                     symbol="QQQ261116P00400000", direction="PUT")
    if led.get_decision("dec-2")["snapshots"][0]["reject_code"] != "risk_blocked:daily_loss":
        print("FAIL: reject code must persist"); ok = False

    # Broken ledger -> drop counted, never raises.
    class _Boom:
        def record_snapshot(self, **kw):
            raise RuntimeError("disk full")

        def record_event(self, **kw):
            raise RuntimeError("disk full")

    cap2 = ResearchCapture(ledger=_Boom(), enabled=True)
    if cap2.on_candidate(decision_id="z", accepted=True) is not None:
        print("FAIL: failing write must return None"); ok = False
    cap2.on_fill("z", "o", qty=1, price=1.0)
    if cap2.dropped.get("on_candidate") != 1 or cap2.dropped.get("on_fill") != 1:
        print("FAIL: drops must be counted", cap2.dropped); ok = False

    led.close()
    print("research_capture self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
