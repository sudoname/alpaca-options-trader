"""
Oracle Decision Kernel — schema (frozen inputs + immutable Decision).

These dataclasses are the *contract* between the decision kernel and everything
around it. They are deliberately transport-agnostic: a live scan, a paper scan
and a historical replay all populate the SAME ``Snapshot`` shape, so a Decision
produced from a frozen historical snapshot is comparable, field-for-field, to
one produced live. That comparability is the acceptance test for Upgrade 1
(``decision_live == decision_backtest``).

Naming follows Oracle's own vocabulary (p_call / p_put / p_no_trade from
``oracle_voting``; expected_move / implied_move / move_edge; theoretical_ev /
pop from ``entry_ev_stamp``; conviction from ``oracle/conviction.py``). We do
not invent new probabilities or rename existing ones to fit the interface.

Determinism / equality:
  * ``Decision`` compares by a canonical JSON *fingerprint* (sorted keys,
    rounded floats) rather than object identity, so two decisions built from
    equal inputs on different code paths are ``==``. ``fingerprint()`` is the
    stable string used by the parity tests and by any dedup ledger.
  * Every dataclass here is frozen. Mutable inputs (ctx / contract dicts) are
    normalized to plain dicts on the way in via the ``make`` helpers.

Nothing here does I/O. Fail-open: the ``make`` helpers coerce junk to safe
defaults and never raise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Version tags stamped onto every Decision so a sweep / ledger can bucket by the
# code + config generation that produced it. Bump MODEL_VERSION when the tally
# or head math changes; CONFIG_VERSION is derived from the DecisionConfig.
ORACLE_MODEL_VERSION = "oracle-3.0-kernel-1"

# Canonical action / direction vocabularies (closed sets; anything else is
# coerced to the safe abstain state).
ACTION_ENTER = "enter"
ACTION_SKIP = "skip"          # a directional signal formed but a gate vetoed it
ACTION_NO_TRADE = "no_trade"  # the system chose to abstain (no clean direction)
_ACTIONS = (ACTION_ENTER, ACTION_SKIP, ACTION_NO_TRADE)

DIR_CALL = "call"
DIR_PUT = "put"
_DIRECTIONS = (DIR_CALL, DIR_PUT)


# --------------------------------------------------------------------------- #
# Coercion helpers (fail-open; never raise)
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:  # pragma: no cover
        return None


def _plain_dict(value: Any) -> Dict[str, Any]:
    """Return a shallow plain-dict copy of a mapping, else empty dict."""
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _tuple_of_dicts(value: Any) -> Tuple[Dict[str, Any], ...]:
    if isinstance(value, (str, bytes, dict)) or value is None:
        if isinstance(value, dict):
            return (_plain_dict(value),)
        return ()
    try:
        return tuple(_plain_dict(v) for v in value if isinstance(v, Mapping))
    except TypeError:
        return ()


def _tuple_of(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return ()


def _canonical(obj: Any) -> Any:
    """Recursively canonicalize for a stable fingerprint: sort dict keys, round
    floats to 6 dp, turn tuples into lists. Deterministic and json-safe."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    return obj


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Snapshot:
    """Point-in-time inputs to a single decision. Populated identically by the
    live scan and by a historical replay. ``ctx`` is the evidence context the
    Oracle head reads (the dict ``explain_context.build_explain_context`` /
    ``_build_evidence_ctx`` produce). ``prices`` is the recent underlying close
    series the direction tally consumes."""

    timestamp: str
    symbol: str
    strategy_mode: str = "intraday"
    ctx: Dict[str, Any] = field(default_factory=dict)
    prices: Tuple[float, ...] = ()
    momentum: Optional[float] = None
    volatility: Optional[float] = None
    market_regime: Optional[str] = None
    news: Tuple[Any, ...] = ()
    contracts: Tuple[Dict[str, Any], ...] = ()
    quote: Optional[Dict[str, Any]] = None
    model_version: str = ORACLE_MODEL_VERSION

    @staticmethod
    def make(symbol: str, timestamp: str, *, strategy_mode: str = "intraday",
             ctx: Optional[Mapping[str, Any]] = None,
             prices: Sequence[float] = (),
             momentum: Any = None, volatility: Any = None,
             market_regime: Any = None, news: Sequence[Any] = (),
             contracts: Sequence[Mapping[str, Any]] = (),
             quote: Optional[Mapping[str, Any]] = None,
             model_version: str = ORACLE_MODEL_VERSION) -> "Snapshot":
        prices_t = tuple(
            f for f in (_to_float(p) for p in _tuple_of(prices)) if f is not None
        )
        return Snapshot(
            timestamp=_to_str(timestamp) or "",
            symbol=(_to_str(symbol) or "").upper(),
            strategy_mode=_to_str(strategy_mode) or "intraday",
            ctx=_plain_dict(ctx),
            prices=prices_t,
            momentum=_to_float(momentum),
            volatility=_to_float(volatility),
            market_regime=_to_str(market_regime),
            news=_tuple_of(news),
            contracts=_tuple_of_dicts(contracts),
            quote=(_plain_dict(quote) or None),
            model_version=_to_str(model_version) or ORACLE_MODEL_VERSION,
        )


@dataclass(frozen=True)
class PortfolioState:
    """The account-level facts the kernel needs to *size* and *veto* — never to
    pick direction. Buying power and existing exposure gate the trade; open
    symbols drive the duplicate guard. All optional / fail-open."""

    buying_power: Optional[float] = None
    open_symbols: Tuple[str, ...] = ()
    net_delta: Optional[float] = None
    net_gamma: Optional[float] = None
    net_theta: Optional[float] = None
    net_vega: Optional[float] = None
    position_count: int = 0

    @staticmethod
    def make(*, buying_power: Any = None,
             open_symbols: Sequence[str] = (),
             net_delta: Any = None, net_gamma: Any = None,
             net_theta: Any = None, net_vega: Any = None,
             position_count: Any = 0) -> "PortfolioState":
        syms = tuple(
            s for s in ((_to_str(x) or "").upper() for x in _tuple_of(open_symbols)) if s
        )
        return PortfolioState(
            buying_power=_to_float(buying_power),
            open_symbols=syms,
            net_delta=_to_float(net_delta),
            net_gamma=_to_float(net_gamma),
            net_theta=_to_float(net_theta),
            net_vega=_to_float(net_vega),
            position_count=_to_int(position_count),
        )


@dataclass(frozen=True)
class StrategyState:
    """Per-symbol strategy memory that legitimately influences a decision
    (cooldowns, last direction, consecutive losses). Distinct from PortfolioState
    so a replay can hold it fixed. All optional / fail-open."""

    last_direction: Optional[str] = None
    cooldown_until: Optional[str] = None
    consecutive_losses: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(*, last_direction: Any = None, cooldown_until: Any = None,
             consecutive_losses: Any = 0,
             extras: Optional[Mapping[str, Any]] = None) -> "StrategyState":
        return StrategyState(
            last_direction=_to_str(last_direction),
            cooldown_until=_to_str(cooldown_until),
            consecutive_losses=_to_int(consecutive_losses),
            extras=_plain_dict(extras),
        )


@dataclass(frozen=True)
class DecisionConfig:
    """Frozen thresholds + flags + weights the kernel reads. This mirrors the
    ``smart_trader`` ``_flag/_f2/_i2`` values so a live decision and a replay use
    the same knobs. ``params`` is stored as a sorted tuple so equal content
    hashes identically and ``config_version`` is content-derived."""

    params: Tuple[Tuple[str, Any], ...] = ()

    @staticmethod
    def make(params: Optional[Mapping[str, Any]] = None) -> "DecisionConfig":
        p = tuple(sorted((str(k), v) for k, v in _plain_dict(params).items()))
        return DecisionConfig(params=p)

    def get(self, name: str, default: Any = None) -> Any:
        for k, v in self.params:
            if k == name:
                return v
        return default

    def get_float(self, name: str, default: float) -> float:
        f = _to_float(self.get(name))
        return default if f is None else f

    def get_int(self, name: str, default: int) -> int:
        return _to_int(self.get(name, default), default)

    def get_bool(self, name: str, default: bool = False) -> bool:
        v = self.get(name, default)
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.params}

    @property
    def config_version(self) -> str:
        """Short stable digest of the param set (deterministic across runs)."""
        blob = json.dumps(_canonical(self.as_dict()), sort_keys=True,
                          separators=(",", ":"), default=str)
        # djb2 — small, dependency-free, stable across interpreters.
        h = 5381
        for ch in blob:
            h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
        return f"cfg-{h:08x}"


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Decision:
    """Immutable result of one decision. Equality/hash are by canonical
    fingerprint (not identity), so a Decision from the live path equals a
    Decision from the backtest path when the inputs match."""

    timestamp: str
    symbol: str
    strategy_mode: str
    action: str                          # enter | skip | no_trade
    direction: Optional[str]             # call | put | None
    p_call: Optional[float] = None
    p_put: Optional[float] = None
    p_no_trade: Optional[float] = None
    expected_move: Optional[float] = None
    implied_move: Optional[float] = None
    move_edge: Optional[float] = None
    selected_contract: Optional[Dict[str, Any]] = None
    theoretical_ev: Optional[float] = None
    pop: Optional[float] = None
    conviction: Optional[float] = None
    size: int = 0
    invalidation: Optional[float] = None
    vetoes: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    model_version: str = ORACLE_MODEL_VERSION
    config_version: str = ""

    # -- canonical form / equality ---------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["vetoes"] = list(self.vetoes)
        d["reasons"] = list(self.reasons)
        return d

    def fingerprint(self) -> str:
        """Stable JSON digest used for equality, dedup and parity tests.

        Decision *identity* depends only on the semantic decision fields. The
        recorded ``selected_contract`` may carry incidental metadata that a
        live feed or a storage row decorates it with (e.g. ``_row_id``, a fetch
        timestamp, a source tag); such keys are excluded from the fingerprint so
        that ``decision_live == decision_backtest`` holds whenever the FACTS
        match. Metadata is identified by the leading-underscore convention. The
        full contract is preserved in ``to_dict()`` / the stored record."""
        d = self.to_dict()
        sc = d.get("selected_contract")
        if isinstance(sc, Mapping):
            d = dict(d)
            d["selected_contract"] = {
                k: v for k, v in sc.items() if not str(k).startswith("_")
            }
        return json.dumps(_canonical(d), sort_keys=True,
                          separators=(",", ":"), default=str)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Decision):
            return NotImplemented
        return self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def is_actionable(self) -> bool:
        return self.action == ACTION_ENTER and self.direction in _DIRECTIONS

    def with_veto(self, veto: str) -> "Decision":
        """Return a copy demoted to SKIP with ``veto`` appended (kernel gates
        use this so a directional decision can be vetoed without losing its
        provenance). Never raises."""
        v = _to_str(veto) or "veto"
        return replace(self, action=ACTION_SKIP, size=0,
                       vetoes=self.vetoes + (v,))


def make_no_trade(symbol: str, timestamp: str, strategy_mode: str,
                  *, reasons: Sequence[str] = (),
                  p_no_trade: Optional[float] = None,
                  config_version: str = "",
                  model_version: str = ORACLE_MODEL_VERSION) -> Decision:
    """Canonical abstain Decision. The system is ALWAYS allowed to NO-TRADE."""
    return Decision(
        timestamp=_to_str(timestamp) or "",
        symbol=(_to_str(symbol) or "").upper(),
        strategy_mode=_to_str(strategy_mode) or "intraday",
        action=ACTION_NO_TRADE,
        direction=None,
        p_no_trade=_to_float(p_no_trade),
        reasons=tuple(str(r) for r in _tuple_of(reasons)),
        model_version=model_version,
        config_version=config_version,
    )


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Coercion / fail-open on junk.
    if Snapshot.make("aapl", "2024-01-02T16:00:00",
                     prices=["x", 1, None, 2.5]).prices != (1.0, 2.5):
        print("FAIL: prices coercion"); ok = False
    if Snapshot.make("AAA", "t", contracts=42).contracts != ():
        print("FAIL: junk contracts should be empty"); ok = False
    if Snapshot.make("AAA", "t", ctx=None).ctx != {}:
        print("FAIL: None ctx should be empty dict"); ok = False

    # Symbol is upper-cased; mode defaults.
    s = Snapshot.make("spy", "t")
    if s.symbol != "SPY" or s.strategy_mode != "intraday":
        print("FAIL: symbol/mode defaults", s.symbol, s.strategy_mode); ok = False

    # DecisionConfig content-based version + typed getters.
    c1 = DecisionConfig.make({"a": 1, "b": 2.0})
    c2 = DecisionConfig.make({"b": 2.0, "a": 1})
    if c1 != c2 or c1.config_version != c2.config_version:
        print("FAIL: config equality/version"); ok = False
    if c1.get_float("b", 9.0) != 2.0 or c1.get_int("a", 0) != 1:
        print("FAIL: config typed getters"); ok = False
    if c1.get_bool("missing", True) is not True:
        print("FAIL: config bool default"); ok = False

    # Decision fingerprint equality across independent construction (this is the
    # decision_live == decision_backtest invariant in miniature).
    d1 = Decision(timestamp="t", symbol="SPY", strategy_mode="intraday",
                  action=ACTION_ENTER, direction=DIR_CALL, p_call=0.6,
                  p_put=0.3, p_no_trade=0.1, theoretical_ev=12.3456789,
                  selected_contract={"strike": 500, "type": "call"}, size=2,
                  config_version="cfg-x")
    d2 = Decision(timestamp="t", symbol="SPY", strategy_mode="intraday",
                  action=ACTION_ENTER, direction=DIR_CALL, p_call=0.6,
                  p_put=0.3, p_no_trade=0.1,
                  theoretical_ev=12.3456789 + 1e-9,   # within float rounding
                  selected_contract={"type": "call", "strike": 500}, size=2,
                  config_version="cfg-x")
    if d1 != d2:
        print("FAIL: decision fingerprint equality"); ok = False
    if hash(d1) != hash(d2):
        print("FAIL: decision hash equality"); ok = False
    if len({d1, d2}) != 1:
        print("FAIL: decisions should dedupe in a set"); ok = False

    # A real difference must break equality.
    d3 = replace(d1, direction=DIR_PUT)
    if d1 == d3:
        print("FAIL: differing decisions compare equal"); ok = False

    # with_veto demotes to skip and records provenance without mutating d1.
    dv = d1.with_veto("risk_engine")
    if dv.action != ACTION_SKIP or dv.size != 0 or "risk_engine" not in dv.vetoes:
        print("FAIL: with_veto demotion"); ok = False
    if d1.action != ACTION_ENTER:
        print("FAIL: with_veto mutated original"); ok = False
    if d1.is_actionable() is not True or make_no_trade("x", "t", "intraday").is_actionable():
        print("FAIL: is_actionable"); ok = False

    # make_no_trade is always a valid abstain.
    nt = make_no_trade("qqq", "t", "swing", reasons=["no clean direction"])
    if nt.action != ACTION_NO_TRADE or nt.direction is not None:
        print("FAIL: make_no_trade shape"); ok = False

    print("decision.schema self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
