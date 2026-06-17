"""
Phase 11A-1 — Triple Gap (analytics only, pure, no I/O, no network).

The Triple Gap is the combined disagreement between Oracle's model and the
market, across three independent axes:

    vol_gap  = market_iv             - forecast_vol
    move_gap = market_expected_move  - oracle_expected_move
    ev_gap   = oracle_expected_value - market_neutral_expected_value

A larger gap means Oracle and the market disagree more, which is where an edge
(if any) must live. Each raw gap is normalised to a 0-100 component score and
combined into a single ``triple_gap_score`` via configurable weights
(default 0.30 / 0.30 / 0.40). The score is later bucketed and compared to
realised candidate outcomes (calibration_reports.py) to test whether
disagreement actually predicts profitability.

``market_neutral_expected_value`` is rarely available today, so when it is
missing we fall back to a ``0.0`` baseline and tag ``ev_gap_source =
"zero_baseline"`` (vs ``"model_baseline"`` when a real value is supplied) so
downstream analytics can tell the two apart.

STRICTLY analytics: this module only computes numbers from numbers. It never
touches the network, never reads or writes a file, and never influences any
real or paper trade. Functions never raise on bad input — they return a
``TripleGapResult`` tagged ``insufficient_data`` instead.
"""

from dataclasses import dataclass, asdict
from typing import Optional

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_data"

EV_GAP_MODEL_BASELINE = "model_baseline"
EV_GAP_ZERO_BASELINE = "zero_baseline"

# Full-scale constants for normalising each raw gap to 0-100. A gap at (or
# beyond) full scale maps to a component score of 100; smaller gaps scale
# linearly. Chosen as documented, conservative defaults — they are pure
# constants, not tunables, so the score stays comparable across runs.
VOL_GAP_FULL_SCALE = 0.10   # 0.10 = 10 vol points of IV-vs-forecast disagreement
MOVE_GAP_FULL_SCALE = 1.0   # fallback $ scale when oracle_expected_move is absent
EV_GAP_FULL_SCALE = 50.0    # $50 of Oracle-over-market EV edge = full score


def _to_float(value) -> Optional[float]:
    """Tolerant float coercion (None on anything non-numeric)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class TripleGapConfig:
    enabled: bool = True
    vol_weight: float = 0.30
    move_weight: float = 0.30
    ev_weight: float = 0.40

    @staticmethod
    def from_env(path: str = ".env") -> "TripleGapConfig":
        """Resolve config the project way (shell > .env > default). Fail-open."""
        try:
            from config_loader import ConfigLoader
            cfg = ConfigLoader(path=path)
            return TripleGapConfig(
                enabled=cfg.get_bool("USE_TRIPLE_GAP", True),
                vol_weight=cfg.get_float("TRIPLE_GAP_VOL_WEIGHT", 0.30),
                move_weight=cfg.get_float("TRIPLE_GAP_MOVE_WEIGHT", 0.30),
                ev_weight=cfg.get_float("TRIPLE_GAP_EV_WEIGHT", 0.40),
            )
        except Exception:
            return TripleGapConfig()


@dataclass
class TripleGapResult:
    symbol: str = ""
    strategy: str = ""
    vol_gap: Optional[float] = None
    move_gap: Optional[float] = None
    ev_gap: Optional[float] = None
    vol_gap_score: Optional[float] = None
    move_gap_score: Optional[float] = None
    ev_gap_score: Optional[float] = None
    triple_gap_score: Optional[float] = None
    ev_gap_source: str = EV_GAP_ZERO_BASELINE
    status: str = STATUS_INSUFFICIENT
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Component normalisers (each pure; None in -> None out)
# ---------------------------------------------------------------------------
def normalize_vol_gap(vol_gap: Optional[float]) -> Optional[float]:
    """|IV - forecast| scaled by VOL_GAP_FULL_SCALE, clamped to 0-100."""
    if vol_gap is None:
        return None
    return round(_clamp01(abs(vol_gap) / VOL_GAP_FULL_SCALE) * 100.0, 1)


def normalize_move_gap(move_gap: Optional[float],
                       oracle_expected_move: Optional[float] = None
                       ) -> Optional[float]:
    """|market move - oracle move| as a fraction of the oracle move (so it is
    scale-free per underlying); falls back to a fixed $ scale when the oracle
    move is missing/non-positive. Clamped to 0-100."""
    if move_gap is None:
        return None
    denom = oracle_expected_move if (oracle_expected_move is not None
                                     and oracle_expected_move > 0) else None
    scale = denom if denom is not None else MOVE_GAP_FULL_SCALE
    if scale <= 0:
        return None
    return round(_clamp01(abs(move_gap) / scale) * 100.0, 1)


def normalize_ev_gap(ev_gap: Optional[float]) -> Optional[float]:
    """Oracle-over-market EV edge scaled by EV_GAP_FULL_SCALE, clamped to
    0-100. Floored at 0: only a POSITIVE edge (Oracle thinks the trade is
    worth more than the market-neutral baseline) earns score."""
    if ev_gap is None:
        return None
    return round(_clamp01(ev_gap / EV_GAP_FULL_SCALE) * 100.0, 1)


def _weighted_score(components, weights) -> Optional[float]:
    """Weighted average over the components that are present; weights
    renormalise across whichever components survived. None when none survive."""
    num = 0.0
    den = 0.0
    for value, weight in zip(components, weights):
        if value is None:
            continue
        w = max(0.0, float(weight))
        num += w * value
        den += w
    if den <= 0:
        return None
    return round(num / den, 1)


def compute_triple_gap(symbol: str = "",
                       strategy: str = "",
                       market_iv: Optional[float] = None,
                       forecast_vol: Optional[float] = None,
                       market_expected_move: Optional[float] = None,
                       oracle_expected_move: Optional[float] = None,
                       oracle_expected_value: Optional[float] = None,
                       market_neutral_expected_value: Optional[float] = None,
                       config: Optional[TripleGapConfig] = None
                       ) -> TripleGapResult:
    """Compute the three gaps, their 0-100 component scores and the weighted
    ``triple_gap_score``. Never raises; missing inputs simply drop out and the
    surviving weights renormalise. Returns ``insufficient_data`` when no
    component can be computed."""
    cfg = config or TripleGapConfig()
    res = TripleGapResult(symbol=str(symbol or ""), strategy=str(strategy or ""))

    iv = _to_float(market_iv)
    fv = _to_float(forecast_vol)
    mkt_move = _to_float(market_expected_move)
    ora_move = _to_float(oracle_expected_move)
    ora_ev = _to_float(oracle_expected_value)
    mkt_ev = _to_float(market_neutral_expected_value)

    # Raw gaps (only when both operands exist).
    if iv is not None and fv is not None:
        res.vol_gap = round(iv - fv, 6)
    if mkt_move is not None and ora_move is not None:
        res.move_gap = round(mkt_move - ora_move, 6)
    if ora_ev is not None:
        if mkt_ev is None:
            res.ev_gap = round(ora_ev - 0.0, 6)
            res.ev_gap_source = EV_GAP_ZERO_BASELINE
        else:
            res.ev_gap = round(ora_ev - mkt_ev, 6)
            res.ev_gap_source = EV_GAP_MODEL_BASELINE

    # Component scores.
    res.vol_gap_score = normalize_vol_gap(res.vol_gap)
    res.move_gap_score = normalize_move_gap(res.move_gap, ora_move)
    res.ev_gap_score = normalize_ev_gap(res.ev_gap)

    res.triple_gap_score = _weighted_score(
        (res.vol_gap_score, res.move_gap_score, res.ev_gap_score),
        (cfg.vol_weight, cfg.move_weight, cfg.ev_weight),
    )

    if res.triple_gap_score is None:
        res.status = STATUS_INSUFFICIENT
        res.reason = "no gap component computable from supplied signals"
    else:
        res.status = STATUS_OK
    return res


# ---------------------------------------------------------------------------
# Self-test (no creds, no network)
# ---------------------------------------------------------------------------
def _self_test() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print("FAIL:", msg)
            ok = False

    # Raw gap math + signs.
    r = compute_triple_gap(
        symbol="SPY", strategy="debit_call_spread",
        market_iv=0.25, forecast_vol=0.20,
        market_expected_move=12.0, oracle_expected_move=10.0,
        oracle_expected_value=25.0)
    check(abs(r.vol_gap - 0.05) < 1e-9, f"vol_gap should be 0.05, got {r.vol_gap}")
    check(abs(r.move_gap - 2.0) < 1e-9, f"move_gap should be 2.0, got {r.move_gap}")
    check(abs(r.ev_gap - 25.0) < 1e-9, f"ev_gap should be 25.0, got {r.ev_gap}")
    check(r.ev_gap_source == EV_GAP_ZERO_BASELINE, "missing baseline -> zero_baseline")
    check(r.status == STATUS_OK, f"status should be ok, got {r.status}")

    # Component normalisation: vol 0.05/0.10 = 50; move 2/10 = 20; ev 25/50 = 50.
    check(r.vol_gap_score == 50.0, f"vol score 50, got {r.vol_gap_score}")
    check(r.move_gap_score == 20.0, f"move score 20, got {r.move_gap_score}")
    check(r.ev_gap_score == 50.0, f"ev score 50, got {r.ev_gap_score}")
    # Weighted: (0.30*50 + 0.30*20 + 0.40*50) / 1.0 = 41.0.
    check(r.triple_gap_score == 41.0, f"triple 41.0, got {r.triple_gap_score}")

    # Clamping: gaps beyond full scale cap at 100.
    big = compute_triple_gap(market_iv=1.0, forecast_vol=0.0,
                             oracle_expected_value=10_000.0)
    check(big.vol_gap_score == 100.0, "vol score clamps to 100")
    check(big.ev_gap_score == 100.0, "ev score clamps to 100")

    # EV gap floored at 0 for a negative edge.
    neg = compute_triple_gap(oracle_expected_value=-30.0)
    check(neg.ev_gap_score == 0.0, f"negative ev edge floors at 0, got {neg.ev_gap_score}")

    # Real (model) baseline tag.
    base = compute_triple_gap(oracle_expected_value=20.0,
                              market_neutral_expected_value=5.0)
    check(base.ev_gap == 15.0, f"ev_gap should be 15.0, got {base.ev_gap}")
    check(base.ev_gap_source == EV_GAP_MODEL_BASELINE, "supplied baseline -> model_baseline")

    # Weight renormalisation: only vol present -> triple == vol score.
    only_vol = compute_triple_gap(market_iv=0.30, forecast_vol=0.20)
    check(only_vol.move_gap_score is None, "move score None when no move inputs")
    check(only_vol.ev_gap_score is None, "ev score None when no ev input")
    check(only_vol.triple_gap_score == only_vol.vol_gap_score,
          "single-component triple == that component")

    # Custom weights renormalise correctly (vol+ev present, move missing).
    cfg = TripleGapConfig(vol_weight=0.30, move_weight=0.30, ev_weight=0.40)
    mix = compute_triple_gap(market_iv=0.30, forecast_vol=0.20,
                             oracle_expected_value=25.0, config=cfg)
    # vol=100? 0.10/0.10=100; ev=50. (0.30*100 + 0.40*50)/(0.70) = 50/0.7=71.4
    check(mix.vol_gap_score == 100.0, f"vol score 100, got {mix.vol_gap_score}")
    check(mix.triple_gap_score == 71.4,
          f"renormalised triple should be 71.4, got {mix.triple_gap_score}")

    # No inputs -> insufficient.
    empty = compute_triple_gap()
    check(empty.status == STATUS_INSUFFICIENT, "empty -> insufficient_data")
    check(empty.triple_gap_score is None, "empty -> no triple score")

    print("triple_gap self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
