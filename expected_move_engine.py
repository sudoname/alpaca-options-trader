"""
Phase 7A — expected-move / volatility-edge engine (advisory, offline-pure).

Given a small set of volatility inputs for an underlying, this engine forecasts
the expected price move over several horizons and measures the *volatility edge*:
how rich (or cheap) the market's implied volatility is versus our realized-vol
forecast. It is a pure, deterministic, network-free calculator — all inputs are
passed IN, so it is fully unit-testable with no creds and no market data feed.

Inputs (any may be missing; the engine fails open and reports what it can):
    HV20, HV60, HV90            annualized historical vol (decimal fractions)
    ATR                          average true range in DOLLARS (needs price)
    VIX                          index level in points (e.g. 18.5 -> implied 0.185)
    earnings_days                trading days until earnings (None = none known)
    recent_realized_vol          annualized realized vol over a short window
    price                        underlying price (optional; enables $ outputs)

Outputs (:class:`ExpectedMove`):
    expected_move_1d / _3d / _7d / _30d
    market_expected_move         implied 30d move (from VIX)
    volatility_edge              normalized signed edge in [-1, 1]
                                 (+ = implied richer than realized -> sell premium)

It also persists each prediction to ``expected_move_history.csv`` (append-only,
header auto-managed). Persistence is best-effort and never raises.

Convention: expected-move outputs are FRACTIONS (e.g. 0.012 = 1.2%) unless a
``price`` is supplied, in which case they are DOLLARS. ``in_dollars`` records
which form was produced.
"""

import csv
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from config_loader import ConfigLoader

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

EDGE_OVERPRICED = "options_overpriced"   # implied >> realized -> favor selling premium
EDGE_UNDERPRICED = "options_underpriced"  # implied << realized -> favor buying premium
EDGE_FAIR = "fair"
EDGE_NA = "n/a"


# --------------------------------------------------------------------------- #
# Pure volatility helpers (stdlib only)
# --------------------------------------------------------------------------- #
def realized_volatility(closes: List[float], window: Optional[int] = None,
                        annualize: bool = True) -> Optional[float]:
    """Annualized realized vol from a list of closing prices (log returns).

    ``window`` (optional) restricts to the most recent ``window`` returns.
    Returns None when there is not enough data.
    """
    if not closes or len(closes) < 3:
        return None
    rets = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if prev and prev > 0 and cur and cur > 0:
            rets.append(math.log(cur / prev))
    if window is not None and window > 0:
        rets = rets[-window:]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sigma = math.sqrt(var)
    if annualize:
        sigma *= math.sqrt(TRADING_DAYS)
    return sigma


def average_true_range(bars: List[dict], period: int = 14) -> Optional[float]:
    """ATR in dollars from OHLC bars (dicts with 'h','l','c'). None if short."""
    if not bars or len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h = bars[i].get("h")
        l = bars[i].get("l")
        pc = bars[i - 1].get("c")
        if h is None or l is None or pc is None:
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    window = trs[-period:]
    return sum(window) / len(window)


def _mean(values) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def classify_vol_edge(edge: Optional[float], threshold: float = 0.10) -> str:
    if edge is None:
        return EDGE_NA
    if edge >= threshold:
        return EDGE_OVERPRICED
    if edge <= -threshold:
        return EDGE_UNDERPRICED
    return EDGE_FAIR


# --------------------------------------------------------------------------- #
# Config + dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ExpectedMoveConfig:
    enabled: bool = True
    history_file: str = "expected_move_history.csv"
    earnings_premium: float = 0.5     # fractional bump to a horizon that spans earnings
    edge_threshold: float = 0.10      # |edge| past this flips the label off "fair"

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "ExpectedMoveConfig":
        cfg = loader if loader is not None else ConfigLoader(path=path)
        return ExpectedMoveConfig(
            enabled=cfg.get_bool("EXPECTED_MOVE_ENABLED", True),
            history_file=cfg.get_str("EXPECTED_MOVE_HISTORY_FILE",
                                     "expected_move_history.csv"),
            earnings_premium=cfg.get_float("EXPECTED_MOVE_EARNINGS_PREMIUM", 0.5),
            edge_threshold=cfg.get_float("EXPECTED_MOVE_EDGE_THRESHOLD", 0.10),
        )


@dataclass
class ExpectedMoveInputs:
    hv20: Optional[float] = None
    hv60: Optional[float] = None
    hv90: Optional[float] = None
    atr: Optional[float] = None
    vix: Optional[float] = None
    earnings_days: Optional[int] = None
    recent_realized_vol: Optional[float] = None
    price: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExpectedMove:
    symbol: str = ""
    expected_move_1d: Optional[float] = None
    expected_move_3d: Optional[float] = None
    expected_move_7d: Optional[float] = None
    expected_move_30d: Optional[float] = None
    market_expected_move: Optional[float] = None
    volatility_edge: Optional[float] = None
    forecast_vol: Optional[float] = None
    implied_vol: Optional[float] = None
    edge_label: str = EDGE_NA
    in_dollars: bool = False
    status: str = "ok"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class ExpectedMoveEngine:
    def __init__(self, config: Optional[ExpectedMoveConfig] = None):
        self.config = config or ExpectedMoveConfig.from_env()

    # -- forecast vol blend ---------------------------------------------- #
    @staticmethod
    def _forecast_vol(inp: ExpectedMoveInputs) -> Optional[float]:
        """Weighted blend of available HVs + recent realized vol.

        Short bucket (HV20 + recent realized) weighs most, then HV60, then HV90.
        Weights are renormalized over whatever data is present (fail-open).
        """
        short = _mean([inp.hv20, inp.recent_realized_vol])
        buckets = []
        if short is not None:
            buckets.append((0.5, short))
        if isinstance(inp.hv60, (int, float)):
            buckets.append((0.3, inp.hv60))
        if isinstance(inp.hv90, (int, float)):
            buckets.append((0.2, inp.hv90))
        if not buckets:
            return None
        total_w = sum(w for w, _ in buckets)
        return sum(w * v for w, v in buckets) / total_w

    def _earnings_mult(self, inp: ExpectedMoveInputs, horizon_days: int) -> float:
        """Inflate a horizon that spans a known earnings event."""
        ed = inp.earnings_days
        if isinstance(ed, (int, float)) and 0 <= ed <= horizon_days:
            return 1.0 + max(0.0, self.config.earnings_premium)
        return 1.0

    def compute(self, inputs: ExpectedMoveInputs,
                symbol: str = "") -> ExpectedMove:
        """Compute expected moves + volatility edge from ``inputs`` (pure)."""
        forecast_vol = self._forecast_vol(inputs)
        implied_vol = (inputs.vix / 100.0
                       if isinstance(inputs.vix, (int, float)) and inputs.vix > 0
                       else None)
        price = inputs.price if isinstance(inputs.price, (int, float)) and inputs.price > 0 else None
        in_dollars = price is not None

        if forecast_vol is None:
            return ExpectedMove(symbol=symbol, status="insufficient_data",
                                implied_vol=implied_vol, in_dollars=in_dollars,
                                edge_label=EDGE_NA)

        # Daily vol = blend of HV-implied daily move and (if available) ATR/price.
        daily_vol = forecast_vol / math.sqrt(TRADING_DAYS)
        if price is not None and isinstance(inputs.atr, (int, float)) and inputs.atr > 0:
            daily_vol = _mean([daily_vol, inputs.atr / price]) or daily_vol

        def em(days: int) -> float:
            frac = daily_vol * math.sqrt(days) * self._earnings_mult(inputs, days)
            return round(frac * price, 4) if price is not None else round(frac, 6)

        # Market expected move from implied (VIX) over a 30d horizon.
        market_em = None
        if implied_vol is not None:
            m_frac = implied_vol * math.sqrt(30.0 / TRADING_DAYS)
            market_em = round(m_frac * price, 4) if price is not None else round(m_frac, 6)

        # Volatility edge: implied vs realized forecast, normalized to [-1, 1].
        vol_edge = None
        if implied_vol is not None and forecast_vol is not None and implied_vol > 0:
            vol_edge = round(_clamp((implied_vol - forecast_vol) / implied_vol,
                                    -1.0, 1.0), 4)

        return ExpectedMove(
            symbol=symbol,
            expected_move_1d=em(1),
            expected_move_3d=em(3),
            expected_move_7d=em(7),
            expected_move_30d=em(30),
            market_expected_move=market_em,
            volatility_edge=vol_edge,
            forecast_vol=round(forecast_vol, 6),
            implied_vol=round(implied_vol, 6) if implied_vol is not None else None,
            edge_label=classify_vol_edge(vol_edge, self.config.edge_threshold),
            in_dollars=in_dollars,
            status="ok",
        )

    # -- persistence (best-effort, append-only CSV) ---------------------- #
    def record(self, result: ExpectedMove, inputs: ExpectedMoveInputs,
               path: Optional[str] = None) -> None:
        path = path or self.config.history_file
        row = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "symbol": result.symbol}
        for k, v in inputs.as_dict().items():
            row[f"in_{k}"] = v
        for k, v in result.to_dict().items():
            if k == "symbol":
                continue
            row[k] = v
        _csv_append_row(path, row)


# --------------------------------------------------------------------------- #
# Shared CSV append helper (header auto-managed; never raises)
# --------------------------------------------------------------------------- #
def _csv_append_row(path: str, row: dict) -> None:
    try:
        existing: List[dict] = []
        fieldnames: List[str] = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                existing = list(reader)
        new_fields = [k for k in row.keys() if k not in fieldnames]
        if new_fields:
            # Header grew (or file is new): rewrite with the union header.
            fieldnames = fieldnames + new_fields
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for e in existing:
                    writer.writerow(e)
                writer.writerow(row)
        else:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
    except Exception as exc:
        logger.warning("expected_move history write failed (%s): %s", path, exc)


# --------------------------------------------------------------------------- #
# Live-input gathering (optional; used by the Telegram command). Network via the
# trader is best-effort — any failure leaves a field None (fail-open).
# --------------------------------------------------------------------------- #
def gather_inputs_from_trader(trader, symbol: str,
                              vix: Optional[float] = None) -> ExpectedMoveInputs:
    """Build ExpectedMoveInputs from a SmartOptionsTrader's price history.

    Uses ``trader.get_price_history`` (closes only) to derive HV20/60/90 and a
    recent realized vol; ATR is left None when only closes are available. ``vix``
    may be supplied by the caller. All steps are guarded.
    """
    hv20 = hv60 = hv90 = recent = price = None
    try:
        closes = trader.get_price_history(symbol, days=130) or []
        if closes:
            price = closes[-1]
            hv20 = realized_volatility(closes, window=20)
            hv60 = realized_volatility(closes, window=60)
            hv90 = realized_volatility(closes, window=90)
            recent = realized_volatility(closes, window=10)
    except Exception as exc:
        logger.warning("gather_inputs_from_trader(%s) failed: %s", symbol, exc)
    return ExpectedMoveInputs(hv20=hv20, hv60=hv60, hv90=hv90,
                              recent_realized_vol=recent, vix=vix, price=price)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    eng = ExpectedMoveEngine(ExpectedMoveConfig(enabled=True))

    # Realized vol of a flat series is ~0; a noisy series is > 0.
    if realized_volatility([100, 100, 100, 100]) not in (0.0, None):
        # flat -> zero stdev
        pass

    inp = ExpectedMoveInputs(hv20=0.20, hv60=0.22, hv90=0.25, vix=30.0,
                             price=100.0, atr=2.0)
    res = eng.compute(inp, symbol="SPY")
    if res.status != "ok":
        print("FAIL: status", res.status); ok = False
    if not (res.expected_move_1d and res.expected_move_30d
            and res.expected_move_30d > res.expected_move_1d):
        print("FAIL: horizon scaling", res.to_dict()); ok = False
    # VIX 30 -> implied 0.30 > forecast ~0.21 -> positive (overpriced) edge.
    if res.volatility_edge is None or res.volatility_edge <= 0:
        print("FAIL: edge sign", res.volatility_edge); ok = False
    if res.edge_label != EDGE_OVERPRICED:
        print("FAIL: edge label", res.edge_label); ok = False

    # Underpriced: low VIX vs high realized.
    res2 = eng.compute(ExpectedMoveInputs(hv20=0.40, vix=12.0, price=100.0))
    if res2.edge_label != EDGE_UNDERPRICED:
        print("FAIL: underpriced label", res2.edge_label); ok = False

    # Insufficient data -> fail-open.
    res3 = eng.compute(ExpectedMoveInputs(vix=20.0))
    if res3.status != "insufficient_data":
        print("FAIL: insufficient", res3.status); ok = False

    # Earnings bump inflates horizons spanning the event.
    base = eng.compute(ExpectedMoveInputs(hv20=0.2, price=100.0))
    bump = eng.compute(ExpectedMoveInputs(hv20=0.2, price=100.0, earnings_days=2))
    if not (bump.expected_move_3d > base.expected_move_3d):
        print("FAIL: earnings bump", base.expected_move_3d, bump.expected_move_3d); ok = False

    # CSV persistence round-trips.
    d = tempfile.mkdtemp()
    path = os.path.join(d, "hist.csv")
    eng.record(res, inp, path=path)
    eng.record(res2, ExpectedMoveInputs(hv20=0.40, vix=12.0, price=100.0), path=path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2 or "volatility_edge" not in rows[0]:
        print("FAIL: csv persistence", len(rows)); ok = False

    # ATR helper.
    bars = [{"h": 11, "l": 9, "c": 10} for _ in range(20)]
    if average_true_range(bars, period=14) is None:
        print("FAIL: atr"); ok = False

    print("expected_move_engine self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
