"""Shadow strategy engine: observe-only strategies logged to the research ledger.

Pluggable strategies evaluate the same symbol universe as the live bot every
scan cycle and log their would-be trades as ``kind='shadow_candidate'`` rows
via :class:`research_experiments.ShadowComparator`. They never place orders,
never touch the live decision path, and their outcomes are resolved later as
``class: simulated`` underlying-move counterfactuals
(``research_outcomes.OutcomeReconciler.resolve_shadow``).

Default state: OFF. The runner is only constructed when
``ENABLE_SHADOW_STRATEGIES`` is truthy AND the research ledger is active.

Strategies
----------
* ``simons_meanrev`` — Simons-style cross-sectional mean reversion:
  z = (close - SMA(w)) / STD(w); oversold (z <= -Z_ENTRY) -> CALL,
  overbought (z >= +Z_ENTRY) -> PUT. Among threshold-passers in a cycle only
  the top-K by |z| are accepted; the rest are logged ``shadow:not_top_k``.
* ``soros_momentum`` — Soros-style momentum with volume confirmation:
  |5-day return| > MOM_THRESH and last volume > VOL_MULT x avg(volume, 20),
  with an exhaustion filter (|3-day return| > 2 x daily sigma x sqrt(3)).

Honest limits: shadow outcomes are underlying-move counterfactuals, not
option P/L. Good for measuring directional edge; useless for cost/exit tuning.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional

__all__ = [
    "ShadowStrategy",
    "SimonsMeanReversion",
    "SorosMomentumVolume",
    "ShadowStrategyRunner",
    "build_strategies",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(val) -> Optional[float]:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _sma(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _sma(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _env_float(key: str, default: float) -> float:
    out = _f(os.environ.get(key))
    return out if out is not None else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, ""))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# strategy interface
# ---------------------------------------------------------------------------

class ShadowStrategy:
    """Base class for observe-only strategies.

    ``evaluate(ctx)`` receives a ShadowContext dict:

        {symbol, closes: List[float], volumes: Optional[List[float]],
         underlying_price: float, regime: Optional[str],
         baseline_candidate: Optional[dict]}

    and returns ``None`` (nothing worth logging) or a ledger-ready candidate:

        {accepted: bool, reject_code: Optional[str], symbol, underlying,
         underlying_price, direction: 'CALL'|'PUT', score: float,
         signals: dict, regime, extra: dict}

    ``underlying_price`` is REQUIRED on every returned candidate — the
    counterfactual resolver prices outcomes from it.
    """

    name: str = "base"
    variant: str = "base"

    def __init__(self, params: Optional[Dict] = None):
        self.params: Dict = dict(params or {})

    def evaluate(self, ctx: Dict) -> Optional[Dict]:  # pragma: no cover
        raise NotImplementedError

    # shared scaffolding ---------------------------------------------------
    def _candidate(self, ctx: Dict, *, direction: str, score: float,
                   signals: Dict, accepted: bool = True,
                   reject_code: Optional[str] = None) -> Dict:
        return {
            "accepted": accepted,
            "reject_code": reject_code,
            "symbol": ctx["symbol"],
            "underlying": ctx["symbol"],
            "underlying_price": ctx["underlying_price"],
            "direction": direction,
            "score": float(score),
            "signals": signals,
            "regime": ctx.get("regime"),
            "extra": {"strategy": self.name},
        }


class SimonsMeanReversion(ShadowStrategy):
    """Cross-sectional mean reversion on daily closes (z-score of last close).

    Per-symbol: emits a threshold-passing candidate when |z| >= z_entry
    (oversold -> CALL, overbought -> PUT); |z| < 0.5 is not logged at all.
    The cross-sectional top-K selection is applied by the runner, which
    downgrades passers outside the top-K to ``shadow:not_top_k``.
    """

    name = "simons_mean_reversion"
    variant = "simons_meanrev"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.window = int(self.params.get("window", 20))
        self.z_entry = float(self.params.get("z_entry", 1.5))
        self.top_k = int(self.params.get("top_k", 3))

    def evaluate(self, ctx: Dict) -> Optional[Dict]:
        closes = ctx.get("closes") or []
        if len(closes) < self.window:
            return None
        tail = [c for c in closes[-self.window:] if _f(c) is not None]
        if len(tail) < self.window:
            return None
        last = tail[-1]
        sma = _sma(tail)
        std = _std(tail)
        if std <= 0:
            return None
        z = (last - sma) / std
        if abs(z) < 0.5:
            return None  # noise control: not worth a ledger row
        direction = "CALL" if z < 0 else "PUT"
        signals = {"z": round(z, 4), "sma": round(sma, 4),
                   "std": round(std, 4), "window": self.window}
        cand = self._candidate(ctx, direction=direction, score=abs(z),
                               signals=signals,
                               accepted=abs(z) >= self.z_entry,
                               reject_code=(None if abs(z) >= self.z_entry
                                            else "shadow:below_z_entry"))
        return cand


class SorosMomentumVolume(ShadowStrategy):
    """Momentum with volume confirmation and an exhaustion filter."""

    name = "soros_momentum_volume"
    variant = "soros_momentum"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.mom_thresh = float(self.params.get("mom_thresh", 0.02))
        self.vol_mult = float(self.params.get("vol_mult", 1.2))
        self.vol_window = int(self.params.get("vol_window", 20))

    def evaluate(self, ctx: Dict) -> Optional[Dict]:
        closes = ctx.get("closes") or []
        if len(closes) < 21:
            return None
        c = [x for x in closes if _f(x) is not None]
        if len(c) < 21 or c[-6] <= 0 or c[-4] <= 0:
            return None
        ret5 = c[-1] / c[-6] - 1.0
        if abs(ret5) <= self.mom_thresh:
            return None  # no momentum signal: nothing to log
        ret3 = c[-1] / c[-4] - 1.0
        # daily sigma from the trailing 20 daily returns
        rets = [c[i] / c[i - 1] - 1.0 for i in range(len(c) - 20, len(c))
                if c[i - 1] > 0]
        sigma = _std(rets)
        direction = "CALL" if ret5 > 0 else "PUT"

        volumes = ctx.get("volumes")
        vol_ratio = None
        signals = {"ret5": round(ret5, 5), "ret3": round(ret3, 5),
                   "sigma": round(sigma, 5)}

        if not volumes or len(volumes) < self.vol_window:
            signals["vol_ratio"] = None
            return self._candidate(ctx, direction=direction, score=abs(ret5),
                                   signals=signals, accepted=False,
                                   reject_code="shadow:no_volume_data")

        avg_vol = _sma([v for v in volumes[-self.vol_window:]])
        vol_ratio = (volumes[-1] / avg_vol) if avg_vol > 0 else 0.0
        signals["vol_ratio"] = round(vol_ratio, 4)

        # exhaustion: 3-day move more than 2x the expected 3-day sigma
        # (1e-6 floor so float-noise sigma on near-constant series can't fire)
        if sigma > 1e-6 and abs(ret3) > 2.0 * sigma * math.sqrt(3):
            return self._candidate(ctx, direction=direction, score=abs(ret5),
                                   signals=signals, accepted=False,
                                   reject_code="shadow:exhausted")

        if vol_ratio <= self.vol_mult:
            return self._candidate(ctx, direction=direction, score=abs(ret5),
                                   signals=signals, accepted=False,
                                   reject_code="shadow:weak_volume")

        return self._candidate(ctx, direction=direction, score=abs(ret5),
                               signals=signals, accepted=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class ShadowStrategyRunner:
    """Runs shadow strategies over the scan-cycle universe, fail-open.

    * ``fetch_bars(symbol, days) -> {'closes': [...], 'volumes': [...]}``
    * ``fetch_price(symbol) -> Optional[float]``

    One bars fetch per symbol per cycle, shared across strategies. Every
    decision is submitted through a per-variant
    :class:`research_experiments.ShadowComparator`; the entire cycle is
    wrapped so an error can never reach the caller.
    """

    def __init__(self, ledger, strategies: List[ShadowStrategy],
                 fetch_bars: Callable[[str, int], Dict],
                 fetch_price: Callable[[str], Optional[float]],
                 *, bars_days: int = 40, regime_fn: Optional[Callable] = None):
        from research_experiments import ShadowComparator
        self.strategies = list(strategies)
        self.fetch_bars = fetch_bars
        self.fetch_price = fetch_price
        self.bars_days = int(bars_days)
        self.regime_fn = regime_fn
        self.errors = 0
        self.cycles = 0
        self._comparators = {
            s.variant: ShadowComparator(ledger, variant=s.variant)
            for s in self.strategies
        }

    # ------------------------------------------------------------------
    def run_cycle(self, symbols: List[str],
                  baseline_candidates: Optional[List[Dict]] = None) -> Dict:
        """Evaluate all strategies over ``symbols``. Never raises."""
        summary = {"evaluated": 0, "submitted": 0, "errors": 0}
        try:
            summary = self._run_cycle(symbols, baseline_candidates or [])
        except Exception:
            self.errors += 1
            summary["errors"] += 1
        self.cycles += 1
        return summary

    def _run_cycle(self, symbols: List[str],
                   baseline_candidates: List[Dict]) -> Dict:
        summary = {"evaluated": 0, "submitted": 0, "errors": 0}
        base_by_sym = {}
        for c in baseline_candidates:
            sym = (c or {}).get("sym") or (c or {}).get("symbol")
            if sym:
                base_by_sym[sym] = c

        regime = None
        if self.regime_fn is not None:
            try:
                regime = self.regime_fn()
            except Exception:
                regime = None

        contexts: List[Dict] = []
        for sym in list(dict.fromkeys(symbols or [])):
            try:
                bars = self.fetch_bars(sym, self.bars_days) or {}
                closes = list(bars.get("closes") or [])
                volumes = bars.get("volumes")
                volumes = list(volumes) if volumes else None
                price = None
                if closes:
                    price = _f(closes[-1])
                live = self.fetch_price(sym)
                if _f(live) is not None:
                    price = _f(live)
                if price is None or price <= 0 or not closes:
                    continue
                contexts.append({
                    "symbol": sym,
                    "closes": closes,
                    "volumes": volumes,
                    "underlying_price": price,
                    "regime": regime,
                    "baseline_candidate": base_by_sym.get(sym),
                })
            except Exception:
                summary["errors"] += 1
                self.errors += 1

        for strat in self.strategies:
            cands: List[Dict] = []
            for ctx in contexts:
                summary["evaluated"] += 1
                try:
                    cand = strat.evaluate(ctx)
                except Exception:
                    summary["errors"] += 1
                    self.errors += 1
                    continue
                if cand is not None:
                    cands.append(cand)

            if isinstance(strat, SimonsMeanReversion):
                cands = self._apply_top_k(cands, strat.top_k)

            comp = self._comparators[strat.variant]
            for cand in cands:
                if comp.submit(cand):
                    summary["submitted"] += 1
        return summary

    @staticmethod
    def _apply_top_k(cands: List[Dict], top_k: int) -> List[Dict]:
        """Keep only the top-K accepted candidates by score; downgrade rest."""
        accepted = [c for c in cands if c.get("accepted")]
        accepted.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        keep = set(id(c) for c in accepted[:max(0, top_k)])
        for c in accepted:
            if id(c) not in keep:
                c["accepted"] = False
                c["reject_code"] = "shadow:not_top_k"
        return cands

    # ------------------------------------------------------------------
    def drain(self) -> int:
        total = 0
        for comp in self._comparators.values():
            try:
                total += comp.drain()
            except Exception:
                pass
        return total

    def close(self, timeout_s: float = 2.0) -> None:
        for comp in self._comparators.values():
            try:
                comp.close(timeout_s)
            except Exception:
                pass

    def accounting(self) -> Dict:
        return {
            "cycles": self.cycles,
            "errors": self.errors,
            "variants": {v: comp.accounting()
                         for v, comp in self._comparators.items()},
        }


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def build_strategies(names: Optional[List[str]] = None) -> List[ShadowStrategy]:
    """Build strategies by variant name, params pulled from the environment."""
    available = {
        "simons_meanrev": lambda: SimonsMeanReversion({
            "window": _env_int("SHADOW_SIMONS_WINDOW", 20),
            "z_entry": _env_float("SHADOW_SIMONS_Z_ENTRY", 1.5),
            "top_k": _env_int("SHADOW_SIMONS_TOP_K", 3),
        }),
        "soros_momentum": lambda: SorosMomentumVolume({
            "mom_thresh": _env_float("SHADOW_SOROS_MOM_THRESH", 0.02),
            "vol_mult": _env_float("SHADOW_SOROS_VOL_MULT", 1.2),
        }),
    }
    if not names:
        names = list(available)
    out: List[ShadowStrategy] = []
    for n in names:
        n = (n or "").strip()
        if n in available:
            out.append(available[n]())
    return out


# ---------------------------------------------------------------------------
# offline self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:  # pragma: no cover - exercised by `python shadow_strategies.py`
    import tempfile

    # synthetic fixtures ------------------------------------------------
    flat = [100.0] * 30
    # oversold: steady, then a sharp drop at the end
    oversold = [100.0 + 0.05 * i for i in range(25)] + [101, 100, 97, 94, 90]
    overbought = [100.0 - 0.05 * i for i in range(25)] + [99, 100, 103, 106, 110]
    # momentum up with rising volume, modest daily moves (no exhaustion)
    momo = [100.0 * (1.006 ** i) for i in range(30)]
    momo_vol = [1_000_000.0] * 29 + [2_000_000.0]
    weak_vol = [1_000_000.0] * 30
    # exhaustion: huge 3-day spike after a quiet series
    exhausted = [100.0] * 27 + [104, 109, 115]
    exh_vol = [1_000_000.0] * 29 + [3_000_000.0]

    simons = SimonsMeanReversion({"window": 20, "z_entry": 1.5, "top_k": 1})
    soros = SorosMomentumVolume({"mom_thresh": 0.02, "vol_mult": 1.2})

    def ctx(sym, closes, volumes=None):
        return {"symbol": sym, "closes": closes, "volumes": volumes,
                "underlying_price": closes[-1], "regime": "test",
                "baseline_candidate": None}

    # Simons unit behavior
    c = simons.evaluate(ctx("OV", oversold))
    assert c and c["direction"] == "CALL" and c["accepted"], c
    c = simons.evaluate(ctx("OB", overbought))
    assert c and c["direction"] == "PUT" and c["accepted"], c
    assert simons.evaluate(ctx("FLAT", flat)) is None

    # Soros unit behavior
    c = soros.evaluate(ctx("MOMO", momo, momo_vol))
    assert c and c["accepted"] and c["direction"] == "CALL", c
    c = soros.evaluate(ctx("WEAK", momo, weak_vol))
    assert c and not c["accepted"] and c["reject_code"] == "shadow:weak_volume", c
    c = soros.evaluate(ctx("EXH", exhausted, exh_vol))
    assert c and not c["accepted"] and c["reject_code"] == "shadow:exhausted", c
    c = soros.evaluate(ctx("NOVOL", momo, None))
    assert c and not c["accepted"] and c["reject_code"] == "shadow:no_volume_data", c

    # Runner end-to-end against a real (temp) ledger
    from research_ledger import ResearchLedger
    with tempfile.TemporaryDirectory() as td:
        led = ResearchLedger(os.path.join(td, "t.db"))
        data = {
            "OV": {"closes": oversold, "volumes": weak_vol},
            "OB": {"closes": overbought, "volumes": weak_vol},
            "MOMO": {"closes": momo, "volumes": momo_vol},
            "BAD": None,  # fetch returns nothing
        }

        def fetch_bars(sym, days):
            if sym == "BOOM":
                raise RuntimeError("boom")
            return data.get(sym)

        def fetch_price(sym):
            d = data.get(sym)
            return d["closes"][-1] if d else None

        runner = ShadowStrategyRunner(led, [simons, soros],
                                      fetch_bars, fetch_price)
        s = runner.run_cycle(["OV", "OB", "MOMO", "BAD", "BOOM"], [])
        runner.drain()
        runner.close()
        assert s["submitted"] >= 3, s
        assert s["errors"] >= 1, s  # BOOM counted, never raised

        rows = led.snapshots(kind="shadow_candidate")
        assert rows, "no shadow rows written"
        variants = {r["strategy_version"] for r in rows}
        assert "shadow:simons_meanrev" in variants, variants
        assert "shadow:soros_momentum" in variants, variants
        # top_k=1: OV and OB both pass z threshold; one must be not_top_k
        sim_rows = [r for r in rows
                    if r["strategy_version"] == "shadow:simons_meanrev"]
        rejects = {r["reject_code"] for r in sim_rows}
        assert "shadow:not_top_k" in rejects, rejects
        n_acc = sum(1 for r in sim_rows if r["accepted"])
        assert n_acc == 1, sim_rows
        # baseline isolation
        assert not led.snapshots(kind="candidate")
        led.close()  # Windows: release the db file before temp-dir cleanup

    print("shadow_strategies selftest: PASS")


if __name__ == "__main__":
    _selftest()
