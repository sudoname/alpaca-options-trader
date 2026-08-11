"""
Oracle Lab — experiment runner (deterministic, offline).

An *experiment* applies a candidate decision rule to a point-in-time dataset
(rows from ``oracle.lab.dataset``) and measures the result with
``oracle.lab.metrics``. It is the atomic unit the parameter-sweep and
walk-forward harnesses compose.

Nothing here trades. A rule turns a feature row into a hypothetical trade record
whose PnL is derived from the row's *already-observed forward label* — so an
experiment answers "had we followed this rule at these decision points, what
would the realized distribution have looked like?" without any live effect.

Determinism: ``run_experiment`` is a pure function of (config, dataset,
strategy_fn). A seed is threaded to the strategy for any stochastic tie-break,
but the default rule is fully deterministic. Re-running with the same inputs
yields a byte-identical result dict.

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure compute; the only optional I/O is JSON persistence of the result.
  * Fail-open on a bad row (skip it); never raise in the research path.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from oracle.lab import metrics as _metrics

_RESULTS_DIR_DEFAULT = os.path.join("oracle", "lab", "results")

# Default notional per hypothetical trade (dollars of exposure). Realized PnL of
# a directional option proxy = notional * signed forward return fraction.
_DEFAULT_NOTIONAL = 1000.0


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


# --------------------------------------------------------------------------- #
# Config / result records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen, hashable description of one experiment. ``params`` is coerced to a
    sorted tuple of (k, v) internally so two configs with the same content are
    equal and hash identically (deterministic sweep keys)."""

    experiment_id: str
    seed: int = 0
    params: Tuple[Tuple[str, Any], ...] = ()
    symbols: Tuple[str, ...] = ()
    start: Optional[str] = None
    end: Optional[str] = None
    mode: str = "intraday"

    @staticmethod
    def make(experiment_id: str, *, seed: int = 0,
             params: Optional[Dict[str, Any]] = None,
             symbols: Optional[Sequence[str]] = None,
             start: Optional[str] = None, end: Optional[str] = None,
             mode: str = "intraday") -> "ExperimentConfig":
        p = tuple(sorted((str(k), v) for k, v in (params or {}).items()))
        s = tuple(str(x).upper() for x in (symbols or []))
        return ExperimentConfig(experiment_id=str(experiment_id), seed=int(seed),
                                params=p, symbols=s, start=start, end=end,
                                mode=str(mode))

    def params_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.params}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = self.params_dict()
        d["symbols"] = list(self.symbols)
        return d


@dataclass
class ExperimentResult:
    experiment_id: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    n_trades: int
    n_rows: int
    breakdowns: Dict[str, Any] = field(default_factory=dict)
    trades: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Default decision rule
# --------------------------------------------------------------------------- #
def default_strategy(row: dict, params: Dict[str, Any],
                     rng: random.Random) -> Optional[dict]:
    """Momentum threshold rule -> hypothetical trade record, or None (NO-TRADE).

    Params:
      momentum_threshold_pct  enter long above +T, short below -T (default 1.0)
      notional                dollars of exposure per trade (default 1000)
      min_vol_pct / max_vol_pct  optional realized-vol gate (skip outside band)

    The realized PnL uses the row's forward label return, so a CALL profits from
    an up move and a PUT profits from a down move. Direction is an OUTPUT of the
    rule, never an input — consistent with Oracle's core philosophy."""
    if not isinstance(row, dict):
        return None
    ctx = row.get("ctx") or {}
    thr = _to_float(params.get("momentum_threshold_pct"))
    thr = 1.0 if thr is None else thr
    notional = _to_float(params.get("notional")) or _DEFAULT_NOTIONAL

    mom = _to_float(ctx.get("momentum_5d_pct"))
    if mom is None:
        return None

    vol = _to_float(ctx.get("realized_vol_pct"))
    min_vol = _to_float(params.get("min_vol_pct"))
    max_vol = _to_float(params.get("max_vol_pct"))
    if vol is not None:
        if min_vol is not None and vol < min_vol:
            return None
        if max_vol is not None and vol > max_vol:
            return None

    if mom > thr:
        direction = "call"
    elif mom < -thr:
        direction = "put"
    else:
        return None  # abstain — the system is always allowed to NO-TRADE

    ret_pct = _to_float(row.get("label_return_pct"))
    if ret_pct is None:
        return None
    signed = ret_pct if direction == "call" else -ret_pct
    pnl = round(notional * signed / 100.0, 6)

    return {
        "symbol": row.get("symbol"),
        "direction": direction,
        "pnl": pnl,
        "return_pct": round(signed, 6),
        "regime": ctx.get("regime"),
        "strategy_mode": row.get("mode"),
        "conviction": min(1.0, abs(mom) / max(thr, 1e-9) / 4.0),
        "as_of": row.get("as_of"),
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_experiment(cfg: ExperimentConfig, dataset: Sequence[dict], *,
                   strategy_fn: Optional[Callable] = None,
                   keep_trades: bool = True) -> ExperimentResult:
    """Apply ``strategy_fn`` (default ``default_strategy``) to every dataset row
    and summarize. Deterministic in (cfg, dataset, strategy_fn)."""
    strategy_fn = strategy_fn or default_strategy
    rng = random.Random(cfg.seed)
    params = cfg.params_dict()

    if isinstance(dataset, (str, bytes, dict)) or dataset is None:
        rows = []
    else:
        try:
            rows = [r for r in dataset if isinstance(r, dict)]
        except TypeError:
            rows = []
    # Stable ordering so the seeded rng consumption is reproducible.
    rows = sorted(rows, key=lambda r: (str(r.get("as_of") or ""),
                                       str(r.get("symbol") or "")))
    trades: List[dict] = []
    for r in rows:
        try:
            t = strategy_fn(r, params, rng)
        except Exception:  # pragma: no cover - fail-open per row
            t = None
        if t:
            trades.append(t)

    m = _metrics.compute_metrics(trades)
    breakdowns = {
        "by_direction": _metrics.breakdown_by_direction(trades),
        "by_regime": _metrics.breakdown_by_regime(trades),
        "by_conviction": _metrics.breakdown_by_conviction(trades),
    }
    return ExperimentResult(
        experiment_id=cfg.experiment_id,
        config=cfg.to_dict(),
        metrics=m,
        n_trades=len(trades),
        n_rows=len(rows),
        breakdowns=breakdowns,
        trades=trades if keep_trades else [],
    )


def save_result(result: ExperimentResult, *,
                results_dir: str = _RESULTS_DIR_DEFAULT) -> str:
    """Persist an experiment result to ``<results_dir>/<experiment_id>.json``.
    Returns the path. Creates the directory if needed. Fail-open."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{result.experiment_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, default=str, indent=2, sort_keys=True)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[lab.experiment] save failed: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Synthetic dataset: two up rows (momentum high, forward up) and one down.
    dataset = [
        {"symbol": "AAA", "as_of": "2024-01-02T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": 3.0, "realized_vol_pct": 20.0,
                 "regime": "trending"}, "label_return_pct": 5.0},
        {"symbol": "BBB", "as_of": "2024-01-03T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": 2.5, "realized_vol_pct": 25.0,
                 "regime": "trending"}, "label_return_pct": 4.0},
        {"symbol": "CCC", "as_of": "2024-01-04T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": -3.0, "realized_vol_pct": 30.0,
                 "regime": "choppy"}, "label_return_pct": -2.0},
        # Below threshold -> abstain (NO-TRADE).
        {"symbol": "DDD", "as_of": "2024-01-05T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": 0.2, "realized_vol_pct": 15.0},
         "label_return_pct": 9.0},
    ]

    cfg = ExperimentConfig.make("selftest", seed=7,
                                params={"momentum_threshold_pct": 1.0,
                                        "notional": 1000.0},
                                symbols=["AAA", "BBB", "CCC", "DDD"])
    res = run_experiment(cfg, dataset)

    # Three trades taken (the 0.2%-momentum row abstains).
    if res.n_trades != 3:
        print("FAIL: n_trades", res.n_trades); ok = False
    if res.n_rows != 4:
        print("FAIL: n_rows", res.n_rows); ok = False

    # PnLs: call +5% ->  +50, call +4% -> +40, put on -2% -> +20 (all wins).
    pnls = sorted(t["pnl"] for t in res.trades)
    if pnls != [20.0, 40.0, 50.0]:
        print("FAIL: pnls", pnls); ok = False
    if res.metrics["win_rate"] != 1.0:
        print("FAIL: win_rate", res.metrics["win_rate"]); ok = False

    # Direction breakdown: two calls, one put.
    bd = res.breakdowns["by_direction"]
    if bd.get("call", {}).get("trade_count") != 2:
        print("FAIL: call count", bd); ok = False
    if bd.get("put", {}).get("trade_count") != 1:
        print("FAIL: put count", bd); ok = False

    # Frozen config equality / hashing is content-based.
    cfg2 = ExperimentConfig.make("selftest", seed=7,
                                 params={"notional": 1000.0,
                                         "momentum_threshold_pct": 1.0},
                                 symbols=["AAA", "BBB", "CCC", "DDD"])
    if cfg != cfg2 or hash(cfg) != hash(cfg2):
        print("FAIL: config equality/hash"); ok = False

    # Determinism: identical result on re-run.
    if run_experiment(cfg, dataset).to_dict() != res.to_dict():
        print("FAIL: non-deterministic result"); ok = False

    # A vol gate outside the band abstains everything.
    cfg_gate = ExperimentConfig.make("gate", params={"max_vol_pct": 5.0})
    if run_experiment(cfg_gate, dataset).n_trades != 0:
        print("FAIL: vol gate should abstain all"); ok = False

    # Junk tolerance.
    for junk in (None, 42, "x", [None, {"bad": 1}]):
        try:
            run_experiment(cfg, junk)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk dataset", junk, exc); ok = False

    print("lab.experiment self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
