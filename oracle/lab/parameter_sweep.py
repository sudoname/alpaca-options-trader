"""
Oracle Lab — deterministic parameter sweep (ANALYTICS ONLY, offline).

A *sweep* runs one experiment per point of a cartesian ``param_grid`` (each grid
point merged over a base config) and ranks the resulting
``oracle.lab.experiment.ExperimentResult`` objects by a chosen metric. It is the
composition layer above ``run_experiment`` and the feed below
``walk_forward`` — it enumerates candidates; walk-forward decides whether a
candidate survives out-of-sample.

Determinism:
  * Grid points are enumerated in a fixed order: keys sorted, then
    ``itertools.product`` over each key's values in the caller-given order. The
    same ``param_grid`` always yields the same ordered list of points.
  * Each point gets a stable ``experiment_id`` derived from the base id + a
    canonical hash of its params, so re-running writes/loads the same keys.
  * Ranking is a total order: primary = score (None -> -inf), tie-broken by the
    canonical params string. Re-running a sweep yields a byte-identical ranking.

Methodology guard (per spec — "never selects on full-sample only"):
  ``sweep`` returns the *ranked in-sample* candidates and every ``SweepResult``
  is stamped ``in_sample_only=True``. There is deliberately NO ``select_best``
  that collapses the sweep to a single winner on full-sample metrics; choosing a
  survivor is the job of ``walk_forward`` (out-of-sample). ``top_k`` is provided
  only to shortlist candidates to *hand to* walk-forward.

Design rules (mirror the rest of oracle/lab):
  * Pure compute; the only optional I/O is JSON persistence.
  * Fail-open: a bad grid point is skipped, never raised, in the research path.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from oracle.lab.experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)

_RESULTS_DIR_DEFAULT = os.path.join("oracle", "lab", "results")

# Metrics where a larger value is better (used to pick a sane default direction).
_HIGHER_IS_BETTER = {
    "expectancy", "total_pnl", "win_rate", "profit_factor", "sharpe",
    "sortino", "calmar", "avg_win", "avg_mfe", "realized_ev_capture",
    "trade_count",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _canon_params(params: Dict[str, Any]) -> str:
    """Canonical, order-independent string for a param mapping (ranking
    tie-break + stable experiment-id suffix)."""
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        return repr(sorted((str(k), str(v)) for k, v in params.items()))


def _params_suffix(params: Dict[str, Any]) -> str:
    return hashlib.sha1(_canon_params(params).encode("utf-8")).hexdigest()[:10]


def grid_points(param_grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Deterministic cartesian product of ``param_grid``.

    Keys are sorted; each key's values are taken in the given order. An empty or
    malformed grid yields a single empty point (i.e. run the base config once).
    Values given as a scalar (not a list/tuple) are treated as a 1-element axis.
    """
    if not isinstance(param_grid, dict) or not param_grid:
        return [{}]
    keys = sorted(str(k) for k in param_grid.keys())
    axes: List[Sequence[Any]] = []
    for k in keys:
        v = param_grid[k]
        if isinstance(v, (list, tuple)):
            axes.append(list(v))
        else:
            axes.append([v])  # scalar axis
    if any(len(a) == 0 for a in axes):
        return [{}]
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def _score_of(result: ExperimentResult, rank_by: str) -> float:
    v = result.metrics.get(rank_by)
    if v is None:
        return float("-inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class SweepResult:
    """One ranked grid point. ``in_sample_only`` is always True — a sweep never
    certifies a winner; walk-forward does."""

    rank: int
    experiment_id: str
    params: Dict[str, Any]
    score: float
    rank_by: str
    n_trades: int
    n_rows: int
    metrics: Dict[str, Any]
    in_sample_only: bool = True
    result: Optional[Dict[str, Any]] = None  # full ExperimentResult.to_dict()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SweepReport:
    base_experiment_id: str
    rank_by: str
    higher_is_better: bool
    n_points: int
    n_evaluated: int
    results: List[SweepResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d

    def top_k(self, k: int) -> List[SweepResult]:
        """Shortlist the ``k`` best candidates to hand to walk-forward. This is
        NOT a full-sample winner selection — it only narrows the field."""
        return self.results[: max(0, int(k))]


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def sweep(param_grid: Dict[str, Sequence[Any]],
          base_cfg: ExperimentConfig,
          dataset: Sequence[dict], *,
          strategy_fn: Optional[Callable] = None,
          rank_by: str = "expectancy",
          higher_is_better: Optional[bool] = None,
          min_trades: int = 1,
          keep_trades: bool = False,
          keep_result: bool = False) -> SweepReport:
    """Run one experiment per grid point (base params overridden by the point)
    and return a deterministically-ranked ``SweepReport``.

    Ranking: candidates with ``n_trades >= min_trades`` always rank above those
    below it; within each group, by ``score`` (respecting ``higher_is_better``),
    tie-broken by the canonical params string. This is a total order, so the
    ranking is reproducible.
    """
    rank_by = str(rank_by)
    if higher_is_better is None:
        higher_is_better = rank_by in _HIGHER_IS_BETTER

    points = grid_points(param_grid)
    base_params = base_cfg.params_dict()

    evaluated: List[Tuple[Dict[str, Any], ExperimentResult]] = []
    for pt in points:
        try:
            merged = dict(base_params)
            merged.update(pt)
            suffix = _params_suffix(merged)
            cfg = ExperimentConfig.make(
                f"{base_cfg.experiment_id}__{suffix}",
                seed=base_cfg.seed, params=merged,
                symbols=list(base_cfg.symbols),
                start=base_cfg.start, end=base_cfg.end, mode=base_cfg.mode)
            res = run_experiment(cfg, dataset, strategy_fn=strategy_fn,
                                 keep_trades=keep_trades)
        except Exception:  # pragma: no cover - fail-open per point
            continue
        evaluated.append((pt, res))

    def _sort_key(item: Tuple[Dict[str, Any], ExperimentResult]):
        pt, res = item
        raw = _score_of(res, rank_by)
        score = raw if higher_is_better else -raw
        meets = 1 if res.n_trades >= min_trades else 0
        # Sort DESC by (meets_min, score); tie-break ASC by canonical params so
        # ordering is total and stable.
        return (-meets, -score, _canon_params(res.config.get("params", {})))

    evaluated.sort(key=_sort_key)

    results: List[SweepResult] = []
    for i, (pt, res) in enumerate(evaluated):
        results.append(SweepResult(
            rank=i,
            experiment_id=res.experiment_id,
            params=res.config.get("params", {}),
            score=_score_of(res, rank_by),
            rank_by=rank_by,
            n_trades=res.n_trades,
            n_rows=res.n_rows,
            metrics=res.metrics,
            in_sample_only=True,
            result=res.to_dict() if keep_result else None,
        ))

    return SweepReport(
        base_experiment_id=base_cfg.experiment_id,
        rank_by=rank_by,
        higher_is_better=bool(higher_is_better),
        n_points=len(points),
        n_evaluated=len(evaluated),
        results=results,
    )


def save_report(report: SweepReport, *,
                results_dir: str = _RESULTS_DIR_DEFAULT) -> str:
    """Persist a sweep report to ``<results_dir>/<base_id>__sweep.json``.
    Fail-open; returns the path or ''."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir,
                            f"{report.base_experiment_id}__sweep.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, default=str, indent=2,
                      sort_keys=True)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[lab.parameter_sweep] save failed: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Grid enumeration is deterministic and cartesian.
    g = grid_points({"b": [1, 2], "a": ["x", "y"]})
    # keys sorted -> a, b; product in given order.
    expected = [
        {"a": "x", "b": 1}, {"a": "x", "b": 2},
        {"a": "y", "b": 1}, {"a": "y", "b": 2},
    ]
    if g != expected:
        print("FAIL: grid_points order", g); ok = False
    if grid_points({}) != [{}]:
        print("FAIL: empty grid"); ok = False
    if grid_points({"a": 5}) != [{"a": 5}]:
        print("FAIL: scalar axis"); ok = False

    # Synthetic dataset with a clear momentum edge (up rows trend up).
    dataset = [
        {"symbol": "AAA", "as_of": "2024-01-02T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": 3.0, "regime": "trending"},
         "label_return_pct": 5.0},
        {"symbol": "BBB", "as_of": "2024-01-03T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": 2.0, "regime": "trending"},
         "label_return_pct": 4.0},
        {"symbol": "CCC", "as_of": "2024-01-04T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": -3.0, "regime": "choppy"},
         "label_return_pct": -2.0},
        {"symbol": "DDD", "as_of": "2024-01-05T16:00:00", "mode": "intraday",
         "ctx": {"momentum_5d_pct": 0.5, "regime": "choppy"},
         "label_return_pct": 9.0},
    ]

    base = ExperimentConfig.make("sweep_selftest", seed=3,
                                 params={"notional": 1000.0},
                                 symbols=["AAA", "BBB", "CCC", "DDD"])
    grid = {"momentum_threshold_pct": [1.0, 2.5, 10.0]}

    rep = sweep(grid, base, dataset, rank_by="expectancy")
    if rep.n_points != 3 or rep.n_evaluated != 3:
        print("FAIL: n_points/evaluated", rep.n_points, rep.n_evaluated)
        ok = False
    if len(rep.results) != 3:
        print("FAIL: results length", len(rep.results)); ok = False

    # Ranks are 0..n-1 and monotonic in the (meets_min, score) key.
    if [r.rank for r in rep.results] != [0, 1, 2]:
        print("FAIL: rank sequence", [r.rank for r in rep.results]); ok = False

    # The 10.0 threshold takes zero trades -> must rank LAST (below min_trades).
    last = rep.results[-1]
    if last.params.get("momentum_threshold_pct") != 10.0 or last.n_trades != 0:
        print("FAIL: zero-trade point should rank last", last.to_dict())
        ok = False

    # Every candidate is stamped in-sample-only (methodology guard).
    if not all(r.in_sample_only for r in rep.results):
        print("FAIL: in_sample_only flag"); ok = False

    # Determinism: identical report on re-run.
    rep2 = sweep(grid, base, dataset, rank_by="expectancy")
    if rep.to_dict() != rep2.to_dict():
        print("FAIL: non-deterministic sweep"); ok = False

    # top_k shortlists without collapsing to a single full-sample winner.
    top2 = rep.top_k(2)
    if len(top2) != 2 or top2[0].rank != 0 or top2[1].rank != 1:
        print("FAIL: top_k", [r.rank for r in top2]); ok = False

    # A losing metric direction flips ranking (lower max_drawdown is better).
    rep_dd = sweep(grid, base, dataset, rank_by="max_drawdown")
    if rep_dd.higher_is_better is not False:
        print("FAIL: max_drawdown default direction"); ok = False

    # Junk tolerance: bad grid / dataset never raise.
    for junk in (None, 42, "x", []):
        try:
            sweep(junk, base, dataset)          # type: ignore[arg-type]
            sweep(grid, base, junk)             # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("lab.parameter_sweep self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
