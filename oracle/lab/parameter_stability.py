"""
Oracle Lab — parameter-stability analysis (ANALYTICS ONLY, offline).

A profitable backtest is worthless if it sits on a knife-edge: a rule whose edge
evaporates when a threshold moves by one tick was curve-fit to noise. This module
asks the robustness question the sweep can't: is the winning point on a *plateau*
(neighbours are also good) or a *spike* (neighbours fall off a cliff)?

It consumes a ``SweepReport`` (from ``oracle.lab.parameter_sweep``) plus the
original ``param_grid`` (needed to know each axis's ordered values, hence which
points are adjacent) and produces:

  * per-axis sensitivity  — mean absolute score change per one-step move along
    that axis, holding the other axes fixed (a numeric gradient along the axis).
  * best-point neighbourhood — the best point's immediate neighbours (differ in
    exactly one axis by one step) and how far their scores drop from the peak.
  * plateau / robust-region size — fraction of grid points within a tolerance of
    the best score (a wide plateau => the edge is not a fluke).
  * ``is_spike`` — the headline flag: the best point is an isolated peak whose
    neighbours drop by more than ``spike_drop_frac`` of the peak.

Pure, deterministic, fail-open. No sweep is re-run here; it only analyses one.
A convenience ``analyze_grid`` runs the sweep for you.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from oracle.lab.parameter_sweep import (
    SweepReport,
    _canon_params,
    grid_points,
    sweep,
)

_RESULTS_DIR_DEFAULT = os.path.join("oracle", "lab", "results")


def _axis_values(param_grid: Dict[str, Sequence[Any]]) -> Dict[str, List[Any]]:
    """Ordered, de-duplicated value list for each axis (caller order preserved).
    Scalars become a 1-element axis; keys sorted for deterministic iteration."""
    axes: Dict[str, List[Any]] = {}
    if not isinstance(param_grid, dict):
        return axes
    for k in sorted(str(x) for x in param_grid.keys()):
        v = param_grid[k]
        seq = list(v) if isinstance(v, (list, tuple)) else [v]
        seen: List[Any] = []
        for item in seq:
            if item not in seen:
                seen.append(item)
        axes[k] = seen
    return axes


def _point_key(params: Dict[str, Any], axes: Dict[str, List[Any]]) -> Tuple:
    """Positional key of a point within the grid lattice: the index of each
    axis value. Points that share this key modulo one axis are neighbours."""
    key: List[int] = []
    for k in axes:
        vals = axes[k]
        v = params.get(k)
        try:
            key.append(vals.index(v))
        except ValueError:
            key.append(-1)
    return tuple(key)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class AxisSensitivity:
    axis: str
    n_values: int
    mean_abs_step: float       # mean |Δscore| per one-step move along this axis
    max_abs_step: float
    range_score: float         # max-min score attributable to moving this axis

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityResult:
    rank_by: str
    higher_is_better: bool
    n_points: int
    best_params: Dict[str, Any]
    best_score: float
    score_mean: float
    score_std: float
    score_cv: Optional[float]              # std / |mean|
    plateau_tol: float
    plateau_size: int                      # points within tol of best
    plateau_frac: float
    n_neighbors: int
    neighbor_mean_drop: float              # mean (best - neighbor) score drop
    neighbor_max_drop: float
    worst_neighbor_params: Optional[Dict[str, Any]]
    spike_drop_frac: float
    is_spike: bool
    axis_sensitivity: List[AxisSensitivity] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axis_sensitivity"] = [a.to_dict() for a in self.axis_sensitivity]
        return d


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def _oriented(score: float, higher_is_better: bool) -> float:
    """Map a raw score to a 'bigger==better' axis so drop math is uniform."""
    return score if higher_is_better else -score


def analyze(report: SweepReport,
            param_grid: Dict[str, Sequence[Any]], *,
            plateau_tol_frac: float = 0.25,
            spike_drop_frac: float = 0.5) -> StabilityResult:
    """Analyse a completed ``SweepReport`` for parameter stability.

    ``plateau_tol_frac`` — a point counts as "on the plateau" if its oriented
    score is within this fraction of the best oriented score's magnitude.
    ``spike_drop_frac`` — the best point is a spike if its worst one-step
    neighbour drops by more than this fraction of the best oriented score.
    """
    hib = bool(report.higher_is_better)
    results = list(report.results or [])
    axes = _axis_values(param_grid)

    # Map lattice key -> oriented score (skip unscored / -inf).
    scored: Dict[Tuple, float] = {}
    params_by_key: Dict[Tuple, Dict[str, Any]] = {}
    raw_scores: List[float] = []
    for r in results:
        s = r.score
        if s is None or s == float("-inf") or s == float("inf"):
            continue
        key = _point_key(r.params, axes)
        os_ = _oriented(float(s), hib)
        scored[key] = os_
        params_by_key[key] = r.params
        raw_scores.append(float(s))

    n = len(scored)
    if n == 0:
        return StabilityResult(
            rank_by=report.rank_by, higher_is_better=hib, n_points=0,
            best_params={}, best_score=0.0, score_mean=0.0, score_std=0.0,
            score_cv=None, plateau_tol=0.0, plateau_size=0, plateau_frac=0.0,
            n_neighbors=0, neighbor_mean_drop=0.0, neighbor_max_drop=0.0,
            worst_neighbor_params=None, spike_drop_frac=float(spike_drop_frac),
            is_spike=False, axis_sensitivity=[])

    # Best point = the sweep's rank-0 (already oriented by the sweep).
    best = results[0]
    best_key = _point_key(best.params, axes)
    best_os = scored.get(best_key)
    if best_os is None:  # best was unscored -> fall back to max oriented
        best_key = max(scored, key=lambda k: scored[k])
        best_os = scored[best_key]

    mean = sum(scored.values()) / n
    var = sum((v - mean) ** 2 for v in scored.values()) / n
    std = var ** 0.5
    cv = (std / abs(mean)) if mean != 0 else None

    # Plateau: oriented score within tol of best.
    tol = abs(best_os) * float(plateau_tol_frac)
    plateau = [k for k, v in scored.items() if (best_os - v) <= tol]
    plateau_size = len(plateau)

    # Neighbours of the best point: differ in exactly one axis by one lattice
    # step. Score drop = best_os - neighbour_os.
    axis_names = list(axes.keys())
    neighbor_drops: List[Tuple[float, Tuple]] = []
    for ai in range(len(axis_names)):
        for step in (-1, 1):
            nk = list(best_key)
            nk[ai] = best_key[ai] + step
            nkey = tuple(nk)
            if nkey in scored:
                neighbor_drops.append((best_os - scored[nkey], nkey))

    n_neighbors = len(neighbor_drops)
    if n_neighbors:
        drops = [d for d, _ in neighbor_drops]
        mean_drop = sum(drops) / n_neighbors
        max_drop, worst_key = max(neighbor_drops, key=lambda x: x[0])
        worst_params = params_by_key.get(worst_key)
    else:
        mean_drop = 0.0
        max_drop = 0.0
        worst_params = None

    # Spike: an isolated peak whose worst neighbour falls off a cliff. Only
    # meaningful when it actually has neighbours and a positive peak.
    is_spike = bool(n_neighbors and best_os > 0
                    and max_drop > abs(best_os) * float(spike_drop_frac))

    # Per-axis sensitivity: for each axis, walk successive lattice steps while
    # holding the other axes fixed, average |Δ oriented score|.
    axis_sens: List[AxisSensitivity] = []
    for ai, name in enumerate(axis_names):
        steps: List[float] = []
        span_scores: List[float] = []
        # Group keys by their "other axes" signature.
        groups: Dict[Tuple, List[Tuple[int, float]]] = {}
        for key, val in scored.items():
            sig = tuple(x for j, x in enumerate(key) if j != ai)
            groups.setdefault(sig, []).append((key[ai], val))
        for sig, pts in groups.items():
            pts.sort(key=lambda x: x[0])
            for j in range(len(pts) - 1):
                steps.append(abs(pts[j + 1][1] - pts[j][1]))
            span_scores.extend(v for _, v in pts)
        mean_step = (sum(steps) / len(steps)) if steps else 0.0
        max_step = max(steps) if steps else 0.0
        rng = (max(span_scores) - min(span_scores)) if span_scores else 0.0
        axis_sens.append(AxisSensitivity(
            axis=name, n_values=len(axes[name]),
            mean_abs_step=round(mean_step, 6), max_abs_step=round(max_step, 6),
            range_score=round(rng, 6)))

    return StabilityResult(
        rank_by=report.rank_by,
        higher_is_better=hib,
        n_points=n,
        best_params=best.params,
        best_score=round(float(best.score), 6),
        score_mean=round(mean if hib else -mean, 6),
        score_std=round(std, 6),
        score_cv=(round(cv, 6) if cv is not None else None),
        plateau_tol=round(tol, 6),
        plateau_size=plateau_size,
        plateau_frac=round(plateau_size / n, 6),
        n_neighbors=n_neighbors,
        neighbor_mean_drop=round(mean_drop, 6),
        neighbor_max_drop=round(max_drop, 6),
        worst_neighbor_params=worst_params,
        spike_drop_frac=float(spike_drop_frac),
        is_spike=is_spike,
        axis_sensitivity=axis_sens,
    )


def analyze_grid(param_grid: Dict[str, Sequence[Any]],
                 base_cfg, dataset: Sequence[dict], *,
                 strategy_fn=None,
                 rank_by: str = "expectancy",
                 higher_is_better: Optional[bool] = None,
                 min_trades: int = 1,
                 plateau_tol_frac: float = 0.25,
                 spike_drop_frac: float = 0.5) -> StabilityResult:
    """Convenience: run the sweep then analyse it in one call."""
    report = sweep(param_grid, base_cfg, dataset, strategy_fn=strategy_fn,
                   rank_by=rank_by, higher_is_better=higher_is_better,
                   min_trades=min_trades)
    return analyze(report, param_grid, plateau_tol_frac=plateau_tol_frac,
                   spike_drop_frac=spike_drop_frac)


def save_result(result: StabilityResult, base_experiment_id: str, *,
                results_dir: str = _RESULTS_DIR_DEFAULT) -> str:
    """Persist to ``<results_dir>/<base_id>__stability.json``. Fail-open."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir,
                            f"{base_experiment_id}__stability.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, default=str, indent=2,
                      sort_keys=True)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[lab.parameter_stability] save failed: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _make_row(day: int, mom: float, ret: float) -> dict:
    return {
        "symbol": "AAA",
        "as_of": f"2024-01-{day:02d}T16:00:00",
        "mode": "intraday",
        "ctx": {"momentum_5d_pct": mom, "regime": "trending"},
        "label_return_pct": ret,
    }


def _self_test() -> int:
    from oracle.lab.experiment import ExperimentConfig

    ok = True

    # Axis-value extraction: sorted keys, order preserved, dedup, scalar axis.
    axes = _axis_values({"b": [1, 1, 2], "a": [3]})
    if list(axes.keys()) != ["a", "b"] or axes["b"] != [1, 2] or axes["a"] != [3]:
        print("FAIL: axis values", axes); ok = False

    # --- Plateau case: a real momentum edge over a range of thresholds. ------ #
    # Up rows trend up, down rows trend down; any threshold in [1,2,3] harvests
    # the same edge -> neighbours of the best point are close (a plateau).
    plateau_ds = []
    for d in range(1, 13):
        if d % 2 == 1:
            plateau_ds.append(_make_row(d, 4.0, 4.0))
        else:
            plateau_ds.append(_make_row(d, -4.0, -4.0))
    base = ExperimentConfig.make("stab_plateau", seed=1,
                                 params={"notional": 1000.0}, symbols=["AAA"])
    grid = {"momentum_threshold_pct": [1.0, 2.0, 3.0]}
    res = analyze_grid(grid, base, plateau_ds, rank_by="expectancy")
    if res.n_points != 3:
        print("FAIL: plateau n_points", res.n_points); ok = False
    if res.n_neighbors < 1:
        print("FAIL: plateau best has no neighbours"); ok = False
    if res.is_spike:
        print("FAIL: plateau wrongly flagged as spike", res.to_dict()); ok = False
    # All three thresholds harvest the identical edge -> wide plateau.
    if res.plateau_frac < 0.99:
        print("FAIL: plateau_frac too small", res.plateau_frac); ok = False

    # Determinism.
    res2 = analyze_grid(grid, base, plateau_ds, rank_by="expectancy")
    if res.to_dict() != res2.to_dict():
        print("FAIL: stability non-deterministic"); ok = False

    # --- Spike case: only one threshold works; neighbours fall off a cliff. -- #
    # Construct returns so a threshold of exactly 2.5 harvests a clean edge but
    # 1.0 (takes extra losing low-momentum trades) and 5.0 (takes none) are far
    # worse -> the best point is an isolated peak.
    spike_ds = [
        _make_row(1, 3.0, 6.0),    # |mom|=3 winner (only 2.5<|3|<5)
        _make_row(2, -3.0, -6.0),  # |mom|=3 winner
        _make_row(3, 1.5, -8.0),   # |mom|=1.5 loser (only thr=1.0 takes it)
        _make_row(4, -1.5, 8.0),   # |mom|=1.5 loser for a put
        _make_row(5, 3.0, 6.0),
        _make_row(6, -3.0, -6.0),
        _make_row(7, 1.5, -8.0),
        _make_row(8, -1.5, 8.0),
    ]
    grid_s = {"momentum_threshold_pct": [1.0, 2.5, 5.0]}
    res_s = analyze_grid(grid_s, base, spike_ds, rank_by="expectancy",
                         spike_drop_frac=0.5, min_trades=1)
    # Best should be the 2.5 threshold (clean winners only).
    if res_s.best_params.get("momentum_threshold_pct") != 2.5:
        print("FAIL: spike best not 2.5", res_s.best_params); ok = False
    if not res_s.is_spike:
        print("FAIL: spike not flagged", res_s.to_dict()); ok = False
    if res_s.neighbor_max_drop <= 0:
        print("FAIL: spike neighbour drop", res_s.neighbor_max_drop); ok = False

    # Axis sensitivity reported for the swept axis.
    names = [a.axis for a in res_s.axis_sensitivity]
    if "momentum_threshold_pct" not in names:
        print("FAIL: axis sensitivity missing", names); ok = False

    # Empty / junk sweep -> empty result, no raise.
    empty = SweepReport(base_experiment_id="e", rank_by="expectancy",
                        higher_is_better=True, n_points=0, n_evaluated=0,
                        results=[])
    er = analyze(empty, grid)
    if er.n_points != 0 or er.is_spike:
        print("FAIL: empty report handling", er.to_dict()); ok = False

    print("lab.parameter_stability self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
