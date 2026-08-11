"""
Oracle Lab — walk-forward / out-of-sample validation (ANALYTICS ONLY, offline).

Where ``parameter_sweep`` *enumerates* candidates on the full sample,
walk-forward *certifies* one. It answers the only question that matters for
robustness: does a rule that looked good on data it was tuned on still work on
data it never saw?

Protocol per fold (TRAIN -> VALIDATE -> FREEZE -> TEST -> ROLL):
  * The dataset is sorted by time (``as_of``) and cut into ``n_folds`` contiguous
    TEST windows that tile the back portion of the series.
  * For each fold, everything strictly BEFORE the test window is the
    IN-SAMPLE region; ``train_frac`` of that region is TRAIN, the remainder is
    VALIDATE. We sweep the ``param_grid`` on VALIDATE, FREEZE the winner, then
    score it once on the untouched TEST window.
  * Chronological only — TEST is always in the future relative to the data used
    to choose the parameters, so there is no look-ahead in parameter selection.

Aggregation:
  * IS metrics = pooled validate-window trades of the chosen params.
  * OOS metrics = pooled test-window trades of the chosen params.
  * ``oos_collapse`` fires when OOS expectancy is materially worse than IS
    (drop beyond ``collapse_frac`` of the IS expectancy, or IS positive while
    OOS non-positive) — the headline "did it overfit?" flag.

Determinism: pure function of (dataset, base_cfg, param_grid, n_folds,
train_frac, seed). Fail-open: a degenerate fold contributes nothing rather than
raising.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from oracle.lab import metrics as _metrics
from oracle.lab.experiment import ExperimentConfig, run_experiment
from oracle.lab.parameter_sweep import sweep

_RESULTS_DIR_DEFAULT = os.path.join("oracle", "lab", "results")


def _as_rows(dataset: Sequence[dict]) -> List[dict]:
    if isinstance(dataset, (str, bytes, dict)) or dataset is None:
        return []
    try:
        rows = [r for r in dataset if isinstance(r, dict)]
    except TypeError:
        return []
    return sorted(rows, key=lambda r: (str(r.get("as_of") or ""),
                                       str(r.get("symbol") or "")))


def _expectancy(trades: Sequence[dict]) -> float:
    return _metrics.expectancy(trades)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class FoldResult:
    fold: int
    train_range: Tuple[Optional[str], Optional[str]]
    validate_range: Tuple[Optional[str], Optional[str]]
    test_range: Tuple[Optional[str], Optional[str]]
    n_train: int
    n_validate: int
    n_test: int
    chosen_params: Dict[str, Any]
    is_metrics: Dict[str, Any]        # validate-window metrics of chosen params
    oos_metrics: Dict[str, Any]       # test-window metrics of chosen params
    is_expectancy: float
    oos_expectancy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WalkForwardResult:
    base_experiment_id: str
    rank_by: str
    n_folds: int
    train_frac: float
    collapse_frac: float
    folds: List[FoldResult] = field(default_factory=list)
    is_metrics: Dict[str, Any] = field(default_factory=dict)   # pooled validate
    oos_metrics: Dict[str, Any] = field(default_factory=dict)  # pooled test
    is_expectancy: float = 0.0
    oos_expectancy: float = 0.0
    oos_capture_ratio: Optional[float] = None   # oos_exp / is_exp when is_exp>0
    oos_collapse: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["folds"] = [f.to_dict() for f in self.folds]
        return d


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def _fold_windows(n: int, n_folds: int) -> List[Tuple[int, int, int]]:
    """Return ``(in_sample_end, test_start, test_end)`` index triples.

    The back portion of the series is tiled into ``n_folds`` contiguous TEST
    windows; each fold's IN-SAMPLE region is everything before its TEST window.
    Test windows partition the second half of the series so that fold 0 keeps a
    non-empty in-sample region.
    """
    if n < 2 or n_folds < 1:
        return []
    # Reserve the first half (at least 1 row) for the earliest in-sample region.
    test_region_start = max(1, n // 2)
    test_rows = n - test_region_start
    if test_rows < 1:
        return []
    folds = min(n_folds, test_rows)
    base = test_rows // folds
    rem = test_rows % folds
    windows: List[Tuple[int, int, int]] = []
    cursor = test_region_start
    for i in range(folds):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            continue
        t_start = cursor
        t_end = cursor + size
        windows.append((t_start, t_start, t_end))  # in_sample_end == test_start
        cursor = t_end
    return windows


def _range(rows: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    if not rows:
        return (None, None)
    return (str(rows[0].get("as_of") or ""), str(rows[-1].get("as_of") or ""))


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(dataset: Sequence[dict],
                 base_cfg: ExperimentConfig,
                 param_grid: Dict[str, Sequence[Any]], *,
                 n_folds: int = 4,
                 train_frac: float = 0.6,
                 strategy_fn: Optional[Callable] = None,
                 rank_by: str = "expectancy",
                 higher_is_better: Optional[bool] = None,
                 min_trades: int = 1,
                 collapse_frac: float = 0.5) -> WalkForwardResult:
    """Chronological walk-forward. For each fold: sweep ``param_grid`` on the
    VALIDATE slice, freeze the winner, score once on TEST. Pool IS (validate)
    and OOS (test) trades across folds and flag ``oos_collapse``."""
    rows = _as_rows(dataset)
    windows = _fold_windows(len(rows), max(1, int(n_folds)))

    fold_results: List[FoldResult] = []
    pooled_is: List[dict] = []
    pooled_oos: List[dict] = []

    for i, (is_end, t_start, t_end) in enumerate(windows):
        in_sample = rows[:is_end]
        test = rows[t_start:t_end]
        if not in_sample or not test:
            continue
        # Split in-sample chronologically into TRAIN | VALIDATE.
        cut = max(1, min(len(in_sample) - 1,
                         int(round(len(in_sample) * float(train_frac)))))
        train = in_sample[:cut]
        validate = in_sample[cut:] or in_sample[-1:]

        # Choose params on VALIDATE only (never on TEST, never on full sample).
        rep = sweep(param_grid, base_cfg, validate, strategy_fn=strategy_fn,
                    rank_by=rank_by, higher_is_better=higher_is_better,
                    min_trades=min_trades, keep_trades=True)
        if not rep.results:
            continue
        chosen = rep.results[0].params

        # Score the frozen winner on VALIDATE (IS) and TEST (OOS).
        frozen = ExperimentConfig.make(
            f"{base_cfg.experiment_id}__wf{i}",
            seed=base_cfg.seed, params=chosen, symbols=list(base_cfg.symbols),
            mode=base_cfg.mode)
        is_res = run_experiment(frozen, validate, strategy_fn=strategy_fn,
                                keep_trades=True)
        oos_res = run_experiment(frozen, test, strategy_fn=strategy_fn,
                                 keep_trades=True)

        pooled_is.extend(is_res.trades)
        pooled_oos.extend(oos_res.trades)

        fold_results.append(FoldResult(
            fold=i,
            train_range=_range(train),
            validate_range=_range(validate),
            test_range=_range(test),
            n_train=len(train),
            n_validate=len(validate),
            n_test=len(test),
            chosen_params=chosen,
            is_metrics=is_res.metrics,
            oos_metrics=oos_res.metrics,
            is_expectancy=is_res.metrics.get("expectancy", 0.0),
            oos_expectancy=oos_res.metrics.get("expectancy", 0.0),
        ))

    is_metrics = _metrics.compute_metrics(pooled_is)
    oos_metrics = _metrics.compute_metrics(pooled_oos)
    is_exp = is_metrics.get("expectancy", 0.0) or 0.0
    oos_exp = oos_metrics.get("expectancy", 0.0) or 0.0

    capture = None
    if is_exp > 0:
        capture = round(oos_exp / is_exp, 6)

    # Collapse: IS profitable but OOS gives most of it back (or goes negative).
    collapse = False
    if is_exp > 0:
        if oos_exp <= 0:
            collapse = True
        elif oos_exp < is_exp * float(collapse_frac):
            collapse = True

    return WalkForwardResult(
        base_experiment_id=base_cfg.experiment_id,
        rank_by=str(rank_by),
        n_folds=len(fold_results),
        train_frac=float(train_frac),
        collapse_frac=float(collapse_frac),
        folds=fold_results,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        is_expectancy=round(is_exp, 6),
        oos_expectancy=round(oos_exp, 6),
        oos_capture_ratio=capture,
        oos_collapse=collapse,
    )


def save_result(result: WalkForwardResult, *,
                results_dir: str = _RESULTS_DIR_DEFAULT) -> str:
    """Persist to ``<results_dir>/<base_id>__walkforward.json``. Fail-open."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir,
                            f"{result.base_experiment_id}__walkforward.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, default=str, indent=2,
                      sort_keys=True)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[lab.walk_forward] save failed: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _make_row(day: int, mom: float, ret: float, regime: str = "trending") -> dict:
    return {
        "symbol": "AAA",
        "as_of": f"2024-01-{day:02d}T16:00:00",
        "mode": "intraday",
        "ctx": {"momentum_5d_pct": mom, "regime": regime},
        "label_return_pct": ret,
    }


def _self_test() -> int:
    ok = True

    # Window math: 10 rows, 4 folds -> tile the back 5 rows into 4 windows.
    w = _fold_windows(10, 4)
    if not w or w[0][0] != 5:
        print("FAIL: fold windows", w); ok = False
    # Every test window starts where its in-sample region ends (no overlap).
    if any(is_end != t_start for (is_end, t_start, _t_end) in w):
        print("FAIL: is_end != test_start", w); ok = False
    # Windows tile contiguously to the end.
    if w and (w[0][1] != 5 or w[-1][2] != 10):
        print("FAIL: windows do not tile to end", w); ok = False

    # --- Robust rule: momentum edge persists into OOS (no collapse). --------- #
    robust = []
    for d in range(1, 13):
        if d % 2 == 1:
            robust.append(_make_row(d, 3.0, 4.0))     # up call -> +4%
        else:
            robust.append(_make_row(d, -3.0, -3.0))   # down put -> +3%
    grid = {"momentum_threshold_pct": [1.0, 2.0]}
    base = ExperimentConfig.make("wf_robust", seed=1,
                                 params={"notional": 1000.0}, symbols=["AAA"])
    r = walk_forward(robust, base, grid, n_folds=3, train_frac=0.6)
    if r.n_folds < 1:
        print("FAIL: robust produced no folds"); ok = False
    if r.oos_collapse:
        print("FAIL: robust flagged collapse", r.oos_expectancy,
              r.is_expectancy); ok = False
    if r.oos_expectancy <= 0:
        print("FAIL: robust OOS non-positive", r.oos_expectancy); ok = False

    # Determinism.
    r2 = walk_forward(robust, base, grid, n_folds=3, train_frac=0.6)
    if r.to_dict() != r2.to_dict():
        print("FAIL: walk_forward non-deterministic"); ok = False

    # --- Overfit rule: IS looks good, OOS reverses -> collapse fires. -------- #
    overfit = []
    for d in range(1, 9):
        overfit.append(_make_row(d, 3.0, 5.0))        # in-sample: call wins big
    for d in range(9, 15):
        overfit.append(_make_row(d, 3.0, -6.0))       # OOS: same signal loses
    o = walk_forward(overfit, base, grid, n_folds=2, train_frac=0.6)
    if not o.oos_collapse:
        print("FAIL: overfit not flagged as collapse",
              o.is_expectancy, o.oos_expectancy); ok = False
    if o.is_expectancy <= 0:
        print("FAIL: overfit IS should be positive", o.is_expectancy); ok = False

    # --- Separation invariant: no test row leaks into the validate pool. ----- #
    # The last fold's TEST window is the tail of the series; its rows must not
    # appear in any fold's chosen-params IS scoring window.
    if r.folds:
        test_as_ofs = set()
        val_as_ofs = set()
        for f in r.folds:
            # ranges are (first, last) as_of strings; rebuild membership by
            # re-deriving from the sorted rows is overkill — instead assert the
            # test range starts at/after the validate range end for each fold.
            v_end = f.validate_range[1]
            t_start = f.test_range[0]
            if v_end is not None and t_start is not None and t_start < v_end:
                print("FAIL: test starts before validate ends", f.to_dict())
                ok = False

    # Degenerate inputs never raise.
    for junk in (None, 42, "x", [], [{"bad": 1}]):
        try:
            walk_forward(junk, base, grid)   # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk dataset", junk, exc); ok = False

    print("lab.walk_forward self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
