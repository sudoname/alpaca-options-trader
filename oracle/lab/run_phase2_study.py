"""
Oracle Lab — Phase-2 offline study harness (ANALYTICS ONLY, no creds/network).

Phase-1 built the Lab; Phase-2 *runs* it. This harness ties the existing lab
pieces into one deterministic, offline pipeline that produces the out-of-sample
evidence a promotion is supposed to depend on:

    build_dataset(point-in-time)      (oracle.lab.dataset)
        -> walk_forward (IS vs OOS)   (oracle.lab.walk_forward)
        -> parameter_stability        (oracle.lab.parameter_stability)
        -> compute/format report      (oracle.lab.reports)
        -> results/<id>__phase2_study.json

Nothing here trades, sizes, prices, blocks, or alters a real/paper order, reads
creds, or hits the network. It only reads a caller-supplied dataset (or builds
one from a caller-supplied ``market_view_factory``) and writes a JSON report to
``oracle/lab/results/`` when asked. The live path in ``smart_trader.py`` is
byte-identical whether or not this module is imported.

Flag: ``ENABLE_ORACLE_LAB`` documents intent to use the offline research surface.
Because the Lab has no live effect, the flag is advisory here (logged, not
enforced) — the study is safe to run offline regardless.

Determinism: ``run_study`` is a pure function of (dataset, base_cfg, param_grid,
n_folds, train_frac, seed, rank_by). Re-running yields a byte-identical result.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

from oracle.lab.dataset import build_dataset
from oracle.lab.experiment import ExperimentConfig, run_experiment
from oracle.lab.parameter_stability import analyze as _analyze_stability
from oracle.lab.parameter_sweep import sweep
from oracle.lab.reports import compute_lab_report, format_lab_report
from oracle.lab.walk_forward import _as_rows, walk_forward

_RESULTS_DIR_DEFAULT = os.path.join("oracle", "lab", "results")


def lab_enabled() -> bool:
    """Advisory read of ``ENABLE_ORACLE_LAB`` (fail-open false). The study is
    offline research, so this only documents intent — it never blocks a run."""
    val = str(os.environ.get("ENABLE_ORACLE_LAB", "")).strip().lower()
    return val in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Dataset construction (thin wrapper -> plain dict rows the harness consumes)
# --------------------------------------------------------------------------- #
def build_study_dataset(symbols: Sequence[str], as_ofs: Sequence[datetime], *,
                        market_view_factory: Callable[[datetime], Any],
                        horizon: timedelta,
                        mode: str = "intraday",
                        feature_fn: Optional[Callable] = None,
                        price_fn: Optional[Callable] = None,
                        flat_band_pct: float = 0.0) -> List[dict]:
    """Build point-in-time rows via ``oracle.lab.dataset.build_dataset`` and
    return them as plain dicts (the shape the experiment/sweep layers expect)."""
    rows = build_dataset(symbols, as_ofs,
                         market_view_factory=market_view_factory,
                         horizon=horizon, mode=mode, feature_fn=feature_fn,
                         price_fn=price_fn, flat_band_pct=flat_band_pct)
    return [r.to_dict() for r in rows]


# --------------------------------------------------------------------------- #
# Pooled IS / OOS trades for the report breakdowns
# --------------------------------------------------------------------------- #
def _slice_by_range(rows: List[dict], rng: Any) -> List[dict]:
    """Rows whose ``as_of`` falls in the inclusive [lo, hi] window a fold
    recorded. Deterministic; empty when the range is missing."""
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        return []
    lo, hi = rng[0], rng[1]
    if not lo or not hi:
        return []
    lo, hi = str(lo), str(hi)
    return [r for r in rows if lo <= str(r.get("as_of") or "") <= hi]


def _pooled_trades(dataset: Sequence[dict], base_cfg: ExperimentConfig,
                   wf, *, strategy_fn: Optional[Callable] = None):
    """Re-score each fold's FROZEN params on its recorded validate/test window
    to recover the pooled IS/OOS trade lists (walk_forward keeps only metrics).
    Reuses the fold's own ``chosen_params`` — no parameter selection here."""
    rows = _as_rows(dataset)
    pooled_is: List[dict] = []
    pooled_oos: List[dict] = []
    for f in getattr(wf, "folds", []) or []:
        frozen = ExperimentConfig.make(
            f"{base_cfg.experiment_id}__wf{f.fold}", seed=base_cfg.seed,
            params=f.chosen_params, symbols=list(base_cfg.symbols),
            mode=base_cfg.mode)
        val = _slice_by_range(rows, f.validate_range)
        test = _slice_by_range(rows, f.test_range)
        pooled_is.extend(run_experiment(frozen, val, strategy_fn=strategy_fn,
                                        keep_trades=True).trades)
        pooled_oos.extend(run_experiment(frozen, test, strategy_fn=strategy_fn,
                                         keep_trades=True).trades)
    return pooled_is, pooled_oos


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #
def run_study(dataset: Sequence[dict], *,
              base_cfg: ExperimentConfig,
              param_grid: Dict[str, Sequence[Any]],
              n_folds: int = 4,
              train_frac: float = 0.6,
              rank_by: str = "expectancy",
              higher_is_better: Optional[bool] = None,
              min_trades: int = 1,
              collapse_frac: float = 0.5,
              strategy_fn: Optional[Callable] = None,
              plateau_tol_frac: float = 0.25,
              spike_drop_frac: float = 0.5,
              results_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the full offline Phase-2 study on ``dataset`` and return a JSON-safe
    result (walk-forward + stability + report + rendered text).

    ``results_dir`` (optional) persists the result to
    ``<results_dir>/<experiment_id>__phase2_study.json``. Pure/deterministic
    over its inputs; fail-open (a degenerate stage contributes empty rather than
    raising).
    """
    # 1) Out-of-sample certification (chronological walk-forward).
    wf = walk_forward(dataset, base_cfg, param_grid, n_folds=n_folds,
                      train_frac=train_frac, strategy_fn=strategy_fn,
                      rank_by=rank_by, higher_is_better=higher_is_better,
                      min_trades=min_trades, collapse_frac=collapse_frac)

    # 2) In-sample robustness (plateau vs spike of the param grid). This sweep
    #    is full-sample by design (a stability diagnostic); every SweepResult is
    #    stamped in_sample_only=True, so it never certifies a winner — that is
    #    the walk-forward's job.
    sweep_rep = sweep(param_grid, base_cfg, _as_rows(dataset),
                      strategy_fn=strategy_fn, rank_by=rank_by,
                      higher_is_better=higher_is_better, min_trades=min_trades)
    stability = _analyze_stability(sweep_rep, param_grid,
                                   plateau_tol_frac=plateau_tol_frac,
                                   spike_drop_frac=spike_drop_frac)

    # 3) Pooled IS/OOS trades (for the report's breakdown section) + report.
    is_trades, oos_trades = _pooled_trades(dataset, base_cfg, wf,
                                           strategy_fn=strategy_fn)
    report = compute_lab_report(wf, oos_trades=oos_trades, is_trades=is_trades,
                                stability=stability)

    out: Dict[str, Any] = {
        "experiment_id": base_cfg.experiment_id,
        "lab_enabled": lab_enabled(),
        "n_rows": len(_as_rows(dataset)),
        "walk_forward": wf.to_dict(),
        "stability": stability.to_dict(),
        "sweep": sweep_rep.to_dict(),
        "report": report,
        "report_text": format_lab_report(report),
    }
    if results_dir:
        _persist(out, base_cfg.experiment_id, results_dir)
    return out


def _persist(out: Dict[str, Any], experiment_id: str, results_dir: str) -> str:
    """Write the study result to JSON. Fail-open; returns path or ''."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{experiment_id}__phase2_study.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, default=str, indent=2, sort_keys=True)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[lab.run_phase2_study] save failed: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _demo_dataset() -> List[dict]:
    """A small, self-contained momentum dataset (offline; no MarketView)."""
    rows: List[dict] = []
    for d in range(1, 25):
        if d % 2 == 1:
            mom, ret = 3.0, 4.0     # up signal -> call wins
        else:
            mom, ret = -3.0, -3.0   # down signal -> put wins
        rows.append({
            "symbol": "DEMO",
            "as_of": f"2024-01-{d:02d}T16:00:00",
            "mode": "intraday",
            "ctx": {"momentum_5d_pct": mom, "regime": "trending"},
            "label_return_pct": ret,
        })
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Oracle Lab Phase-2 offline study")
    p.add_argument("--write", action="store_true",
                   help="persist the study JSON to oracle/lab/results/")
    p.add_argument("--results-dir", default=_RESULTS_DIR_DEFAULT)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--train-frac", type=float, default=0.6)
    args = p.parse_args(argv)

    if not lab_enabled():
        print("[run_phase2_study] note: ENABLE_ORACLE_LAB is not set "
              "(offline research runs anyway; flag documents intent).")

    base = ExperimentConfig.make("phase2_demo", seed=1,
                                 params={"notional": 1000.0}, symbols=["DEMO"])
    grid = {"momentum_threshold_pct": [1.0, 2.0]}
    out = run_study(_demo_dataset(), base_cfg=base, param_grid=grid,
                    n_folds=args.n_folds, train_frac=args.train_frac,
                    results_dir=(args.results_dir if args.write else None))
    _safe_print(out["report_text"])
    _safe_print(f"\nverdict={out['report']['verdict']} "
                f"oos_collapse={out['report']['is_vs_oos']['oos_collapse']}")
    return 0


def _safe_print(text: str) -> None:
    """Print tolerant of a narrow console encoding (Windows cp1252 can't render
    the report's emoji header). Falls back to a replaced-char encode."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, "replace").decode(enc, "replace"))


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

    base = ExperimentConfig.make("study_selftest", seed=1,
                                 params={"notional": 1000.0}, symbols=["AAA"])
    grid = {"momentum_threshold_pct": [1.0, 2.0]}

    # --- Robust study: OOS survives -> no collapse, IS-vs-OOS present. ------- #
    robust = []
    for d in range(1, 25):
        if d % 2 == 1:
            robust.append(_make_row(d, 3.0, 4.0))
        else:
            robust.append(_make_row(d, -3.0, -3.0))
    out = run_study(robust, base_cfg=base, param_grid=grid, n_folds=3)

    # The report MUST carry an IS-vs-OOS section (acceptance).
    if "is_vs_oos" not in out.get("report", {}):
        print("FAIL: study report missing IS-vs-OOS"); ok = False
    if out["report"]["is_vs_oos"]["oos_collapse"] is not False:
        print("FAIL: robust study flagged collapse"); ok = False
    if "In-sample vs Out-of-sample" not in out.get("report_text", ""):
        print("FAIL: report_text missing IS-vs-OOS header"); ok = False

    # Sweep candidates are all stamped in-sample-only (methodology guard).
    sweep_results = out.get("sweep", {}).get("results", [])
    if not sweep_results or not all(r.get("in_sample_only") for r in sweep_results):
        print("FAIL: sweep not in_sample_only"); ok = False

    # OOS breakdowns unlocked (pooled OOS trades recovered).
    oos_bd = out["report"].get("breakdowns", {}).get("oos", {})
    if "by_direction" not in oos_bd or not oos_bd["by_direction"]:
        print("FAIL: OOS breakdown missing", oos_bd); ok = False

    # Determinism: identical JSON-safe result on re-run.
    out2 = run_study(robust, base_cfg=base, param_grid=grid, n_folds=3)
    if json.dumps(out, sort_keys=True, default=str) != \
            json.dumps(out2, sort_keys=True, default=str):
        print("FAIL: non-deterministic study"); ok = False

    # --- Overfit study: OOS collapses -> verdict OVERFIT. ------------------- #
    overfit = [_make_row(d, 3.0, 5.0) for d in range(1, 13)]       # IS winners
    overfit += [_make_row(d, 3.0, -6.0) for d in range(13, 25)]    # OOS losers
    o = run_study(overfit, base_cfg=base, param_grid=grid, n_folds=1)
    if o["report"]["is_vs_oos"]["oos_collapse"] is not True:
        print("FAIL: overfit study did not flag collapse",
              o["report"]["is_vs_oos"]); ok = False
    if o["report"]["verdict"] != "OVERFIT":
        print("FAIL: overfit verdict", o["report"]["verdict"]); ok = False

    # --- build_study_dataset path (point-in-time, no leak) is wired. -------- #
    try:
        from market_view import HistoricalMarketView, make_bar
        dates = [f"2024-02-{i:02d}" for i in range(1, 13)]
        prices = [100.0 + i for i in range(12)]
        bars = [make_bar(d, p, p + 0.5, p - 0.5, p, 1_000_000)
                for d, p in zip(dates, prices)]

        def factory(as_of):
            return HistoricalMarketView(as_of, daily={"SPY": list(bars)})

        as_ofs = [datetime(2024, 2, i, 16, 0) for i in range(2, 11)]
        rows = build_study_dataset(["SPY"], as_ofs, market_view_factory=factory,
                                   horizon=timedelta(days=1))
        if not rows:
            print("FAIL: build_study_dataset produced no rows"); ok = False
        if any("ctx" not in r for r in rows):
            print("FAIL: study rows missing ctx"); ok = False
    except Exception as exc:  # pragma: no cover - MarketView optional in some envs
        print("FAIL: build_study_dataset path raised", exc); ok = False

    # Junk tolerance: degenerate dataset never raises.
    for junk in (None, 42, "x", [], [{"bad": 1}]):
        try:
            run_study(junk, base_cfg=base, param_grid=grid)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk dataset", junk, exc); ok = False

    print("lab.run_phase2_study self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(_self_test())
    sys.exit(main())
