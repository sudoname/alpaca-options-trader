"""
Oracle Lab — walk-forward / robustness report (ANALYTICS ONLY, offline).

Where ``walk_forward`` *produces* the in-sample-vs-out-of-sample numbers and
``parameter_stability`` *produces* the plateau-vs-spike numbers, this module
*presents* them: a ``compute_/format_/generate_`` trio (mirrors
``oracle_prob_calibration.py``) that turns a ``WalkForwardResult`` (plus optional
pooled trades and a ``StabilityResult``) into

  * an IS-vs-OOS headline (expectancy in-sample vs out-of-sample, the capture
    ratio, and the ``oos_collapse`` "did it overfit?" flag),
  * per-fold IS/OOS expectancy with the frozen params,
  * OOS breakdowns by regime / direction (CALL / PUT) / catalyst / strategy_mode
    / EV bucket / PoP bucket (reusing ``oracle.lab.metrics.breakdown_*``),
  * an optional parameter-stability summary (plateau size, spike flag).

The verdict is deliberately conservative and evidence-first:
  ``ROBUST``            OOS survived (no collapse) with a positive OOS edge,
  ``OVERFIT``           IS looked good but OOS collapsed,
  ``NO_OOS_EDGE``       ran cleanly but OOS expectancy is non-positive,
  ``INSUFFICIENT_DATA`` too few folds / OOS trades to judge.

STRICTLY analytics: pure functions over already-computed research objects. No
network, no creds, no trading side effects, fail-open to INSUFFICIENT_DATA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from oracle.lab import metrics as _metrics

VERDICT_ROBUST = "ROBUST"
VERDICT_OVERFIT = "OVERFIT"
VERDICT_NO_EDGE = "NO_OOS_EDGE"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

MIN_OOS_TRADES = 10          # pooled test trades needed for a non-insufficient verdict
ANALYTICS_FOOTER = "Analytics only — never sizes, blocks or alters a trade."
STUDY_QUESTION = (
    "Did the tuned rule survive out-of-sample, or did the edge collapse?")


# --------------------------------------------------------------------------- #
# Small, None-safe formatting helpers (mirror oracle_prob_calibration._num/_pct)
# --------------------------------------------------------------------------- #
def _num(v: Any, nd: int = 3) -> str:
    try:
        if v is None:
            return "n/a"
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _pct(v: Any, nd: int = 1) -> str:
    try:
        if v is None:
            return "n/a"
        return f"{float(v) * 100.0:.{nd}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Normalize a dataclass-like result (``.to_dict()``) or a plain mapping."""
    if obj is None:
        return {}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return dict(obj.to_dict())
        except Exception:  # pragma: no cover - defensive
            return {}
    if isinstance(obj, dict):
        return dict(obj)
    return {}


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
def _oos_breakdowns(trades: Optional[Sequence[dict]]) -> Dict[str, Any]:
    """OOS breakdowns reusing the pure metrics helpers. Empty when no trades."""
    rows = [t for t in (trades or []) if isinstance(t, dict)]
    if not rows:
        return {}
    return {
        "by_regime": _metrics.breakdown_by_regime(rows),
        "by_direction": _metrics.breakdown_by_direction(rows),
        "by_catalyst": _metrics.breakdown_by_catalyst(rows),
        "by_mode": _metrics.breakdown_by_mode(rows),
        "by_ev": _metrics.breakdown_by_ev(rows),
        "by_pop": _metrics.breakdown_by_pop(rows),
    }


def _verdict(n_folds: int, oos_trade_count: int, oos_exp: float,
             collapse: bool) -> str:
    if n_folds < 1 or oos_trade_count < MIN_OOS_TRADES:
        return VERDICT_INSUFFICIENT
    if collapse:
        return VERDICT_OVERFIT
    if oos_exp > 0:
        return VERDICT_ROBUST
    return VERDICT_NO_EDGE


def compute_lab_report(wf: Any, *,
                       oos_trades: Optional[Sequence[dict]] = None,
                       is_trades: Optional[Sequence[dict]] = None,
                       stability: Any = None) -> Dict[str, Any]:
    """Assemble a JSON-safe study report from a ``WalkForwardResult``.

    ``wf`` may be a ``WalkForwardResult`` (or its ``.to_dict()``). ``oos_trades``
    / ``is_trades`` (the pooled test / validate trades the walk-forward scored)
    are optional and only unlock the breakdown section — the IS-vs-OOS headline
    comes straight from ``wf``. ``stability`` is an optional ``StabilityResult``.

    Pure over its inputs; never raises (fail-open to INSUFFICIENT_DATA).
    """
    w = _as_dict(wf)
    is_metrics = dict(w.get("is_metrics") or {})
    oos_metrics = dict(w.get("oos_metrics") or {})
    n_folds = int(w.get("n_folds") or 0)
    is_exp = float(w.get("is_expectancy") or 0.0)
    oos_exp = float(w.get("oos_expectancy") or 0.0)
    collapse = bool(w.get("oos_collapse"))
    oos_tc = int(oos_metrics.get("trade_count") or 0)

    folds_out: List[dict] = []
    for f in (w.get("folds") or []):
        if not isinstance(f, dict):
            continue
        folds_out.append({
            "fold": f.get("fold"),
            "chosen_params": f.get("chosen_params", {}),
            "n_validate": f.get("n_validate"),
            "n_test": f.get("n_test"),
            "is_expectancy": f.get("is_expectancy"),
            "oos_expectancy": f.get("oos_expectancy"),
            "validate_range": f.get("validate_range"),
            "test_range": f.get("test_range"),
        })

    report: Dict[str, Any] = {
        "record_type": "lab_walkforward_report",
        "base_experiment_id": w.get("base_experiment_id"),
        "rank_by": w.get("rank_by"),
        "n_folds": n_folds,
        "train_frac": w.get("train_frac"),
        "collapse_frac": w.get("collapse_frac"),
        # IS-vs-OOS headline (the acceptance-critical section).
        "is_vs_oos": {
            "is_expectancy": round(is_exp, 6),
            "oos_expectancy": round(oos_exp, 6),
            "oos_capture_ratio": w.get("oos_capture_ratio"),
            "oos_collapse": collapse,
            "is_trade_count": is_metrics.get("trade_count"),
            "oos_trade_count": oos_tc,
        },
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "folds": folds_out,
        "breakdowns": {"oos": _oos_breakdowns(oos_trades)},
        "verdict": _verdict(n_folds, oos_tc, oos_exp, collapse),
    }
    if is_trades is not None:
        report["breakdowns"]["is"] = _oos_breakdowns(is_trades)

    st = _as_dict(stability)
    if st:
        report["stability"] = {
            "best_params": st.get("best_params"),
            "best_score": st.get("best_score"),
            "plateau_frac": st.get("plateau_frac"),
            "plateau_size": st.get("plateau_size"),
            "n_neighbors": st.get("n_neighbors"),
            "neighbor_max_drop": st.get("neighbor_max_drop"),
            "is_spike": st.get("is_spike"),
        }
    return report


# --------------------------------------------------------------------------- #
# Format
# --------------------------------------------------------------------------- #
def _fmt_breakdown_block(title: str, bd: Dict[str, Any]) -> List[str]:
    if not bd:
        return []
    lines = [f"*{title}:*"]
    for label in sorted(bd.keys()):
        m = bd[label] or {}
        lines.append(
            f"`{label}`: n `{m.get('trade_count')}`, "
            f"exp `{_num(m.get('expectancy'))}`, "
            f"win `{_pct(m.get('win_rate'))}`, "
            f"pf `{_num(m.get('profit_factor'))}`")
    return lines


def format_lab_report(report: Dict[str, Any]) -> str:
    """Human-readable study report. Pure formatting; matches the calibration
    report idiom (header / question / sections / verdict / footer)."""
    header = "🧪 *Oracle Lab — Walk-Forward Study* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if not report or report.get("n_folds", 0) == 0:
        return "\n".join([
            header, "", f"_{STUDY_QUESTION}_", "",
            "No completed walk-forward folds.",
            f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", footer,
        ])

    ivo = report.get("is_vs_oos", {}) or {}
    lines = [
        header, "",
        f"_{STUDY_QUESTION}_", "",
        f"*Experiment:* `{report.get('base_experiment_id')}` · "
        f"folds `{report.get('n_folds')}` · rank_by `{report.get('rank_by')}`",
        "",
        "*In-sample vs Out-of-sample:*",
        f"IS expectancy `{_num(ivo.get('is_expectancy'))}` "
        f"(n `{ivo.get('is_trade_count')}`)",
        f"OOS expectancy `{_num(ivo.get('oos_expectancy'))}` "
        f"(n `{ivo.get('oos_trade_count')}`)",
        f"OOS capture ratio `{_num(ivo.get('oos_capture_ratio'))}`",
        f"OOS collapse `{ivo.get('oos_collapse')}`",
    ]

    folds = report.get("folds") or []
    if folds:
        lines += ["", "*Per fold (IS -> OOS expectancy):*"]
        for f in folds:
            lines.append(
                f"fold `{f.get('fold')}`: "
                f"`{_num(f.get('is_expectancy'))}` -> "
                f"`{_num(f.get('oos_expectancy'))}`  "
                f"params `{f.get('chosen_params')}`")

    oos_bd = (report.get("breakdowns", {}) or {}).get("oos", {}) or {}
    if oos_bd:
        lines += [""]
        lines += _fmt_breakdown_block("OOS by regime",
                                      oos_bd.get("by_regime", {}))
        lines += _fmt_breakdown_block("OOS by direction (CALL/PUT)",
                                      oos_bd.get("by_direction", {}))
        lines += _fmt_breakdown_block("OOS by catalyst",
                                      oos_bd.get("by_catalyst", {}))
        lines += _fmt_breakdown_block("OOS by strategy_mode",
                                      oos_bd.get("by_mode", {}))
        lines += _fmt_breakdown_block("OOS by EV bucket",
                                      oos_bd.get("by_ev", {}))
        lines += _fmt_breakdown_block("OOS by PoP bucket",
                                      oos_bd.get("by_pop", {}))

    st = report.get("stability")
    if st:
        lines += [
            "",
            "*Parameter stability:*",
            f"best `{st.get('best_params')}` · "
            f"plateau_frac `{_num(st.get('plateau_frac'))}` · "
            f"spike `{st.get('is_spike')}`",
        ]

    lines += ["", f"*Verdict:* `{report.get('verdict')}`", "", footer]
    return "\n".join(lines)


def generate_lab_report_text(wf: Any, *,
                             oos_trades: Optional[Sequence[dict]] = None,
                             is_trades: Optional[Sequence[dict]] = None,
                             stability: Any = None) -> str:
    """Top-level convenience: compute then format in one call."""
    return format_lab_report(compute_lab_report(
        wf, oos_trades=oos_trades, is_trades=is_trades, stability=stability))


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
    from oracle.lab.experiment import ExperimentConfig, run_experiment
    from oracle.lab.walk_forward import walk_forward

    ok = True

    base = ExperimentConfig.make("report_selftest", seed=1,
                                 params={"notional": 1000.0}, symbols=["AAA"])
    grid = {"momentum_threshold_pct": [1.0, 2.0]}

    # --- Robust rule: OOS survives -> verdict ROBUST, collapse False. -------- #
    robust = []
    for d in range(1, 17):
        if d % 2 == 1:
            robust.append(_make_row(d, 3.0, 4.0))     # up call wins
        else:
            robust.append(_make_row(d, -3.0, -3.0))   # down put wins
    r = walk_forward(robust, base, grid, n_folds=3, train_frac=0.6)
    rep = compute_lab_report(r)

    # IS-vs-OOS section MUST be present (acceptance).
    if "is_vs_oos" not in rep:
        print("FAIL: report missing IS-vs-OOS section"); ok = False
    ivo = rep.get("is_vs_oos", {})
    for k in ("is_expectancy", "oos_expectancy", "oos_capture_ratio",
              "oos_collapse", "oos_trade_count"):
        if k not in ivo:
            print("FAIL: IS-vs-OOS missing key", k); ok = False
    if ivo.get("oos_collapse") is not False:
        print("FAIL: robust wrongly flagged collapse", ivo); ok = False

    # Formatted text carries the headline label + a verdict line.
    txt = format_lab_report(rep)
    if "In-sample vs Out-of-sample" not in txt:
        print("FAIL: formatted report missing IS-vs-OOS header"); ok = False
    if "Verdict:" not in txt:
        print("FAIL: formatted report missing verdict"); ok = False

    # Determinism: same inputs -> byte-identical report + text.
    rep2 = compute_lab_report(walk_forward(robust, base, grid,
                                           n_folds=3, train_frac=0.6))
    if rep != rep2:
        print("FAIL: non-deterministic report"); ok = False
    if generate_lab_report_text(r) != txt:
        print("FAIL: non-deterministic report text"); ok = False

    # --- Overfit rule: IS good, OOS reverses -> verdict OVERFIT. ------------- #
    # 24 rows so pooled OOS clears MIN_OOS_TRADES: days 1-12 the signal wins,
    # days 13-24 the SAME signal loses -> IS positive, OOS collapses.
    overfit = []
    for d in range(1, 13):
        overfit.append(_make_row(d, 3.0, 5.0))        # in-sample: call wins big
    for d in range(13, 25):
        overfit.append(_make_row(d, 3.0, -6.0))       # OOS: same signal loses
    o = walk_forward(overfit, base, grid, n_folds=1, train_frac=0.6)
    o_rep = compute_lab_report(o)
    if o_rep["is_vs_oos"]["oos_collapse"] is not True:
        print("FAIL: overfit not flagged collapse", o_rep["is_vs_oos"]); ok = False
    if o_rep["verdict"] != VERDICT_OVERFIT:
        print("FAIL: overfit verdict", o_rep["verdict"]); ok = False

    # --- Breakdowns unlock when pooled OOS trades are supplied. -------------- #
    oos_trades = run_experiment(
        ExperimentConfig.make("bd", seed=1,
                              params={"momentum_threshold_pct": 1.0}),
        robust, keep_trades=True).trades
    rep_bd = compute_lab_report(r, oos_trades=oos_trades)
    bd = rep_bd.get("breakdowns", {}).get("oos", {})
    if "by_direction" not in bd or not bd["by_direction"]:
        print("FAIL: OOS direction breakdown missing", bd); ok = False
    txt_bd = format_lab_report(rep_bd)
    if "OOS by direction" not in txt_bd:
        print("FAIL: formatted breakdown missing"); ok = False

    # --- Empty / junk -> INSUFFICIENT, renders, never raises. ---------------- #
    empty = compute_lab_report({})
    if empty["verdict"] != VERDICT_INSUFFICIENT:
        print("FAIL: empty verdict", empty["verdict"]); ok = False
    if VERDICT_INSUFFICIENT not in format_lab_report(empty):
        print("FAIL: empty text missing verdict"); ok = False
    for junk in (None, 42, "x", []):
        try:
            format_lab_report(compute_lab_report(junk))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("lab.reports self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
