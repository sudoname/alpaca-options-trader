"""
Oracle 3.0 — Phase 3: probability-of-profit RECALIBRATION with temporal
integrity.

`pop_calibration.py` (10G-B) only *reports* predicted-vs-realized win rates.
This module *fits* a correction — a monotone (isotonic) mapping from the raw
model PoP to an empirically calibrated PoP — under the honesty rules the
master prompt demands:

  * PRIOR SESSIONS ONLY: the calibrator applied to a session is fit
    exclusively on observations from strictly earlier sessions whose outcome
    window CLOSED before that session began (reusing
    `research_validation.purge_overlapping`, plus an optional embargo).
    The first sessions therefore run uncalibrated — reported as such, never
    silently backfilled.
  * OUTCOME LABELS ALIGNED TO THE EVALUATED POLICY: an observation's label is
    net outcome > 0 under ONE named exit policy. Fitting mixes policies only
    if the caller does not name one; when ``exit_policy`` is given, rows from
    other policies are excluded and counted, never blended.
  * EVIDENCE CLASSES: realized outcomes only by default. Explicitly labeled
    simulated outcomes join the fit only on request
    (``include_simulated=True``) and are counted in the output.
  * POOLED / UNCALIBRATED FALLBACK: a cohort (e.g. regime) with fewer than
    ``min_cohort_n`` fitted examples falls back to the pooled calibrator;
    if even pooled evidence is under ``min_pooled_n`` the raw probability is
    passed through unchanged. Every calibrated number carries its ``source``
    ('cohort' | 'pooled' | 'uncalibrated') and the sample size it was fit on.
  * UNTOUCHED FINAL EVALUATION PERIOD: the last ``final_eval_frac`` of
    sessions is flagged ``final_eval=True`` and EXCLUDED from the summary.
    Evaluate it once, at the very end, via ``final_eval_report``.
  * CALIBRATION =/= PROFITABILITY: this module reports Brier scores and
    reliability tables only. Profitability belongs to
    `research_validation.performance_metrics`, computed separately.

Pure python, deterministic, no I/O, no network. Observations use the
`research_validation` dict shape plus optional ``exit_policy`` and
``provenance`` fields (exit-policy replays from `exit_replay.py` supply both).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from research_validation import (
    _day,
    brier_score,
    calibration_table,
    purge_overlapping,
)

#: Fallback sources, strongest first.
SOURCE_COHORT = "cohort"
SOURCE_POOLED = "pooled"
SOURCE_UNCALIBRATED = "uncalibrated"

DEFAULT_MIN_COHORT_N = 30
DEFAULT_MIN_POOLED_N = 20


# --------------------------------------------------------------------------- #
# Isotonic (PAV) calibrator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone step function fit by pool-adjacent-violators on
    (raw p, binary outcome) pairs. Deterministic; prediction clamps to [0,1].
    ``blocks`` is a tuple of (p_lo, p_hi, mean, n) with non-decreasing means."""
    blocks: Tuple[Tuple[float, float, float, int], ...]
    n: int

    @classmethod
    def fit(cls, pairs: Sequence[Tuple[float, float]]) -> "IsotonicCalibrator":
        pts = sorted((max(0.0, min(1.0, float(p))), 1.0 if y else 0.0)
                     for p, y in pairs)
        # Pool ties first: identical raw p values must share one block,
        # otherwise sorted (p, 0) rows precede (p, 1) rows and PAV sees a
        # spurious non-violation. Each group: [sum_y, n, p].
        grouped: List[List[float]] = []
        for p, y in pts:
            if grouped and grouped[-1][2] == p:
                grouped[-1][0] += y
                grouped[-1][1] += 1
            else:
                grouped.append([y, 1, p])
        # Each block: [sum_y, n, p_lo, p_hi]
        blocks: List[List[float]] = []
        for sum_y, cnt, p in grouped:
            blocks.append([sum_y, cnt, p, p])
            while len(blocks) >= 2 and \
                    blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                b = blocks.pop()
                blocks[-1][0] += b[0]
                blocks[-1][1] += b[1]
                blocks[-1][3] = b[3]
        return cls(blocks=tuple((b[2], b[3], b[0] / b[1], int(b[1]))
                                for b in blocks),
                   n=len(pts))

    def predict(self, p) -> Optional[float]:
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None
        if not self.blocks:
            return None
        p = max(0.0, min(1.0, p))
        for lo, hi, mean, _ in self.blocks:
            if p <= hi:
                return round(mean, 6)
        return round(self.blocks[-1][2], 6)


# --------------------------------------------------------------------------- #
# Observation hygiene
# --------------------------------------------------------------------------- #
def _usable(o: Dict, exit_policy: Optional[str],
            include_simulated: bool) -> Optional[str]:
    """None when the observation can enter the fit/evaluation; otherwise the
    reason it is excluded (so exclusions are countable, never silent)."""
    if not isinstance(o, dict):
        return "not_a_dict"
    if o.get("p_win") is None:
        return "no_prediction"
    if o.get("net_pnl_pct") is None:
        return "no_outcome"
    if exit_policy is not None and o.get("exit_policy") != exit_policy:
        return "exit_policy_mismatch"
    if o.get("simulated") and not include_simulated:
        return "simulated_excluded"
    return None


def _label(o: Dict) -> float:
    return 1.0 if float(o["net_pnl_pct"]) > 0 else 0.0


# --------------------------------------------------------------------------- #
# Fit + fallback
# --------------------------------------------------------------------------- #
def fit_pop_calibrators(train: Sequence[Dict], *,
                        cohort_key: str = "regime",
                        min_cohort_n: int = DEFAULT_MIN_COHORT_N,
                        min_pooled_n: int = DEFAULT_MIN_POOLED_N,
                        exit_policy: Optional[str] = None,
                        include_simulated: bool = False) -> Dict:
    """Fit a pooled calibrator plus per-cohort calibrators on usable training
    observations. Cohorts below ``min_cohort_n`` are recorded with their
    sample size but carry no calibrator (they will fall back)."""
    excluded: Dict[str, int] = {}
    usable: List[Dict] = []
    for o in train:
        why = _usable(o, exit_policy, include_simulated)
        if why is None:
            usable.append(o)
        else:
            excluded[why] = excluded.get(why, 0) + 1

    pooled_pairs = [(float(o["p_win"]), _label(o)) for o in usable]
    pooled = (IsotonicCalibrator.fit(pooled_pairs)
              if len(pooled_pairs) >= min_pooled_n else None)

    cohorts: Dict[str, Dict] = {}
    by_cohort: Dict[str, List[Dict]] = {}
    for o in usable:
        by_cohort.setdefault(str(o.get(cohort_key) or "unknown"), []).append(o)
    for name, rows in sorted(by_cohort.items()):
        cal = (IsotonicCalibrator.fit([(float(o["p_win"]), _label(o))
                                       for o in rows])
               if len(rows) >= min_cohort_n else None)
        cohorts[name] = {"calibrator": cal, "n": len(rows)}

    return {
        "pooled": {"calibrator": pooled, "n": len(pooled_pairs)},
        "cohorts": cohorts,
        "cohort_key": cohort_key,
        "min_cohort_n": min_cohort_n,
        "min_pooled_n": min_pooled_n,
        "exit_policy": exit_policy,
        "include_simulated": include_simulated,
        "n_simulated_in_fit": sum(1 for o in usable if o.get("simulated")),
        "excluded": excluded,
    }


def calibrate_p(model: Dict, p_raw, cohort=None) -> Dict:
    """Apply the fitted model with explicit fallback: cohort -> pooled ->
    uncalibrated. Always reports the source and the sample size behind it."""
    try:
        p = float(p_raw)
    except (TypeError, ValueError):
        return {"p_raw": None, "p_calibrated": None,
                "source": SOURCE_UNCALIBRATED, "n_fit": 0}
    c = model["cohorts"].get(str(cohort)) if cohort is not None else None
    if c and c["calibrator"] is not None:
        return {"p_raw": p, "p_calibrated": c["calibrator"].predict(p),
                "source": SOURCE_COHORT, "n_fit": c["n"]}
    pooled = model["pooled"]
    if pooled["calibrator"] is not None:
        return {"p_raw": p, "p_calibrated": pooled["calibrator"].predict(p),
                "source": SOURCE_POOLED, "n_fit": pooled["n"]}
    return {"p_raw": p, "p_calibrated": round(max(0.0, min(1.0, p)), 6),
            "source": SOURCE_UNCALIBRATED, "n_fit": pooled["n"]}


# --------------------------------------------------------------------------- #
# Session walk-forward: fit on prior sessions only
# --------------------------------------------------------------------------- #
def _session_start_iso(rows: List[Dict]) -> Optional[str]:
    starts = [o.get("t_decision") for o in rows if o.get("t_decision")]
    return min(starts) if starts else None


def recalibrate_walk_forward(obs: Sequence[Dict], *,
                             cohort_key: str = "regime",
                             min_cohort_n: int = DEFAULT_MIN_COHORT_N,
                             min_pooled_n: int = DEFAULT_MIN_POOLED_N,
                             embargo_minutes: float = 0.0,
                             final_eval_frac: float = 0.2,
                             exit_policy: Optional[str] = None,
                             include_simulated: bool = False) -> Dict:
    """Session-by-session recalibration with no temporal leakage.

    For every session (UTC day of ``t_decision``), fit calibrators on the
    usable observations of strictly earlier sessions whose outcome closed
    before (session start - embargo), then calibrate that session's raw
    predictions. The last ``final_eval_frac`` of sessions is flagged
    ``final_eval`` and kept out of the summary — evaluate once via
    ``final_eval_report``.

    Returns ``{"rows", "sessions", "summary", ...}`` where each row is the
    original observation plus ``p_calibrated``, ``source``, ``n_fit``,
    ``session`` and ``final_eval``.
    """
    excluded: Dict[str, int] = {}
    usable: List[Dict] = []
    for o in obs:
        why = _usable(o, exit_policy, include_simulated)
        if why is None:
            usable.append(o)
        else:
            excluded[why] = excluded.get(why, 0) + 1

    by_session: Dict[str, List[Dict]] = {}
    for o in usable:
        d = _day(o.get("t_decision"))
        if d is None:
            excluded["no_timestamp"] = excluded.get("no_timestamp", 0) + 1
            continue
        by_session.setdefault(d, []).append(o)
    sessions = sorted(by_session)
    n_final = int(len(sessions) * final_eval_frac) if sessions else 0
    final_set = set(sessions[len(sessions) - n_final:]) if n_final else set()

    rows: List[Dict] = []
    session_meta: List[Dict] = []
    history: List[Dict] = []
    for day in sessions:
        todays = sorted(by_session[day], key=lambda o: o.get("t_decision") or "")
        boundary = _session_start_iso(todays)
        train, purged = purge_overlapping(history, boundary, embargo_minutes)
        model = fit_pop_calibrators(
            train, cohort_key=cohort_key, min_cohort_n=min_cohort_n,
            min_pooled_n=min_pooled_n, exit_policy=exit_policy,
            include_simulated=include_simulated)
        src_counts: Dict[str, int] = {}
        for o in todays:
            cal = calibrate_p(model, o.get("p_win"), o.get(cohort_key))
            row = dict(o)
            row.update(cal)
            row["session"] = day
            row["final_eval"] = day in final_set
            rows.append(row)
            src_counts[cal["source"]] = src_counts.get(cal["source"], 0) + 1
        session_meta.append({
            "session": day, "n": len(todays), "train_n": model["pooled"]["n"],
            "purged": purged, "sources": src_counts,
            "final_eval": day in final_set,
        })
        history.extend(todays)

    dev = [r for r in rows if not r["final_eval"]]
    return {
        "rows": rows,
        "sessions": session_meta,
        "n_sessions": len(sessions),
        "n_final_eval_sessions": n_final,
        "final_eval_untouched": True,
        "embargo_minutes": embargo_minutes,
        "exit_policy": exit_policy,
        "include_simulated": include_simulated,
        "n_simulated": sum(1 for r in rows if r.get("simulated")),
        "excluded": excluded,
        "summary": calibration_summary(dev),
        "note": ("summary covers development sessions only; run "
                 "final_eval_report(result['rows']) once, at the very end"),
    }


# --------------------------------------------------------------------------- #
# Evaluation — calibration ONLY (profitability lives in research_validation)
# --------------------------------------------------------------------------- #
def calibration_summary(rows: Sequence[Dict]) -> Dict:
    """Brier raw vs calibrated plus reliability tables. Deliberately carries
    no P&L figures: calibration and profitability are evaluated separately."""
    raw = [r for r in rows if r.get("p_win") is not None
           and r.get("net_pnl_pct") is not None]
    cal_rows = [dict(r, p_win=r["p_calibrated"]) for r in raw
                if r.get("p_calibrated") is not None]
    by_source: Dict[str, int] = {}
    n_fits = []
    for r in raw:
        by_source[r.get("source", SOURCE_UNCALIBRATED)] = \
            by_source.get(r.get("source", SOURCE_UNCALIBRATED), 0) + 1
        n_fits.append(int(r.get("n_fit") or 0))
    return {
        "n": len(raw),
        "brier_raw": brier_score(raw).get("brier"),
        "brier_calibrated": brier_score(cal_rows).get("brier"),
        "reliability_raw": calibration_table(raw),
        "reliability_calibrated": calibration_table(cal_rows),
        "sources": by_source,
        "avg_n_fit": (sum(n_fits) / len(n_fits)) if n_fits else None,
    }


def final_eval_report(rows: Sequence[Dict]) -> Dict:
    """One-shot evaluation of the held-out final sessions. Call exactly once,
    by explicit choice, after all development is frozen."""
    final = [r for r in rows if r.get("final_eval")]
    return dict(calibration_summary(final),
                n_sessions=len({r.get("session") for r in final}),
                period="final_eval")


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network, no files)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import random

    ok = True

    # PAV: overconfident 0.8s with a 50% hit rate calibrate to ~0.5.
    rng = random.Random(3)
    pairs = [(0.8, 1 if rng.random() < 0.5 else 0) for _ in range(400)]
    iso = IsotonicCalibrator.fit(pairs)
    if abs(iso.predict(0.8) - 0.5) > 0.08:
        print("FAIL: isotonic recenters overconfidence", iso.predict(0.8))
        ok = False

    # Monotone: predictions never decrease in p.
    pairs = [(rng.random(), 1 if rng.random() < 0.5 else 0) for _ in range(300)]
    iso = IsotonicCalibrator.fit(pairs)
    grid = [iso.predict(x / 100.0) for x in range(101)]
    if any(a > b + 1e-12 for a, b in zip(grid, grid[1:])):
        print("FAIL: monotonicity")
        ok = False

    # Fallback chain with sample sizes.
    train = [{"p_win": 0.7, "net_pnl_pct": 1.0, "regime": "trend",
              "t_decision": "2026-01-01T10:00:00+00:00"} for _ in range(25)]
    model = fit_pop_calibrators(train, min_cohort_n=30, min_pooled_n=20)
    c = calibrate_p(model, 0.7, "trend")
    if c["source"] != SOURCE_POOLED or c["n_fit"] != 25:
        print("FAIL: pooled fallback", c)
        ok = False
    model2 = fit_pop_calibrators(train[:5], min_cohort_n=30, min_pooled_n=20)
    c2 = calibrate_p(model2, 0.7, "trend")
    if c2["source"] != SOURCE_UNCALIBRATED or c2["p_calibrated"] != 0.7:
        print("FAIL: uncalibrated fallback", c2)
        ok = False

    print("pop_recalibration self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
