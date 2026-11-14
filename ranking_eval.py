"""
Contract-ranking evaluation by NET replay outcomes (Phase 5 of the master
prompt): "Evaluate IV, delta, and liquidity improvements in contract ranking
by net replay outcomes."

Input: opportunity GROUPS. One group = the candidate contracts that were all
simultaneously available for one entry opportunity, each carrying the fields
the live selector had at decision time plus the NET outcome the replay
assigned to that contract under the frozen exit policy and cost model:

    {
      "symbol": str,
      "score": float,            # existing selector score (baseline ranking)
      "iv": float|None,          # as carried on the option dict
      "delta": float|None,
      "greeks_quality": {field: "broker"|"fallback"}|None,   # Phase 5b labels
      "bid": float|None, "ask": float|None,
      "quote_age_s": float|None,
      "volume": int|None, "open_interest": int|None,
      "net_outcome": float|None  # net $ P&L from replay; None = unfilled
    }

Ranking variants (each picks ONE contract per group — argmax of its key):

    baseline       — existing selector score, unchanged (frozen comparator).
    observed_only  — baseline score, but candidates whose iv or delta labels
                     are 'fallback' are down-weighted; prefers contracts with
                     broker-observed pricing inputs.
    liquidity_tilt — baseline score scaled by a spread/OI liquidity factor.
    delta_band     — baseline score, restricted to |delta| in [0.45, 0.65]
                     (falls back to baseline pick when no candidate qualifies,
                     so the variant is always comparable per opportunity).

Accounting (outcome-state aware — a missing outcome is NEVER $0):
candidates may carry `outcome_state` (research_export.py labels). The value
of a pick is decided by `outcome_value`:

    confirmed_no_fill / policy_skip      -> $0 (legitimately nothing happened)
    numeric net_outcome                  -> realized dollars (fills-based or
                                            reported net from the export)
    counterfactual_required / unresolved
      / missing_coverage / unlabeled None-> MISSING: excluded from totals and
                                            counted in `missing_reasons`.

Expectancy is reported per EVALUABLE opportunity (realized + legitimate
zeros) and per filled pick; `coverage` shows how much of the sample was
evaluable. Paired analysis (`paired_vs_baseline`) compares each variant to
the frozen baseline only on groups where BOTH picks are evaluable, reports
per-session (UTC date) dollar differences and the pick-change rate, and a
provenance block records input hash, code revision, config constants, and
cost assumptions so a run is reproducible.

Offline-first: deterministic fixtures, a JSONL loader (one group per line:
{"group_id": ..., "candidates": [...]}; research_export.py --groups-out is
compatible), and a CLI. No credentials, no network.
"""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

FALLBACK_PENALTY = 0.5          # observed_only: multiplier per fallback field
DELTA_BAND = (0.45, 0.65)       # delta_band variant
LIQ_SPREAD_SCALE = 10.0         # liquidity_tilt: spread_pct normalizer (%)
LIQ_OI_SATURATION = 500.0       # liquidity_tilt: OI saturation level

VARIANT_ORDER = ("baseline", "observed_only", "liquidity_tilt", "delta_band")

# outcome_states where $0 is the TRUE value (nothing happened, by evidence):
ZERO_STATES = ("confirmed_no_fill", "policy_skip")


def outcome_value(c: Dict) -> Tuple[Optional[float], str]:
    """Dollar value of picking candidate `c`, with an honesty label.

    Returns (value, status):
      (0.0 or net, "zero:<state>")  state confirms nothing happened ($0 true)
      (net,        "realized")      numeric net_outcome (fills/reported $)
      (None,       "missing:<why>") no evidence — EXCLUDED, never $0
    """
    state = c.get("outcome_state")
    net = _num(c.get("net_outcome"))
    if isinstance(state, str) and state.lower() in ZERO_STATES:
        return (net if net is not None else 0.0, f"zero:{state.lower()}")
    if net is not None:
        return (net, "realized")
    if isinstance(state, str) and state:
        return (None, f"missing:{state.lower()}")
    return (None, "missing:unlabeled")


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v)


def _spread_pct(c: Dict) -> Optional[float]:
    bid, ask = _num(c.get("bid")), _num(c.get("ask"))
    if bid is None or ask is None or ask <= 0 or bid <= 0 or bid > ask:
        return None
    return (ask - bid) / ask * 100.0


# --------------------------------------------------------------------------- #
# Variant ranking keys
# --------------------------------------------------------------------------- #
def _key_baseline(c: Dict) -> float:
    return _num(c.get("score")) or 0.0


def _key_observed_only(c: Dict) -> float:
    """Down-weight candidates whose iv/delta are fallback estimates.

    Uses the Phase-5b per-field labels. A candidate with no labels at all is
    treated as all-fallback (legacy rows predate the labels), which is the
    conservative reading: unlabeled means unobserved.
    """
    score = _key_baseline(c)
    quality = c.get("greeks_quality") or {}
    for field in ("iv", "delta"):
        if quality.get(field) != "broker":
            score *= FALLBACK_PENALTY
    return score


def _key_liquidity_tilt(c: Dict) -> float:
    """Baseline score scaled by spread tightness and open-interest depth.

    spread factor: 1 at 0% spread, linearly down to 0 at LIQ_SPREAD_SCALE%.
    OI factor: saturating min(oi, SAT)/SAT in [0, 1].
    Unknown spread or OI contributes a neutral-but-pessimistic 0.5 factor so
    missing liquidity evidence is never rewarded over observed evidence.
    """
    score = _key_baseline(c)
    sp = _spread_pct(c)
    sp_factor = max(0.0, 1.0 - sp / LIQ_SPREAD_SCALE) if sp is not None else 0.5
    oi = _num(c.get("open_interest"))
    oi_factor = min(oi, LIQ_OI_SATURATION) / LIQ_OI_SATURATION if oi is not None and oi >= 0 else 0.5
    return score * (0.5 + 0.5 * sp_factor) * (0.5 + 0.5 * oi_factor)


def _pick(candidates: List[Dict], key) -> Optional[Dict]:
    """Argmax with a deterministic tie-break (lexicographically smaller
    symbol wins) so picks are reproducible regardless of input order."""
    best, best_k = None, None
    for c in candidates:
        k = key(c)
        if best_k is None or k > best_k or (
                k == best_k
                and str(c.get("symbol") or "") < str(best.get("symbol") or "")):
            best, best_k = c, k
    return best


def _pick_delta_band(candidates: List[Dict]) -> Optional[Dict]:
    lo, hi = DELTA_BAND
    in_band = [c for c in candidates
               if (_num(c.get("delta")) is not None
                   and lo <= abs(_num(c.get("delta"))) <= hi)]
    if in_band:
        return _pick(in_band, _key_baseline)
    return _pick(candidates, _key_baseline)  # comparable fallback


def pick_for_variant(variant: str, candidates: List[Dict]) -> Optional[Dict]:
    if not candidates:
        return None
    if variant == "baseline":
        return _pick(candidates, _key_baseline)
    if variant == "observed_only":
        return _pick(candidates, _key_observed_only)
    if variant == "liquidity_tilt":
        return _pick(candidates, _key_liquidity_tilt)
    if variant == "delta_band":
        return _pick_delta_band(candidates)
    raise ValueError(f"unknown variant {variant!r}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_ranking(groups: Iterable[Dict],
                     variants=VARIANT_ORDER) -> Dict:
    """Matched comparison of ranking variants on identical opportunity groups.

    Same groups, same outcomes, same costs — only the pick changes. Values
    come from `outcome_value`: realized dollars and evidence-backed $0s
    (confirmed_no_fill / policy_skip) enter the totals; MISSING outcomes
    (counterfactual_required, unresolved, missing_coverage, bare None) are
    EXCLUDED and reported in `missing_reasons` — never silently $0.
    Expectancy is per evaluable opportunity; `coverage` shows how much of
    the sample backed the number.
    """
    groups = list(groups)
    report = {"n_groups": len(groups), "variants": {}, "verdict": "ok"}
    if not groups:
        report["verdict"] = "insufficient_evidence"
        return report
    for variant in variants:
        n_filled = 0
        n_evaluable = 0
        total = 0.0
        wins = 0
        losses = 0
        gross_win = 0.0
        gross_loss = 0.0
        picks = []
        picks_detail = []
        missing_reasons: Dict[str, int] = {}
        for g in groups:
            cands = g.get("candidates") or []
            pick = pick_for_variant(variant, cands)
            if pick is None:
                picks.append(None)
                picks_detail.append({"group_id": g.get("group_id"),
                                     "symbol": None,
                                     "status": "missing:empty_group",
                                     "value": None})
                missing_reasons["empty_group"] = (
                    missing_reasons.get("empty_group", 0) + 1)
                continue
            picks.append(pick.get("symbol"))
            value, status = outcome_value(pick)
            picks_detail.append({"group_id": g.get("group_id"),
                                 "symbol": pick.get("symbol"),
                                 "candidate_id": pick.get("candidate_id"),
                                 "status": status, "value": value})
            if value is None:
                why = status.split(":", 1)[1]
                missing_reasons[why] = missing_reasons.get(why, 0) + 1
                continue
            n_evaluable += 1
            total += value
            if status == "realized":
                n_filled += 1
            if value > 0:
                wins += 1
                gross_win += value
            elif value < 0:
                losses += 1
                gross_loss += -value
        n = len(groups)
        n_missing = n - n_evaluable
        report["variants"][variant] = {
            "n_groups": n,
            "n_filled": n_filled,
            "n_evaluable": n_evaluable,
            "n_missing": n_missing,
            "missing_reasons": missing_reasons,
            "coverage": n_evaluable / n if n else None,
            "fill_rate": n_filled / n_evaluable if n_evaluable else None,
            "total_net": total,
            "expectancy_per_opportunity": (total / n_evaluable
                                           if n_evaluable else None),
            "expectancy_per_fill": total / n_filled if n_filled else None,
            "win_rate_filled": wins / n_filled if n_filled else None,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0
                             else (None if gross_win == 0 else float("inf")),
            "picks": picks,
            "picks_detail": picks_detail,
        }
    return report


# --------------------------------------------------------------------------- #
# Paired per-session comparison vs the frozen baseline
# --------------------------------------------------------------------------- #
def paired_vs_baseline(groups: Iterable[Dict],
                       variants=VARIANT_ORDER) -> Dict:
    """Per-group paired dollar difference (variant − baseline).

    A pair enters only when BOTH picks are evaluable (realized or a
    legitimate zero) — a missing counterfactual excludes the pair and is
    counted in `excluded_reasons` rather than biasing the diff toward $0.
    Session key = UTC date of the group's event_at_utc (empty when absent).
    """
    groups = list(groups)
    out: Dict[str, Dict] = {}
    for variant in variants:
        if variant == "baseline":
            continue
        n_pairs = 0
        n_changed = 0
        total_diff = 0.0
        sessions: Dict[str, float] = {}
        excluded: Dict[str, int] = {}
        for g in groups:
            cands = g.get("candidates") or []
            base = pick_for_variant("baseline", cands)
            var = pick_for_variant(variant, cands)
            if base is None or var is None:
                excluded["empty_group"] = excluded.get("empty_group", 0) + 1
                continue
            bv, bs = outcome_value(base)
            vv, vs = outcome_value(var)
            if bv is None or vv is None:
                why = (bs if bv is None else vs).split(":", 1)[1]
                excluded[why] = excluded.get(why, 0) + 1
                continue
            n_pairs += 1
            if var.get("symbol") != base.get("symbol"):
                n_changed += 1
            diff = vv - bv
            total_diff += diff
            day = str(g.get("event_at_utc") or "")[:10]
            sessions[day] = sessions.get(day, 0.0) + diff
        out[variant] = {
            "n_pairs": n_pairs,
            "pick_change_rate": n_changed / n_pairs if n_pairs else None,
            "total_diff": total_diff,
            "mean_diff": total_diff / n_pairs if n_pairs else None,
            "session_diffs": dict(sorted(sessions.items())),
            "n_sessions": len(sessions),
            "n_sessions_positive": sum(1 for d in sessions.values() if d > 0),
            "n_sessions_negative": sum(1 for d in sessions.values() if d < 0),
            "excluded_reasons": excluded,
        }
    return out


def build_provenance(groups_path: Optional[str],
                     groups: List[Dict]) -> Dict:
    """Reproducibility block: what data, what code, what assumptions."""
    def _sha256_file(path):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    git_rev = None
    try:
        import subprocess
        git_rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            timeout=5).stdout.strip() or None
    except Exception:
        git_rev = None

    n_multi = sum(1 for g in groups
                  if len(g.get("candidates") or []) > 1)
    dates = sorted({str(g.get("event_at_utc") or "")[:10]
                    for g in groups if g.get("event_at_utc")})
    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": groups_path,
        "input_sha256": _sha256_file(groups_path) if groups_path else None,
        "code_git_rev": git_rev,
        "code_sha256": _sha256_file(os.path.abspath(__file__)),
        "n_groups": len(groups),
        "n_multi_candidate": n_multi,
        "n_single_candidate": len(groups) - n_multi,
        "single_candidate_note": (
            "single-candidate groups cannot differentiate variants; they "
            "contribute parity checks only"),
        "session_dates": {"first": dates[0] if dates else None,
                          "last": dates[-1] if dates else None,
                          "n": len(dates)},
        "config": {
            "FALLBACK_PENALTY": FALLBACK_PENALTY,
            "DELTA_BAND": list(DELTA_BAND),
            "LIQ_SPREAD_SCALE": LIQ_SPREAD_SCALE,
            "LIQ_OI_SATURATION": LIQ_OI_SATURATION,
            "ZERO_STATES": list(ZERO_STATES),
        },
        "cost_assumptions": (
            "net outcomes consumed as provided by the export (fills-based "
            "or reported net dollars, fees already included once); no "
            "additional costs modeled here; missing outcomes are excluded "
            "from totals, never treated as $0"),
    }


def _fmt(x, pct=False, money=False):
    if x is None:
        return "--"
    if x == float("inf"):
        return "inf"
    if pct:
        return f"{x * 100:.0f}%"
    if money:
        return f"{x:+.2f}"
    return f"{x:.2f}"


def format_report(report: Dict, paired: Optional[Dict] = None,
                  provenance: Optional[Dict] = None) -> str:
    lines = [f"ranking_eval: {report['n_groups']} opportunity groups "
             f"(verdict={report['verdict']})"]
    hdr = (f"{'variant':<15}{'eval':>6}{'miss':>6}{'fills':>7}{'cov%':>7}"
           f"{'total$':>10}{'EV/opp':>9}{'EV/fill':>9}{'win%':>7}{'PF':>7}")
    lines.append(hdr)
    for v, s in report.get("variants", {}).items():
        lines.append(f"{v:<15}{s['n_evaluable']:>6}{s['n_missing']:>6}"
                     f"{s['n_filled']:>7}{_fmt(s['coverage'], pct=True):>7}"
                     f"{_fmt(s['total_net'], money=True):>10}"
                     f"{_fmt(s['expectancy_per_opportunity'], money=True):>9}"
                     f"{_fmt(s['expectancy_per_fill'], money=True):>9}"
                     f"{_fmt(s['win_rate_filled'], pct=True):>7}"
                     f"{_fmt(s['profit_factor']):>7}")
        if s.get("missing_reasons"):
            why = ", ".join(f"{k}={n}" for k, n
                            in sorted(s["missing_reasons"].items()))
            lines.append(f"{'':<15}  missing: {why}")
    if paired:
        lines.append("")
        lines.append("paired vs baseline (pairs where BOTH picks evaluable):")
        lines.append(f"{'variant':<15}{'pairs':>7}{'chg%':>7}{'diff$':>10}"
                     f"{'mean$':>9}{'sess':>6}{'+':>4}{'-':>4}")
        for v, p in paired.items():
            lines.append(f"{v:<15}{p['n_pairs']:>7}"
                         f"{_fmt(p['pick_change_rate'], pct=True):>7}"
                         f"{_fmt(p['total_diff'], money=True):>10}"
                         f"{_fmt(p['mean_diff'], money=True):>9}"
                         f"{p['n_sessions']:>6}{p['n_sessions_positive']:>4}"
                         f"{p['n_sessions_negative']:>4}")
            if p.get("excluded_reasons"):
                why = ", ".join(f"{k}={n}" for k, n
                                in sorted(p["excluded_reasons"].items()))
                lines.append(f"{'':<15}  excluded pairs: {why}")
    if provenance:
        lines.append("")
        lines.append("provenance:")
        for k in ("run_at_utc", "input_path", "input_sha256", "code_git_rev",
                  "code_sha256", "n_groups", "n_multi_candidate",
                  "n_single_candidate", "session_dates", "config",
                  "cost_assumptions"):
            lines.append(f"  {k}: {provenance.get(k)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# IO: JSONL groups + deterministic fixture
# --------------------------------------------------------------------------- #
def load_groups_jsonl(path: str) -> List[Dict]:
    """One JSON object per line: {'group_id': ..., 'candidates': [...]}."""
    groups = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("candidates"), list):
                groups.append(row)
    return groups


def fixture_groups() -> List[Dict]:
    """Deterministic fixture exercising every variant's discriminator,
    outcome-state accounting, and the paired/session analysis."""
    def c(sym, score, iv_q, delta, bid, ask, oi, outcome, state=None):
        d = {"symbol": sym, "score": score, "iv": 0.25, "delta": delta,
             "greeks_quality": {"iv": iv_q, "delta": iv_q,
                                "gamma": iv_q, "theta": iv_q, "vega": iv_q},
             "bid": bid, "ask": ask, "volume": 100, "open_interest": oi,
             "net_outcome": outcome}
        if state is not None:
            d["outcome_state"] = state
        return d
    return [
        # G1: highest-score contract has fallback greeks and loses; the
        # broker-observed one wins — separates baseline vs observed_only.
        {"group_id": "g1", "event_at_utc": "2026-11-10T15:00:00+00:00",
         "candidates": [
            c("A1", 9.0, "fallback", 0.70, 1.00, 1.40, 50, -60.0),
            c("A2", 7.0, "broker", 0.55, 1.00, 1.05, 800, +40.0)]},
        # G2: wide-spread/thin-OI top score loses; liquid one wins —
        # separates baseline vs liquidity_tilt.
        {"group_id": "g2", "event_at_utc": "2026-11-10T16:00:00+00:00",
         "candidates": [
            c("B1", 8.0, "broker", 0.80, 0.50, 1.00, 5, -30.0),
            c("B2", 7.5, "broker", 0.50, 1.00, 1.04, 900, +25.0)]},
        # G3: top score is out of the delta band and CONFIRMED no-fill — a
        # legitimate $0 by evidence (order submitted, never filled, canceled);
        # in-band candidate fills and wins — delta_band + zero-state accounting.
        {"group_id": "g3", "event_at_utc": "2026-11-11T15:00:00+00:00",
         "candidates": [
            c("C1", 9.5, "broker", 0.90, 1.00, 1.05, 400, None,
              state="confirmed_no_fill"),
            c("C2", 8.0, "broker", 0.60, 1.00, 1.05, 400, +15.0)]},
        # G4: everyone agrees (same pick) and it loses — shared downside.
        {"group_id": "g4", "event_at_utc": "2026-11-11T16:00:00+00:00",
         "candidates": [
            c("D1", 6.0, "broker", 0.55, 1.00, 1.05, 300, -10.0)]},
        # G5: top-score alternative was never traded and has NO quote series
        # (counterfactual_required) — its value is MISSING, not $0; the
        # selected runner-up realized +20. Baseline must EXCLUDE this group
        # from totals and the paired analysis must drop the pair.
        {"group_id": "g5", "event_at_utc": "2026-11-11T17:00:00+00:00",
         "candidates": [
            c("E1", 9.0, "fallback", 0.90, 1.00, 1.30, 20, None,
              state="counterfactual_required"),
            c("E2", 7.0, "broker", 0.55, 1.00, 1.05, 700, +20.0)]},
    ]


# --------------------------------------------------------------------------- #
# Self-test + CLI
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        if not cond:
            print("FAIL", name)
            ok = False

    rep = evaluate_ranking(fixture_groups())
    v = rep["variants"]

    # Baseline picks the top raw scores: A1, B1, C1, D1, E1.
    check("baseline picks", v["baseline"]["picks"] == [
        "A1", "B1", "C1", "D1", "E1"])
    # Realized fills only (A1, B1, D1); C1 is a legitimate zero, E1 missing.
    check("baseline fills", v["baseline"]["n_filled"] == 3)
    check("baseline evaluable", v["baseline"]["n_evaluable"] == 4)
    check("baseline missing excluded (never $0)",
          v["baseline"]["n_missing"] == 1
          and v["baseline"]["missing_reasons"] == {
              "counterfactual_required": 1})
    check("baseline total", abs(v["baseline"]["total_net"] - (-100.0)) < 1e-9)
    # EV per EVALUABLE opportunity: -100 / 4 (zero counted, missing excluded).
    check("baseline EV/opp", abs(
        v["baseline"]["expectancy_per_opportunity"] - (-25.0)) < 1e-9)
    check("baseline coverage", abs(v["baseline"]["coverage"] - 0.8) < 1e-9)

    # observed_only penalizes A1's fallback labels -> picks A2 in G1,
    # and escapes the missing E1 in G5 -> E2 realized.
    check("observed_only picks A2", v["observed_only"]["picks"][0] == "A2")
    check("observed_only picks E2", v["observed_only"]["picks"][4] == "E2")
    check("observed_only full coverage",
          v["observed_only"]["n_missing"] == 0)

    # liquidity_tilt prefers the tight/deep B2 in G2.
    check("liquidity_tilt picks B2", v["liquidity_tilt"]["picks"][1] == "B2")

    # delta_band excludes C1 (0.90) -> picks C2 in G3 and fills it.
    check("delta_band picks C2", v["delta_band"]["picks"][2] == "C2")
    check("delta_band fills", v["delta_band"]["n_filled"] == 5)

    # Deterministic tie-break: equal keys -> lexicographically smaller
    # symbol, regardless of candidate order.
    t1 = [{"symbol": "ZZZ", "score": 5.0, "net_outcome": 1.0},
          {"symbol": "AAA", "score": 5.0, "net_outcome": 2.0}]
    check("tie-break is order-independent",
          _pick(t1, _key_baseline)["symbol"] == "AAA"
          and _pick(list(reversed(t1)), _key_baseline)["symbol"] == "AAA")

    # Paired analysis: only pairs where BOTH picks are evaluable; g5 is
    # excluded because baseline's pick (E1) is missing a counterfactual.
    paired = paired_vs_baseline(fixture_groups())
    po = paired["observed_only"]
    check("paired pairs exclude missing", po["n_pairs"] == 4
          and po["excluded_reasons"] == {"counterfactual_required": 1})
    check("paired diff", abs(po["total_diff"] - 100.0) < 1e-9)
    check("paired change rate", abs(po["pick_change_rate"] - 0.25) < 1e-9)
    check("paired sessions", po["n_sessions"] == 2
          and abs(po["session_diffs"]["2026-11-10"] - 100.0) < 1e-9
          and po["n_sessions_positive"] == 1)
    pd = paired["delta_band"]
    check("delta_band paired diff", abs(pd["total_diff"] - 170.0) < 1e-9)

    # outcome_value semantics directly.
    check("zero-state value", outcome_value(
        {"net_outcome": None, "outcome_state": "confirmed_no_fill"})
        == (0.0, "zero:confirmed_no_fill"))
    check("policy_skip value", outcome_value(
        {"net_outcome": None, "outcome_state": "policy_skip"})
        == (0.0, "zero:policy_skip"))
    check("missing unlabeled", outcome_value({"net_outcome": None})
          == (None, "missing:unlabeled"))
    check("missing counterfactual", outcome_value(
        {"net_outcome": None, "outcome_state": "counterfactual_required"})
        == (None, "missing:counterfactual_required"))
    check("realized value", outcome_value({"net_outcome": -12.5})
          == (-12.5, "realized"))

    # Provenance block: reproducibility fields present.
    prov = build_provenance(None, fixture_groups())
    check("provenance code hash", bool(prov["code_sha256"]))
    check("provenance counts", prov["n_groups"] == 5
          and prov["n_multi_candidate"] == 4
          and prov["n_single_candidate"] == 1)
    check("provenance sessions", prov["session_dates"] == {
        "first": "2026-11-10", "last": "2026-11-11", "n": 2})
    check("provenance config", prov["config"]["ZERO_STATES"] == [
        "confirmed_no_fill", "policy_skip"])

    # Report renders with paired + provenance without raising.
    text = format_report(rep, paired=paired, provenance=prov)
    check("report renders", "paired vs baseline" in text
          and "provenance:" in text and "missing:" in text)

    # Empty input: labeled insufficient, never fabricated.
    check("empty verdict",
          evaluate_ranking([])["verdict"] == "insufficient_evidence")

    print("ranking_eval self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--groups", help="JSONL file of opportunity groups "
                                     "({'group_id', 'candidates': [...]})")
    ap.add_argument("--fixture", action="store_true",
                    help="run on the built-in deterministic fixture")
    ap.add_argument("--json", help="also write the full machine-readable "
                                   "report (variants + paired + provenance) "
                                   "to this path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _self_test()
    if args.groups:
        groups = load_groups_jsonl(args.groups)
    elif args.fixture:
        groups = fixture_groups()
    else:
        ap.error("one of --groups / --fixture / --selftest is required")
        return 2
    rep = evaluate_ranking(groups)
    paired = paired_vs_baseline(groups)
    prov = build_provenance(args.groups, groups)
    print(format_report(rep, paired=paired, provenance=prov))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"report": rep, "paired_vs_baseline": paired,
                       "provenance": prov}, fh, indent=2, default=str)
        print(f"json report -> {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
