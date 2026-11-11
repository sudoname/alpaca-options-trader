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

Accounting (Phase-1 conventions): a picked contract with net_outcome=None is
an UNFILLED opportunity and contributes $0 at the opportunity level, counted
separately so fill rates stay visible. Expectancy is reported both per
opportunity (nonfills = 0) and per filled pick.

Offline-first: deterministic fixtures, a JSONL loader (one group per line:
{"group_id": ..., "candidates": [...]}), and a CLI. No credentials, no
network.
"""

import json
import math
from typing import Dict, Iterable, List, Optional

FALLBACK_PENALTY = 0.5          # observed_only: multiplier per fallback field
DELTA_BAND = (0.45, 0.65)       # delta_band variant
LIQ_SPREAD_SCALE = 10.0         # liquidity_tilt: spread_pct normalizer (%)
LIQ_OI_SATURATION = 500.0       # liquidity_tilt: OI saturation level

VARIANT_ORDER = ("baseline", "observed_only", "liquidity_tilt", "delta_band")


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
    best, best_k = None, None
    for c in candidates:
        k = key(c)
        if best_k is None or k > best_k:
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

    Same groups, same outcomes, same costs — only the pick changes. Nonfilled
    picks count $0 per opportunity (Phase-1 accounting) and are reported in
    the fill rate, so a variant cannot look better by picking unfillable
    contracts.
    """
    groups = list(groups)
    report = {"n_groups": len(groups), "variants": {}, "verdict": "ok"}
    if not groups:
        report["verdict"] = "insufficient_evidence"
        return report
    for variant in variants:
        n_filled = 0
        total = 0.0
        wins = 0
        losses = 0
        gross_win = 0.0
        gross_loss = 0.0
        picks = []
        for g in groups:
            cands = g.get("candidates") or []
            pick = pick_for_variant(variant, cands)
            if pick is None:
                picks.append(None)
                continue
            picks.append(pick.get("symbol"))
            outcome = _num(pick.get("net_outcome"))
            if outcome is None:
                continue  # unfilled: $0 at the opportunity level
            n_filled += 1
            total += outcome
            if outcome > 0:
                wins += 1
                gross_win += outcome
            elif outcome < 0:
                losses += 1
                gross_loss += -outcome
        n = len(groups)
        report["variants"][variant] = {
            "n_groups": n,
            "n_filled": n_filled,
            "fill_rate": n_filled / n if n else None,
            "total_net": total,
            "expectancy_per_opportunity": total / n if n else None,
            "expectancy_per_fill": total / n_filled if n_filled else None,
            "win_rate_filled": wins / n_filled if n_filled else None,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0
                             else (None if gross_win == 0 else float("inf")),
            "picks": picks,
        }
    return report


def format_report(report: Dict) -> str:
    lines = [f"ranking_eval: {report['n_groups']} opportunity groups "
             f"(verdict={report['verdict']})"]
    hdr = (f"{'variant':<15}{'fills':>7}{'fill%':>8}{'total$':>10}"
           f"{'EV/opp':>9}{'EV/fill':>9}{'win%':>7}{'PF':>7}")
    lines.append(hdr)
    for v, s in report.get("variants", {}).items():
        def fmt(x, pct=False, money=False):
            if x is None:
                return "--"
            if x == float("inf"):
                return "inf"
            if pct:
                return f"{x * 100:.0f}%"
            if money:
                return f"{x:+.2f}"
            return f"{x:.2f}"
        lines.append(f"{v:<15}{s['n_filled']:>7}{fmt(s['fill_rate'], pct=True):>8}"
                     f"{fmt(s['total_net'], money=True):>10}"
                     f"{fmt(s['expectancy_per_opportunity'], money=True):>9}"
                     f"{fmt(s['expectancy_per_fill'], money=True):>9}"
                     f"{fmt(s['win_rate_filled'], pct=True):>7}"
                     f"{fmt(s['profit_factor']):>7}")
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
    """Deterministic fixture exercising every variant's discriminator."""
    def c(sym, score, iv_q, delta, bid, ask, oi, outcome):
        return {"symbol": sym, "score": score, "iv": 0.25, "delta": delta,
                "greeks_quality": {"iv": iv_q, "delta": iv_q,
                                   "gamma": iv_q, "theta": iv_q, "vega": iv_q},
                "bid": bid, "ask": ask, "volume": 100, "open_interest": oi,
                "net_outcome": outcome}
    return [
        # G1: highest-score contract has fallback greeks and loses; the
        # broker-observed one wins — separates baseline vs observed_only.
        {"group_id": "g1", "candidates": [
            c("A1", 9.0, "fallback", 0.70, 1.00, 1.40, 50, -60.0),
            c("A2", 7.0, "broker", 0.55, 1.00, 1.05, 800, +40.0)]},
        # G2: wide-spread/thin-OI top score loses; liquid one wins —
        # separates baseline vs liquidity_tilt.
        {"group_id": "g2", "candidates": [
            c("B1", 8.0, "broker", 0.80, 0.50, 1.00, 5, -30.0),
            c("B2", 7.5, "broker", 0.50, 1.00, 1.04, 900, +25.0)]},
        # G3: top score is out of the delta band and unfilled (None);
        # in-band candidate fills and wins — delta_band + nonfill accounting.
        {"group_id": "g3", "candidates": [
            c("C1", 9.5, "broker", 0.90, 1.00, 1.05, 400, None),
            c("C2", 8.0, "broker", 0.60, 1.00, 1.05, 400, +15.0)]},
        # G4: everyone agrees (same pick) and it loses — shared downside.
        {"group_id": "g4", "candidates": [
            c("D1", 6.0, "broker", 0.55, 1.00, 1.05, 300, -10.0)]},
    ]


# --------------------------------------------------------------------------- #
# Self-test + CLI
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    rep = evaluate_ranking(fixture_groups())
    v = rep["variants"]

    # Baseline picks the top raw scores: A1, B1, C1(unfilled), D1.
    if v["baseline"]["picks"] != ["A1", "B1", "C1", "D1"]:
        print("FAIL baseline picks", v["baseline"]["picks"]); ok = False
    if v["baseline"]["n_filled"] != 3:  # C1 unfilled
        print("FAIL baseline fills", v["baseline"]["n_filled"]); ok = False
    if abs(v["baseline"]["total_net"] - (-100.0)) > 1e-9:
        print("FAIL baseline total", v["baseline"]["total_net"]); ok = False
    # Nonfill counts as $0 per opportunity: -100 / 4 groups.
    if abs(v["baseline"]["expectancy_per_opportunity"] - (-25.0)) > 1e-9:
        print("FAIL baseline EV/opp", v["baseline"]); ok = False

    # observed_only penalizes A1's fallback labels -> picks A2 in G1.
    if v["observed_only"]["picks"][0] != "A2":
        print("FAIL observed_only pick", v["observed_only"]["picks"]); ok = False

    # liquidity_tilt prefers the tight/deep B2 in G2.
    if v["liquidity_tilt"]["picks"][1] != "B2":
        print("FAIL liquidity_tilt pick", v["liquidity_tilt"]["picks"]); ok = False

    # delta_band excludes C1 (0.90) -> picks C2 in G3 and fills it.
    if v["delta_band"]["picks"][2] != "C2":
        print("FAIL delta_band pick", v["delta_band"]["picks"]); ok = False
    if v["delta_band"]["n_filled"] != 4:
        print("FAIL delta_band fills", v["delta_band"]["n_filled"]); ok = False

    # Empty input: labeled insufficient, never fabricated.
    if evaluate_ranking([])["verdict"] != "insufficient_evidence":
        print("FAIL empty verdict"); ok = False

    print("ranking_eval self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--groups", help="JSONL file of opportunity groups "
                                     "({'group_id', 'candidates': [...]})")
    ap.add_argument("--fixture", action="store_true",
                    help="run on the built-in deterministic fixture")
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
    print(format_report(evaluate_ranking(groups)))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
