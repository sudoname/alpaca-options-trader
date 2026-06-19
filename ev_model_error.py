"""
Phase 12 — EV Model Error (analytics only, fail-open).

Where ``ev_calibration``/``calibration_reports`` ask whether higher EV *ranks*
better, this report asks the blunter question the EV-first redesign needs:

    How far is the EV model's dollar prediction from the realized dollar PnL,
    and is that error concentrated in a particular structure, direction, exit
    reason, or EV bucket?

For every closed trade that carries a frozen entry-time EV stamp (the Phase 10D
Best-EV spreads and Phase 10H single-leg scheduler trades, loaded verbatim via
``ev_attribution.load_closed_records``) it computes the signed per-trade error
``realized_pnl - expected_ev`` and reports, overall and per breakdown:

    trades, avg_expected_ev, avg_realized_pnl, bias (avg signed error),
    mean_abs_error, profit_factor.

Breakdowns: by strategy, by CALL vs PUT (single legs) / SPREAD, by exit reason,
and by entry-EV bucket (the shared Phase 10E ``EV_BUCKETS``).

Verdict (overall, once >= MIN_TRADES resolved):
    EV_CALIBRATED   |bias| within max($10, 25% of |avg_expected_ev|)
    EV_OVERPREDICTS bias strongly negative (model promised more than realized)
    EV_UNDERPREDICTS bias strongly positive (model promised less than realized)
    INSUFFICIENT_DATA  fewer than MIN_TRADES resolved

STRICTLY analytics: never opens, closes, sizes, prices, blocks or alters any
real or paper trade and never touches the network. Every reader fails open.
"""

import re
from typing import List, Optional

import ev_attribution as eva
import oracle_analytics as oa
from ev_attribution import ANALYTICS_FOOTER, EV_BUCKETS
from oracle_analytics import AnalyticsConfig

MIN_TRADES = 10
ABS_TOLERANCE = 10.0
REL_TOLERANCE = 0.25

VERDICT_CALIBRATED = "EV_CALIBRATED"
VERDICT_OVERPREDICTS = "EV_OVERPREDICTS"
VERDICT_UNDERPREDICTS = "EV_UNDERPREDICTS"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

EV_ERROR_QUESTION = "How far is predicted EV from realized PnL, and where?"

# Spread strategy names imply structure, not a single-leg direction.
_SPREAD_HINT = "spread"
_CONDOR_HINT = "condor"
# OCC option symbol: <root><YYMMDD><C|P><8-digit strike>.
_OCC_RE = re.compile(r"\d{6}([CP])\d{8}$")


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------
def load_ev_error_records(records: Optional[List[dict]] = None,
                          config: Optional[AnalyticsConfig] = None,
                          attribution_path: Optional[str] = None) -> List[dict]:
    """Closed records carrying BOTH a realized PnL and an entry-time EV stamp."""
    if records is None:
        records = eva.load_closed_records(config=config,
                                          attribution_path=attribution_path)
    return [r for r in records if isinstance(r, dict)
            and eva._ev(r) is not None and oa._trade_pnl(r) is not None]


# ---------------------------------------------------------------------------
# Breakdown keys (tolerant of label variants; never raise)
# ---------------------------------------------------------------------------
def _strategy_key(r: dict) -> str:
    val = r.get("strategy") or r.get("strategy_name")
    return str(val) if val else "unknown"


def _call_put_key(r: dict) -> str:
    """CALL / PUT for single legs, SPREAD for multi-leg structures, else unknown."""
    strat = (r.get("strategy") or r.get("strategy_name") or "").lower()
    if _SPREAD_HINT in strat or _CONDOR_HINT in strat:
        return "SPREAD"
    for key in ("option_type", "contract_type", "right", "call_put", "side"):
        v = r.get(key)
        if isinstance(v, str) and v:
            low = v.lower()
            if low in ("c", "call"):
                return "CALL"
            if low in ("p", "put"):
                return "PUT"
    sym = str(r.get("symbol") or r.get("option_symbol") or "")
    m = _OCC_RE.search(sym)
    if m:
        return "CALL" if m.group(1) == "C" else "PUT"
    return "unknown"


def _exit_reason_key(r: dict) -> str:
    val = (r.get("exit_reason") or r.get("close_reason")
           or r.get("outcome") or r.get("status"))
    return str(val) if val else "unknown"


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------
def _avg(values) -> Optional[float]:
    vals = [oa._to_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _error_block(rows: List[dict]) -> dict:
    """Predicted-vs-realized error stats for one group of closed trades."""
    stats = eva.bucket_stats(rows)
    expected = _avg(eva._ev(r) for r in rows)
    realized = stats["average_pnl"] if rows else None
    errors = []
    for r in rows:
        ev = eva._ev(r)
        pnl = oa._trade_pnl(r)
        if ev is not None and pnl is not None:
            errors.append(pnl - ev)
    bias = round(sum(errors) / len(errors), 4) if errors else None
    mae = round(sum(abs(e) for e in errors) / len(errors), 4) if errors else None
    return {
        "trades": stats["trades"],
        "avg_expected_ev": expected,
        "avg_realized_pnl": realized,
        "bias": bias,                 # signed: realized - expected
        "mean_abs_error": mae,
        "profit_factor": stats["profit_factor"],
    }


def _group_table(rows: List[dict], key_fn) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    return {k: _error_block(v) for k, v in groups.items()}


def _bucket_table(rows: List[dict]) -> dict:
    out = {}
    for label, _, _ in EV_BUCKETS:
        sub = [r for r in rows
               if eva.bucket_label(eva._ev(r), EV_BUCKETS) == label]
        out[label] = _error_block(sub)
    return out


def classify_bias(avg_expected_ev, bias) -> str:
    """Verdict from the overall signed bias, with a magnitude tolerance."""
    if bias is None or avg_expected_ev is None:
        return VERDICT_INSUFFICIENT
    tol = max(ABS_TOLERANCE, REL_TOLERANCE * abs(avg_expected_ev))
    if abs(bias) <= tol:
        return VERDICT_CALIBRATED
    return VERDICT_UNDERPREDICTS if bias > 0 else VERDICT_OVERPREDICTS


def compute_ev_model_error(records: Optional[List[dict]] = None,
                           config: Optional[AnalyticsConfig] = None,
                           attribution_path: Optional[str] = None) -> dict:
    """Overall + per-breakdown EV prediction error. Never raises."""
    rows = load_ev_error_records(records=records, config=config,
                                 attribution_path=attribution_path)
    overall = _error_block(rows)
    verdict = (VERDICT_INSUFFICIENT if overall["trades"] < MIN_TRADES
               else classify_bias(overall["avg_expected_ev"], overall["bias"]))
    return {
        "question": EV_ERROR_QUESTION,
        "sample_size": overall["trades"],
        "overall": overall,
        "by_strategy": _group_table(rows, _strategy_key),
        "by_call_put": _group_table(rows, _call_put_key),
        "by_exit_reason": _group_table(rows, _exit_reason_key),
        "by_ev_bucket": _bucket_table(rows),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Telegram formatting (pure)
# ---------------------------------------------------------------------------
def _grp_line(label: str, b: dict) -> str:
    return (f"`{label}`: `{b['trades']}` trades, "
            f"exp `{eva._money(b['avg_expected_ev'])}` -> "
            f"real `{eva._money(b['avg_realized_pnl'])}` "
            f"(bias `{eva._money(b['bias'])}`, "
            f"MAE `{eva._money(b['mean_abs_error'])}`, "
            f"PF `{eva._pf_str(b['profit_factor'])}`)")


def _section(title: str, table: dict, *, order=None) -> List[str]:
    lines = [f"*{title}*"]
    items = ([(k, table[k]) for k in order if k in table] if order
             else sorted(table.items(),
                         key=lambda kv: kv[1].get("trades", 0), reverse=True))
    seen = False
    for label, b in items:
        if b.get("trades", 0) == 0:
            continue
        lines.append(_grp_line(label, b))
        seen = True
    if not seen:
        lines.append("_no data_")
    return lines


def format_ev_model_error(report: dict) -> str:
    """Telegram-ready EV_MODEL_ERROR. Pure formatting."""
    header = "🎯 *EV Model Error* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if report.get("sample_size", 0) == 0:
        return "\n".join([
            header, "",
            f"_{EV_ERROR_QUESTION}_",
            "",
            "No closed trades carrying an entry EV stamp yet.",
            f"*Verdict:* `{VERDICT_INSUFFICIENT}`",
            "", footer,
        ])
    o = report["overall"]
    lines = [
        header, "",
        f"_{EV_ERROR_QUESTION}_",
        "",
        f"*Overall:* `{o['trades']}` trades, expected "
        f"`{eva._money(o['avg_expected_ev'])}` -> realized "
        f"`{eva._money(o['avg_realized_pnl'])}`",
        f"Bias `{eva._money(o['bias'])}` · MAE `{eva._money(o['mean_abs_error'])}` · "
        f"PF `{eva._pf_str(o['profit_factor'])}`",
        "",
    ]
    lines += _section("By strategy:", report["by_strategy"])
    lines += [""]
    lines += _section("By CALL vs PUT:", report["by_call_put"],
                      order=["CALL", "PUT", "SPREAD", "unknown"])
    lines += [""]
    lines += _section("By exit reason:", report["by_exit_reason"])
    lines += [""]
    lines += _section("By EV bucket:", report["by_ev_bucket"],
                      order=[b[0] for b in EV_BUCKETS])
    lines += [
        "",
        f"*Verdict:* `{report['verdict']}`",
        f"Sample size: `{report['sample_size']}`",
        "", footer,
    ]
    return "\n".join(lines)


def generate_ev_model_error_text(config: Optional[AnalyticsConfig] = None,
                                 attribution_path: Optional[str] = None) -> str:
    """Top-level entry for the EV_MODEL_ERROR Telegram command."""
    return format_ev_model_error(
        compute_ev_model_error(config=config, attribution_path=attribution_path))


# ---------------------------------------------------------------------------
# Self-test (no creds, no network)
# ---------------------------------------------------------------------------
def _self_test() -> int:
    ok = True
    # Synthetic closed records carrying EV stamp + realized pnl.
    recs = [
        {"strategy": "bullish_put_credit_spread", "expected_value": 12.0,
         "pnl": 20.0, "max_loss": 100.0, "exit_reason": "take_profit",
         "symbol": "SPY"},
        {"strategy": "bullish_put_credit_spread", "expected_value": 8.0,
         "pnl": -50.0, "max_loss": 100.0, "exit_reason": "stop_loss",
         "symbol": "SPY"},
        {"strategy": "single_leg", "expected_value": 5.0, "pnl": 5.0,
         "max_loss": 200.0, "exit_reason": "take_profit",
         "symbol": "COST260717C00930000"},
        {"strategy": "single_leg", "expected_value": 5.0, "pnl": -10.0,
         "max_loss": 200.0, "exit_reason": "stop_loss",
         "symbol": "AVGO260717P00410000"},
    ]
    report = compute_ev_model_error(records=recs)
    if report["sample_size"] != 4:
        print("FAIL: sample size", report["sample_size"]); ok = False

    # CALL vs PUT classification works for OCC symbols + spreads.
    cp = report["by_call_put"]
    if "CALL" not in cp or "PUT" not in cp or "SPREAD" not in cp:
        print("FAIL: call/put/spread split", list(cp)); ok = False

    # Bias = realized - expected; here realized avg < expected avg overall.
    o = report["overall"]
    if o["bias"] is None or o["mean_abs_error"] is None:
        print("FAIL: bias/mae missing"); ok = False

    # Verdict math: small sample -> insufficient.
    if report["verdict"] != VERDICT_INSUFFICIENT:
        print("FAIL: <10 trades should be INSUFFICIENT", report["verdict"]); ok = False

    # classify_bias edges.
    if classify_bias(40.0, 5.0) != VERDICT_CALIBRATED:
        print("FAIL: within tolerance should be CALIBRATED"); ok = False
    if classify_bias(40.0, -30.0) != VERDICT_OVERPREDICTS:
        print("FAIL: strong negative bias should OVERPREDICT"); ok = False
    if classify_bias(40.0, 30.0) != VERDICT_UNDERPREDICTS:
        print("FAIL: strong positive bias should UNDERPREDICT"); ok = False

    # Empty store -> clean insufficient report + footer present.
    empty = format_ev_model_error(compute_ev_model_error(records=[]))
    if ANALYTICS_FOOTER not in empty or VERDICT_INSUFFICIENT not in empty:
        print("FAIL: empty report malformed"); ok = False

    _ = format_ev_model_error(report)  # never raises

    print("ev_model_error self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
