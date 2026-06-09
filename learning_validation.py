"""
Phase 9B — Learning validation layer (advisory, read-only, offline-pure).

For every *completed* (closed) simulated trade this module records what the
**Oracle** policy would have decided versus what the **RL** advisor recommended,
then measures how each side would have performed. It answers: *is the RL
advisor actually better (or worse) than the Oracle score policy on the trades we
have seen?*

It reads only historical artifacts (the closed simulated spread book, and — when
present — an RL decision per trade) and writes a derived validation dataset
``learning_validation.csv``. It is STRICTLY advisory / analytics: no order
placement, no execution, no gating, no strategy changes — nothing here can open,
modify, block or close a real or paper position. Every reader fails open
(missing / empty / malformed -> "no data") and every public function returns a
plain structure that is safe to format even when inputs are empty.

Oracle vs RL decisions are expressed as a binary policy over each completed
trade::

    TAKE  — the policy would have entered this trade
    SKIP  — the policy would have passed

* Oracle decision  : TAKE when ``oracle_score >= recommended_min_oracle_score``
                     (the Phase 8C recommendation; falls back to a constant when
                     undefined), else SKIP. A trade with no score defaults TAKE
                     (it was, in fact, taken).
* RL decision      : taken from an explicit RL field on the trade record
                     (``rl_recommendation`` / ``rl_decision`` / ``rl_action`` /
                     ``recommended_action``) when present, else from an optional
                     id-keyed ``rl_lookup`` mapping, else ``None`` (unknown).

Metrics (Req 7): Oracle / RL win-rate, Oracle / RL profit-factor, agreement and
disagreement rate (over trades where both decisions are known), and sample size.

Public API:

    build_validation_records()      -> list of per-trade comparison dicts
    write_validation_csv()          -> bool (writes learning_validation.csv)
    compute_rl_performance()        -> metrics dict
    generate_rl_performance_text()  -> Telegram string (RL_PERFORMANCE)
    generate_validation_stats_text()-> Telegram string (VALIDATION_STATS)
"""

import csv
import logging
from typing import Optional, List, Dict

import oracle_analytics as oa
import threshold_engine as te
from oracle_analytics import AnalyticsConfig

logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_FILE = "learning_validation.csv"

# Fallback used only when the data-driven minimum score is undefined (no data).
DEFAULT_MIN_ORACLE_SCORE = 60.0

TAKE = "TAKE"
SKIP = "SKIP"

# Tokens recognised when normalising an RL recommendation off a trade record.
_TAKE_TOKENS = {"take", "accept", "call", "put", "buy", "long", "yes", "1",
                "true", "trade", "enter", "strong_accept"}
_SKIP_TOKENS = {"skip", "reject", "no", "0", "false", "avoid", "pass", "hold",
                "reject_candidate"}

# CSV schema (Req 6).
VALIDATION_CSV_FIELDS = ["trade_id", "symbol", "oracle_decision", "rl_decision",
                         "oracle_score", "volatility_edge", "strategy", "pnl",
                         "win_loss"]

_RL_FIELDS = ("rl_recommendation", "rl_decision", "rl_action",
              "recommended_action")


# --------------------------------------------------------------------------- #
# Decision derivation
# --------------------------------------------------------------------------- #
def _normalize_decision(value) -> Optional[str]:
    """Coerce an arbitrary RL recommendation token to TAKE / SKIP / None."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in _SKIP_TOKENS:
        return SKIP
    if s in _TAKE_TOKENS:
        return TAKE
    return None


def _oracle_decision(trade: dict, min_oracle_score: float) -> str:
    """TAKE when the trade's oracle_score clears the minimum, else SKIP.

    A trade with no recorded score defaults to TAKE (it was actually taken).
    """
    score = oa._trade_oracle(trade)
    if score is None:
        return TAKE
    return TAKE if score >= float(min_oracle_score) else SKIP


def _rl_decision(trade: dict, rl_lookup: Optional[Dict[str, str]]) -> Optional[str]:
    """RL decision for a trade: record field first, then ``rl_lookup`` by id."""
    for k in _RL_FIELDS:
        d = _normalize_decision(trade.get(k))
        if d is not None:
            return d
    if rl_lookup:
        tid = str(trade.get("id") or trade.get("trade_id") or "")
        if tid and tid in rl_lookup:
            return _normalize_decision(rl_lookup[tid])
    return None


# --------------------------------------------------------------------------- #
# Req 5 / Req 6 — per-trade comparison records + dataset
# --------------------------------------------------------------------------- #
def build_validation_records(config: Optional[AnalyticsConfig] = None,
                             trades: Optional[List[dict]] = None,
                             rl_lookup: Optional[Dict[str, str]] = None,
                             min_oracle_score: Optional[float] = None,
                             recommendations: Optional[dict] = None) -> List[dict]:
    """One comparison record per closed trade (Req 5).

    Each record carries: trade_id, symbol, date, strategy,
    oracle_decision (a.k.a. oracle_recommendation), rl_decision (a.k.a.
    rl_recommendation), oracle_score, volatility_edge, pnl, win_loss and
    actual_outcome.
    """
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)

    if min_oracle_score is None:
        rec = recommendations if recommendations is not None else \
            te.compute_recommendations(config, closed)
        min_oracle_score = rec.get("recommended_min_oracle_score")
        if min_oracle_score is None:
            min_oracle_score = DEFAULT_MIN_ORACLE_SCORE

    records = []
    for i, t in enumerate(closed):
        pnl = oa._trade_pnl(t)
        win = oa._is_win(t)
        outcome = "win" if win else "loss"
        records.append({
            "trade_id": str(t.get("id") or t.get("trade_id") or i),
            "symbol": str(t.get("symbol") or "").strip().upper(),
            "date": t.get("date") or t.get("closed_at") or t.get("timestamp") or "",
            "strategy": t.get("strategy") or "unknown",
            "oracle_score": oa._trade_oracle(t),
            "volatility_edge": oa._trade_edge(t),
            "oracle_decision": _oracle_decision(t, min_oracle_score),
            "rl_decision": _rl_decision(t, rl_lookup),
            "pnl": pnl if pnl is not None else 0.0,
            "win_loss": outcome,
            "actual_outcome": outcome,
        })
    return records


def write_validation_csv(records: List[dict],
                         path: Optional[str] = None) -> bool:
    """Write ``records`` to the validation CSV. Returns True on success."""
    path = path or DEFAULT_VALIDATION_FILE
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=VALIDATION_CSV_FIELDS)
            w.writeheader()
            for r in records:
                w.writerow({k: ("" if r.get(k) is None else r.get(k))
                            for k in VALIDATION_CSV_FIELDS})
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("validation CSV write failed (%s): %s", path, exc)
        return False


# --------------------------------------------------------------------------- #
# Req 7 — performance metrics
# --------------------------------------------------------------------------- #
def _policy_stats(records: List[dict], decision_field: str) -> dict:
    """Win-rate / profit-factor over the trades a policy would have TAKEn."""
    taken = [r for r in records if r.get(decision_field) == TAKE]
    n = len(taken)
    wins = sum(1 for r in taken if r["win_loss"] == "win")
    pf = te._profit_factor([{"pnl": r["pnl"]} for r in taken]) if taken else None
    return {
        "take_count": n,
        "win_rate": (wins / n) if n else 0.0,
        "profit_factor": pf,
    }


def compute_rl_performance(config: Optional[AnalyticsConfig] = None,
                           trades: Optional[List[dict]] = None,
                           records: Optional[List[dict]] = None,
                           rl_lookup: Optional[Dict[str, str]] = None) -> dict:
    """Oracle vs RL win-rate / profit-factor + agreement / disagreement (Req 7)."""
    if records is None:
        records = build_validation_records(config, trades, rl_lookup=rl_lookup)

    oracle = _policy_stats(records, "oracle_decision")
    rl = _policy_stats(records, "rl_decision")

    # agreement is only meaningful where both decisions are known.
    comparable = [r for r in records
                  if r.get("oracle_decision") in (TAKE, SKIP)
                  and r.get("rl_decision") in (TAKE, SKIP)]
    n_cmp = len(comparable)
    agree = sum(1 for r in comparable
                if r["oracle_decision"] == r["rl_decision"])

    return {
        "sample_size": len(records),
        "rl_coverage": n_cmp,            # trades with a known RL decision
        "oracle_win_rate": oracle["win_rate"],
        "rl_win_rate": rl["win_rate"],
        "oracle_profit_factor": oracle["profit_factor"],
        "rl_profit_factor": rl["profit_factor"],
        "oracle_take_count": oracle["take_count"],
        "rl_take_count": rl["take_count"],
        "agreement_rate": (agree / n_cmp) if n_cmp else 0.0,
        "disagreement_rate": (1.0 - agree / n_cmp) if n_cmp else 0.0,
    }


# --------------------------------------------------------------------------- #
# Req 8 — Telegram formatting
# --------------------------------------------------------------------------- #
def _pf_str(pf) -> str:
    if pf is None:
        return "n/a"
    if pf == float("inf"):
        return "∞"
    return "%.2f" % pf


def _pct(value) -> str:
    return "%.1f%%" % ((value or 0.0) * 100.0)


def format_rl_performance(metrics: dict) -> str:
    """RL_PERFORMANCE Telegram body."""
    if not metrics or metrics.get("sample_size", 0) == 0:
        return ("*RL Performance*\n\nNo completed trades yet.\n"
                "_(Advisory only — no trades placed.)_")
    lines = [
        "*RL Performance — Oracle vs RL*",
        "",
        f"Oracle win rate: *{_pct(metrics['oracle_win_rate'])}*  "
        f"(take {metrics['oracle_take_count']})",
        f"RL win rate: *{_pct(metrics['rl_win_rate'])}*  "
        f"(take {metrics['rl_take_count']})",
        "",
        f"Oracle PF: *{_pf_str(metrics['oracle_profit_factor'])}*",
        f"RL PF: *{_pf_str(metrics['rl_profit_factor'])}*",
        "",
        f"Agreement rate: *{_pct(metrics['agreement_rate'])}*",
        f"Disagreement rate: *{_pct(metrics['disagreement_rate'])}*",
        "",
        f"Sample size: *{metrics['sample_size']}*  "
        f"(RL-labelled {metrics['rl_coverage']})",
        "",
        "_(Advisory only — no trades placed.)_",
    ]
    return "\n".join(lines)


def format_validation_stats(records: List[dict], metrics: dict,
                            sample: int = 5) -> str:
    """VALIDATION_STATS Telegram body — counts + a few recent comparisons."""
    if not records:
        return ("*Validation Stats*\n\nNo completed trades yet.\n"
                "_(Advisory only — no trades placed.)_")

    oracle_take = sum(1 for r in records if r["oracle_decision"] == TAKE)
    oracle_skip = sum(1 for r in records if r["oracle_decision"] == SKIP)
    rl_take = sum(1 for r in records if r["rl_decision"] == TAKE)
    rl_skip = sum(1 for r in records if r["rl_decision"] == SKIP)
    rl_unknown = sum(1 for r in records if r["rl_decision"] not in (TAKE, SKIP))

    lines = [
        "*Validation Stats*",
        "",
        f"Completed trades: *{len(records)}*",
        f"Oracle: TAKE {oracle_take} / SKIP {oracle_skip}",
        f"RL: TAKE {rl_take} / SKIP {rl_skip} / unknown {rl_unknown}",
        f"Agreement: *{_pct(metrics['agreement_rate'])}*  "
        f"Disagreement: *{_pct(metrics['disagreement_rate'])}*",
        "",
        "*Recent trades:*",
    ]
    for r in records[-sample:][::-1]:
        rl = r["rl_decision"] or "n/a"
        lines.append(
            f"`{r['symbol'] or '?'}` {r['oracle_decision']}/{rl} "
            f"→ {r['win_loss']} ({r['pnl']:+.0f})"
        )
    lines.append("")
    lines.append("_(Advisory only — no trades placed.)_")
    return "\n".join(lines)


def generate_rl_performance_text(config: Optional[AnalyticsConfig] = None,
                                 trades: Optional[List[dict]] = None,
                                 rl_lookup: Optional[Dict[str, str]] = None) -> str:
    """Top-level entry for the RL_PERFORMANCE Telegram command."""
    records = build_validation_records(config, trades, rl_lookup=rl_lookup)
    metrics = compute_rl_performance(records=records)
    return format_rl_performance(metrics)


def generate_validation_stats_text(config: Optional[AnalyticsConfig] = None,
                                   trades: Optional[List[dict]] = None,
                                   rl_lookup: Optional[Dict[str, str]] = None,
                                   write_csv: bool = True,
                                   csv_path: Optional[str] = None) -> str:
    """Top-level entry for the VALIDATION_STATS Telegram command.

    Refreshes ``learning_validation.csv`` (best-effort) and returns the summary.
    """
    config = config or AnalyticsConfig.from_env()
    records = build_validation_records(config, trades, rl_lookup=rl_lookup)
    if write_csv:
        write_validation_csv(records, csv_path)
    metrics = compute_rl_performance(records=records)
    return format_validation_stats(records, metrics)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; synthetic data only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import os
    import tempfile

    ok = True
    cfg = AnalyticsConfig(spread_trades_file="/nonexistent/lv_st.json",
                          expected_move_file="/nonexistent/lv_em.csv",
                          training_dataset_file="/nonexistent/lv_ds.csv")

    # --- empty -> safe zeros, never raises ---
    metrics = compute_rl_performance(cfg)
    if metrics["sample_size"] != 0 or metrics["agreement_rate"] != 0.0:
        print("FAIL: empty metrics", metrics); ok = False
    if "No completed trades" not in format_rl_performance(metrics):
        print("FAIL: empty rl perf text"); ok = False

    # --- decision normalisation ---
    if _normalize_decision("SKIP") != SKIP or _normalize_decision("CALL") != TAKE:
        print("FAIL: normalize"); ok = False
    if _normalize_decision("") is not None or _normalize_decision("???") is not None:
        print("FAIL: normalize unknown"); ok = False

    # --- synthetic closed trades w/ explicit RL fields ---
    trades = [
        # Oracle TAKE (score 85 >= 60), RL TAKE, WIN
        {"id": "a1", "symbol": "SPY", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 85, "volatility_edge": 0.035,
         "pnl": 120.0, "rl_recommendation": "TAKE"},
        # Oracle TAKE (72 >= 60), RL SKIP, LOSS  -> RL correctly skipped a loser
        {"id": "a2", "symbol": "QQQ", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 72, "volatility_edge": 0.01,
         "pnl": -80.0, "rl_recommendation": "SKIP"},
        # Oracle SKIP (45 < 60), RL TAKE, WIN
        {"id": "a3", "symbol": "META", "strategy": "iron_condor",
         "status": "closed", "oracle_score": 45, "volatility_edge": -0.01,
         "pnl": 40.0, "rl_recommendation": "TAKE"},
    ]
    records = build_validation_records(cfg, trades=trades,
                                       min_oracle_score=60.0)
    if len(records) != 3:
        print("FAIL: record count", len(records)); ok = False
    dec = {r["trade_id"]: (r["oracle_decision"], r["rl_decision"], r["win_loss"])
           for r in records}
    if dec["a1"] != (TAKE, TAKE, "win"):
        print("FAIL: a1", dec["a1"]); ok = False
    if dec["a2"] != (TAKE, SKIP, "loss"):
        print("FAIL: a2", dec["a2"]); ok = False
    if dec["a3"] != (SKIP, TAKE, "win"):
        print("FAIL: a3", dec["a3"]); ok = False

    metrics = compute_rl_performance(records=records)
    # Oracle TAKEs: a1(win),a2(loss) -> 50%; RL TAKEs: a1(win),a3(win) -> 100%
    if abs(metrics["oracle_win_rate"] - 0.5) > 1e-9:
        print("FAIL: oracle win rate", metrics["oracle_win_rate"]); ok = False
    if abs(metrics["rl_win_rate"] - 1.0) > 1e-9:
        print("FAIL: rl win rate", metrics["rl_win_rate"]); ok = False
    # RL PF: only wins among RL TAKEs -> inf; Oracle PF: 120 / 80 = 1.5
    if metrics["rl_profit_factor"] != float("inf"):
        print("FAIL: rl pf", metrics["rl_profit_factor"]); ok = False
    if metrics["oracle_profit_factor"] != 1.5:
        print("FAIL: oracle pf", metrics["oracle_profit_factor"]); ok = False
    # agreement: a1 agree(TAKE/TAKE); a2 disagree; a3 disagree -> 1/3
    if abs(metrics["agreement_rate"] - (1 / 3)) > 1e-9:
        print("FAIL: agreement", metrics["agreement_rate"]); ok = False
    if abs(metrics["disagreement_rate"] - (2 / 3)) > 1e-9:
        print("FAIL: disagreement", metrics["disagreement_rate"]); ok = False
    if metrics["rl_coverage"] != 3 or metrics["sample_size"] != 3:
        print("FAIL: coverage", metrics); ok = False

    # --- unknown RL decision is excluded from agreement ---
    trades2 = [dict(trades[0]), dict(trades[1])]
    del trades2[0]["rl_recommendation"]          # unknown RL for this one
    recs2 = build_validation_records(cfg, trades=trades2, min_oracle_score=60.0)
    m2 = compute_rl_performance(records=recs2)
    if m2["rl_coverage"] != 1:  # only the second trade has an RL label
        print("FAIL: rl_coverage exclude unknown", m2["rl_coverage"]); ok = False

    # --- rl_lookup by id when no field present ---
    recs3 = build_validation_records(cfg, trades=trades2,
                                     rl_lookup={"a1": "TAKE"},
                                     min_oracle_score=60.0)
    if next(r for r in recs3 if r["trade_id"] == "a1")["rl_decision"] != TAKE:
        print("FAIL: rl_lookup"); ok = False

    # --- CSV round-trips with the documented header ---
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, DEFAULT_VALIDATION_FILE)
    if not write_validation_csv(records, csv_path):
        print("FAIL: write csv"); ok = False
    back = oa.read_csv_rows(csv_path)
    if len(back) != 3 or list(back[0].keys()) != VALIDATION_CSV_FIELDS:
        print("FAIL: csv header/rows", back[:1]); ok = False

    # --- formatting never raises on real records ---
    if "Validation Stats" not in format_validation_stats(records, metrics):
        print("FAIL: validation stats text"); ok = False
    if "RL Performance" not in format_rl_performance(metrics):
        print("FAIL: rl perf text"); ok = False

    print("learning_validation self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
