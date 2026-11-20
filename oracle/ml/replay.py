"""Oracle 4.0 Phase 6b -- paired portfolio replay over the frozen dataset.

Replays the stream of decisions the production bot ACTUALLY traded
(``outcomes.status == 'mark_derived'``) through a deterministic portfolio
engine, under several paired selection variants:

  * ``baseline_frozen``   -- every recorded trade (acceptance: parity with
                             the outcome store must be exact).
  * ``baseline_quality``  -- baseline minus trades failing a simple entry
                             quality gate (wide spread / missing quote).
  * ``valuation_ranked``  -- only trades the Phase 5 scenario valuation
                             independently selected (``decision == 'trade'``).
  * ``classical_gated``   -- baseline gated by out-of-fold 3-class
                             (down/neutral/up) multinomial-logistic direction
                             probabilities (purged session folds); the gate
                             abstains when the forecast is neutral-dominant or
                             the directional edge is below the pre-registered
                             margin -- neutral is never treated as a put
                             signal (Oracle 4.0.1).
  * ``neural_gated``      -- same 3-class gate using the Phase 4 multitask
                             MLP direction head (torch-optional; honest
                             ``unavailable`` without).
  * ``combined``          -- valuation AND classical gates.

Rules (master prompt, Phase 6):
  * every variant sees the same opportunities, timestamps, capital and cost
    model; gates can only DROP recorded trades -- counterfactual fills are
    never fabricated.  Selections without a recorded outcome are counted as
    ``outcome_unavailable``, never simulated.
  * full portfolio replay with cash and concurrency constraints, not
    independent-trade summation.  Structural skips are counted.
  * positions are carried at COST between entry and exit because no daily
    option marks exist in the dataset; this limitation is stated in the
    artifact (``mtm_note``) rather than hidden.
  * friction comes from the Phase 6a execution-calibration artifact
    (paper entry-vs-mid slippage), swept at 1.0x / 1.5x / 2.0x; a stop
    overshoot stress degrades hard-stop exits by the calibrated
    (p90 - p50) overshoot.
  * per-session paired deltas vs baseline get session block-bootstrap CIs.
  * daily Sharpe uses sqrt(252) annualisation over EVERY session in the
    replay window; zero-trade sessions contribute a 0 return.
  * pre-registered holdout sessions are refused at the DECISION level
    (an in-window decision's exit may legitimately land on a holdout day).
  * deterministic, offline, research-only; no broker/network access and no
    production imports.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from oracle.lab.purged_validation import (
    Sample,
    Standardizer,
    dataset_meta,
    load_samples,
    make_folds,
    metrics_fingerprint,
    paired_session_bootstrap,
)
from oracle.ml.baselines import MultinomialLogisticModel, build_design
from oracle.ml.evidence import kish_ess
from oracle.ml.mlp import direction_class
from oracle.ml.schema import parse_utc
from oracle.ml.valuation import parse_occ_symbol

__all__ = [
    "Trade",
    "load_traded_outcomes",
    "load_valuation_selection",
    "load_session_calendar",
    "load_friction",
    "crosscheck_inputs",
    "gate_direction",
    "compute_oof_classical",
    "compute_oof_neural",
    "run_portfolio",
    "sharpe_ratio",
    "max_drawdown",
    "profit_factor",
    "run_replay",
    "main",
]

DEFAULT_HOLDOUT = ("2026-11-12", "2026-11-13")
TRADING_DAYS_PER_YEAR = 252
QUALITY_MAX_SPREAD_PCT = 0.10
FRICTION_MULTS = (1.0, 1.5, 2.0)

MTM_NOTE = (
    "No per-day option marks exist in the dataset, so open positions are "
    "carried at entry cost between entry and exit; all P&L is recognised at "
    "exit. Intraday/overnight drawdown on open positions is therefore "
    "UNDERSTATED. This is a documented limitation, not an assumption that "
    "the risk is zero."
)
COUNTERFACTUAL_NOTE = (
    "Decisions the production bot did not trade carry no realised P&L. "
    "Variants that would have traded them count outcome_unavailable; their "
    "fills are never fabricated."
)


# --------------------------------------------------------------------------- #
# trades
# --------------------------------------------------------------------------- #

@dataclass
class Trade:
    """One recorded production trade with a resolved mark-derived outcome."""

    decision_id: str
    session: str
    entry_at: datetime
    exit_at: datetime
    symbol: Optional[str]
    underlying: Optional[str]
    option_type: Optional[str]
    fill_price: float
    fill_qty: float
    gross_pnl_pct: float
    reason: Optional[str]
    spread_pct: Optional[float] = None
    has_quote: Optional[bool] = None

    @property
    def premium(self) -> float:
        return self.fill_price * self.fill_qty * 100.0


def load_traded_outcomes(
    dataset_path: str,
    holdout: Sequence[str] = DEFAULT_HOLDOUT,
) -> Tuple[List[Trade], Dict[str, Any]]:
    """Load the actually-traded stream (mark_derived outcomes + features).

    Holdout refusal happens here, at the decision level: any trade whose
    ENTRY session is pre-registered holdout is excluded and counted.
    """
    holdout_set = set(holdout)
    report: Dict[str, Any] = {
        "rows_scanned": 0,
        "trades_kept": 0,
        "holdout_excluded": 0,
        "dropped_bad_timestamp": 0,
        "dropped_missing_fill": 0,
        "holdout_sessions": sorted(holdout_set),
        "outcome_status_counts": {},
    }
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        where = ""
        args: List[str] = []
        if holdout_set:
            where = ("WHERE f.session NOT IN (%s) "
                     % ",".join("?" * len(holdout_set)))
            args = sorted(holdout_set)
        for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM outcomes o "
                "JOIN features f USING(decision_id) "
                + where + "GROUP BY status", args):
            report["outcome_status_counts"][row["status"]] = row["n"]

        trades: List[Trade] = []
        cur = conn.execute(
            """SELECT o.decision_id, o.gross_pnl_pct, o.reason,
                      o.fill_price, o.fill_qty, o.outcome_at,
                      f.session, f.as_of, f.symbol, f.underlying,
                      f.features_json, f.quality_json
                 FROM outcomes o JOIN features f USING(decision_id)
                WHERE o.status = 'mark_derived'
                  AND o.gross_pnl_pct IS NOT NULL
                ORDER BY f.as_of, o.decision_id""")
        for row in cur:
            report["rows_scanned"] += 1
            if row["session"] in holdout_set:
                report["holdout_excluded"] += 1
                continue
            entry_at = parse_utc(row["as_of"])
            exit_at = parse_utc(row["outcome_at"])
            if entry_at is None or exit_at is None:
                report["dropped_bad_timestamp"] += 1
                continue
            if exit_at < entry_at:
                exit_at = entry_at
            if row["fill_price"] is None or row["fill_qty"] is None:
                report["dropped_missing_fill"] += 1
                continue
            feats = _loads(row["features_json"])
            quality = _loads(row["quality_json"])
            occ = parse_occ_symbol(row["symbol"])
            trades.append(Trade(
                decision_id=row["decision_id"],
                session=row["session"],
                entry_at=entry_at,
                exit_at=exit_at,
                symbol=row["symbol"],
                underlying=row["underlying"],
                option_type=occ["option_type"] if occ else None,
                fill_price=float(row["fill_price"]),
                fill_qty=float(row["fill_qty"]),
                gross_pnl_pct=float(row["gross_pnl_pct"]),
                reason=row["reason"],
                spread_pct=_opt_float(feats.get("spread_pct")),
                has_quote=(bool(quality["has_quote"])
                           if "has_quote" in quality else None),
            ))
            report["trades_kept"] += 1
        return trades, report
    finally:
        conn.close()


def _loads(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _opt_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# external evidence (valuation selections, friction calibration)
# --------------------------------------------------------------------------- #

def load_valuation_selection(path: str) -> Tuple[Set[str], Dict[str, Any]]:
    """Decision_ids the Phase 5 valuation independently chose to trade."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    scans = doc.get("result", {}).get("scans", [])
    selected: Set[str] = set()
    n_trade = 0
    for scan in scans:
        if scan.get("decision") != "trade":
            continue
        n_trade += 1
        did = (scan.get("selected") or {}).get("decision_id")
        if did:
            selected.add(did)
    report = {
        "path": path,
        "report_fingerprint": doc.get("report_fingerprint"),
        "dataset_hash": doc.get("dataset_hash"),
        "git_sha": doc.get("git_sha"),
        "scans": len(scans),
        "trade_scans": n_trade,
        "selected_ids": len(selected),
    }
    return selected, report


def load_session_calendar(
    dataset_path: str,
    holdout: Sequence[str] = DEFAULT_HOLDOUT,
) -> List[str]:
    """Every distinct feature session in the dataset, minus holdout.

    Master prompt (Oracle 4.0.1 Stage F): replay must use the SAME session
    calendar including no-trade days -- risk metrics computed only over
    traded sessions overstate Sharpe and understate drawdown duration.
    Zero-trade sessions contribute a 0 equity return in ``run_portfolio``.
    """
    holdout_set = set(holdout)
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT session FROM features ORDER BY session")
        return [r[0] for r in rows
                if r[0] is not None and r[0] not in holdout_set]
    finally:
        conn.close()


def load_friction(path: Optional[str]) -> Dict[str, Any]:
    """Friction/stress parameters from the Phase 6a calibration artifact.

    one_way_frac: mean paper entry-vs-mid slippage as a fraction of premium,
    applied once at entry and once at exit (round trip = 2x).
    overshoot_extra_pts: (p90 - p50) of hard-stop overshoot in points of
    premium, the stress degradation for dynamic_stop_loss exits.
    """
    if not path or not os.path.exists(path):
        return {
            "source": None,
            "one_way_frac": 0.02,
            "overshoot_extra_pts": 10.0,
            "note": ("calibration artifact missing; conservative defaults "
                     "used (2% one-way friction, +10pt stop stress)"),
        }
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    one_way = (doc.get("slippage", {}).get("entry_vs_mid", {})
               .get("quantiles", {}).get("mean"))
    oq = (doc.get("stop_overshoot", {}).get("dynamic_stop_loss", {})
          .get("overshoot_quantiles_pts", {}))
    extra = None
    if oq.get("p90") is not None and oq.get("p50") is not None:
        extra = float(oq["p90"]) - float(oq["p50"])
    return {
        "source": path,
        "source_fingerprint": doc.get("fingerprint"),
        "one_way_frac": float(one_way) if one_way is not None else 0.02,
        "overshoot_extra_pts": float(extra) if extra is not None else 10.0,
        "note": ("one_way_frac = mean paper entry-vs-mid slippage; paper "
                 "fills bound internal consistency only, so the 1.5x/2.0x "
                 "sweeps are the live-cost hedge"),
    }


def crosscheck_inputs(
    meta: Dict[str, Any],
    valuation_report: Optional[Dict[str, Any]],
    friction: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate input artifacts against each other BEFORE they are used.

    Master prompt (Oracle 4.0.1 Stage F): artifact hashes and dataset /
    model / config provenance must be cross-checked; a valuation selection
    computed on a DIFFERENT dataset must be refused, not silently merged.

    Raises ``ValueError`` on a positive mismatch.  Artifacts that simply do
    not carry the field (older fixtures) are reported as ``unknown`` --
    absence of evidence is surfaced, not punished retroactively.
    """
    checks: List[Dict[str, Any]] = []
    ds_hash = meta.get("dataset_hash")

    if valuation_report is not None:
        v_hash = valuation_report.get("dataset_hash")
        if v_hash is None or ds_hash is None:
            status = "unknown"
        elif str(v_hash) == str(ds_hash):
            status = "match"
        else:
            status = "mismatch"
        checks.append({
            "name": "valuation_dataset_hash_vs_dataset",
            "expected": ds_hash,
            "observed": v_hash,
            "status": status,
        })

    checks.append({
        "name": "execution_calibration_fingerprint_recorded",
        "expected": None,
        "observed": friction.get("source_fingerprint"),
        "status": ("unknown" if friction.get("source_fingerprint") is None
                   else "recorded"),
    })

    mismatches = [c for c in checks if c["status"] == "mismatch"]
    if mismatches:
        raise ValueError(
            "input crosscheck failed; refusing to mix artifacts: "
            + "; ".join(
                f"{c['name']} expected={c['expected']} observed={c['observed']}"
                for c in mismatches))
    return {
        "checks": checks,
        "ok": True,
        "note": ("mismatched artifacts raise before any replay computation; "
                 "'unknown' means the artifact predates provenance fields"),
    }


# --------------------------------------------------------------------------- #
# out-of-fold direction gates
# --------------------------------------------------------------------------- #

#: Pre-registered abstention margin for the 3-class direction gate
#: (Oracle 4.0.1).  Fixed BEFORE any evaluation on this dataset; the gate
#: endorses a direction only when the forecast's directional edge
#: (p_favoured - p_opposed) reaches this margin.  Not tuned on outcomes.
DIRECTION_MARGIN = 0.10


def gate_direction(option_type: Optional[str],
                   p_down: float,
                   p_neutral: float,
                   p_up: float,
                   margin: float = DIRECTION_MARGIN) -> str:
    """3-class direction gate verdict: ``pass`` / ``gated_out`` / ``abstain``.

    Oracle 4.0.1 defect fix: the old binary gate (``p_up < 0.5`` => put)
    merged neutral and bearish forecasts, so a neutral-dominant forecast
    was treated as a PUT signal.  Here:

      * a call passes only when UP is the argmax AND ``p_up - p_down``
        reaches the pre-registered ``margin``;
      * a put passes only when DOWN is the argmax AND ``p_down - p_up``
        reaches the margin;
      * the opposing argmax gates the trade out (direction contradicted);
      * everything else -- neutral argmax, or a directional argmax with an
        edge below the margin -- is an explicit ABSTAIN: the model declines
        to endorse the trade without contradicting it.  Neutral is NEVER a
        put signal.

    Unknown option type passes (the gate only acts on evidence it has).
    Counterexample the old gate got wrong: a put with
    (p_down, p_neutral, p_up) = (0.25, 0.45, 0.30) has p_up < 0.5 and
    used to pass; the true forecast is "probably goes nowhere", so the
    correct verdict is abstain (no-trade) despite low upward probability.
    """
    probs = (p_down, p_neutral, p_up)
    argmax = max(range(3), key=lambda i: probs[i])
    if option_type == "call":
        if argmax == 2 and (p_up - p_down) >= margin:
            return "pass"
        if argmax == 0:
            return "gated_out"
        return "abstain"
    if option_type == "put":
        if argmax == 0 and (p_down - p_up) >= margin:
            return "pass"
        if argmax == 2:
            return "gated_out"
        return "abstain"
    return "pass"


def _nonholdout_samples(dataset_path: str, horizon: str,
                        holdout: Sequence[str]) -> List[Sample]:
    samples = load_samples(dataset_path, horizon, "direction")
    hold = set(holdout)
    return [s for s in samples if s.session not in hold]


def _ok(s: Sample) -> bool:
    return s.label_status == "ok" and s.label_value is not None


def compute_oof_classical(
    dataset_path: str,
    holdout: Sequence[str] = DEFAULT_HOLDOUT,
    horizon: str = "60m",
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Any]]:
    """Out-of-fold 3-class direction probabilities per decision_id.

    Oracle 4.0.1: multinomial logistic over (down, neutral, up) -- the
    binary up-vs-not-up collapse merged neutral and bearish outcomes.
    Purged session folds (Phase 2 parameters). Only decisions that land in
    some fold's TEST block get a prediction; everything else is honestly
    uncovered -- the replay passes those trades through un-gated and counts
    them as ``no_prediction``.
    """
    samples = _nonholdout_samples(dataset_path, horizon, holdout)
    folds, fold_sessions, _ = make_folds(
        samples, test_block_sessions=1, min_train_sessions=2,
        embargo_sessions=1, holdout_sessions=0)
    by_id = {s.decision_id: s for s in samples}
    names = sorted({k for s in samples for k in s.features})

    preds: Dict[str, Tuple[float, float, float]] = {}
    report: Dict[str, Any] = {
        "model": "multinomial_logistic",
        "classes": ["down", "neutral", "up"],
        "horizon": horizon,
        "folds_total": len(folds),
        "folds_used": 0,
        "folds_skipped": 0,
        "sessions": fold_sessions,
        "predictions": 0,
    }
    for fold in folds:
        train = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids]
        y_train = [direction_class(float(s.label_value)) for s in train]
        if len(set(y_train)) < 2 or not test:
            report["folds_skipped"] += 1
            continue
        std = Standardizer().fit([s.features for s in train], names)
        model = MultinomialLogisticModel().fit(
            build_design(train, std, std.feature_names).x, y_train)
        dm = build_design(test, std, std.feature_names)
        p = model.predict_proba(dm.x)
        for s, row in zip(test, p):
            preds[s.decision_id] = (float(row[0]), float(row[1]),
                                    float(row[2]))
        report["folds_used"] += 1
    report["predictions"] = len(preds)
    return preds, report


def compute_oof_neural(
    dataset_path: str,
    holdout: Sequence[str] = DEFAULT_HOLDOUT,
    horizon: str = "60m",
    max_epochs: int = 20,
    seed: int = 42,
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Any]]:
    """Out-of-fold MLP 3-class direction probabilities per decision_id.

    Oracle 4.0.1: the gate consumes the full (down, neutral, up) row from
    the direction head, not the collapsed ``p_up`` scalar.
    Models are retrained per purged fold -- the saved Phase 4 final model
    would leak its own test sessions.  Aux heads are passed empty (masked),
    so only the direction loss trains; that is the head the gate uses.
    Returns ``({}, report)`` with status ``unavailable`` when torch is
    missing: unknown coverage is not zero.
    """
    from oracle.ml import mlp as mlp_mod

    if not mlp_mod.TORCH_AVAILABLE:
        return {}, {
            "model": "multitask_mlp",
            "status": "unavailable",
            "why": mlp_mod.TORCH_WHY,
        }

    samples = _nonholdout_samples(dataset_path, horizon, holdout)
    folds, fold_sessions, _ = make_folds(
        samples, test_block_sessions=1, min_train_sessions=2,
        embargo_sessions=1, holdout_sessions=0)
    by_id = {s.decision_id: s for s in samples}
    names = sorted({k for s in samples for k in s.features})
    cfg = mlp_mod.MLPConfig(max_epochs=max_epochs)

    preds: Dict[str, Tuple[float, float, float]] = {}
    report: Dict[str, Any] = {
        "model": "multitask_mlp",
        "classes": ["down", "neutral", "up"],
        "status": "ok",
        "horizon": horizon,
        "max_epochs": max_epochs,
        "folds_total": len(folds),
        "folds_used": 0,
        "folds_skipped": 0,
        "sessions": fold_sessions,
        "predictions": 0,
    }
    for fold in folds:
        train_all = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids]
        if not test or len(train_all) < 10:
            report["folds_skipped"] += 1
            continue
        if len({1 if s.label_value > 0 else 0 for s in train_all}) < 2:
            report["folds_skipped"] += 1
            continue
        test_start = min(s.as_of for s in test)
        # chronological val tail = last train session (fallback: all train)
        last_sess = max(s.session for s in train_all)
        fit_rows = [s for s in train_all if s.session != last_sess]
        val_rows = [s for s in train_all if s.session == last_sess]
        if len(fit_rows) < 10 or len(val_rows) < 10:
            fit_rows, val_rows = train_all, train_all
        std = Standardizer().fit([s.features for s in fit_rows], names)
        x_fit = build_design(fit_rows, std, std.feature_names).x
        x_val = build_design(val_rows, std, std.feature_names).x
        x_test = build_design(test, std, std.feature_names).x
        hd_fit = mlp_mod.assemble_head_data(
            fit_rows, {}, {}, {}, {}, test_start)
        hd_val = mlp_mod.assemble_head_data(
            val_rows, {}, {}, {}, {}, test_start)
        model, _info = mlp_mod.train_mlp(
            x_fit, hd_fit, x_val, hd_val, cfg, seed=seed, device="cpu")
        out = mlp_mod.predict_heads(model, x_test, device="cpu")
        for s, row in zip(test, out["direction_probs"]):
            preds[s.decision_id] = (float(row[0]), float(row[1]),
                                    float(row[2]))
        report["folds_used"] += 1
    report["predictions"] = len(preds)
    return preds, report


# --------------------------------------------------------------------------- #
# portfolio engine
# --------------------------------------------------------------------------- #

def sharpe_ratio(returns: Sequence[float]) -> Optional[float]:
    """Daily Sharpe annualised by sqrt(252); None when undefined."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return None
    return (mean / math.sqrt(var)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity: Sequence[float]) -> float:
    """Max peak-to-trough fraction on the (end-of-session) equity curve."""
    peak = -math.inf
    worst = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            worst = min(worst, (e - peak) / peak)
    return worst


def profit_factor(pnls: Sequence[float]) -> Optional[float]:
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses <= 0:
        return None
    return wins / losses


def run_portfolio(
    trades: Sequence[Trade],
    sessions: Sequence[str],
    starting_cash: float = 100_000.0,
    max_concurrent: int = 5,
    one_way_frac: float = 0.0,
    friction_mult: float = 1.0,
    overshoot_extra_pts: float = 0.0,
) -> Dict[str, Any]:
    """Deterministic event-driven replay of ``trades`` under constraints.

    * entry: debit = premium + premium*one_way*mult; refused (and counted)
      when cash is short or ``max_concurrent`` is reached.
    * exit: credit = max(premium*(1 + gross/100), 0) - premium*one_way*mult;
      hard-stop exits are degraded by ``overshoot_extra_pts`` points of
      premium when stress is on.
    * equity at each session close = cash + cost of open positions (see
      MTM_NOTE); zero-trade sessions yield a 0 return.
    """
    one_way = one_way_frac * friction_mult
    events: List[Tuple[datetime, int, str, Trade]] = []
    for t in sorted(trades, key=lambda t: (t.entry_at, t.decision_id)):
        events.append((t.entry_at, 0, t.decision_id, t))
        events.append((t.exit_at, 1, t.decision_id, t))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    all_days = sorted(set(sessions)
                      | {t.exit_at.date().isoformat() for t in trades})

    cash = starting_cash
    open_cost: Dict[str, float] = {}
    skipped: Dict[str, str] = {}
    per_trade_pnl: Dict[str, float] = {}
    entry_count = 0
    gross_pnl = 0.0
    friction_paid = 0.0
    notional_traded = 0.0
    pnl_by_exit_day: Dict[str, float] = {}
    equity_curve: List[Tuple[str, float]] = []
    exposure_samples: List[float] = []
    ei = 0

    for day in all_days:
        while ei < len(events) and events[ei][0].date().isoformat() <= day:
            _at, kind, did, t = events[ei]
            ei += 1
            if kind == 0:  # entry
                fee = t.premium * one_way
                cost = t.premium + fee
                if did in skipped:
                    continue
                if len(open_cost) >= max_concurrent:
                    skipped[did] = "concurrency"
                    continue
                if cost > cash:
                    skipped[did] = "cash"
                    continue
                cash -= cost
                friction_paid += fee
                notional_traded += t.premium
                open_cost[did] = cost
                entry_count += 1
            else:  # exit
                if did in skipped or did not in open_cost:
                    continue
                g_pct = t.gross_pnl_pct
                if overshoot_extra_pts and t.reason == "dynamic_stop_loss":
                    g_pct = g_pct - overshoot_extra_pts
                value = max(t.premium * (1.0 + g_pct / 100.0), 0.0)
                fee = t.premium * one_way
                cash += value - fee
                friction_paid += fee
                cost = open_cost.pop(did)
                pnl = (value - fee) - cost
                per_trade_pnl[did] = pnl
                gross_pnl += t.premium * g_pct / 100.0
                exit_day = t.exit_at.date().isoformat()
                pnl_by_exit_day[exit_day] = (
                    pnl_by_exit_day.get(exit_day, 0.0) + pnl)
        equity = cash + sum(open_cost.values())
        equity_curve.append((day, equity))
        exposure_samples.append(
            sum(open_cost.values()) / equity if equity > 0 else 0.0)

    eq_values = [starting_cash] + [e for _, e in equity_curve]
    returns = [(eq_values[i] - eq_values[i - 1]) / eq_values[i - 1]
               for i in range(1, len(eq_values)) if eq_values[i - 1] > 0]
    pnls = list(per_trade_pnl.values())
    skip_counts: Dict[str, int] = {}
    for why in skipped.values():
        skip_counts[why] = skip_counts.get(why, 0) + 1

    return {
        "n_candidate_trades": len(trades),
        "n_executed": entry_count,
        "skipped": skip_counts,
        "net_pnl": round(sum(pnls), 6),
        "gross_pnl": round(gross_pnl, 6),
        "friction_paid": round(friction_paid, 6),
        "win_rate": (sum(1 for p in pnls if p > 0) / len(pnls)
                     if pnls else None),
        "expectancy_per_trade": (sum(pnls) / len(pnls) if pnls else None),
        "profit_factor": profit_factor(pnls),
        "sharpe_daily_ann": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(
            [float(starting_cash)] + [e for _, e in equity_curve]),
        "turnover": notional_traded / starting_cash,
        "avg_exposure": (sum(exposure_samples) / len(exposure_samples)
                         if exposure_samples else 0.0),
        "final_equity": equity_curve[-1][1] if equity_curve else starting_cash,
        "sessions": list(all_days),
        "equity_curve": [[d, round(e, 6)] for d, e in equity_curve],
        "pnl_by_exit_day": {k: round(v, 6)
                            for k, v in sorted(pnl_by_exit_day.items())},
        "per_trade_pnl_sum_check": round(
            (equity_curve[-1][1] - starting_cash) if equity_curve else 0.0, 6),
    }


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #

def _quality_pass(t: Trade) -> bool:
    if t.has_quote is False:
        return False
    if t.spread_pct is not None and t.spread_pct > QUALITY_MAX_SPREAD_PCT:
        return False
    return True


def build_variants(
    trades: Sequence[Trade],
    valuation_ids: Optional[Set[str]],
    classical_preds: Dict[str, float],
    neural_preds: Dict[str, float],
    neural_status: str,
) -> Dict[str, Dict[str, Any]]:
    """Paired selection: each variant keeps a subset of the SAME trades."""
    variants: Dict[str, Dict[str, Any]] = {}

    variants["baseline_frozen"] = {
        "trades": list(trades),
        "accounting": {"n": len(trades)},
    }

    kept_q = [t for t in trades if _quality_pass(t)]
    variants["baseline_quality"] = {
        "trades": kept_q,
        "accounting": {
            "n": len(kept_q),
            "dropped_quality": len(trades) - len(kept_q),
            "max_spread_pct": QUALITY_MAX_SPREAD_PCT,
        },
    }

    if valuation_ids is not None:
        kept_v = [t for t in trades if t.decision_id in valuation_ids]
        traded_ids = {t.decision_id for t in trades}
        unavailable = len(valuation_ids - traded_ids)
        variants["valuation_ranked"] = {
            "trades": kept_v,
            "accounting": {
                "n": len(kept_v),
                "selected_total": len(valuation_ids),
                "selected_with_recorded_outcome": len(kept_v),
                "outcome_unavailable": unavailable,
                "note": COUNTERFACTUAL_NOTE,
            },
        }

    def _gated(preds: Dict[str, Tuple[float, float, float]],
               ) -> Tuple[List[Trade], Dict[str, int]]:
        # no_prediction (uncovered row -> trade passes through un-gated) is
        # accounted separately from abstained (the model SAW the row and
        # explicitly declined to endorse a direction -> trade dropped).
        kept: List[Trade] = []
        acct = {"gated_out": 0, "no_prediction": 0, "gate_passed": 0,
                "abstained": 0}
        for t in trades:
            p = preds.get(t.decision_id)
            if p is None:
                acct["no_prediction"] += 1
                kept.append(t)  # gate only acts where evidence exists
                continue
            verdict = gate_direction(t.option_type, p[0], p[1], p[2])
            if verdict == "pass":
                acct["gate_passed"] += 1
                kept.append(t)
            elif verdict == "abstain":
                acct["abstained"] += 1
            else:
                acct["gated_out"] += 1
        return kept, acct

    kept_c, acct_c = _gated(classical_preds)
    variants["classical_gated"] = {
        "trades": kept_c,
        "accounting": {"n": len(kept_c), **acct_c},
    }

    if neural_status == "ok":
        kept_n, acct_n = _gated(neural_preds)
        variants["neural_gated"] = {
            "trades": kept_n,
            "accounting": {"n": len(kept_n), **acct_n},
        }
    else:
        variants["neural_gated"] = {
            "trades": None,
            "accounting": {"status": "unavailable"},
        }

    if valuation_ids is not None:
        kept_cb, acct_cb = _gated(classical_preds)
        kept_cb = [t for t in kept_cb if t.decision_id in valuation_ids]
        variants["combined"] = {
            "trades": kept_cb,
            "accounting": {"n": len(kept_cb),
                           "classical_gate": acct_cb,
                           "and_valuation_selected": True},
        }
    return variants


# --------------------------------------------------------------------------- #
# top-level replay
# --------------------------------------------------------------------------- #

def _per_session_delta(challenger: Dict[str, float],
                       baseline: Dict[str, float]) -> Dict[str, float]:
    days = sorted(set(challenger) | set(baseline))
    return {d: challenger.get(d, 0.0) - baseline.get(d, 0.0) for d in days}


def run_replay(
    dataset_path: str,
    valuation_path: Optional[str],
    execution_calibration_path: Optional[str],
    out_path: Optional[str] = None,
    holdout: Sequence[str] = DEFAULT_HOLDOUT,
    starting_cash: float = 100_000.0,
    max_concurrent: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
    include_neural: bool = True,
    mlp_epochs: int = 20,
) -> Dict[str, Any]:
    trades, trade_report = load_traded_outcomes(dataset_path, holdout)
    traded_sessions = sorted({t.session for t in trades})
    calendar = load_session_calendar(dataset_path, holdout)
    # full feature-session calendar including no-trade days (Stage F);
    # fall back to traded sessions only when the dataset carries none.
    sessions = calendar or traded_sessions
    friction = load_friction(execution_calibration_path)
    meta = dataset_meta(dataset_path)

    valuation_ids: Optional[Set[str]] = None
    valuation_report: Optional[Dict[str, Any]] = None
    if valuation_path and os.path.exists(valuation_path):
        valuation_ids, valuation_report = load_valuation_selection(
            valuation_path)

    # refuse mismatched artifacts BEFORE any computation (raises ValueError)
    input_crosschecks = crosscheck_inputs(meta, valuation_report, friction)

    classical_preds, classical_report = compute_oof_classical(
        dataset_path, holdout)
    if include_neural:
        neural_preds, neural_report = compute_oof_neural(
            dataset_path, holdout, max_epochs=mlp_epochs, seed=seed)
    else:
        neural_preds, neural_report = {}, {
            "model": "multitask_mlp", "status": "unavailable",
            "why": "disabled by --no-neural"}

    variants = build_variants(
        trades, valuation_ids, classical_preds, neural_preds,
        neural_report.get("status", "unavailable"))

    # --- acceptance: baseline parity (frictionless engine vs outcome store)
    parity_engine = run_portfolio(
        trades, sessions, starting_cash=1e12, max_concurrent=10**9,
        one_way_frac=0.0, friction_mult=1.0)
    independent_gross = sum(
        t.premium * t.gross_pnl_pct / 100.0 for t in trades)
    floored_gross = sum(
        max(t.premium * (1.0 + t.gross_pnl_pct / 100.0), 0.0) - t.premium
        for t in trades)
    parity = {
        "n_trades_engine": parity_engine["n_executed"],
        "n_trades_outcomes": len(trades),
        "engine_gross_pnl": parity_engine["gross_pnl"],
        "independent_gross_pnl": round(independent_gross, 6),
        "engine_net_pnl_frictionless": parity_engine["net_pnl"],
        "independent_net_pnl_floored": round(floored_gross, 6),
        "max_abs_diff": round(max(
            abs(parity_engine["gross_pnl"] - independent_gross),
            abs(parity_engine["net_pnl"] - floored_gross)), 9),
    }
    parity["ok"] = (parity["n_trades_engine"] == parity["n_trades_outcomes"]
                    and parity["max_abs_diff"] < 1e-6)

    # --- paired runs across friction multipliers
    results: Dict[str, Any] = {}
    per_session_base: Dict[str, Dict[str, float]] = {}
    for name, spec in variants.items():
        if spec["trades"] is None:
            results[name] = {"status": "unavailable",
                             "accounting": spec["accounting"]}
            continue
        by_mult: Dict[str, Any] = {}
        for mult in FRICTION_MULTS:
            m = run_portfolio(
                spec["trades"], sessions,
                starting_cash=starting_cash,
                max_concurrent=max_concurrent,
                one_way_frac=friction["one_way_frac"],
                friction_mult=mult)
            by_mult[f"{mult:.1f}"] = m
            if name == "baseline_frozen":
                per_session_base[f"{mult:.1f}"] = m["pnl_by_exit_day"]
        stress = run_portfolio(
            spec["trades"], sessions,
            starting_cash=starting_cash,
            max_concurrent=max_concurrent,
            one_way_frac=friction["one_way_frac"],
            friction_mult=1.0,
            overshoot_extra_pts=friction["overshoot_extra_pts"])
        results[name] = {
            "status": "ok",
            "accounting": spec["accounting"],
            "by_friction_mult": by_mult,
            "stop_overshoot_stress": {
                "extra_pts": friction["overshoot_extra_pts"],
                "net_pnl": stress["net_pnl"],
                "delta_vs_unstressed": round(
                    stress["net_pnl"]
                    - by_mult["1.0"]["net_pnl"], 6),
            },
        }

    # --- paired per-session deltas vs baseline (friction 1.0x)
    paired: Dict[str, Any] = {}
    base_days = per_session_base.get("1.0", {})
    for name, res in results.items():
        if name == "baseline_frozen" or res.get("status") != "ok":
            continue
        days = res["by_friction_mult"]["1.0"]["pnl_by_exit_day"]
        delta = _per_session_delta(days, base_days)
        nonzero = [abs(v) for v in delta.values() if v != 0.0]
        paired[name] = {
            "per_session_delta": {k: round(v, 6) for k, v in delta.items()},
            "bootstrap": paired_session_bootstrap(
                delta, n_boot=n_boot, seed=seed),
            # clustered-uncertainty honesty: sessions where the challenger
            # made no different choice carry zero delta and inflate apparent
            # n; the Kish ESS over |delta| shows the effective sample.
            "n_nonzero_delta_sessions": len(nonzero),
            "ess_abs_delta": round(kish_ess(nonzero), 4),
        }

    # --- matched cohort: identical prediction/quote-eligible trades
    # (Stage F) -- separates "no prediction fallback" from challenger
    # abstention, so the model comparison is apples-to-apples instead of
    # being obscured by rows the challenger never saw.
    cohort = [t for t in trades
              if t.decision_id in classical_preds and _quality_pass(t)]
    excluded_no_prediction = sum(
        1 for t in trades if t.decision_id not in classical_preds)
    excluded_quote_ineligible = sum(
        1 for t in trades
        if t.decision_id in classical_preds and not _quality_pass(t))
    cohort_kept: List[Trade] = []
    cohort_acct = {"gate_passed": 0, "abstained": 0, "gated_out": 0}
    for t in cohort:
        p = classical_preds[t.decision_id]
        verdict = gate_direction(t.option_type, p[0], p[1], p[2])
        if verdict == "pass":
            cohort_acct["gate_passed"] += 1
            cohort_kept.append(t)
        elif verdict == "abstain":
            cohort_acct["abstained"] += 1
        else:
            cohort_acct["gated_out"] += 1

    def _slim(m: Dict[str, Any]) -> Dict[str, Any]:
        return {k: m[k] for k in (
            "n_executed", "net_pnl", "profit_factor", "sharpe_daily_ann",
            "max_drawdown", "pnl_by_exit_day")}

    matched_cohort: Dict[str, Any]
    if cohort:
        base_m = run_portfolio(
            cohort, sessions, starting_cash=starting_cash,
            max_concurrent=max_concurrent,
            one_way_frac=friction["one_way_frac"], friction_mult=1.0)
        chal_m = run_portfolio(
            cohort_kept, sessions, starting_cash=starting_cash,
            max_concurrent=max_concurrent,
            one_way_frac=friction["one_way_frac"], friction_mult=1.0)
        c_delta = _per_session_delta(chal_m["pnl_by_exit_day"],
                                     base_m["pnl_by_exit_day"])
        c_nonzero = [abs(v) for v in c_delta.values() if v != 0.0]
        matched_cohort = {
            "status": "ok",
            "n_cohort": len(cohort),
            "excluded_no_prediction": excluded_no_prediction,
            "excluded_quote_ineligible": excluded_quote_ineligible,
            "challenger_gate": cohort_acct,
            "baseline_on_cohort": _slim(base_m),
            "challenger_on_cohort": _slim(chal_m),
            "paired": {
                "per_session_delta": {k: round(v, 6)
                                      for k, v in c_delta.items()},
                "bootstrap": paired_session_bootstrap(
                    c_delta, n_boot=n_boot, seed=seed),
                "n_nonzero_delta_sessions": len(c_nonzero),
                "ess_abs_delta": round(kish_ess(c_nonzero), 4),
            },
            "note": ("baseline and challenger evaluated on the IDENTICAL "
                     "prediction-covered, quote-eligible cohort at 1.0x "
                     "friction; no-prediction fallback rows are excluded "
                     "here (counted), so challenger abstention is not "
                     "conflated with coverage gaps"),
        }
    else:
        matched_cohort = {
            "status": "empty",
            "n_cohort": 0,
            "excluded_no_prediction": excluded_no_prediction,
            "excluded_quote_ineligible": excluded_quote_ineligible,
            "note": "no prediction-covered quote-eligible trades",
        }
    artifact: Dict[str, Any] = {
        "generated_for": "oracle4_phase6_paired_replay",
        "phase": 6,
        "seed": seed,
        "config": {
            "starting_cash": starting_cash,
            "max_concurrent": max_concurrent,
            "friction_mults": list(FRICTION_MULTS),
            "n_boot": n_boot,
            "quality_max_spread_pct": QUALITY_MAX_SPREAD_PCT,
            "mlp_epochs": mlp_epochs,
        },
        "inputs": {
            "dataset": dataset_path,
            "dataset_meta": meta,
            "valuation": valuation_report,
            "execution_calibration": friction,
        },
        "holdout_refusal": {
            "sessions": sorted(set(holdout)),
            "decisions_excluded": trade_report["holdout_excluded"],
            "note": ("holdout refusal is at the DECISION level; exits of "
                     "in-window decisions may land on holdout days"),
        },
        "trade_universe": trade_report,
        "sessions_replayed": sessions,
        "session_calendar": {
            "n_sessions": len(sessions),
            "n_traded_sessions": len(traded_sessions),
            "no_trade_sessions": sorted(set(sessions)
                                        - set(traded_sessions)),
            "note": ("replay runs over the full feature-session calendar; "
                     "zero-trade sessions contribute a 0 equity return to "
                     "Sharpe/drawdown instead of being dropped"),
        },
        "input_crosschecks": input_crosschecks,
        "mtm_note": MTM_NOTE,
        "missing_data_accounting": {
            "outcome_status_counts_nonholdout":
                trade_report["outcome_status_counts"],
            "counterfactual_note": COUNTERFACTUAL_NOTE,
            "classical_no_prediction":
                results["classical_gated"]["accounting"].get(
                    "no_prediction"),
            "neural": neural_report,
            "selection_bias_note": (
                "the replay universe is the baseline bot's own fills, so "
                "baseline coverage is 100% by construction while every "
                "challenger can only drop trades; challenger upside that "
                "required trading NON-baseline picks is invisible here and "
                "shows up only as outcome_unavailable counts."),
        },
        "oof_gates": {
            "classical": classical_report,
            "neural": {k: v for k, v in neural_report.items()},
        },
        "parity": parity,
        "variants": results,
        "paired_vs_baseline": paired,
        "matched_cohort": matched_cohort,
        "risk_metrics_limited": {
            "limited": True,
            "reasons": [
                ("positions are carried at entry cost between entry and "
                 "exit (no daily option marks), so intraday/overnight "
                 "drawdown and Sharpe volatility are UNDERSTATED"),
            ] + ([
                ("short sample: fewer than 30 sessions, so sqrt(252) "
                 "annualised Sharpe is an extrapolation, not an estimate"),
            ] if len(sessions) < 30 else []),
            "note": ("drawdown/Sharpe here are LIMITED estimates and are "
                     "not sufficient for promotion on their own; promotion "
                     "requires a validated forward-evidence artifact"),
        },
        "event_ordering_note": (
            "events are replayed in deterministic (timestamp, "
            "entry-before-exit, decision_id) order; because entries sort "
            "BEFORE exits at equal timestamps, cash credited by a "
            "same-timestamp exit can never fund that entry -- no "
            "same-timestamp cash reuse (conservative)"),
        "provenance_note": (
            "All P&L derives from PAPER marks recorded by the production "
            "bot; no counterfactual fills were fabricated and no broker or "
            "network access occurred."),
    }
    artifact["fingerprint"] = metrics_fingerprint(artifact)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=1, sort_keys=True)
    return artifact


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Oracle 4.0 Phase 6b paired portfolio replay")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--valuation", default=None)
    ap.add_argument("--execution-calibration", default=None)
    ap.add_argument("--out", default="artifacts/oracle4/replay.json")
    ap.add_argument("--holdout", nargs="*", default=list(DEFAULT_HOLDOUT))
    ap.add_argument("--starting-cash", type=float, default=100_000.0)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--mlp-epochs", type=int, default=20)
    args = ap.parse_args(argv)

    artifact = run_replay(
        dataset_path=args.dataset,
        valuation_path=args.valuation,
        execution_calibration_path=args.execution_calibration,
        out_path=args.out,
        holdout=args.holdout,
        starting_cash=args.starting_cash,
        max_concurrent=args.max_concurrent,
        n_boot=args.n_boot,
        seed=args.seed,
        include_neural=not args.no_neural,
        mlp_epochs=args.mlp_epochs,
    )
    parity = artifact["parity"]
    print(f"parity ok={parity['ok']} "
          f"(n={parity['n_trades_engine']}, "
          f"max_abs_diff={parity['max_abs_diff']})")
    for name, res in artifact["variants"].items():
        if res.get("status") != "ok":
            print(f"{name:18s} status={res.get('status')}")
            continue
        m = res["by_friction_mult"]["1.0"]
        print(f"{name:18s} n={m['n_executed']:4d} "
              f"net={m['net_pnl']:12.2f} pf={m['profit_factor']} "
              f"sharpe={m['sharpe_daily_ann']}")
    print(f"fingerprint={artifact['fingerprint']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
