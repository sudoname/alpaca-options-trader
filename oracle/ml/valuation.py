"""Oracle 4.0 Phase 5 — scenario valuation + all-candidate ranking.

Research-only.  Nothing in the production trading path imports this.

Pipeline per scan (= one (session, underlying, time-bucket) group of
eligible same-side candidates):

  1. **Scenarios** — past-only circular block bootstrap of step log
     returns of the underlying, built from ``underlying_obs`` strictly at
     or before ``as_of``.  Blocks preserve the joint dependence of
     return, volatility and timing inside each block; marginal quantiles
     are never sampled independently.  One scenario matrix is generated
     per scan and shared by **every** candidate in it (common random
     numbers), so cross-candidate EV differences are not Monte-Carlo
     noise.  An EWMA vol forecast (RiskMetrics lambda) rescales the
     sampled returns; the scale is clamped and the clamp recorded as an
     out-of-distribution flag.
  2. **Repricing** — each contract is repriced along every path with the
     European BSM formula at the entry implied vol solved from the
     observed mid (sticky-strike constant IV).  This is a *physical*
     scenario of the underlying priced with a *risk-neutral* formula; the
     model-implied ITM probability is never reported as empirical PoP.
     The CRR American premium over BSM at entry is recorded per candidate
     as the approximation error (``american_gap_pct``).
  3. **Exit policy** — the production ``exit_manager`` semantics are
     simulated on the path marks: stop loss (−10 %) takes precedence over
     take profit (+20 %), then time exit at the simulation horizon (or
     expiry → intrinsic settlement).  Triggers use the mark vs the entry
     fill price exactly like production ``pnl_percent``.  Trailing stops
     start inactive in production and are not simulated (documented
     approximation).  Step granularity means intrabar breaches are seen
     at the step close (documented approximation; the Phase 1 overshoot
     study quantifies real stop overshoot).
  4. **Costs** — executable prices embed spread + slippage (entry fills
     at mid + half-spread + slippage; exit sells at mark − half-spread −
     slippage, floored at zero) and OCC/commission fees are subtracted
     once; nothing is double counted and the contract multiplier (100)
     is applied exactly once.  Exit spread is proxied by the entry
     spread (exit quotes are unavailable in the store; marked as an
     approximation).  Friction stress reprices the same paths at 1.0x /
     1.5x / 2.0x frictions (spread, slippage, fees scale together) with
     an explicit cost decomposition.
  5. **Opportunity EV** — ``ev_opportunity = p_fill * E[net | filled]``
     with the existing conservative ``FillModel`` prior for p_fill and
     partial-fill quantity scaling; known non-fills (missing/unusable
     quotes) contribute exactly zero.
  6. **Ranking** — every eligible same-side candidate is valued; ranking
     is by expected executable net dollars then capital efficiency with
     deterministic tie-breaks, subject to liquidity / uncertainty /
     downside gates.  Abstention (EV = 0) is an explicit option:
     negative or unknown EV candidates can never beat it.  The legacy
     ``baseline_score`` ordering is reported as a separate comparator,
     never blended.

Usage:
  python -m oracle.ml.valuation --dataset artifacts/oracle4/dataset_v1.sqlite \
      --out artifacts/oracle4/valuation.json --seed 42 --n-paths 512
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cost_model import CONTRACT_MULTIPLIER, CostConfig
from oracle.execution.client import OrderRequest, Quote
from oracle.execution.fill_model import FillModel
from pricing_diagnostics import (
    QUOTE_OK,
    bs_price,
    classify_quote,
    crr_american_price,
    solve_iv,
)

__all__ = [
    "ValuationConfig",
    "Candidate",
    "ScenarioSet",
    "parse_occ_symbol",
    "load_candidates",
    "load_underlying_obs",
    "generate_scenarios",
    "reprice_candidate",
    "value_candidate",
    "rank_candidates",
    "main",
]

RISK_GATE_REASONS = (
    "ev_not_positive", "uncertain_ev", "wide_spread", "excess_downside",
)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ValuationConfig:
    """All knobs; defaults mirror production exit/cost behaviour."""
    n_paths: int = 512                 # clamped to [256, 2048]
    step_minutes: float = 5.0
    max_steps: int = 78                # one regular session of 5-min steps
    block_len: int = 6                 # 30 min blocks keep local vol clusters
    min_history_returns: int = 30
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    stop_loss_pct: float = 10.0        # exit_manager defaults
    take_profit_pct: float = 20.0
    qty: int = 1
    scan_bucket_min: float = 5.0
    friction_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0)
    crr_steps: int = 200
    ewma_lambda: float = 0.94
    vol_scale_bounds: Tuple[float, float] = (0.5, 2.0)
    max_spread_pct: float = 0.40       # liquidity gate
    uncertainty_k: float = 1.0         # require EV > k * MC standard error
    max_es95_frac_of_capital: float = 1.0   # long premium is the max loss

    def clamped_paths(self) -> int:
        return max(256, min(2048, int(self.n_paths)))


# --------------------------------------------------------------------------- #
# candidates from the feature store
# --------------------------------------------------------------------------- #

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ_symbol(symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse an OCC option symbol; None when it is not one."""
    if not symbol:
        return None
    m = _OCC_RE.match(symbol.strip().upper())
    if not m:
        return None
    root, ymd, cp, strike8 = m.groups()
    try:
        exp = datetime.strptime(ymd, "%y%m%d").date()
    except ValueError:
        return None
    strike = int(strike8) / 1000.0
    if strike <= 0:
        return None
    return {
        "underlying": root,
        "expiration": exp.isoformat(),
        "option_type": "call" if cp == "C" else "put",
        "strike": strike,
    }


@dataclass
class Candidate:
    decision_id: str
    symbol: Optional[str]
    underlying: str
    session: str
    as_of: datetime
    kind: str
    features: Dict[str, Optional[float]]
    contract: Optional[Dict[str, Any]]   # parsed OCC fields or None

    @property
    def quote(self) -> Tuple[Optional[float], Optional[float]]:
        """(bid, ask) reconstructed exactly from mid +/- spread/2."""
        mid = self.features.get("quote_mid")
        spread = self.features.get("spread_abs")
        if mid is None or spread is None or spread < 0:
            return None, None
        return mid - spread / 2.0, mid + spread / 2.0


def _parse_utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_candidates(dataset_path: str,
                    sessions: Optional[Sequence[str]] = None
                    ) -> List[Candidate]:
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        out: List[Candidate] = []
        allow = set(sessions) if sessions else None
        cur = conn.execute(
            "SELECT decision_id, symbol, underlying, session, as_of, kind,"
            "       features_json FROM features ORDER BY as_of, decision_id")
        for row in cur:
            if allow is not None and row["session"] not in allow:
                continue
            as_of = _parse_utc(row["as_of"])
            if as_of is None:
                continue
            out.append(Candidate(
                decision_id=row["decision_id"],
                symbol=row["symbol"],
                underlying=row["underlying"],
                session=row["session"],
                as_of=as_of,
                kind=row["kind"],
                features=json.loads(row["features_json"]),
                contract=parse_occ_symbol(row["symbol"]),
            ))
        return out
    finally:
        conn.close()


def load_underlying_obs(dataset_path: str
                        ) -> Dict[str, List[Tuple[datetime, float]]]:
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    try:
        out: Dict[str, List[Tuple[datetime, float]]] = {}
        cur = conn.execute(
            "SELECT underlying, event_at, price FROM underlying_obs "
            "ORDER BY underlying, event_at")
        for u, event_at, price in cur:
            t = _parse_utc(event_at)
            if t is None or price is None or price <= 0:
                continue
            out.setdefault(u, []).append((t, float(price)))
        return out
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# scenarios: past-only block bootstrap with common random numbers
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioSet:
    prices: np.ndarray                 # (n_paths, n_steps + 1); col 0 == s0
    step_minutes: float
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return self.prices.shape[1] - 1


def _step_returns(obs: Sequence[Tuple[datetime, float]], as_of: datetime,
                  step_minutes: float) -> np.ndarray:
    """Past-only step log returns, LOCF-resampled per session day.

    Overnight gaps are excluded (returns never span session boundaries),
    so sampled paths describe intrasession evolution only.
    """
    by_day: Dict[str, List[Tuple[datetime, float]]] = {}
    for t, p in obs:
        if t > as_of:
            break                       # obs are time-ordered
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append((t, p))

    step_s = step_minutes * 60.0
    rets: List[float] = []
    for day in sorted(by_day):
        series = by_day[day]
        t0 = series[0][0]
        grid_vals: List[float] = []
        idx = 0
        k = 0
        last: Optional[float] = None
        end = series[-1][0]
        while True:
            t_k = t0.timestamp() + k * step_s
            if t_k > end.timestamp():
                break
            while idx < len(series) and series[idx][0].timestamp() <= t_k:
                last = series[idx][1]
                idx += 1
            if last is not None and last > 0:
                grid_vals.append(last)
            k += 1
        for a, b in zip(grid_vals, grid_vals[1:]):
            if a > 0 and b > 0:
                rets.append(math.log(b / a))
    return np.asarray(rets, dtype=np.float64)


def _ewma_vol(returns: np.ndarray, lam: float) -> Optional[float]:
    if returns.size < 2:
        return None
    var = float(returns[0] ** 2)
    for r in returns[1:]:
        var = lam * var + (1.0 - lam) * float(r) ** 2
    return math.sqrt(var) if var > 0 else None


def scenario_seed(base_seed: int, underlying: str, bucket_key: str) -> int:
    raw = hashlib.sha256(
        f"{base_seed}|{underlying}|{bucket_key}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % (2 ** 63 - 1)


def generate_scenarios(obs: Sequence[Tuple[datetime, float]],
                       as_of: datetime, s0: float,
                       config: ValuationConfig,
                       seed: int) -> Tuple[Optional[ScenarioSet], str]:
    """(scenarios, reason).  reason != 'ok' means no scenarios."""
    if s0 is None or s0 <= 0:
        return None, "missing_underlying_price"
    rets = _step_returns(obs, as_of, config.step_minutes)
    if rets.size < config.min_history_returns:
        return None, "insufficient_history"

    hist_vol = float(np.std(rets))
    ewma = _ewma_vol(rets, config.ewma_lambda)
    scale, clamped = 1.0, False
    if ewma is not None and hist_vol > 0:
        raw_scale = ewma / hist_vol
        lo, hi = config.vol_scale_bounds
        scale = min(max(raw_scale, lo), hi)
        clamped = (scale != raw_scale)

    n_paths = config.clamped_paths()
    n_steps = config.max_steps
    rng = np.random.default_rng(seed)
    blk = max(1, int(config.block_len))
    n_blocks = (n_steps + blk - 1) // blk
    starts = rng.integers(0, rets.size, size=(n_paths, n_blocks))
    offsets = np.arange(blk)
    idx = (starts[:, :, None] + offsets[None, None, :]) % rets.size
    sampled = rets[idx].reshape(n_paths, n_blocks * blk)[:, :n_steps]
    sampled = sampled * scale

    log_paths = np.cumsum(sampled, axis=1)
    prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    prices[:, 0] = s0
    prices[:, 1:] = s0 * np.exp(log_paths)
    meta = {
        "n_source_returns": int(rets.size),
        "hist_step_vol": hist_vol,
        "ewma_step_vol": ewma,
        "vol_scale": scale,
        "vol_scale_clamped_ood": clamped,
        "rv_forecast": ("ewma" if ewma is not None
                        else "unavailable_scale_1"),
        "block_len": blk,
        "seed": int(seed),
    }
    return ScenarioSet(prices=prices, step_minutes=config.step_minutes,
                       meta=meta), "ok"


# --------------------------------------------------------------------------- #
# vectorized BSM (verified against pricing_diagnostics.bs_price in tests)
# --------------------------------------------------------------------------- #

_ERF = np.vectorize(math.erf, otypes=[np.float64])


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(x / math.sqrt(2.0)))


def _bs_price_vec(S: np.ndarray, K: float, T: float, r: float, q: float,
                  sigma: float, opt_type: str) -> np.ndarray:
    """European BSM over a vector of spots at one (K, T, sigma).

    Degenerate T<=0 or sigma<=0 returns discounted forward intrinsic,
    matching the scalar ``pricing_diagnostics.bs_price`` exactly.
    """
    S = np.asarray(S, dtype=np.float64)
    sign = 1.0 if opt_type == "call" else -1.0
    if T <= 0 or sigma <= 0:
        tt = max(T, 0.0)
        fwd = S * math.exp(-q * tt) - K * math.exp(-r * tt)
        return np.maximum(sign * fwd, 0.0)
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    return sign * (disc_s * _norm_cdf(sign * d1)
                   - disc_k * _norm_cdf(sign * d2))


# --------------------------------------------------------------------------- #
# per-candidate valuation
# --------------------------------------------------------------------------- #

def _abstain(cand: Candidate, reason: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "decision_id": cand.decision_id,
        "symbol": cand.symbol,
        "underlying": cand.underlying,
        "session": cand.session,
        "status": "abstain",
        "reason": reason,
        "ev_opportunity": 0.0,
        "baseline_score": cand.features.get("baseline_score"),
    }
    out.update(extra)
    return out


def reprice_candidate(cand: Candidate, scen: ScenarioSet,
                      config: ValuationConfig
                      ) -> Tuple[Optional[Dict[str, Any]],
                                 Optional[Dict[str, Any]]]:
    """Reprice the contract along the shared paths.

    Returns (priced, abstain_row).  Exactly one is non-None.  ``priced``
    carries the mark matrix and entry context; costs/exits are applied by
    :func:`value_candidate` so friction stress can reuse the repricing.
    """
    bid, ask = cand.quote
    if bid is None or ask is None:
        return None, _abstain(cand, "missing_quote")
    status = classify_quote(bid, ask)
    if status != QUOTE_OK:
        return None, _abstain(cand, f"unusable_quote:{status.lower()}")
    if cand.contract is None:
        return None, _abstain(cand, "unparseable_symbol")
    # The scenario paths are generated from the scan's stored underlying.
    # A contract whose OCC root names a different underlying cannot be
    # repriced on those paths (observed in ~0.5% of real rows where the
    # upstream logger stored candidates under the scan symbol).
    root = cand.contract.get("underlying")
    if cand.underlying and root and root != cand.underlying:
        return None, _abstain(cand, "underlying_mismatch")

    dte = cand.features.get("dte_days")
    if dte is None:
        return None, _abstain(cand, "missing_dte")
    if dte <= 0:
        return None, _abstain(cand, "expired")
    s0 = cand.features.get("underlying_price")
    if s0 is None or s0 <= 0:
        return None, _abstain(cand, "missing_underlying_price")

    K = float(cand.contract["strike"])
    opt_type = cand.contract["option_type"]
    T0 = float(dte) / 365.0
    r, q = config.risk_free_rate, config.dividend_yield
    mid = (bid + ask) / 2.0

    iv_res = solve_iv(mid, s0, K, T0, r, q, opt_type)
    if iv_res.get("status") != "ok" or not iv_res.get("iv"):
        return None, _abstain(
            cand, f"iv_unresolved:{iv_res.get('status', 'unknown')}")
    sigma = float(iv_res["iv"])

    # approximation error vs the American reference at entry
    try:
        amer = crr_american_price(s0, K, T0, r, q, sigma, opt_type,
                                  steps=config.crr_steps)
        euro = bs_price(s0, K, T0, r, q, sigma, opt_type)
        gap_pct = ((amer - euro) / euro * 100.0) if euro > 0 else None
    except ValueError:
        gap_pct = None

    dte_minutes = float(dte) * 1440.0
    n_steps = min(scen.n_steps,
                  max(1, int(dte_minutes // scen.step_minutes)))
    expires_in_sim = (n_steps < scen.n_steps) or \
        (dte_minutes <= scen.n_steps * scen.step_minutes)

    marks = np.empty((scen.prices.shape[0], n_steps + 1), dtype=np.float64)
    marks[:, 0] = mid
    minutes_per_year = 365.0 * 1440.0
    for k in range(1, n_steps + 1):
        T_k = T0 - (k * scen.step_minutes) / minutes_per_year
        if expires_in_sim and k == n_steps:
            T_k = min(T_k, 0.0)         # settle at intrinsic on expiry
        marks[:, k] = _bs_price_vec(scen.prices[:, k], K, T_k, r, q,
                                    sigma, opt_type)

    return {
        "marks": marks,
        "bid": bid, "ask": ask, "mid": mid,
        "sigma": sigma,
        "american_gap_pct": gap_pct,
        "expires_in_sim": expires_in_sim,
        "n_steps": n_steps,
    }, None


def _simulate_exit(marks: np.ndarray, entry_fill: float,
                   config: ValuationConfig,
                   expires_in_sim: bool
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Vectorized exit policy on the mark path.

    Precedence per production exit_manager: stop > target, else time exit
    at the horizon (expiry settles at intrinsic in the final mark).
    Returns (exit_step, exit_mark, reason_counts).
    """
    n_paths, n_cols = marks.shape
    steps = marks[:, 1:]                           # decisions after entry
    stop_lvl = entry_fill * (1.0 - config.stop_loss_pct / 100.0)
    tgt_lvl = entry_fill * (1.0 + config.take_profit_pct / 100.0)
    hit = (steps <= stop_lvl) | (steps >= tgt_lvl)
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), n_cols - 2)
    exit_step = first + 1                          # 1-based mark column
    rows = np.arange(n_paths)
    exit_mark = marks[rows, exit_step]

    is_stop = any_hit & (exit_mark <= stop_lvl)
    is_tgt = any_hit & ~is_stop
    time_exit = ~any_hit
    reasons = {
        "stop_loss": int(is_stop.sum()),
        "take_profit": int(is_tgt.sum()),
        ("expiry" if expires_in_sim else "time_exit"): int(time_exit.sum()),
    }
    return exit_step, exit_mark, reasons


def value_candidate(cand: Candidate, priced: Dict[str, Any],
                    config: ValuationConfig,
                    friction: float = 1.0,
                    cost_config: Optional[CostConfig] = None,
                    fill_model: Optional[FillModel] = None
                    ) -> Dict[str, Any]:
    """Cost + exit-policy valuation of one repriced candidate.

    Net P&L uses executable entry/exit prices (spread + slippage embedded)
    and subtracts only fees, which are not embedded — costs are charged
    exactly once, with the 100x contract multiplier applied exactly once.
    """
    cc = cost_config or CostConfig()
    bid, ask, mid = priced["bid"], priced["ask"], priced["mid"]
    marks: np.ndarray = priced["marks"]
    qty = max(1, int(config.qty))

    spread = max(ask - bid, cc.min_spread_floor)
    half = spread / 2.0 * friction
    slip = cc.slippage_per_contract * friction
    fee_leg = (cc.occ_fee_per_contract
               + cc.commission_per_contract) * friction

    entry_fill = mid + half + slip                 # == ask + slip at 1.0x
    if entry_fill <= 0:
        return _abstain(cand, "nonpositive_entry_fill")

    exit_step, exit_mark, reasons = _simulate_exit(
        marks, entry_fill, config, priced["expires_in_sim"])
    settled = priced["expires_in_sim"] & \
        (exit_step == priced["n_steps"])           # expiry: cash settlement
    exit_frict = np.where(settled, 0.0, half + slip)
    exit_fill = np.maximum(exit_mark - exit_frict, 0.0)
    exit_fees = np.where(settled, 0.0, fee_leg)

    mult = CONTRACT_MULTIPLIER
    gross = (exit_mark - mid) * mult * qty         # frictionless reference
    fees_total = (fee_leg + exit_fees) * qty
    net = (exit_fill - entry_fill) * mult * qty - fees_total

    # fill probability from the existing conservative prior
    fm = fill_model or FillModel()
    est = fm.estimate_fill(
        OrderRequest(symbol=cand.symbol or "", side="buy", qty=qty,
                     order_type="market"),
        Quote(symbol=cand.symbol or "", bid=bid, ask=ask))
    p_fill = float(est.fill_probability or 0.0)
    fill_frac = 1.0
    if est.partial_qty is not None and qty > 0:
        fill_frac = max(0.0, min(1.0, est.partial_qty / qty))
    if p_fill <= 0.0:
        return _abstain(cand, "no_fill_expected",
                        fill_status=est.status, fill_reasons=est.reasons)

    mean_net = float(np.mean(net))
    n = net.size
    mc_se = float(np.std(net, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    tail_n = max(1, int(math.ceil(0.05 * n)))
    es95 = float(np.mean(np.sort(net)[:tail_n]))
    capital = entry_fill * mult * qty
    ev_opportunity = p_fill * fill_frac * mean_net

    return {
        "decision_id": cand.decision_id,
        "symbol": cand.symbol,
        "underlying": cand.underlying,
        "session": cand.session,
        "status": "ok",
        "reason": None,
        "friction_multiplier": friction,
        "entry": {"bid": bid, "ask": ask, "mid": mid,
                  "fill_price": entry_fill, "sigma": priced["sigma"],
                  "american_gap_pct": priced["american_gap_pct"]},
        "per_fill": {
            "net_mean": mean_net,
            "net_median": float(np.median(net)),
            "net_std": float(np.std(net, ddof=1)) if n > 1 else 0.0,
            "net_p05": float(np.quantile(net, 0.05)),
            "net_p95": float(np.quantile(net, 0.95)),
            "gross_mean": float(np.mean(gross)),
            "pop": float(np.mean(net > 0.0)),
            "es95": es95,
            "mc_se": mc_se,
            "mean_exit_step": float(np.mean(exit_step)),
            "exit_reasons": reasons,
        },
        "costs": {
            "spread_entry": half * mult * qty,
            "spread_exit_mean": float(np.mean(
                np.where(settled, 0.0, half))) * mult * qty,
            "slippage_round_trip_mean": (slip + float(np.mean(
                np.where(settled, 0.0, slip)))) * mult * qty,
            "fees_mean": float(np.mean(fees_total)),
            "exit_spread_source": "entry_spread_proxy",
        },
        "fill": {"p_fill": p_fill, "fill_frac": fill_frac,
                 "status": est.status},
        "capital_required": capital,
        "ev_opportunity": ev_opportunity,
        "ev_per_dollar": ev_opportunity / capital if capital > 0 else None,
        "n_paths": int(n),
        "baseline_score": cand.features.get("baseline_score"),
        "approximations": [
            "sticky_strike_constant_iv",
            "exit_spread_equals_entry_spread",
            "step_granularity_exits",
            "european_bsm_marks_american_gap_recorded",
            "calendar_minute_theta_decay",
        ],
    }


# --------------------------------------------------------------------------- #
# ranking with explicit abstention
# --------------------------------------------------------------------------- #

def rank_candidates(valuations: Sequence[Dict[str, Any]],
                    config: Optional[ValuationConfig] = None
                    ) -> Dict[str, Any]:
    """Rank one scan's valuations; abstention (EV=0) is always on the menu.

    Deterministic: gates produce recorded reasons; ties break on
    capital efficiency, then symbol, then decision_id.  The legacy
    baseline_score ordering is reported separately as a comparator.
    """
    cfg = config or ValuationConfig()
    eligible: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    for v in valuations:
        did = v["decision_id"]
        if v["status"] != "ok":
            excluded.append({"decision_id": did, "reason": v["reason"]})
            continue
        ev = v["ev_opportunity"]
        pf = v["per_fill"]
        spread_pct = ((v["entry"]["ask"] - v["entry"]["bid"])
                      / v["entry"]["mid"] if v["entry"]["mid"] > 0
                      else float("inf"))
        if ev <= 0.0:
            excluded.append({"decision_id": did,
                             "reason": "ev_not_positive"})
        elif ev <= cfg.uncertainty_k * pf["mc_se"]:
            excluded.append({"decision_id": did, "reason": "uncertain_ev"})
        elif spread_pct > cfg.max_spread_pct:
            excluded.append({"decision_id": did, "reason": "wide_spread"})
        elif (v["capital_required"] > 0 and -pf["es95"]
              > cfg.max_es95_frac_of_capital * v["capital_required"]):
            excluded.append({"decision_id": did,
                             "reason": "excess_downside"})
        else:
            eligible.append(v)

    ranked = sorted(
        eligible,
        key=lambda v: (-v["ev_opportunity"], -(v["ev_per_dollar"] or 0.0),
                       v["symbol"] or "", v["decision_id"]))

    scored = [v for v in valuations
              if v.get("baseline_score") is not None]
    baseline = sorted(
        scored,
        key=lambda v: (-v["baseline_score"], v["symbol"] or "",
                       v["decision_id"]))

    selected = ranked[0] if ranked else None
    return {
        "decision": "trade" if selected else "abstain",
        "selected_decision_id": selected["decision_id"] if selected else None,
        "selected_symbol": selected["symbol"] if selected else None,
        "selected_ev": selected["ev_opportunity"] if selected else 0.0,
        "abstention_ev": 0.0,
        "ranking": [
            {"decision_id": v["decision_id"], "symbol": v["symbol"],
             "ev_opportunity": v["ev_opportunity"],
             "ev_per_dollar": v["ev_per_dollar"],
             "pop": v["per_fill"]["pop"],
             "es95": v["per_fill"]["es95"],
             "capital_required": v["capital_required"],
             "baseline_score": v.get("baseline_score")}
            for v in ranked],
        "excluded": sorted(excluded,
                           key=lambda e: (e["reason"], e["decision_id"])),
        "baseline_top_decision_id": (
            baseline[0]["decision_id"] if baseline else None),
        "baseline_order": [v["decision_id"] for v in baseline],
        "n_valued": len(valuations),
        "n_eligible": len(ranked),
    }


# --------------------------------------------------------------------------- #
# CLI: value every candidate over the fold sessions (holdout untouched)
# --------------------------------------------------------------------------- #

def _py(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    return obj


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _scan_key(cand: Candidate, bucket_min: float) -> Tuple[str, str, int]:
    bucket = int(cand.as_of.timestamp() // (bucket_min * 60.0))
    return (cand.session, cand.underlying, bucket)


def run_valuation(dataset_path: str, sessions: Sequence[str],
                  config: ValuationConfig, seed: int) -> Dict[str, Any]:
    cands = load_candidates(dataset_path, sessions=sessions)
    obs_by_u = load_underlying_obs(dataset_path)

    scans: Dict[Tuple[str, str, int], List[Candidate]] = {}
    for c in cands:
        scans.setdefault(_scan_key(c, config.scan_bucket_min), []).append(c)

    fill_model = FillModel()
    cost_cfg = CostConfig()
    scan_rows: List[Dict[str, Any]] = []
    abstain_hist: Dict[str, int] = {}
    pooled_ev: List[float] = []
    pooled_pop: List[float] = []
    stress_net: Dict[str, List[float]] = {
        f"{f:.1f}": [] for f in config.friction_multipliers}
    stress_ev: Dict[str, List[float]] = {
        f"{f:.1f}": [] for f in config.friction_multipliers}
    agree = both = 0
    t_total = t_scen = 0.0
    n_candidates = 0

    for key in sorted(scans):
        session, underlying, bucket = key
        group = sorted(scans[key], key=lambda c: c.decision_id)
        n_candidates += len(group)
        as_of = max(c.as_of for c in group)
        s0 = next((c.features.get("underlying_price") for c in group
                   if c.features.get("underlying_price")), None)
        t0 = time.perf_counter()
        sc_seed = scenario_seed(seed, underlying, str(bucket))
        scen, why = generate_scenarios(
            obs_by_u.get(underlying, []), as_of, s0 or 0.0, config, sc_seed)
        t_scen += time.perf_counter() - t0

        vals: List[Dict[str, Any]] = []
        stressed: Dict[str, List[Dict[str, Any]]] = {
            k: [] for k in stress_net}
        for c in group:
            if scen is None:
                vals.append(_abstain(c, why))
                continue
            priced, abst = reprice_candidate(c, scen, config)
            if abst is not None:
                vals.append(abst)
                continue
            base = None
            for f in config.friction_multipliers:
                v = value_candidate(c, priced, config, friction=f,
                                    cost_config=cost_cfg,
                                    fill_model=fill_model)
                stressed[f"{f:.1f}"].append(v)
                if f == 1.0:
                    base = v
            vals.append(base if base is not None else _abstain(
                c, "no_base_friction"))

        for v in vals:
            if v["status"] == "abstain":
                abstain_hist[v["reason"]] = \
                    abstain_hist.get(v["reason"], 0) + 1
            else:
                pooled_ev.append(v["ev_opportunity"])
                pooled_pop.append(v["per_fill"]["pop"])
        for fk, rows in stressed.items():
            for v in rows:
                if v["status"] == "ok":
                    stress_net[fk].append(v["per_fill"]["net_mean"])
                    stress_ev[fk].append(v["ev_opportunity"])

        rank = rank_candidates(vals, config)
        if rank["selected_decision_id"] and rank["baseline_top_decision_id"]:
            both += 1
            if rank["selected_decision_id"] == \
                    rank["baseline_top_decision_id"]:
                agree += 1
        scan_rows.append({
            "session": session, "underlying": underlying,
            "bucket": bucket, "as_of": as_of.isoformat(),
            "n_candidates": len(group),
            "n_eligible": rank["n_eligible"],
            "decision": rank["decision"],
            "selected": {
                "decision_id": rank["selected_decision_id"],
                "symbol": rank["selected_symbol"],
                "ev_opportunity": rank["selected_ev"]},
            "baseline_top_decision_id": rank["baseline_top_decision_id"],
            "scenario": None if scen is None else scen.meta,
        })
        t_total += time.perf_counter() - t0

    def _stats(xs: List[float]) -> Optional[Dict[str, float]]:
        if not xs:
            return None
        a = np.asarray(xs)
        return {"n": int(a.size), "mean": float(a.mean()),
                "p05": float(np.quantile(a, 0.05)),
                "p50": float(np.quantile(a, 0.50)),
                "p95": float(np.quantile(a, 0.95))}

    stress = {}
    for fk in sorted(stress_net):
        stress[fk] = {"net_mean": _stats(stress_net[fk]),
                      "ev_opportunity": _stats(stress_ev[fk])}
    keys = sorted(stress_net)
    monotone = None
    if all(stress_net[k] for k in keys) and len(keys) >= 2:
        means = [float(np.mean(stress_net[k])) for k in keys]
        monotone = all(a >= b - 1e-9 for a, b in zip(means, means[1:]))

    n_scans = len(scan_rows)
    return {
        "sessions_valued": sorted({r["session"] for r in scan_rows}),
        "counts": {
            "candidates": n_candidates,
            "scans": n_scans,
            "valued_ok": len(pooled_ev),
            "abstained": int(sum(abstain_hist.values())),
            "trade_decisions": sum(
                1 for r in scan_rows if r["decision"] == "trade"),
            "abstain_decisions": sum(
                1 for r in scan_rows if r["decision"] == "abstain"),
        },
        "abstain_reasons": dict(sorted(abstain_hist.items())),
        "ev_summary": _stats(pooled_ev),
        "pop_summary": _stats(pooled_pop),
        "friction_stress": stress,
        "friction_stress_monotone_decreasing": monotone,
        "baseline_agreement": {
            "scans_with_both": both,
            "top1_agree": agree,
            "top1_agree_rate": (agree / both) if both else None,
        },
        "latency": {
            "total_s": t_total,
            "scenario_s": t_scen,
            "s_per_scan": (t_total / n_scans) if n_scans else None,
            "s_per_candidate": (t_total / n_candidates)
            if n_candidates else None,
        },
        "scans": scan_rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from oracle.lab.purged_validation import (
        HORIZON_EMBARGO_SESSIONS, dataset_meta, load_samples, make_folds)

    p = argparse.ArgumentParser(
        description="Oracle 4.0 Phase 5 scenario valuation + ranking "
                    "(research-only; fold sessions only, holdout untouched)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-paths", type=int, default=512)
    p.add_argument("--step-minutes", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=78)
    p.add_argument("--sessions", nargs="*", default=None,
                   help="explicit sessions (must not include holdout)")
    args = p.parse_args(argv)

    config = ValuationConfig(n_paths=args.n_paths,
                             step_minutes=args.step_minutes,
                             max_steps=args.max_steps)

    # derive fold vs holdout sessions exactly like the training CLI
    samples = load_samples(args.dataset, "60m", "direction")
    fold_cfg = {"test_block_sessions": 1, "min_train_sessions": 2,
                "embargo_sessions": HORIZON_EMBARGO_SESSIONS.get("60m", 1),
                "holdout_sessions": 2}
    _folds, fold_sessions, holdout = make_folds(samples, **fold_cfg)
    sessions = args.sessions or list(fold_sessions)
    leak = sorted(set(sessions) & set(holdout))
    if leak:
        p.error(f"sessions {leak} are in the pre-registered holdout")

    report: Dict[str, Any] = {
        "phase": "oracle4_phase5_valuation",
        "dataset_path": args.dataset,
        "dataset_hash": dataset_meta(args.dataset).get("dataset_hash"),
        "git_sha": _git_sha(),
        "seed": args.seed,
        "config": asdict(config),
        "cost_config": asdict(CostConfig()),
        "sessions": {"folds": list(fold_sessions), "holdout": list(holdout),
                     "requested": sessions},
        "result": run_valuation(args.dataset, sessions, config, args.seed),
    }
    clean = _py(report)
    from oracle.lab.purged_validation import metrics_fingerprint
    clean["report_fingerprint"] = metrics_fingerprint(clean)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, sort_keys=True)
    c = clean["result"]["counts"]
    print(f"scans={c['scans']} candidates={c['candidates']} "
          f"ok={c['valued_ok']} abstained={c['abstained']} "
          f"trade={c['trade_decisions']} abstain={c['abstain_decisions']}")
    print(f"fingerprint={clean['report_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
