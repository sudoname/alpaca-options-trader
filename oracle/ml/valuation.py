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
     out-of-distribution flag.  4.0.1-C: when a verified model forecast
     is supplied, scenarios are *forecast-conditioned* — the model's
     realized-vol forecast replaces the EWMA scale target and the
     terminal return distribution is remapped (rank-preserving) onto the
     forecast quantiles at the forecast horizon; achieved-vs-target
     quantiles and interval coverage are recorded in the meta.  Without
     a usable forecast the explicitly labeled EWMA/bootstrap baseline
     runs byte-identically — no hard-coded directional drift, ever.
  2. **Repricing** — each contract is repriced along every path with the
     European BSM formula at the entry implied vol solved from the
     observed mid (sticky-strike constant IV).  This is a *physical*
     scenario of the underlying priced with a *risk-neutral* formula; the
     model-implied ITM probability is never reported as empirical PoP.
     The CRR American premium over BSM at entry is recorded per candidate
     as the approximation error (``american_gap_pct``).
  3. **Exit policy** — the production ``exit_manager`` semantics are
     simulated on the path marks: stop loss takes precedence over take
     profit, then time exit at the simulation horizon.  Stop/target/qty
     come from the immutable candidate snapshot when present; falling
     back to config defaults is explicitly flagged ``default_assumed``
     (never silent).  4.0.1-C: physically settled equity/ETF contracts
     (SPY included) are never "cash settled at intrinsic" — paths that
     reach expiry are liquidated at the last pre-expiry step with full
     exit frictions and fees (documented research convention
     ``pre_expiry_liquidation_research``; the given-up intrinsic delta
     is recorded as a diagnostic).  Triggers use the mark vs the entry
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
     partial-fill quantity scaling.  4.0.1-C ``ev_basis`` taxonomy:
     *confirmed non-fills* (missing/unusable quote, zero fill prob) are
     exactly 0.0; *unavailable* EV (insufficient history, stale quote,
     unresolved IV, unparseable symbol, ...) is ``None`` — unknown is
     never conflated with zero.  Valued rows are ``estimated``.
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
from datetime import datetime, timedelta, timezone
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
    max_gap_steps: float = 3.0         # history gap tolerance (x step)
    friction_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0)
    crr_steps: int = 200
    ewma_lambda: float = 0.94
    vol_scale_bounds: Tuple[float, float] = (0.5, 2.0)
    max_spread_pct: float = 0.40       # liquidity gate
    max_quote_age_s: float = 120.0     # stale-quote gate (snapshot age)
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
                        ) -> Dict[str, List[Tuple[datetime, datetime, float]]]:
    """Load ``(event_at, available_at, price)`` per underlying.

    4.0.1-A fix: ``available_at`` is loaded and used for point-in-time
    filtering (the 4.0 loader selected only ``event_at``, so a late-arriving
    observation could leak into history built "as of" an earlier time).
    Missing ``available_at`` falls back to ``event_at``.
    """
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    try:
        out: Dict[str, List[Tuple[datetime, datetime, float]]] = {}
        cur = conn.execute(
            "SELECT underlying, event_at, available_at, price "
            "FROM underlying_obs ORDER BY underlying, event_at")
        for u, event_at, available_at, price in cur:
            t = _parse_utc(event_at)
            if t is None or price is None or price <= 0:
                continue
            a = _parse_utc(available_at) or t
            out.setdefault(u, []).append((t, a, float(price)))
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


def _normalize_obs(
    obs: Sequence[Tuple],
) -> List[Tuple[datetime, datetime, float]]:
    """Normalize observations to ``(event_at, available_at, price)``.

    Accepts legacy ``(event_at, price)`` 2-tuples (availability assumed at
    event time, the pre-4.0.1 convention) and point-in-time 3-tuples.
    Invalid prices are dropped; output is sorted by event time.
    """
    out: List[Tuple[datetime, datetime, float]] = []
    for o in obs:
        if len(o) == 2:
            t, p = o
            a = t
        else:
            t, a, p = o
        if t is None or p is None or p <= 0:
            continue
        out.append((t, a or t, float(p)))
    out.sort(key=lambda x: x[0])
    return out


def _step_returns(obs: Sequence[Tuple], as_of: datetime,
                  step_minutes: float,
                  max_gap_steps: float = 3.0,
                  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Past-only step log returns from *actual* observation timestamps.

    4.0.1-A rewrite.  The 4.0 version resampled whatever observations
    existed onto a synthetic uniform grid with LOCF, which (a) fabricated
    flat returns when observations were sparse (vol deflation) and
    (b) silently treated daily closes as intraday steps (vol inflation
    downstream).  This version:

      * filters point-in-time on ``available_at <= as_of`` (never event
        time alone), so late-arriving observations are excluded;
      * emits a return only between two real observations whose actual
        spacing is in ``[step, max_gap_steps * step]`` — a gap wider than
        tolerance breaks the chain (counted, never bridged with invented
        values) and daily-on-intraday data yields *no* returns rather
        than rescaled ones;
      * never spans session boundaries (America/New_York sessions), so
        overnight gaps are structurally excluded.

    Returns ``(returns, diagnostics)``; diagnostics make abstention
    auditable (n_obs_pit, n_sessions, n_returns, n_gap_bridged,
    n_chain_breaks, median_spacing_s).
    """
    from oracle.ml.bars import session_date_of

    norm = _normalize_obs(obs)
    by_session: Dict[str, List[Tuple[datetime, float]]] = {}
    n_pit = 0
    for t, a, p in norm:
        if a > as_of or t > as_of:
            continue
        n_pit += 1
        by_session.setdefault(session_date_of(t), []).append((t, p))

    step_s = step_minutes * 60.0
    tol_s = max_gap_steps * step_s
    rets: List[float] = []
    spacings: List[float] = []
    n_bridged = 0
    n_breaks = 0
    for day in sorted(by_session):
        series = by_session[day]
        anchor_t, anchor_p = series[0]
        for t, p in series[1:]:
            dt_s = (t - anchor_t).total_seconds()
            if dt_s < step_s:
                continue                      # not yet a full step ahead
            if dt_s > tol_s:
                n_breaks += 1                 # gap too wide: re-anchor
                anchor_t, anchor_p = t, p
                continue
            if dt_s > 1.5 * step_s:
                n_bridged += 1                # tolerated, counted gap
            rets.append(math.log(p / anchor_p))
            spacings.append(dt_s)
            anchor_t, anchor_p = t, p

    diag = {
        "n_obs_pit": n_pit,
        "n_sessions": len(by_session),
        "n_returns": len(rets),
        "n_gap_bridged": n_bridged,
        "n_chain_breaks": n_breaks,
        "median_spacing_s": (float(np.median(spacings))
                             if spacings else None),
        "step_minutes": float(step_minutes),
        "max_gap_steps": float(max_gap_steps),
    }
    return np.asarray(rets, dtype=np.float64), diag


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


_HORIZON_MINUTES = {"60m": 60.0, "1s": 390.0}


def _horizon_minutes(horizon: Any) -> Optional[float]:
    """Trading minutes for a known forecast horizon label, else None."""
    if not horizon:
        return None
    return _HORIZON_MINUTES.get(str(horizon))


def _usable_forecast(forecast: Optional[Dict[str, Any]],
                     step_minutes: float, n_steps: int
                     ) -> Tuple[Optional[Dict[str, Any]],
                                Optional[Dict[str, Any]]]:
    """(usable forecast dict, fallback note).  Exactly one is non-None
    unless no forecast was supplied at all (both None)."""
    if not forecast:
        return None, None
    if not forecast.get("available"):
        return None, {"fallback_reason": "forecast_unavailable",
                      "detail": forecast.get("reason")}
    horizon_min = _horizon_minutes(forecast.get("horizon"))
    if horizon_min is None:
        return None, {"fallback_reason": "unknown_horizon",
                      "detail": forecast.get("horizon")}
    k_star = int(round(horizon_min / step_minutes))
    truncated = k_star > n_steps
    k_star = max(1, min(k_star, n_steps))
    out = dict(forecast)
    out["_horizon_minutes"] = horizon_min
    out["_k_star"] = k_star
    out["_horizon_truncated"] = truncated
    return out, None


def generate_scenarios(obs: Sequence[Tuple],
                       as_of: datetime, s0: float,
                       config: ValuationConfig,
                       seed: int,
                       forecast: Optional[Dict[str, Any]] = None
                       ) -> Tuple[Optional[ScenarioSet], str]:
    """(scenarios, reason).  reason != 'ok' means no scenarios.

    ``obs`` accepts legacy ``(event_at, price)`` tuples or point-in-time
    ``(event_at, available_at, price)`` tuples (see ``_normalize_obs``).

    4.0.1-C: ``forecast`` is an optional *plain dict* (``Forecast.to_dict``
    shape from :mod:`oracle.ml.forecast`; torch never required here).
    When usable, the realized-vol forecast replaces the EWMA scale target
    and the terminal log-return distribution at the forecast horizon is
    remapped — rank-preserving, by empirical quantile deltas ramped over
    the horizon — onto the forecast quantiles.  Dependence between return
    and volatility inside sampled blocks is preserved (the remap only
    shifts by rank; it never resamples marginals independently).  The rng
    is consumed identically with or without a forecast, so common random
    numbers hold across contracts *and* across forecast variants; a
    missing/unavailable forecast reproduces the baseline byte-for-byte.
    """
    if s0 is None or s0 <= 0:
        return None, "missing_underlying_price"
    rets, hist_diag = _step_returns(obs, as_of, config.step_minutes,
                                    max_gap_steps=config.max_gap_steps)
    if rets.size < config.min_history_returns:
        return None, "insufficient_history"

    n_steps = config.max_steps
    fc, fc_fallback = _usable_forecast(forecast, config.step_minutes,
                                       n_steps)

    hist_vol = float(np.std(rets))
    ewma = _ewma_vol(rets, config.ewma_lambda)
    scale, clamped = 1.0, False
    rv_source = "ewma" if ewma is not None else "unavailable_scale_1"
    rv_fc = None
    if fc is not None:
        try:
            rv_fc = float(fc.get("rv_forecast"))
        except (TypeError, ValueError):
            rv_fc = None
        if rv_fc is not None and not (rv_fc > 0 and math.isfinite(rv_fc)):
            rv_fc = None
    lo, hi = config.vol_scale_bounds
    if rv_fc is not None and hist_vol > 0:
        # model rv is over the forecast horizon; convert to per-step vol
        target_step_vol = rv_fc / math.sqrt(fc["_k_star"])
        raw_scale = target_step_vol / hist_vol
        scale = min(max(raw_scale, lo), hi)
        clamped = (scale != raw_scale)
        rv_source = "model_forecast"
    elif ewma is not None and hist_vol > 0:
        raw_scale = ewma / hist_vol
        scale = min(max(raw_scale, lo), hi)
        clamped = (scale != raw_scale)

    n_paths = config.clamped_paths()
    rng = np.random.default_rng(seed)
    blk = max(1, int(config.block_len))
    n_blocks = (n_steps + blk - 1) // blk
    starts = rng.integers(0, rets.size, size=(n_paths, n_blocks))
    offsets = np.arange(blk)
    idx = (starts[:, :, None] + offsets[None, None, :]) % rets.size
    sampled = rets[idx].reshape(n_paths, n_blocks * blk)[:, :n_steps]
    sampled = sampled * scale

    log_paths = np.cumsum(sampled, axis=1)

    fc_meta: Optional[Dict[str, Any]] = None
    mode = "baseline_bootstrap_ewma"
    if fc is not None:
        k_star = fc["_k_star"]
        notes: List[str] = []
        if fc["_horizon_truncated"]:
            notes.append("horizon_truncated_to_sim")
        # ---- quantile remap (rank-preserving, ramped to the horizon) ----
        pairs: List[Tuple[float, float]] = []
        for qk, qv in (fc.get("quantiles") or {}).items():
            try:
                qf, vf = float(qk), float(qv)
            except (TypeError, ValueError):
                continue
            if 0.0 < qf < 1.0 and math.isfinite(vf):
                pairs.append((qf, vf))
        pairs.sort()
        qs = np.asarray([p[0] for p in pairs])
        vs = np.asarray([p[1] for p in pairs])
        remapped = False
        if pairs and np.any(np.diff(vs) < 0):
            notes.append("nonmonotone_quantiles_remap_skipped")
        elif len(pairs) >= 2:
            R = log_paths[:, k_star - 1]
            emp = np.quantile(R, qs)
            deltas = vs - emp
            order = np.argsort(np.argsort(R, kind="stable"), kind="stable")
            u = (order + 0.5) / n_paths
            delta_i = np.interp(u, qs, deltas)
            ramp = np.minimum(np.arange(1, n_steps + 1), k_star) / k_star
            log_paths = log_paths + delta_i[:, None] * ramp[None, :]
            remapped = True
        else:
            notes.append("insufficient_quantiles_remap_skipped")
        R_new = log_paths[:, k_star - 1]
        achieved = ({f"{q:g}": float(a)
                     for q, a in zip(qs, np.quantile(R_new, qs))}
                    if len(pairs) else {})
        cov = None
        tq = {f"{q:g}": float(v) for q, v in pairs}
        if "0.05" in tq and "0.95" in tq:
            cov = float(np.mean((R_new >= tq["0.05"])
                                & (R_new <= tq["0.95"])))
        mode = "forecast_conditioned"
        fc_meta = {
            "model_id": fc.get("model_id"),
            "input_digest": fc.get("input_digest"),
            "horizon": fc.get("horizon"),
            "horizon_minutes": fc["_horizon_minutes"],
            "k_star": k_star,
            "horizon_truncated": fc["_horizon_truncated"],
            "p_up": fc.get("p_up"),
            "quantile_remap_applied": remapped,
            "target_quantiles": tq,
            "achieved_quantiles": achieved,
            "interval_coverage_05_95": cov,
            "notes": notes,
        }
    elif fc_fallback is not None:
        fc_meta = fc_fallback

    prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    prices[:, 0] = s0
    prices[:, 1:] = s0 * np.exp(log_paths)
    meta = {
        "mode": mode,
        "n_source_returns": int(rets.size),
        "hist_step_vol": hist_vol,
        "ewma_step_vol": ewma,
        "vol_scale": scale,
        "vol_scale_clamped_ood": clamped,
        "rv_forecast": (rv_source if rv_source != "ewma"
                        else "ewma"),
        "block_len": blk,
        "seed": int(seed),
        "history": hist_diag,
        "forecast": fc_meta,
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
    """Explicit abstention row.

    ``ev_opportunity`` follows EV = P(fill) * E(net | filled):
    - confirmed non-fill / confirmed unusable market -> EV is exactly 0.0
      (``ev_basis="confirmed_nonfill"``);
    - everything else (missing inputs, stale data, model refusal) means the
      EV simply could not be estimated -> ``None``
      (``ev_basis="unavailable"``), never a silent zero.
    """
    confirmed = reason.split(":", 1)[0] in (
        "missing_quote", "unusable_quote", "no_fill_expected")
    out = {
        "decision_id": cand.decision_id,
        "symbol": cand.symbol,
        "underlying": cand.underlying,
        "session": cand.session,
        "status": "abstain",
        "reason": reason,
        "ev_opportunity": 0.0 if confirmed else None,
        "ev_basis": "confirmed_nonfill" if confirmed else "unavailable",
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
    age = cand.features.get("quote_age_s")
    if age is not None and float(age) > config.max_quote_age_s:
        return None, _abstain(cand, "stale_quote",
                              quote_age_s=float(age),
                              max_quote_age_s=config.max_quote_age_s)
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
                   stop_loss_pct: float, take_profit_pct: float,
                   expires_in_sim: bool
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Vectorized exit policy on the mark path.

    Precedence per production exit_manager: stop > target, else time exit at
    the horizon.  When the contract expires inside the simulation we enforce
    the documented research convention of PRE-EXPIRY LIQUIDATION: paths that
    have not triggered stop/target are sold at the last mark BEFORE the
    expiry column (physically settled equity/ETF options are never held to
    settlement here, so the intrinsic final mark is diagnostic only).
    Returns (exit_step, exit_mark, reason_counts).
    """
    n_paths, n_cols = marks.shape
    n_steps = n_cols - 1
    steps = marks[:, 1:]                           # decisions after entry
    stop_lvl = entry_fill * (1.0 - stop_loss_pct / 100.0)
    tgt_lvl = entry_fill * (1.0 + take_profit_pct / 100.0)
    if expires_in_sim:
        scan = steps[:, :-1]           # triggers only BEFORE the expiry mark
        liq_col = max(1, n_steps - 1)  # degenerate 1-step: expiry mark col 1
    else:
        scan = steps
        liq_col = n_steps
    if scan.shape[1]:
        hit = (scan <= stop_lvl) | (scan >= tgt_lvl)
        any_hit = hit.any(axis=1)
        first = np.where(any_hit, hit.argmax(axis=1), liq_col - 1)
    else:
        any_hit = np.zeros(n_paths, dtype=bool)
        first = np.full(n_paths, liq_col - 1)
    exit_step = first + 1                          # 1-based mark column
    rows = np.arange(n_paths)
    exit_mark = marks[rows, exit_step]

    is_stop = any_hit & (exit_mark <= stop_lvl)
    is_tgt = any_hit & ~is_stop
    reasons = {
        "stop_loss": int(is_stop.sum()),
        "take_profit": int(is_tgt.sum()),
        ("pre_expiry_liquidation" if expires_in_sim else "time_exit"):
            int((~any_hit).sum()),
    }
    return exit_step, exit_mark, reasons


def _resolve_policy(cand: Candidate, config: ValuationConfig
                    ) -> Tuple[float, float, int, str, bool]:
    """Resolve exit policy + quantity from the immutable candidate snapshot.

    Returns (stop_loss_pct, take_profit_pct, qty, source, defaulted).
    Snapshot values win; config defaults are used only when the snapshot
    carries nothing, and that defaulting is flagged so promotion-grade
    comparisons can exclude silently-defaulted policies.
    """
    def _pos(v: Any) -> Optional[float]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f > 0.0 else None

    defaulted = False
    stop = _pos(cand.features.get("stop_loss_pct"))
    if stop is None:
        stop, defaulted = float(config.stop_loss_pct), True
    tp = _pos(cand.features.get("take_profit_pct"))
    if tp is None:
        tp, defaulted = float(config.take_profit_pct), True
    qty_raw = cand.features.get("qty", cand.features.get("quantity"))
    qty: Optional[int]
    try:
        qty = int(qty_raw) if qty_raw is not None else None
        if qty is not None and qty <= 0:
            qty = None
    except (TypeError, ValueError):
        qty = None
    if qty is None:
        qty, defaulted = max(1, int(config.qty)), True
    source = "default_assumed" if defaulted else "snapshot"
    return stop, tp, qty, source, defaulted


def value_candidate(cand: Candidate, priced: Dict[str, Any],
                    config: ValuationConfig,
                    friction: float = 1.0,
                    cost_config: Optional[CostConfig] = None,
                    fill_model: Optional[FillModel] = None,
                    forecast: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """Cost + exit-policy valuation of one repriced candidate.

    Net P&L uses executable entry/exit prices (spread + slippage embedded)
    and subtracts only fees, which are not embedded — costs are charged
    exactly once, with the 100x contract multiplier applied exactly once.
    Exit policy and quantity come from the candidate snapshot when present
    (config defaults are flagged).  Expiry-in-sim paths are liquidated at
    the last pre-expiry mark with full exit frictions (never free cash
    settlement).  ``forecast`` (optional, already validated upstream) only
    feeds the hazard diagnostic — it is never double-counted into EV.
    """
    cc = cost_config or CostConfig()
    bid, ask, mid = priced["bid"], priced["ask"], priced["mid"]
    marks: np.ndarray = priced["marks"]
    stop_pct, tp_pct, qty, policy_source, policy_defaulted = \
        _resolve_policy(cand, config)

    spread = max(ask - bid, cc.min_spread_floor)
    half = spread / 2.0 * friction
    slip = cc.slippage_per_contract * friction
    fee_leg = (cc.occ_fee_per_contract
               + cc.commission_per_contract) * friction

    entry_fill = mid + half + slip                 # == ask + slip at 1.0x
    if entry_fill <= 0:
        return _abstain(cand, "nonpositive_entry_fill")

    exit_step, exit_mark, reasons = _simulate_exit(
        marks, entry_fill, stop_pct, tp_pct, priced["expires_in_sim"])
    # Every exit is a traded liquidation (pre-expiry liquidation convention):
    # full exit frictions and the exit fee leg apply on every path.
    exit_fill = np.maximum(exit_mark - (half + slip), 0.0)

    mult = CONTRACT_MULTIPLIER
    gross = (exit_mark - mid) * mult * qty         # frictionless reference
    fees_total = 2.0 * fee_leg * qty               # entry + exit leg, once
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

    # horizon bucketing: 0DTE vs swing (promotion evidence is kept separate)
    dte = cand.features.get("dte_days")
    horizon_class = "0dte" if (dte is not None and float(dte) < 1.0) \
        else "swing"

    # settlement diagnostic: how far the enforced pre-expiry liquidation
    # mark sits from the intrinsic settlement column (research-only note;
    # physically settled contracts are never held through settlement here)
    settlement: Optional[Dict[str, Any]] = None
    if priced["expires_in_sim"]:
        n_steps = int(priced["n_steps"])
        n_liq = int(reasons.get("pre_expiry_liquidation", 0))
        resid: Optional[float] = None
        if n_steps >= 2:
            liq_paths = exit_step == (n_steps - 1)
            if np.any(liq_paths):
                resid = float(np.mean(
                    marks[liq_paths, n_steps] - marks[liq_paths, n_steps - 1]))
        settlement = {
            "convention": "pre_expiry_liquidation_research",
            "n_liquidated": n_liq,
            "mean_intrinsic_minus_liq_mark": resid,
            "note": ("physically settled contract exits at the last "
                     "pre-expiry mark with full frictions; intrinsic final "
                     "mark is diagnostic only"),
        }

    # hazard diagnostic: forecast CIF vs simulated exit mix — reported for
    # calibration inspection only, never folded into EV (no double counting)
    hazard_diag: Optional[Dict[str, Any]] = None
    if forecast and forecast.get("available") and forecast.get("cif"):
        try:
            cif_term = {str(cause): float(bins[-1])
                        for cause, bins in forecast["cif"].items()
                        if isinstance(bins, (list, tuple)) and bins}
        except (TypeError, ValueError):
            cif_term = {}
        if cif_term:
            hazard_diag = {
                "sim_exit_reasons": reasons,
                "forecast_cif_terminal": cif_term,
                "note": "diagnostic only; not used in EV "
                        "(no double counting)",
            }

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
            "spread_exit_mean": half * mult * qty,
            "slippage_round_trip_mean": 2.0 * slip * mult * qty,
            "fees_mean": float(fees_total),
            "exit_spread_source": "entry_spread_proxy",
        },
        "fill": {"p_fill": p_fill, "fill_frac": fill_frac,
                 "status": est.status},
        "capital_required": capital,
        "ev_opportunity": ev_opportunity,
        "ev_basis": "estimated",
        "ev_per_dollar": ev_opportunity / capital if capital > 0 else None,
        "n_paths": int(n),
        "policy": {
            "stop_loss_pct": stop_pct,
            "take_profit_pct": tp_pct,
            "qty": qty,
            "source": policy_source,
            "defaulted": policy_defaulted,
        },
        "horizon_class": horizon_class,
        "settlement": settlement,
        "hazard_diagnostic": hazard_diag,
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
                  config: ValuationConfig, seed: int,
                  bars_db: Optional[str] = None,
                  registry_path: Optional[str] = None,
                  model_id: Optional[str] = None) -> Dict[str, Any]:
    """Value every candidate over *sessions*.

    4.0.1-A: when ``bars_db`` is given, per-scan history comes from the
    read-only bars cache (actual timestamped completed bars, PIT via the
    completed-bar rule). Unavailability falls back to the dataset's
    ``underlying_obs`` with the structured reason recorded — never silently.

    4.0.1-C: when ``registry_path``/``model_id`` are given, each scan asks
    the verified ForecastProvider for one forecast (newest candidate's
    features) and conditions the shared scenario set on it.  Any refusal
    falls open to the explicitly labeled baseline — never a silent drift.
    """
    from oracle.ml.bars import BarStore, BarsUnavailable

    cands = load_candidates(dataset_path, sessions=sessions)
    obs_by_u = load_underlying_obs(dataset_path)

    bar_store: Optional[BarStore] = None
    bars_note: Optional[Dict[str, Any]] = None
    bars_fallbacks: Dict[str, Dict[str, Any]] = {}
    bars_source_counts: Dict[str, int] = {"bars": 0, "dataset_obs": 0}
    if bars_db:
        try:
            bar_store = BarStore(bars_db, readonly=True)
        except BarsUnavailable as e:
            bars_note = e.to_dict()

    provider = None
    forecast_note: Optional[Dict[str, Any]] = None
    if registry_path and model_id:
        try:
            from oracle.ml.forecast import get_provider
            provider = get_provider(registry_path, model_id)
        except Exception as e:  # ForecastUnavailable or import failure
            forecast_note = (e.to_dict() if hasattr(e, "to_dict")
                             else {"available": False,
                                   "reason": "PROVIDER_ERROR",
                                   "detail": str(e)})

    scans: Dict[Tuple[str, str, int], List[Candidate]] = {}
    for c in cands:
        scans.setdefault(_scan_key(c, config.scan_bucket_min), []).append(c)

    fill_model = FillModel()
    cost_cfg = CostConfig()
    scan_rows: List[Dict[str, Any]] = []
    abstain_hist: Dict[str, int] = {}
    scan_modes: Dict[str, int] = {}
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
        history_source = "dataset_obs"
        obs: Sequence[Tuple] = obs_by_u.get(underlying, [])
        if bar_store is not None:
            try:
                bars = bar_store.load_bars(
                    underlying, 60,
                    start=as_of - timedelta(days=7), end=as_of,
                    as_of=as_of, rth_only=True,
                    min_bars=config.min_history_returns)
                # (event_at, available_at, price): a bar is available at
                # its END (completed-bar rule), priced at its close.
                obs = [(b.ts_utc, b.end_utc, b.close) for b in bars]
                history_source = "bars"
            except BarsUnavailable as e:
                bars_fallbacks.setdefault(underlying, e.to_dict())
        bars_source_counts[history_source] += 1
        fc_dict: Optional[Dict[str, Any]] = None
        if provider is not None:
            ref = max(group, key=lambda c: (c.as_of, c.decision_id))
            try:
                fc_dict = provider.forecast(ref.features,
                                            as_of=as_of).to_dict()
            except Exception as e:  # refusal -> labeled baseline fallback
                fc_dict = (e.to_dict() if hasattr(e, "to_dict") else None)
        scen, why = generate_scenarios(
            obs, as_of, s0 or 0.0, config, sc_seed, forecast=fc_dict)
        if scen is not None:
            mode = scen.meta.get("mode", "baseline_bootstrap_ewma")
            scan_modes[mode] = scan_modes.get(mode, 0) + 1
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
                                    fill_model=fill_model,
                                    forecast=fc_dict)
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
            "history_source": history_source,
            "scenario": None if scen is None else scen.meta,
        })
        t_total += time.perf_counter() - t0

    if bar_store is not None:
        bar_store.close()

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
        "history": {
            "bars_db": bars_db,
            "source_counts": bars_source_counts,
            "store_note": bars_note,
            "fallbacks": dict(sorted(bars_fallbacks.items())),
        },
        "forecast": {
            "registry": registry_path,
            "model_id": model_id,
            "provider_note": forecast_note,
            "scan_modes": dict(sorted(scan_modes.items())),
        },
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
    p.add_argument("--bars-db", default=None,
                   help="read-only bars cache (oracle.ml.bars); when set, "
                        "scenario history uses actual timestamped completed "
                        "bars with fail-open fallback to dataset obs")
    p.add_argument("--registry", default=None,
                   help="model registry JSON (oracle.ml.registry); with "
                        "--model-id, scenarios are conditioned on verified "
                        "forecasts, falling open to the labeled baseline")
    p.add_argument("--model-id", default=None,
                   help="registered model id to forecast with")
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
        "bars_db": args.bars_db,
        "registry": args.registry,
        "model_id": args.model_id,
        "result": run_valuation(args.dataset, sessions, config, args.seed,
                                bars_db=args.bars_db,
                                registry_path=args.registry,
                                model_id=args.model_id),
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
