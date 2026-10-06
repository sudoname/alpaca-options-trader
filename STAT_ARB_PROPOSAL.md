# Statistical Arbitrage Extension — Proposal (NO BUILD)

Status: **proposal only**. Per scope, the deep-learning model is NOT built now.
This document specifies what we would build, in what order, and the evidence
gates that must pass before any later stage is attempted.

Reference: Guijarro-Ordonez, Pelger, Zanotti, *Deep Learning Statistical
Arbitrage* (arXiv:2106.04028v2).

## 1. What the paper actually did (and what we would NOT be reproducing)

The paper:
- Universe: ~550 largest US stocks per day, CRSP daily data 1998–2016
  (survivorship-bias-free, includes delistings).
- Step 1 — residuals: remove factor structure with Fama-French 5,
  PCA (K≈5 principal components suffice), or IPCA conditional factors.
- Step 2 — signal: parametric baseline = Ornstein-Uhlenbeck fit + threshold
  trading on the residual; learned alternative = CNN feature extractor +
  Transformer over a 30-day lookback of cumulative residual returns.
- Step 3 — portfolio: self-financing long/short residual portfolios,
  daily rebalance, mean-variance style objective (maximize Sharpe directly).
- Results: Sharpe ≈ 4 gross; after 5 bps transaction cost + 1 bp/day
  short-holding cost, Sharpe fell to ≈ 0.94–1.24. Unconditional residual
  means are ≈ unprofitable — ALL the edge is in the signal extraction.
  Signals decay over ~1 week.

Anything we build here differs materially from that setup and must not be
called a reproduction:

| Paper | Here |
|---|---|
| CRSP 1998–2016, delisting-adjusted | Alpaca SIP daily bars (reliable ~2016+), survivorship-biased: our universe is "symbols Oracle trades today" |
| ~550 names/day cross-section | ~92 liquid names (universe list in `session_stats_universe.json`) |
| FF5 / IPCA factors (characteristics data) | PCA factors only (no characteristics vendor); FF5 possible via Ken French library for the market-level factors but with alignment caveats |
| 18 years of data | ~2 years usable at daily frequency (≈500 sessions) — far too little to train a CNN+Transformer without severe overfitting |

The small cross-section and short history are the binding constraints: the
paper's DL edge came from pooling thousands of residual series over decades.

## 2. Proposed staged plan with evidence gates

### Stage 0 — Data foundation (prerequisite, cheap)
- Extend `session_returns.fetch_daily_bars` usage to build and cache a daily
  total-return matrix for the 92-symbol universe (adjustment="all",
  calendar-aligned — the machinery already exists and is tested).
- Exact data requirements for anything beyond toy scale:
  - survivorship-bias-free membership history (e.g., CRSP, or point-in-time
    index constituents) — **not currently available to this project**;
  - ≥ 500 names × ≥ 10 years for any DL stage;
  - borrow-fee/short-availability data for realistic short costs.

### Stage 1 — Residual construction (simple, testable)
- Rolling PCA on the universe correlation matrix (K = 5, estimation window
  252 sessions, strictly trailing — no look-ahead).
- Residual r_i(t) = return_i(t) − beta_i' f(t), betas re-fit monthly on
  trailing data only.
- Validation: residuals must have ~zero mean, near-zero loading on the
  extracted factors out-of-window, and the identity return = beta'f + residual
  must hold to numerical tolerance (same test style as
  `tests/test_session_returns.py::TestIdentity`).

### Stage 2 — Parametric baselines ONLY (no DL)
- OU fit per residual (trailing window) + threshold entry/exit as in the
  paper's baseline; plus two dumb baselines: 5-day residual reversal and
  residual momentum.
- Evaluation with the already-built harness conventions
  (`session_backtest.py`): chronological 60/20/20 with 1-session embargo,
  thresholds fit on train/val only, 1 bp/side equity costs **plus** 1 bp/day
  short-holding cost on the short leg, turnover and exposure reported,
  annualization stated (252).
- **Gate G1**: a baseline must show positive net Sharpe with |t| ≥ 2 on the
  untouched test segment. If no baseline passes, stop — the paper itself
  shows unconditional residuals carry no edge, and with 92 names we have far
  less cross-sectional breadth than they needed.

### Stage 3 — Learned signal (CONDITIONAL — not now)
- Only if G1 passes AND the Stage-0 data requirements (breadth + history)
  are met. CNN+Transformer over L=30 residual lookbacks per the paper.
- **Gate G2**: the learned signal must beat the best Stage-2 baseline on a
  new, never-before-touched OOS period, net of all costs, with uncertainty
  reported. "Beats in-sample" or "beats on the val set" does not count.
- Without CRSP-grade data this stage is expected to be unjustifiable; the
  proposal exists so the decision is explicit rather than implicit.

### Options overlay (speculative, separate gate)
Oracle trades options, not equities. Translating residual signals into
option positions adds spread costs that are 10–100x equity costs (live
episodes show mean net option P&L of −15%/trade at current spreads). Any
options implementation would need its own paper-trading evidence gate; the
equity-level results above would be labeled UNTESTED for options, exactly as
`session_backtest.py` does today.

## 3. Costs and realism commitments
- Equity legs: ≥ 1 bp/side baseline, sensitivity at 5 bps/side (paper's
  number); short legs: +1 bp/day holding cost.
- Fills at official open/close auctions only; no mid-bar fills.
- Gap risk on overnight holds borne in full; no stop assumption overnight.
- All reported numbers net; gross shown only alongside net.

## 4. Why not build the DL model now (summary)
1. Data: no survivorship-bias-free membership, ~2 years × 92 names vs the
   paper's 18 years × ~550 — the model would memorize noise.
2. Evidence order: the paper's own ablation says residual construction is
   commodity; the signal is the edge — so the cheap parametric baselines
   (Stage 2) are the correct first falsification test.
3. Costs: post-cost Sharpe in the paper dropped ~75%; with our breadth the
   prior is that net edge ≈ 0 until proven otherwise.
4. Current empirical priority: the session analysis (this work) already
   found a concrete, significant pattern in Oracle's own live decisions
   (PUT signals held into the overnight session lose −23.7 bp/night,
   t ≈ −6; CALL signals lose rest-of-day, t ≈ −4). Acting on measured
   session effects via the existing flag-gated gate machinery is higher
   value than a new model class.
