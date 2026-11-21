# Oracle: a Selectivity-First Options Trading System

*One page. How it works, and why it is expected to be profitable. (Status: paper trading.)*

---

## How it works

Oracle is an automated single-leg options trader (long calls/puts on liquid
underlyings, SPY/QQQ and screened movers) built around one idea: **most trades
should not be taken**. Every 15 minutes during market hours it runs a pipeline:

1. **Scan.** Pull intraday bars, options chains, and context (IV rank, breadth,
   news) from Alpaca for the watchlist.
2. **Agent ensemble.** Fourteen independent agents (trend, volatility, volume,
   liquidity, news, breadth, candlestick, options structure, relative strength,
   order flow, intraday momentum, level reclaim, options-flow positioning, and
   a learned RL preference) each cast a bull/bear vote with a confidence. Votes
   aggregate into a direction and a conviction score.
3. **Rule/RL gate.** A Q-learning policy, trained offline by replaying every
   historical decision and its outcome, maps the current state (regime, IV,
   conviction…) to CALL / PUT / SKIP with a veto right.
4. **Hard gates.** A candidate only becomes an order if it clears, in order:
   an **expected-value gate** (modeled post-cost EV must be positive), a
   **profitability gate** (strategy-book forward stats must support the setup),
   liquidity/spread limits, duplicate-contract and portfolio-Greek checks, and
   a daily trade cap. In the live record, ~92% of decisions are blocked:
   of 44,353 decisions, only 3,577 traded.
5. **Execute & exit.** Orders go to Alpaca (paper). A unified exit manager
   enforces profit targets, stops, and time-based exits — long options bleed
   theta, so no position is left to decay passively.
6. **Learn.** Every decision — including every SKIP — is written to an episode
   store. Skipped trades are later resolved **counterfactually** (what would
   that contract have returned?), so the system learns from the trades it
   refused as well as the ones it took. The RL gate retrains on this store; a
   heavier ML stack (logistic + MLP with purged folds, negative controls,
   scenario valuation, paired replay against a frozen baseline) runs
   research-side and can only be promoted into the live path by a **human**
   after out-of-sample forward evidence.

## Why it is expected to be profitable

The edge is not prediction accuracy; it is **asymmetry times selectivity**.

- **Convex payoffs.** Long options have capped loss (premium) and uncapped
  upside. In the live paper record the average winner returns **+24.4%** per
  trade versus **−33.8%** for the average loser — so the breakeven win rate is
  ~58%. The system's job is to only trade when the gates say the odds clear
  that bar, and to skip everything else. Rejecting 92% of candidates is the
  product working, not idling.
- **Costs modeled before entry.** The EV gate prices in spread, slippage, and
  theta over the intended hold, so marginal setups die before they cost money.
- **Ensemble + regime awareness.** No single signal drives entries; agreement
  across independent agents in a recognized regime is required, which filters
  the noise trades that destroy option buyers.
- **A learning loop with teeth.** Counterfactual SKIP resolution gives ~10×
  more labeled outcomes than trades alone, and promotion of any new model
  requires paired-replay outperformance plus live forward evidence — changes
  cannot silently make the system worse.

**Honesty clause.** This is a paper-trading system under evaluation. The
figures above are paper fills; real-world slippage will be worse, several
agents currently contribute weak signal, and no profitability is guaranteed.
The design claim is narrower: with capped downside, modeled costs, extreme
selectivity, and human-gated learning, the system is structured so that *if*
an edge exists in its signals, it compounds — and if it doesn't, the gates
keep it from trading much at all.
