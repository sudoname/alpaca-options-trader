# Profitability validation (research only)

Run these against one timestamped, future export. Nothing here places orders
or changes the live profit gate. Keep the original export immutable.

## 1. Exact broker outcome audit

```
python -m oracle.ml.broker_truth \
  --ledger EXPORT/research_ledger.db \
  --activities EXPORT/account_activities.csv \
  --history EXPORT/trading_history.json \
  --out REPORTS/broker_truth.json
```

The ledger's decision ID and entry/exit broker order IDs must identify the
order. Each fill must have a unique activity ID, direction, quantity, price,
and symbol. Order-linked fee rows must cover both orders. Only equal entry
and exit quantities and complete order-linked fees enter
`fill_backed_after_order_fees` and `sum_after_order_fees`. Missing fee links
produce `fill_backed_fee_unknown`; legacy history P&L can only produce
`mark_derived`; open/partial/ambiguous positions remain `unresolved` with
`unmatched_qty`. Unallocated account fees are reported separately. A missing
fee link cannot be assumed to mean a zero fee.

The June account export in this repository predates the current research
ledger. Its fee rows may not identify orders; it cannot establish exact
per-decision after-fee P&L merely by symbol and time.

## 2. Corrected GBM versus frozen baseline

```
python -m oracle.ml.profitability_research \
  --dataset EXPORT/dataset_bt1121.sqlite \
  --execution-calibration EXPORT/execution_calibration.json \
  --valuation EXPORT/valuation.json \
  --out-dir REPORTS/gbm_corrected
```

The valuation selection is optional. The executable requires measured entry
friction and stop overshoot fields, the pinned LightGBM version, parity,
accounting of every baseline opportunity, and the ordinary promotion gate.
Missing predictions are abstentions; positive paired improvement and positive
absolute after-cost P&L are separate fields. No prior 4.0.2 artifact is
substituted. `INSUFFICIENT_DATA` or `DEPENDENCY_UNAVAILABLE` is not a result.
This replay uses previously traded outcomes, including mark-derived paper
outcomes, and cannot infer a trade's unobserved counterfactual return.

## 3. Entry and exit quote study

```
python -m oracle.ml.paired_execution \
  --candidates EXPORT/candidates.jsonl \
  --quotes-dir EXPORT/option_quotes \
  --out REPORTS/paired_execution.json
```

Each candidate row requires `decision_id`, `decision_time` (timezone aware),
`symbol`, positive integer `qty`, `gross_expected_value` (total dollars),
and `levels` with `stop_loss_percent`/`take_profit_percent` (plus optional
`trailing_stop_distance`). Optional `limit_expiry_s` defaults to 30 seconds.
Provide one `<symbol>.jsonl` quote series with `ts`, `bid`, `ask`, sizes,
`quote_age_s`, and provenance. Preserve every future candidate, including
those with no limit fill. The market order uses the first quote after modeled
arrival; the limit uses the EV cap on the decision quote and may be counted
as missed only with sufficient full-TTL quote coverage. Both entries use the
same exit rules, sell at observed bid minus slippage, and charge entry/exit
fees once. Stop overshoot is recorded. Open/partial exits have no realized
P&L. The cap touch at the ask with displayed depth is **a modeled scenario**:
without order queue or actual order IDs it is not proof of a limit fill.

## 4. IV/RV credit spread paper gate

```
IVRV_FEE_PER_CONTRACT_SIDE=0.02 python iv_rv_experiment.py --status
```

Use an observed or conservative positive fee per leg per side. Without a
fee setting, complete leg data, and at least 40 settled trades, the gate
cannot pass. Intrinsic expiration settlement does not resolve assignment,
early exit, or broker fill risk. See `IVRV_EXPERIMENT.md` for the rule.
