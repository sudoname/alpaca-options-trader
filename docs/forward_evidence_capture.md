# Forward evidence collection (paper/research, read-only capture)

These are separate processes from the trading scheduler. Quote capture makes
market-data GET requests; activity capture makes account-activity GET requests.
Neither submits an order. The IV/RV timer invokes the **existing paper-only**
experiment and writes simulated positions/trades; its feature flag defaults off.
Nothing is enabled merely by merging these files.

## Prerequisites

- Use the same deployed revision, `.env`, and broker account as the scheduler.
- Set `ENABLE_DECISION_LEDGER=1` to record immutable accepted/rejected
  decisions and exact entry/exit broker order IDs. The collector cannot create
  decision identities retroactively. Set `DECISION_LEDGER_PATH=research_ledger.db`
  unless an absolute path is used consistently by all services.
- Alpaca data credentials (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) must have
  indicative option quotes. The activity archive requires the same account's
  trading credentials. Specify `--account paper` or `--account live` explicitly;
  use a separate archive for each account. Protect the runtime directory.
- For the IV/RV paper protocol only, set `ENABLE_IVRV_CREDIT_PAPER=1` and a
  measured positive `IVRV_FEE_PER_CONTRACT_SIDE`. This is a simulated spread
  experiment and never places broker orders. Leave the flag off otherwise.

## One-shot checks

```
python quote_capture.py --once --observations --market-hours-only \
  --db research_ledger.db --active-trades active_trades.json \
  --paper-positions iv_rv_paper_positions.json \
  --out-dir evidence_capture/quote_ticks

python broker_activity_capture.py --account paper --once \
  --db evidence_capture/broker_activities.sqlite \
  --csv evidence_capture/account_activities.csv

python -m oracle.ml.export_forward_candidates \
  --ledger research_ledger.db \
  --out evidence_capture/candidates.jsonl
```

The quote poller watches open active positions, open IV/RV spread legs, and
recent candidate/shadow/group contracts. `--observations` records a row at
**each poll time** even when Alpaca's latest quote timestamp is unchanged;
`quote_ts_utc` and growing `quote_age_s` preserve the actual quote's age.
Without observation rows, a 30-second limit TTL cannot be classified as
fully observed. Use `--interval 5` for forward collection during market hours.
The feed is `indicative_latest`, **not guaranteed executable OPRA**. A quote
touch does not prove a limit order filled, even with displayed depth.

Activity capture fetches FILL, FEE, OPEXP and OPASN in ascending 100-item
pages. It revisits the last seven days on every run because fees can post the
next UTC day. `id` is unique in SQLite; re-polls cannot double-book activity.
Missing order IDs on fee rows stay unallocated. A multi-day outage longer
than seven days needs a one-time `--after <UTC timestamp>` backfill. Pagination
errors fail the invocation instead of claiming complete coverage.

## Optional systemd schedule (deploy root `/var/www/alps`)

The repo includes `deploy/alps-evidence-quotes.service` (5-second polling
inside weekday 09:30–16:05 ET), and
`deploy/alps-evidence-activities.service` plus `.timer` (every five minutes).
Copy those units to `/etc/systemd/system`, run `systemctl daemon-reload`,
then enable the quote service and activities timer. Edit the activities unit
to `--account live` **and use a different SQLite/CSV path** if evaluating a
live account. Install the IV/RV scan/settle timers only when intentionally
collecting the paper experiment; they run at 11:00 and 17:00 ET on weekdays.
Market holidays and early closes require operational review, and missed
paper scans must stay missing rather than be backfilled from later quotes.

Do not enable duplicate pollers for the same files. Inspect
`journalctl -u alps-evidence-quotes -u alps-evidence-activities` for poll
errors and gaps. Check candidate counts, observation cadence, quote age,
account scope and fee-without-order-ID counts before evaluating any result.

## After the evidence window

```
python -m oracle.ml.broker_truth \
  --ledger research_ledger.db \
  --activities evidence_capture/account_activities.csv \
  --history trading_history.json \
  --out evidence_capture/broker_truth.json

python -m oracle.ml.paired_execution \
  --candidates evidence_capture/candidates.jsonl \
  --quotes-dir evidence_capture/quote_ticks \
  --out evidence_capture/paired_execution.json

python iv_rv_experiment.py --status
```

Keep fill-backed after-fee, fill-backed fee-unknown, mark-derived, and
unresolved decisions separate. On quote replay, count a limit miss only if
the full TTL was observed densely; stale or missing intervals remain
unresolved. The IV/RV experiment still needs at least 40 settled paper trades
and its predefined t-stat gate; paper results are not live fills. The GBM
60-session/100-fill evidence floor needs a fresh, later dataset build; this
collector supplies input evidence but does not train or promote a model.

Runtime files live under `evidence_capture/` (gitignored). Never commit
account activities, position files, or quote ticks. Back up the ledger and
SQLite archive with normal data retention procedures.
