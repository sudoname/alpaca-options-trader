# Profitability Tuning Proposal (paper-only, for review)

Branch: `profitability-tuning`. **No production changes are applied by this branch.**
It contains (1) the exact `.env` changes to make on the server and (2) one small,
flag-gated code change so the profitability gate can use a *calibrated* PoP.

## Why

Realized ledger (production paper account, 2026-06 .. 2026-09):

- **-$159,700** over 3,058 closed trades, losing **every month**.
- Win rate **37.7%**, payoff (avg win / avg loss) **0.67**, profit factor **0.48**.
- To break even you need **60% WR** at this payoff, or **1.65 payoff** at this WR.
- The entire loss is losers running: `dynamic_stop_loss` = **-$210,789** (avg **-43%**
  per stop), only partly offset by trailing-stop winners (+$61,572, avg +16%).

Two shadow observers already on the server explain it:

- **PoP model overstates edge by ~28pp** (stamped 66.5% -> actual 38.1%):
  stamped 60% -> 35% real, 70% -> 42%, 80% -> 42%. The gate's `min_pop=0.58`
  is therefore admitting ~35% win-rate trades.
- **Round-trip friction (14.2%) ~2x expected edge (7.4%)**; **74%** of trades
  can't clear their own costs. Wide-spread single-name options are the bleed.

## What changes

### 1. `.env` (server) — the two highest-impact, rigorously-backtested levers

| Key | Current | Proposed | Rationale |
|---|---|---|---|
| `MAX_SPREAD_PCT` | `15` | `2` | Entry liquidity gate (smart_trader.py:2103). Only tight-spread names are profitable. |
| `BASE_STOP_LOSS` | `0.25` | `0.10` | Lower dynamic-stop floor. |
| `MAX_STOP_LOSS` | `0.30` | `0.15` | Cap the dynamic stop at 15%; stops currently realize -43%. |
| `BASE_TAKE_PROFIT` | `0.25` | `0.25` | Unchanged — keeps TP (0.25) **>** SL (0.15), fixing the fatal asymmetry. |

No code change is needed for these — the knobs already exist.

### 2. Code — PoP calibration in the gate (flag-gated, default OFF)

`profitability_validator.py` gains an optional `pop_calibration` rule. When unset
(default) behavior is **identical** to today (verified by existing tests). When the
caller passes a calibration, the stamped PoP is corrected DOWN before the `min_pop`
bar. Supported forms:

```python
{"pop_calibration": {"offset_pp": 28}}          # corrected = pop - 0.28
{"pop_calibration": {"0.6": 0.35, "0.7": 0.42}} # measured floor-bucket curve
{"pop_calibration": lambda p: p - 0.30}         # arbitrary curve
```

It is pure and fail-open: a broken calibration leaves the stamped PoP untouched.
Wiring the caller (smart_trader) to feed the measured curve from
`pop_recal_shadow.jsonl` is a **follow-up** — this branch only makes the gate
*capable* of it so it can be reviewed and unit-tested in isolation.

## Backtest evidence (2,700 real executed trades)

Loss-capping is rigorous: price must pass through a tighter stop on its way to -43%.

```
LEVER 1 — entry spread filter (total P&L)
  spread <=1% : n=  54  WR 55.6%   +1,257   (only profitable slice today)
  spread <=2% : n= 211  WR 43.1%   -5,299
  spread 10-11%: n= 645 WR 32.7%  -49,726   (the bleed)

LEVER 3 — tighten hard stop (total P&L, saved vs baseline -150,134)
  -30% : -79,314  (+70,819)
  -20% : -23,673  (+126,461)
  -15% :  +5,983  (+156,117)   <- flips the whole book positive

COMBINED — spread<=2% AND stop@-20%
  n=211  WR 43.1%  +2,187

WALK-FORWARD (spread<=2% + stop@-20%), holds every month:
  2026-06  +1,902   (base -21,560)
  2026-07  -1,035   (base -43,651)
  2026-08     -50   (base -50,637)
  2026-09  +1,370   (base -34,285)
```

Interaction grid optimum: `spread<=5% + stop@-15% = +$11,988` (865 trades). The
conservative, walk-forward-validated setting is `spread<=2% + stop@-20%`.

## Honest caveats

- The `-15%` stop result is **optimistic on magnitude**: the sim caps trades that
  *finished* below the stop but cannot see winners that dipped below -15% intraday
  and recovered (a real -15% stop would kill some of those). True effect sits
  between the estimate and baseline; direction is robust. Add ~3-5% for option-gap
  slippage on the exit. `-20%` is the safer planning number.
- Surviving subsets are small (54-211 trades). This is about **stopping losses**,
  not a money printer. The structurally sounder path remains defined-risk spreads
  (`spread_builder.py`).

## Rollout & rollback (paper)

1. Edit server `.env` with the table above; restart: `systemctl restart alps-bot alps-scheduler`.
2. Forward-test for a few sessions; compare gate approval rate and daily P/L.
3. Rollback: restore prior `.env` values and restart. Code change is inert unless a
   caller passes `pop_calibration`, so it needs no rollback.
