"""
Measure realized max-adverse-excursion (MAE) of WINNING trades.

For each trailing/roll winner we pull the option's own minute bars between entry
and exit and find the lowest price it traded. MAE% = (low - entry)/entry*100.
A winner with MAE worse than a candidate stop level would have been prematurely
stopped out under that stop -> this is the 'cannibalization' rate the backtest
could not measure from stored data.
"""
import sqlite3, random, sys
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv("/var/www/alps/.env")

from alpaca_client import AlpacaOptionsClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

WIN_EXITS = ("dynamic_trailing_stop", "roll_take_profit", "roll_failed_take_profit")

con = sqlite3.connect("episodes.db")
con.row_factory = sqlite3.Row
q = ("select symbol, fill_price, net_pnl_pct, net_pnl_dollars, created_at, closed_at, outcome "
     "from episodes where mode='live-paper' and outcome is not null "
     "and outcome in (%s) and net_pnl_dollars > 0" % ",".join("?" * len(WIN_EXITS)))
winners = [dict(r) for r in con.execute(q, WIN_EXITS)]
con.close()

print(f"total winning trades in pool: {len(winners)}")

# Sample: 40 largest winners (ran furthest -> model's worst-case) + 40 random.
winners.sort(key=lambda r: -(r["net_pnl_dollars"] or 0))
top = winners[:40]
rest = winners[40:]
random.seed(7)
rnd = random.sample(rest, min(40, len(rest)))
sample = {r["symbol"] + str(r["created_at"]): r for r in top + rnd}.values()
sample = list(sample)
print(f"sampling {len(sample)} winners (40 largest + 40 random)\n")

cli = AlpacaOptionsClient()
dc = cli.data_client


def parse_dt(s):
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


maes = []          # (mae_pct, net_pnl_pct, dollars, symbol)
fetched = 0
empty = 0
errors = 0

for r in sample:
    sym = r["symbol"]
    entry = r["fill_price"] or 0
    t0 = parse_dt(r["created_at"])
    t1 = parse_dt(r["closed_at"])
    if not (sym and entry and t0 and t1):
        errors += 1
        continue
    start = t0 - timedelta(minutes=2)
    end = t1 + timedelta(minutes=2)
    try:
        req = OptionBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                start=start, end=end)
        bs = dc.get_option_bars(req)
        bars = bs.data.get(sym, []) if hasattr(bs, "data") else []
    except Exception as e:
        errors += 1
        if errors <= 3:
            print("  fetch error", sym, repr(e)[:120])
        continue
    if not bars:
        empty += 1
        continue
    low = min(b.low for b in bars)
    mae = (low - entry) / entry * 100.0
    maes.append((mae, r["net_pnl_pct"] or 0, r["net_pnl_dollars"] or 0, sym))
    fetched += 1

print(f"coverage: fetched={fetched}  empty={empty}  errors={errors}\n")

if not maes:
    print("No path data returned -- option historical bars unavailable for these contracts.")
    sys.exit(0)

maes.sort()
print("Deepest 10 drawdowns among winners (MAE% | final% | $ | contract):")
for mae, fin, dol, sym in maes[:10]:
    print(f"  {mae:>7.1f}% | {fin:>+6.1f}% | {dol:>+7.0f} | {sym}")

n = len(maes)
print(f"\nMAE distribution over {n} sampled winners:")
for lvl in (25, 30, 35, 40, 50):
    k = sum(1 for m, *_ in maes if m <= -lvl)
    print(f"  dipped below -{lvl:>2}% before recovering: {k:>3}/{n}  = {100*k/n:>5.1f}%")

import statistics
print(f"\n  median MAE: {statistics.median(m for m,*_ in maes):+.1f}%")
print(f"  mean   MAE: {statistics.mean(m for m,*_ in maes):+.1f}%")
