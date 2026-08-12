"""
Final stop-retune estimator using MEASURED winner MAE.

1. Pull minute-bar MAE for a representative RANDOM sample of winners (no
   large-winner bias) -> real cannibalization rate + real $ swing per level.
2. Loser savings from the full population (truncate anything ending worse than
   the overshoot-adjusted trigger).
3. Net projected P/L = baseline + loser_savings - winner_cannibalization_cost,
   using the measured rate scaled to the full 329-winner pool.
"""
import sqlite3, random
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv("/var/www/alps/.env")
from alpaca_client import AlpacaOptionsClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

OVERSHOOT = 69.3 / 50.0
WIN_EXITS = ("dynamic_trailing_stop", "roll_take_profit", "roll_failed_take_profit")
LEVELS = (25.0, 30.0, 35.0, 40.0)

con = sqlite3.connect("episodes.db"); con.row_factory = sqlite3.Row
allrows = [dict(r) for r in con.execute(
    "select symbol, fill_price, qty, net_pnl_pct, net_pnl_dollars, created_at, closed_at, outcome "
    "from episodes where mode='live-paper' and outcome is not null")]
con.close()

def notional(r):
    fp, q = r.get("fill_price") or 0, r.get("qty") or 1
    if fp and q: return fp * 100.0 * q
    p, d = r.get("net_pnl_pct") or 0, r.get("net_pnl_dollars") or 0
    return abs(d) / (abs(p) / 100.0) if p else 0.0

baseline = sum((r.get("net_pnl_dollars") or 0) for r in allrows)
winners = [r for r in allrows if r.get("outcome") in WIN_EXITS and (r.get("net_pnl_dollars") or 0) > 0]
n_win = len(winners)

random.seed(11)
samp = random.sample(winners, min(120, n_win))
print(f"baseline P/L=${baseline:,.0f}  winners_pool={n_win}  representative_sample={len(samp)}")

cli = AlpacaOptionsClient(); dc = cli.data_client
def pdt(s):
    try: return datetime.fromisoformat(str(s))
    except Exception: return None

measured = []  # (mae_pct, gain_dollars, notional)
fetched = empty = errors = 0
for r in samp:
    sym, entry = r["symbol"], r["fill_price"] or 0
    t0, t1 = pdt(r["created_at"]), pdt(r["closed_at"])
    if not (sym and entry and t0 and t1):
        errors += 1; continue
    try:
        bs = dc.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
            start=t0 - timedelta(minutes=2), end=t1 + timedelta(minutes=2)))
        bars = bs.data.get(sym, []) if hasattr(bs, "data") else []
    except Exception:
        errors += 1; continue
    if not bars:
        empty += 1; continue
    mae = (min(b.low for b in bars) - entry) / entry * 100.0
    measured.append((mae, r.get("net_pnl_dollars") or 0, notional(r)))
    fetched += 1

print(f"coverage: fetched={fetched} empty={empty} errors={errors}\n")
ns = len(measured)

# Loser savings (full population), unchanged by sampling.
def loser_savings(S):
    trig = S * OVERSHOOT
    tot = 0.0
    for r in allrows:
        pct, dol = r.get("net_pnl_pct") or 0, r.get("net_pnl_dollars") or 0
        if pct < -trig:
            tot += (-trig / 100.0 * notional(r)) - dol
    return tot

hdr = f"{'stop':>5} {'cannib%':>8} {'loser_save$':>12} {'winner_cost$':>13} {'NET P/L$':>11} {'vs base':>9}"
print(hdr); print("-"*len(hdr))
for S in LEVELS:
    trig = S * OVERSHOOT
    # measured cannibalization on representative sample
    hit = [(g, notl) for (mae, g, notl) in measured if mae <= -S]
    cannib_rate = len(hit) / ns if ns else 0
    # avg $ swing per cannibalized winner: flip gain -> loss at -S% of notional
    if hit:
        avg_swing = sum((-S/100.0*notl) - g for (g, notl) in hit) / len(hit)
    else:
        avg_swing = 0.0
    winner_cost = avg_swing * cannib_rate * n_win   # negative
    save = loser_savings(S)
    net = baseline + save + winner_cost
    print(f"{S:>4.0f}% {100*cannib_rate:>7.1f}% {save:>+12,.0f} {winner_cost:>+13,.0f} "
          f"{net:>+11,.0f} {net-baseline:>+9,.0f}")

print("\n(winner_cost scales measured cannib rate x avg$ swing to the full 329 pool)")
print("(cannibalized winner assumed to exit at -stop%; loser trigger includes 1.386x overshoot)")
