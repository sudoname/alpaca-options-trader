import json, os
from alpaca_client import AlpacaOptionsClient

rejects = json.load(open("/tmp/rejects.json"))
rej_contracts = {r["contract"]: r for r in rejects}

c = AlpacaOptionsClient()
positions = c.trading_client.get_all_positions()

opt = [p for p in positions if getattr(p, "asset_class", None) and "option" in str(p.asset_class).lower()]
# fallback: options symbols are long OCC strings
if not opt:
    opt = [p for p in positions if len(p.symbol) > 12]

def num(x):
    try:
        return float(x)
    except Exception:
        return 0.0

rows = []
for p in opt:
    sym = p.symbol
    upl = num(p.unrealized_pl)
    uplpc = num(p.unrealized_plpc) * 100.0
    avg = num(p.avg_entry_price)
    cur = num(p.current_price)
    is_rej = sym in rej_contracts
    rows.append((sym, is_rej, avg, cur, upl, uplpc))

print("open option positions:", len(opt), "| matched-as-reject:", sum(1 for r in rows if r[1]))
print()
print("=== EXEC-EV WOULD-REJECT positions (did direction justify?) ===")
rj = [r for r in rows if r[1]]
for sym, _, avg, cur, upl, uplpc in sorted(rj, key=lambda x: x[4]):
    verdict = "LOSER (reject justified)" if upl < 0 else ("WINNER (reject wrong)" if upl > 0 else "flat")
    print("%-22s entry=%6.2f now=%6.2f  P/L=$%8.2f (%+6.1f%%)  %s" % (sym, avg, cur, upl, uplpc, verdict))

rj_pl = [r[4] for r in rj]
nr = [r for r in rows if not r[1]]
nr_pl = [r[4] for r in nr]

def summ(name, pls):
    if not pls:
        print("%s: none" % name); return
    wins = sum(1 for x in pls if x > 0); losers = sum(1 for x in pls if x < 0)
    print("%-28s n=%2d  total=$%9.2f  avg=$%7.2f  winners=%d losers=%d  win%%=%.0f%%" % (
        name, len(pls), sum(pls), sum(pls)/len(pls), wins, losers, 100.0*wins/len(pls)))

print()
print("=== SUMMARY (unrealized P&L at close) ===")
summ("EXEC-EV would-REJECT set", rj_pl)
summ("Non-reject (exec-EV OK) set", nr_pl)
