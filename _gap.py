import os, json, urllib.request
from datetime import datetime
from dotenv import load_dotenv
load_dotenv("/var/www/alps/.env")
k = os.getenv("ALPACA_API_KEY"); s = os.getenv("ALPACA_SECRET_KEY")
paper = str(os.getenv("ALPACA_PAPER", "true")).lower() != "false"
base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
h = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}

def g(p):
    return json.load(urllib.request.urlopen(urllib.request.Request(base + p, headers=h), timeout=30))

watch = ["UNH260717C00400000", "HD260717C00310000", "KO260717C00075000", "XOM260717P00145000"]

print("=== Orders for UNH/HD/KO/XOM (how/when did they close?) ===")
orders = g("/v2/orders?status=all&limit=500&direction=desc")
for sym in watch:
    for o in orders:
        if o.get("symbol") == sym:
            print("  %-22s %-5s qty=%s type=%s status=%s filled_avg=%s submitted=%s order_class=%s" % (
                sym, o.get("side"), o.get("qty"), o.get("type"), o.get("status"),
                o.get("filled_avg_price"), str(o.get("submitted_at"))[:19], o.get("order_class")))

print()
print("=== Are they in trading_history (any exit_time)? ===")
try:
    hist = json.load(open("trading_history.json"))
    for sym in watch:
        rows = [t for t in hist.get("trades", []) if t.get("symbol") == sym]
        if rows:
            for t in rows[-2:]:
                print("  %-22s exit=%s outcome=%s pnl%%=%s" % (
                    sym, t.get("exit_time"), t.get("outcome"), t.get("pnl_percent")))
        else:
            print("  %-22s NOT in trading_history" % sym)
except Exception as e:
    print("hist err:", e)

print()
print("=== Are they still tracked as open in active_trades.json? ===")
try:
    at = json.load(open("active_trades.json"))
    syms = {t.get("symbol") for t in at}
    for sym in watch:
        print("  %-22s %s" % (sym, "STILL TRACKED OPEN" if sym in syms else "not tracked"))
    print("  active_trades total:", len(at))
except Exception as e:
    print("active err:", e)

print()
print("=== Are they in realized_pnl_log.json? ===")
try:
    rl = json.load(open("realized_pnl_log.json"))
    syms = {r.get("symbol") for r in rl}
    for sym in watch:
        print("  %-22s %s" % (sym, "logged" if sym in syms else "MISSING from realized log"))
except Exception as e:
    print("rlog err:", e)
