import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv("/var/www/alps/.env")
k = os.getenv("ALPACA_API_KEY"); s = os.getenv("ALPACA_SECRET_KEY")
paper = str(os.getenv("ALPACA_PAPER", "true")).lower() != "false"
base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
h = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}

def g(p):
    return json.load(urllib.request.urlopen(urllib.request.Request(base + p, headers=h), timeout=30))

seven = ["CVS260821C00100000", "MDT260717C00080000", "SCHW260717C00090000",
         "QQQ260710P00719000", "GSK260821C00045000", "ABT260821C00095000",
         "MO260717C00065000"]

pos = g("/v2/positions")
held = {p["symbol"]: p for p in pos}
print("=== Are the 7 reconciled_closed still OPEN at broker? ===")
for sym in seven:
    if sym in held:
        p = held[sym]
        print("  OPEN   %-22s qty=%s avg=%s cur=%s upl=%s" % (
            sym, p.get("qty"), p.get("avg_entry_price"),
            p.get("current_price"), p.get("unrealized_pl")))
    else:
        print("  ABSENT %-22s (not held)" % sym)

print()
print("=== All ORDERS for those 7 symbols (any status, last 100) ===")
orders = g("/v2/orders?status=all&limit=500&direction=desc")
bysym = {}
for o in orders:
    sym = o.get("symbol")
    if sym in seven:
        bysym.setdefault(sym, []).append(o)
for sym in seven:
    os_ = bysym.get(sym, [])
    if not os_:
        print("  %-22s NO ORDERS FOUND (never entered?)" % sym)
        continue
    for o in os_:
        print("  %-22s %-5s qty=%s status=%s filled_qty=%s filled_avg=%s submitted=%s" % (
            sym, o.get("side"), o.get("qty"), o.get("status"),
            o.get("filled_qty"), o.get("filled_avg_price"),
            str(o.get("submitted_at"))[:19]))
