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

# Pull all FILL activities since the start of yesterday (entries) through now.
acts = []
after = "2026-06-08T00:00:00Z"
url = "/v2/account/activities?activity_types=FILL&after=%s&direction=asc&page_size=100" % after
while True:
    batch = g(url)
    if not batch:
        break
    acts.extend(batch)
    if len(batch) < 100:
        break
    last_id = batch[-1]["id"]
    url = ("/v2/account/activities?activity_types=FILL&after=%s&direction=asc"
           "&page_size=100&page_token=%s" % (after, last_id))

print("total_fills=", len(acts))

today = "2026-06-09"
# Group fills by symbol; compute net cash and net qty (long options).
from collections import defaultdict
buys = defaultdict(lambda: [0.0, 0.0])   # symbol -> [qty, cost$]
sells = defaultdict(lambda: [0.0, 0.0])  # symbol -> [qty, proceeds$]
sell_today = defaultdict(float)          # symbol -> sell qty closed TODAY
for a in acts:
    sym = a.get("symbol"); side = a.get("side");
    qty = float(a.get("qty") or 0); price = float(a.get("price") or 0)
    when = str(a.get("transaction_time") or "")
    val = qty * price * 100.0
    if side in ("buy", "buy_to_open", "buy_to_close"):
        buys[sym][0] += qty; buys[sym][1] += val
    elif side in ("sell", "sell_to_close", "sell_to_open"):
        sells[sym][0] += qty; sells[sym][1] += val
        if when.startswith(today):
            sell_today[sym] += qty

# Realized for symbols that had a SELL today: realized = proceeds - cost basis of
# the quantity sold (avg-cost). Position considered closed if net qty ~ 0.
rows = []
for sym in sorted(sell_today):
    bq, bc = buys[sym]; sq, sp = sells[sym]
    avg_cost = (bc / bq) if bq else 0.0     # $ per contract (x100 already in val)
    sold_q = sells[sym][0]
    realized = sp - avg_cost * sold_q       # proceeds - cost basis of sold qty
    net_q = bq - sq
    rows.append((sym, sold_q, sp, avg_cost * sold_q, realized, net_q))

print("%-26s %4s %10s %10s %10s %5s" % ("symbol", "qty", "proceeds", "costbasis", "realized$", "netq"))
tot = 0.0
for sym, q, pr, cb, rl, nq in sorted(rows, key=lambda x: x[4], reverse=True):
    tot += rl
    print("%-26s %4.0f %10.2f %10.2f %+10.2f %5.0f" % (sym, q, pr, cb, rl, nq))
print("-" * 74)
print("STRICT realized $ for symbols sold today: %+.2f (n=%d)" % (tot, len(rows)))
