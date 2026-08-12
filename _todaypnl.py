import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv("/var/www/alps/.env")
k = os.getenv("ALPACA_API_KEY"); s = os.getenv("ALPACA_SECRET_KEY")
paper = str(os.getenv("ALPACA_PAPER", "true")).lower() != "false"
base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
h = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}

def g(p):
    return json.load(urllib.request.urlopen(urllib.request.Request(base + p, headers=h), timeout=30))

a = g("/v2/account")
eq = float(a["equity"]); le = float(a["last_equity"])
print("ACCOUNT day P/L (equity - last_equity): %+.2f (%+.2f%%)" % (eq - le, (eq - le) / le * 100))
print("  equity=%.2f  last_equity=%.2f" % (eq, le))

pos = g("/v2/positions")
upl = sum(float(p["unrealized_pl"]) for p in pos)
print("OPEN unrealized P/L: %+.2f  (%d positions)" % (upl, len(pos)))

from realized_pnl_tracker import RealizedPnLTracker
print("REALIZED today (kill-switch tracker, post-reset): %+.2f" % RealizedPnLTracker().get_today_realized())
