import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv()
k = os.getenv('ALPACA_API_KEY'); s = os.getenv('ALPACA_SECRET_KEY')
paper = str(os.getenv('ALPACA_PAPER', 'true')).lower() != 'false'
base = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
h = {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}
req = urllib.request.Request(base + '/v2/positions', headers=h)
pos = json.load(urllib.request.urlopen(req, timeout=30))
upl = sum(float(p['unrealized_pl']) for p in pos)
print('open_positions=', len(pos))
print('unrealized_pl= %.2f' % upl)
for p in sorted(pos, key=lambda x: float(x['unrealized_pl']))[:5]:
    print('  worst:', p['symbol'], '%.2f' % float(p['unrealized_pl']))
for p in sorted(pos, key=lambda x: -float(x['unrealized_pl']))[:5]:
    print('  best :', p['symbol'], '%.2f' % float(p['unrealized_pl']))
