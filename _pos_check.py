import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv()
k = os.getenv('ALPACA_API_KEY'); s = os.getenv('ALPACA_SECRET_KEY')
paper = str(os.getenv('ALPACA_PAPER', 'true')).lower() != 'false'
base = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
h = {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}
req = urllib.request.Request(base + '/v2/positions', headers=h)
pos = json.load(urllib.request.urlopen(req, timeout=30))
opts = [p for p in pos if p.get('asset_class') == 'us_option']
opts.sort(key=lambda x: float(x['unrealized_pl']), reverse=True)
print('option_positions=', len(opts))
print('%-22s %3s %8s %8s %10s %8s' % ('symbol','qty','avg','cur','mkt_val','unrl_pl'))
tot_mv = tot_pl = 0.0
for p in opts:
    qty = float(p['qty']); avg = float(p['avg_entry_price']); cur = float(p['current_price'])
    mv = float(p['market_value']); pl = float(p['unrealized_pl']); plpc = float(p['unrealized_plpc'])*100
    tot_mv += mv; tot_pl += pl
    print('%-22s %3d %8.2f %8.2f %10.2f %8.2f (%+.1f%%)' % (
        p['symbol'], int(qty), avg, cur, mv, pl, plpc))
print('-'*70)
print('%-22s %3s %8s %8s %10.2f %8.2f' % ('TOTAL','','','', tot_mv, tot_pl))
