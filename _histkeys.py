import json
from datetime import datetime
today = datetime.now().strftime('%Y-%m-%d')
h = json.load(open('trading_history.json'))
c = [t for t in h.get('trades', []) if str(t.get('exit_time', '')).startswith(today)]
print('closed_today=', len(c))
print('KEYS:', sorted(c[0].keys()))
for t in c[:3]:
    print(json.dumps(t, default=str))
