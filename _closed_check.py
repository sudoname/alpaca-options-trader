from realized_pnl_tracker import RealizedPnLTracker
import json, os
t = RealizedPnLTracker()
print('today_realized= %.2f' % t.get_today_realized())
path = 'realized_pnl_log.json'
if os.path.exists(path):
    try:
        data = json.load(open(path))
    except Exception as e:
        data = []
        print('log_read_error:', e)
    print('total_log_entries=', len(data))
    print('%-26s %14s %10s' % ('symbol', 'when', 'pnl'))
    s = 0.0
    for r in data:
        sym = r.get('symbol') or r.get('ticker') or '?'
        when = r.get('timestamp') or r.get('time') or r.get('closed_at') or ''
        pnl = r.get('realized_pnl', r.get('pnl', r.get('amount', 0)))
        try:
            pnl = float(pnl)
        except Exception:
            pnl = 0.0
        s += pnl
        print('%-26s %14s %10.2f' % (str(sym), str(when)[-19:], pnl))
    print('-' * 54)
    print('sum= %.2f  count=%d' % (s, len(data)))
else:
    print('no realized_pnl_log.json')
