import json, os
from datetime import datetime
today = datetime.now().strftime('%Y-%m-%d')

# Source A: trading_history.json (the daily-summary source) -> closed today
hist_closed = []
try:
    h = json.load(open('trading_history.json'))
    hist_closed = [t for t in h.get('trades', [])
                   if str(t.get('exit_time', '')).startswith(today)]
except Exception as e:
    print('history_err:', e)

# Source B: realized_pnl_log.json (kill-switch dollar tracker)
rlog = []
try:
    rlog = [r for r in json.load(open('realized_pnl_log.json'))
            if str(r.get('date', '')).startswith(today) or str(r.get('timestamp','')).startswith(today)]
except Exception as e:
    print('rlog_err:', e)

print('=== A) trading_history closed today: %d ===' % len(hist_closed))
print('%-26s %8s %9s  %s' % ('symbol', 'exit', 'pnl%', 'outcome'))
for t in sorted(hist_closed, key=lambda x: str(x.get('exit_time',''))):
    print('%-26s %8s %+8.1f%%  %s' % (
        str(t.get('symbol')), str(t.get('exit_time',''))[11:19],
        float(t.get('pnl_percent') or 0), t.get('outcome','')))

print()
print('=== B) realized_pnl_log today: %d ===' % len(rlog))
print('%-26s %8s %10s' % ('symbol', 'time', 'amount$'))
for r in sorted(rlog, key=lambda x: str(x.get('timestamp',''))):
    print('%-26s %8s %+10.2f' % (
        str(r.get('symbol')), str(r.get('timestamp',''))[11:19], float(r.get('amount') or 0)))

# Diff by symbol
A = {}
for t in hist_closed:
    A[str(t.get('symbol'))] = A.get(str(t.get('symbol')), 0) + 1
B = {}
for r in rlog:
    B[str(r.get('symbol'))] = B.get(str(r.get('symbol')), 0) + 1

print()
print('=== In A (history) but NOT in B (realized_log) ===')
for s in sorted(set(A) - set(B)):
    print('  %-26s  (history x%d)' % (s, A[s]))
print('=== In B (realized_log) but NOT in A (history) ===')
for s in sorted(set(B) - set(A)):
    print('  %-26s  (rlog x%d)' % (s, B[s]))
print('=== In BOTH (count A vs B) ===')
for s in sorted(set(A) & set(B)):
    print('  %-26s  A=%d B=%d' % (s, A[s], B[s]))
print()
print('totals: A_symbols=%d A_rows=%d | B_symbols=%d B_rows=%d' % (
    len(A), len(hist_closed), len(B), len(rlog)))
