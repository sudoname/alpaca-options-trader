#!/usr/bin/env bash
# Background trade poller for the armed Oracle scheduler.
# Every 15 min: logs STATUS (mode/is_open/trades_today/entered/pdt) plus
# significant events (entries, closes, spreads, portfolio-limit/contract gates).
# Routine "market is closed" heartbeats and below_min_signals skips are filtered.
LOG=/c/Users/yomi/alpaca-options-trader/trade_monitor.log
SRV=root@147.182.242.177
SSH="ssh -o ServerAliveInterval=30 -o ConnectTimeout=20"
echo "[$(date -u +%H:%M:%SZ)] poller started (clean cadence)" >> "$LOG"
while true; do
  $SSH "$SRV" 'cd /var/www/alps && python3 -c "import json;d=json.load(open(\"scheduler_status.json\"));print(\"STATUS mode=%s is_open=%s trades_today=%s entered=%s pdt_remaining=%s\"%(d.get(\"mode\"),d.get(\"is_open\"),d.get(\"trades_today\"),d.get(\"entered_today\"),(d.get(\"pdt\") or {}).get(\"remaining\")))" 2>/dev/null; D=$(date +%Y-%m-%d); grep -E "^\[$D" /var/log/alps-scheduler.log 2>/dev/null | grep -iE "WOULD ENTER|\[ENTER|\[CLOSE|REJECT|\[SPREAD|portfolio limit|no suitable option" | grep -ivE "market is closed" | tail -n 6' \
    | sed "s/^/[$(date -u +%H:%M:%SZ)] /" >> "$LOG" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] ---" >> "$LOG"
  sleep 900
done
