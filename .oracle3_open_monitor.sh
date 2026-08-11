#!/usr/bin/env bash
KEY=~/.ssh/id_ed25519_1
SRV=root@147.182.242.177
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes -i $KEY"
LOG=/c/Users/yomi/alpaca-options-trader/oracle3_open_monitor.log
ts(){ date -u +%FT%TZ; }
echo "[$(ts)] monitor started (server=ec44a6a, shadow flags on); market opens 13:30 UTC" | tee -a "$LOG"
SAW_OPEN=0
while true; do
  NOWH=$((10#$(date -u +%H%M)))
  ST=$($SSH "$SRV" 'cd /var/www/alps && python3 -c "import json;d=json.load(open(\"scheduler_status.json\"));print(\"is_open=%s mode=%s trades_today=%s entered=%s\"%(d.get(\"is_open\"),d.get(\"mode\"),d.get(\"trades_today\"),len(d.get(\"entered_today\") or [])))" 2>/dev/null' 2>/dev/null)
  [ -z "$ST" ] && ST="(status unreachable)"
  echo "[$(ts)] $ST" | tee -a "$LOG"
  case "$ST" in
    *is_open=True*)
      SAW_OPEN=1
      $SSH "$SRV" 'bash /var/www/alps/.oracle3_capture.sh' 2>/dev/null | sed "s/^/[$(ts)]   /" | tee -a "$LOG"
      SLEEP=120 ;;
    *)
      if [ "$SAW_OPEN" -eq 1 ] && [ "$NOWH" -ge 2010 ]; then echo "[$(ts)] post-session & closed; stopping monitor" | tee -a "$LOG"; break; fi
      if [ "$NOWH" -ge 1310 ] && [ "$NOWH" -lt 1330 ]; then SLEEP=60; else SLEEP=900; fi ;;
  esac
  echo "[$(ts)] --- (next poll in ${SLEEP}s)" >> "$LOG"
  sleep "$SLEEP"
done
echo "[$(ts)] monitor exited" | tee -a "$LOG"
