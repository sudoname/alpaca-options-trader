#!/usr/bin/env bash
# Server-side capture helper for the Oracle 3.0 shadow monitor.
# Selects today's slice of the scheduler log (turning on at the first line that
# starts with today's date, then printing everything after -- including lines
# that begin with a marker like [THESIS]/[EXEC EV] rather than a timestamp),
# then content-filters for the shadow telemetry + real entries/closes/errors.
D=$(date -u +%Y-%m-%d)
awk -v d="[$D" 'index($0,d)==1{f=1} f' /var/log/alps-scheduler.log 2>/dev/null \
  | grep -iE '\[THESIS\]|\[EXEC EV\]|would-reject|entering |order placed for|\[CLOSE|traceback|error' \
  | grep -ivE 'market is closed' \
  | tail -n 40
