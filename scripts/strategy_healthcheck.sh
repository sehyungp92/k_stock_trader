#!/bin/sh
# Strategy container healthcheck — verifies heartbeat file freshness.
# Exits 0 (healthy) if a heartbeat file was modified within the last 10 minutes.
# Exits 1 (unhealthy) otherwise.
#
# Used by Docker healthcheck in compose files.
# Heartbeat files: /app/instrumentation/data/heartbeats/heartbeats_YYYY-MM-DD.jsonl

HEARTBEAT_DIR="/app/instrumentation/data/heartbeats"

# During startup (before first heartbeat is written), check if the main
# python process is alive as a fallback.
if [ ! -d "$HEARTBEAT_DIR" ]; then
  pgrep -f "python" > /dev/null 2>&1
  exit $?
fi

# Check for any heartbeat file modified within the last 10 minutes
find "$HEARTBEAT_DIR" -name "*.jsonl" -mmin -10 | grep -q .
