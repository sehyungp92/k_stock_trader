#!/usr/bin/env bash
# Single-VPS OLR/KALCB afternoon artifact gate.
# Runs after the 14:30 KST completed-bar cutoff, then restarts runtime to load OLR final.
set -euo pipefail

ROOT="${OLR_KALCB_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG="${OLR_KALCB_AFTERNOON_LOG:-/var/log/k_stock_trader/olr_kalcb_afternoon_restart.log}"
LOCK="${OLR_KALCB_AFTERNOON_LOCK:-/tmp/olr_kalcb_afternoon_restart.lock}"
ARTIFACT_TIMEOUT="${OLR_KALCB_ARTIFACT_TIMEOUT_SECONDS:-3600}"

mkdir -p "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || {
  echo "another OLR/KALCB afternoon restart is already running" >&2
  exit 75
}

{
  echo "--- $(date -u '+%Y-%m-%d %H:%M:%S UTC') afternoon ---"
  cd "$ROOT"

  echo "Ensuring base services are healthy..."
  docker compose up -d postgres oms

  TRADE_DATE="${OLR_KALCB_TRADE_DATE:-$(docker compose run --rm runtime python -c 'from deployment.olr_kalcb.readiness import krx_trade_date; print(krx_trade_date().isoformat())')}"
  echo "Trade date: $TRADE_DATE"

  afternoon_args=()
  if [ -n "${OLR_KALCB_MIN_AFTERNOON_BARS_PER_SYMBOL:-}" ]; then
    afternoon_args+=(--min-afternoon-bars-per-symbol "$OLR_KALCB_MIN_AFTERNOON_BARS_PER_SYMBOL")
  fi

  echo "Generating OLR final afternoon artifact..."
  timeout "$ARTIFACT_TIMEOUT" docker compose run --rm runtime \
    python scripts/generate_olr_kalcb_artifacts.py afternoon \
      --trade-date "$TRADE_DATE" \
      "${afternoon_args[@]}"

  echo "Running final artifact gate..."
  docker compose run --rm runtime \
    python scripts/run_olr_kalcb_runtime_session.py preflight \
      --trade-date "$TRADE_DATE" \
      --mode artifact_only

  if [ "${OLR_KALCB_RESTART_RUNTIME_AFTER_AFTERNOON:-true}" = "true" ]; then
    : "${OLR_KALCB_BARS_PARQUET:?set OLR_KALCB_BARS_PARQUET before starting runtime}"
    export OLR_KALCB_TRADE_DATE="$TRADE_DATE"
    echo "Recreating runtime service so it loads the final OLR artifact..."
    docker compose up -d --force-recreate runtime
  else
    echo "Runtime restart skipped by OLR_KALCB_RESTART_RUNTIME_AFTER_AFTERNOON=false."
  fi

  echo "Afternoon gate complete."
} >> "$LOG" 2>&1
