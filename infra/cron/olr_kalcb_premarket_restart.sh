#!/usr/bin/env bash
# Single-VPS OLR/KALCB premarket artifact gate.
# Runs after KRX/KIS daily data is refreshed and before runtime starts/restarts.
set -euo pipefail

ROOT="${OLR_KALCB_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG="${OLR_KALCB_PREMARKET_LOG:-/var/log/k_stock_trader/olr_kalcb_premarket_restart.log}"
LOCK="${OLR_KALCB_PREMARKET_LOCK:-/tmp/olr_kalcb_premarket_restart.lock}"
ARTIFACT_TIMEOUT="${OLR_KALCB_ARTIFACT_TIMEOUT_SECONDS:-3600}"
DAILY_UNIVERSE_FILE="${OLR_KALCB_DAILY_UNIVERSE_FILE:-config/olr_kalcb/olr_deployment_universe_103.yaml}"

mkdir -p "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || {
  echo "another OLR/KALCB premarket restart is already running" >&2
  exit 75
}

{
  echo "--- $(date -u '+%Y-%m-%d %H:%M:%S UTC') premarket ---"
  cd "$ROOT"

  echo "Starting base services..."
  docker compose up -d postgres oms

  TRADE_DATE="${OLR_KALCB_TRADE_DATE:-$(docker compose run --rm runtime python -c 'from deployment.olr_kalcb.readiness import krx_trade_date; print(krx_trade_date().isoformat())')}"
  echo "Trade date: $TRADE_DATE"

  echo "Stopping runtime before artifact refresh..."
  docker compose stop runtime || true

  if [ ! -f "$DAILY_UNIVERSE_FILE" ]; then
    echo "approved OLR/KALCB deployment universe file is missing: $DAILY_UNIVERSE_FILE" >&2
    exit 66
  fi
  daily_args=(--daily-universe-file "$DAILY_UNIVERSE_FILE")
  if [ -n "${OLR_KALCB_MAX_DAILY_LAG_DAYS:-}" ]; then
    daily_args+=(--max-daily-lag-days "$OLR_KALCB_MAX_DAILY_LAG_DAYS")
  fi

  echo "Generating KALCB daily and OLR stage1 artifacts..."
  timeout "$ARTIFACT_TIMEOUT" docker compose run --rm runtime \
    python scripts/generate_olr_kalcb_artifacts.py daily \
      --trade-date "$TRADE_DATE" \
      "${daily_args[@]}"

  echo "Running stage1 artifact gate..."
  docker compose run --rm runtime \
    python scripts/run_olr_kalcb_runtime_session.py preflight \
      --trade-date "$TRADE_DATE" \
      --mode artifact_only_stage1

  if [ "${OLR_KALCB_START_RUNTIME_AFTER_PREMARKET:-false}" = "true" ]; then
    : "${OLR_KALCB_BARS_PARQUET:?set OLR_KALCB_BARS_PARQUET before starting runtime}"
    export OLR_KALCB_TRADE_DATE="$TRADE_DATE"
    echo "Starting runtime service..."
    docker compose up -d runtime
  else
    echo "Runtime start skipped; set OLR_KALCB_START_RUNTIME_AFTER_PREMARKET=true to start after the gate."
  fi

  echo "Premarket gate complete."
} >> "$LOG" 2>&1
