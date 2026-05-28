#!/bin/sh
set -eu

MODE="${OLR_KALCB_RUNTIME_MODE:-dry_run}"
TRADE_DATE="${OLR_KALCB_TRADE_DATE:-}"
BARS_PARQUET="${OLR_KALCB_BARS_PARQUET:-}"
MARKET_DATA_SOURCE="${OLR_KALCB_MARKET_DATA_SOURCE:-auto}"

if [ -z "$TRADE_DATE" ]; then
  echo "OLR_KALCB_TRADE_DATE must be set before starting the runtime service" >&2
  exit 64
fi

EFFECTIVE_MARKET_DATA_SOURCE="$MARKET_DATA_SOURCE"
if [ "$EFFECTIVE_MARKET_DATA_SOURCE" = "auto" ]; then
  case "$MODE" in
    paper|live) EFFECTIVE_MARKET_DATA_SOURCE="kis_websocket" ;;
    *) EFFECTIVE_MARKET_DATA_SOURCE="external_completed_bars" ;;
  esac
fi

if [ "$EFFECTIVE_MARKET_DATA_SOURCE" = "external_completed_bars" ] && [ -z "$BARS_PARQUET" ]; then
  echo "OLR_KALCB_BARS_PARQUET must point to completed 5m bars when using external_completed_bars" >&2
  exit 64
fi

SESSION_ROOT="${OLR_KALCB_SESSION_ROOT:-/app/data/paper_live/olr_kalcb/$TRADE_DATE}"
HEALTH_CHECKS_JSON="${OLR_KALCB_HEALTH_CHECKS_JSON:-/app/data/paper_live/olr_kalcb/$TRADE_DATE/health_checks.json}"
ACCOUNT_STATE_JSON="${OLR_KALCB_ACCOUNT_STATE_JSON:-/app/data/paper_live/olr_kalcb/$TRADE_DATE/account_state.json}"
POSITIONS_JSON="${OLR_KALCB_POSITIONS_JSON:-/app/data/paper_live/olr_kalcb/$TRADE_DATE/positions.json}"
POLL_SECONDS="${OLR_KALCB_POLL_SECONDS:-15}"
OMS_URL="${OMS_URL:-http://oms:8000}"
KIS_WS_URL="${OLR_KALCB_KIS_WS_URL:-}"
WS_LEDGER_PATH="${OLR_KALCB_WS_LEDGER_PATH:-}"

set -- python scripts/run_olr_kalcb_runtime_session.py watch-bars \
  --trade-date "$TRADE_DATE" \
  --mode "$MODE" \
  --market-data-source "$MARKET_DATA_SOURCE" \
  --session-root "$SESSION_ROOT" \
  --health-checks-json "$HEALTH_CHECKS_JSON" \
  --account-state-json "$ACCOUNT_STATE_JSON" \
  --positions-json "$POSITIONS_JSON" \
  --poll-seconds "$POLL_SECONDS" \
  --oms-url "$OMS_URL"

if [ -n "$BARS_PARQUET" ]; then
  set -- "$@" --bars-parquet "$BARS_PARQUET"
fi

if [ -n "$KIS_WS_URL" ]; then
  set -- "$@" --kis-ws-url "$KIS_WS_URL"
fi

if [ -n "$WS_LEDGER_PATH" ]; then
  set -- "$@" --ws-ledger-path "$WS_LEDGER_PATH"
fi

exec "$@"
