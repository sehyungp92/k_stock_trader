#!/bin/bash

STACK_SCOPE=${1:-"base"}
ALERT_WEBHOOK=${2:-""}  # Slack/Discord webhook URL
SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

cd "$PROJECT_DIR"

check_service() {
    local service=$1
    local url=$2

    if curl -sf "$url" > /dev/null 2>&1; then
        echo "[OK] $service is healthy"
        return 0
    else
        echo "[FAIL] $service is DOWN"
        if [ -n "$ALERT_WEBHOOK" ]; then
            curl -X POST -H "Content-Type: application/json" \
                -d "{\"text\":\"$STACK_SCOPE: $service is DOWN\"}" \
                "$ALERT_WEBHOOK"
        fi
        return 1
    fi
}

check_compose_service() {
    local service=$1

    if docker compose -f "$COMPOSE_FILE" ps --services --status running | grep -qx "$service"; then
        echo "[OK] Compose service $service is running"
        return 0
    else
        echo "[FAIL] Compose service $service is NOT running"
        if [ -n "$ALERT_WEBHOOK" ]; then
            curl -X POST -H "Content-Type: application/json" \
                -d "{\"text\":\"$STACK_SCOPE: Compose service $service is NOT running\"}" \
                "$ALERT_WEBHOOK"
        fi
        return 1
    fi
}

echo "=== Health Check: $STACK_SCOPE ==="
echo "Time: $(date)"

check_service "OMS" "http://localhost:8000/health"

services=(postgres oms)

case "$STACK_SCOPE" in
    base|single-vps)
        ;;
    pcim)
        services+=(pcim)
        ;;
    olr-kalcb|runtime)
        services+=(runtime)
        ;;
    dashboard)
        services+=(dashboard)
        ;;
    all)
        services+=(pcim runtime dashboard)
        ;;
    *)
        echo "[WARN] Unknown scope '$STACK_SCOPE'; checking base services only"
        ;;
esac

for service in "${services[@]}"; do
    check_compose_service "$service"
done

echo "=== End Health Check ==="
