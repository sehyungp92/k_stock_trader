#!/bin/bash

VPS_NAME=${1:-"unknown"}
ALERT_WEBHOOK=${2:-""}  # Slack/Discord webhook URL

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
                -d "{\"text\":\"$VPS_NAME: $service is DOWN\"}" \
                "$ALERT_WEBHOOK"
        fi
        return 1
    fi
}

check_container() {
    local container=$1

    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "[OK] Container $container is running"
        return 0
    else
        echo "[FAIL] Container $container is NOT running"
        if [ -n "$ALERT_WEBHOOK" ]; then
            curl -X POST -H "Content-Type: application/json" \
                -d "{\"text\":\"$VPS_NAME: Container $container is NOT running\"}" \
                "$ALERT_WEBHOOK"
        fi
        return 1
    fi
}

echo "=== Health Check: $VPS_NAME ==="
echo "Time: $(date)"

# Check OMS
check_service "OMS" "http://localhost:8000/health"

# Check containers based on VPS
if [ "$VPS_NAME" == "VPS1" ]; then
    check_container "strategy_kmp"
    check_container "strategy_nulrimok"
elif [ "$VPS_NAME" == "VPS2" ]; then
    check_container "strategy_kpr"
    check_container "strategy_pcim"
    check_container "trading_db"
fi

echo "=== End Health Check ==="
