#!/bin/bash
# instrumentation/validate_structure.sh

echo "=== Validating instrumentation file structure ==="
FAIL=0

check() {
    if [ ! -e "$1" ]; then
        echo "MISSING: $1"
        FAIL=1
    else
        echo "OK: $1"
    fi
}

# Config files
check "instrumentation/config/instrumentation_config.yaml"
check "instrumentation/config/simulation_policies.yaml"
check "instrumentation/config/regime_classifier_config.yaml"
check "instrumentation/config/process_scoring_rules.yaml"

# Source files
check "instrumentation/src/event_metadata.py"
check "instrumentation/src/market_snapshot.py"
check "instrumentation/src/trade_logger.py"
check "instrumentation/src/missed_opportunity.py"
check "instrumentation/src/process_scorer.py"
check "instrumentation/src/daily_snapshot.py"
check "instrumentation/src/regime_classifier.py"
check "instrumentation/src/sidecar.py"

# Tests
check "instrumentation/tests/test_event_metadata.py"
check "instrumentation/tests/test_market_snapshot.py"
check "instrumentation/tests/test_trade_logger.py"
check "instrumentation/tests/test_missed_opportunity.py"
check "instrumentation/tests/test_process_scorer.py"
check "instrumentation/tests/test_daily_snapshot.py"
check "instrumentation/tests/test_regime_classifier.py"
check "instrumentation/tests/test_sidecar.py"
check "instrumentation/tests/test_integration.py"

# Data directories
check "instrumentation/data"

# Audit report
check "instrumentation/audit_report.md"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "=== All files present ==="
else
    echo ""
    echo "=== VALIDATION FAILED: missing files ==="
    exit 1
fi
