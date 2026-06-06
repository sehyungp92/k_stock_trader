-- ============================================================
-- 008: Durable OMS protective stops
-- Idempotent: safe to run on already-upgraded databases
-- ============================================================

CREATE TABLE IF NOT EXISTS protective_stops (
    stop_id UUID PRIMARY KEY,
    oms_id VARCHAR(20) NOT NULL DEFAULT 'primary',
    strategy_id VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL DEFAULT 'LONG',
    qty INTEGER NOT NULL,
    stop_price NUMERIC(18,4) NOT NULL,
    trigger_price_source VARCHAR(20) NOT NULL DEFAULT 'LAST',
    protection_mode VARCHAR(30) NOT NULL,
    status VARCHAR(40) NOT NULL,
    broker_order_id VARCHAR(50),
    broker_order_date VARCHAR(8),
    entry_intent_id UUID,
    entry_order_id VARCHAR(50),
    exit_intent_id UUID,
    idempotency_key VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    triggered_at TIMESTAMPTZ,
    last_checked_at TIMESTAMPTZ,
    last_price NUMERIC(18,4),
    last_error TEXT,
    failure_count INTEGER NOT NULL DEFAULT 0,
    config_hash VARCHAR(100),
    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE protective_stops
    ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary',
    ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(20) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS symbol VARCHAR(20) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS side VARCHAR(10) NOT NULL DEFAULT 'LONG',
    ADD COLUMN IF NOT EXISTS qty INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS stop_price NUMERIC(18,4) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS trigger_price_source VARCHAR(20) NOT NULL DEFAULT 'LAST',
    ADD COLUMN IF NOT EXISTS protection_mode VARCHAR(30) NOT NULL DEFAULT 'OMS_WATCHER',
    ADD COLUMN IF NOT EXISTS status VARCHAR(40) NOT NULL DEFAULT 'PENDING',
    ADD COLUMN IF NOT EXISTS broker_order_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS broker_order_date VARCHAR(8),
    ADD COLUMN IF NOT EXISTS entry_intent_id UUID,
    ADD COLUMN IF NOT EXISTS entry_order_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS exit_intent_id UUID,
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(255),
    ADD COLUMN IF NOT EXISTS activated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS triggered_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_price NUMERIC(18,4),
    ADD COLUMN IF NOT EXISTS last_error TEXT,
    ADD COLUMN IF NOT EXISTS failure_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS config_hash VARCHAR(100),
    ADD COLUMN IF NOT EXISTS source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS uq_protective_stops_active_allocation
    ON protective_stops(oms_id, strategy_id, symbol)
    WHERE status IN ('PENDING', 'ACTIVE', 'TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED');

CREATE INDEX IF NOT EXISTS idx_protective_stops_oms_status_updated
    ON protective_stops(oms_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_protective_stops_oms_symbol_strategy
    ON protective_stops(oms_id, symbol, strategy_id);

CREATE INDEX IF NOT EXISTS idx_protective_stops_exit_intent
    ON protective_stops(oms_id, exit_intent_id)
    WHERE exit_intent_id IS NOT NULL;
