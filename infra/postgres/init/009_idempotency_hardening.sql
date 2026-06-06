-- ============================================================
-- 009: Durable idempotency hardening
-- Scope decision: idempotency_key remains account-global.
--
-- Rationale: current strategy-generated keys already include strategy, symbol,
-- trade date, side/intent type, signal/reason, and quantity. A shared OMS DB may
-- host multiple OMS IDs, but callers that submit across accounts must namespace
-- idempotency keys explicitly to avoid cross-account duplicate submit risk.
-- Keeping the global unique key is fail-closed and preserves existing data.
-- ============================================================

COMMENT ON COLUMN intents.idempotency_key IS
    'Account-global idempotency key. Multi-account callers must namespace keys before submission.';

ALTER TABLE intents
    ADD COLUMN IF NOT EXISTS reservation_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reservation_owner VARCHAR(80),
    ADD COLUMN IF NOT EXISTS reservation_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reservation_reconcile_status VARCHAR(40),
    ADD COLUMN IF NOT EXISTS reservation_reconcile_message TEXT,
    ADD COLUMN IF NOT EXISTS submit_ref VARCHAR(80),
    ADD COLUMN IF NOT EXISTS planned_side VARCHAR(4),
    ADD COLUMN IF NOT EXISTS planned_qty INTEGER,
    ADD COLUMN IF NOT EXISTS planned_order_type VARCHAR(30),
    ADD COLUMN IF NOT EXISTS planned_limit_price NUMERIC(18,4),
    ADD COLUMN IF NOT EXISTS planned_stop_price NUMERIC(18,4);

CREATE INDEX IF NOT EXISTS idx_intents_oms_status_created
    ON intents(oms_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_intents_oms_idempotency
    ON intents(oms_id, idempotency_key);

CREATE INDEX IF NOT EXISTS idx_intents_oms_order_id
    ON intents(oms_id, order_id)
    WHERE order_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_orders_oms_status_created
    ON orders(oms_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orders_oms_kis_order
    ON orders(oms_id, kis_order_id)
    WHERE kis_order_id IS NOT NULL;
