-- ============================================================
-- 005: Add oms_id scoping for multi-instance Postgres sharing
-- Idempotent: safe to run on both fresh and existing databases
-- ============================================================

-- ---------- COLLISION-CRITICAL TABLES (PK changes) ----------

-- positions: symbol -> (oms_id, symbol)
ALTER TABLE positions ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.key_column_usage
    WHERE table_name = 'positions' AND column_name = 'oms_id' AND constraint_name = 'positions_pkey'
  ) THEN
    ALTER TABLE positions DROP CONSTRAINT positions_pkey;
    ALTER TABLE positions ADD PRIMARY KEY (oms_id, symbol);
  END IF;
END $$;

-- allocations: (symbol, strategy_id) -> (oms_id, symbol, strategy_id)
ALTER TABLE allocations ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.key_column_usage
    WHERE table_name = 'allocations' AND column_name = 'oms_id' AND constraint_name = 'allocations_pkey'
  ) THEN
    ALTER TABLE allocations DROP CONSTRAINT allocations_pkey;
    ALTER TABLE allocations ADD PRIMARY KEY (oms_id, symbol, strategy_id);
  END IF;
END $$;

-- risk_daily_portfolio: trade_date -> (oms_id, trade_date)
ALTER TABLE risk_daily_portfolio ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.key_column_usage
    WHERE table_name = 'risk_daily_portfolio' AND column_name = 'oms_id'
      AND constraint_name = 'risk_daily_portfolio_pkey'
  ) THEN
    ALTER TABLE risk_daily_portfolio DROP CONSTRAINT risk_daily_portfolio_pkey;
    ALTER TABLE risk_daily_portfolio ADD PRIMARY KEY (oms_id, trade_date);
  END IF;
END $$;

-- ---------- UUID-KEYED TABLES (column only, no PK change) ----------

ALTER TABLE intents ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
ALTER TABLE orders ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
ALTER TABLE order_events ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
ALTER TABLE fills ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
ALTER TABLE trades ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
ALTER TABLE risk_daily_strategy ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
ALTER TABLE strategy_state ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';
ALTER TABLE recon_log ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';

-- ---------- INDEXES ----------

DROP INDEX IF EXISTS idx_positions_frozen;
DROP INDEX IF EXISTS idx_positions_updated;
DROP INDEX IF EXISTS idx_allocations_strategy;

CREATE INDEX IF NOT EXISTS idx_positions_oms ON positions(oms_id);
CREATE INDEX IF NOT EXISTS idx_positions_frozen ON positions(oms_id, frozen) WHERE frozen = TRUE;
CREATE INDEX IF NOT EXISTS idx_positions_updated ON positions(oms_id, last_update_at DESC);
CREATE INDEX IF NOT EXISTS idx_allocations_strategy ON allocations(oms_id, strategy_id) WHERE qty > 0;

CREATE INDEX IF NOT EXISTS idx_orders_oms ON orders(oms_id);
CREATE INDEX IF NOT EXISTS idx_intents_oms ON intents(oms_id);
CREATE INDEX IF NOT EXISTS idx_fills_oms ON fills(oms_id);
CREATE INDEX IF NOT EXISTS idx_trades_oms ON trades(oms_id);
CREATE INDEX IF NOT EXISTS idx_order_events_oms ON order_events(oms_id);
CREATE INDEX IF NOT EXISTS idx_recon_log_oms ON recon_log(oms_id);
CREATE INDEX IF NOT EXISTS idx_risk_daily_strategy_oms ON risk_daily_strategy(oms_id);
CREATE INDEX IF NOT EXISTS idx_strategy_state_oms ON strategy_state(oms_id);
