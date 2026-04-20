-- ============================================================
-- 007: Finalize OMS-scoped primary keys for shared DB runtime
-- Idempotent: safe to run on already-upgraded databases
-- ============================================================

ALTER TABLE risk_daily_strategy
    ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';

DO $$
DECLARE
  current_pk TEXT;
BEGIN
  SELECT tc.constraint_name
    INTO current_pk
  FROM information_schema.table_constraints tc
  WHERE tc.table_schema = 'public'
    AND tc.table_name = 'risk_daily_strategy'
    AND tc.constraint_type = 'PRIMARY KEY';

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.key_column_usage
    WHERE table_schema = 'public'
      AND table_name = 'risk_daily_strategy'
      AND constraint_name = current_pk
      AND column_name = 'oms_id'
  ) THEN
    IF current_pk IS NOT NULL THEN
      EXECUTE format('ALTER TABLE risk_daily_strategy DROP CONSTRAINT %I', current_pk);
    END IF;
    ALTER TABLE risk_daily_strategy ADD PRIMARY KEY (oms_id, trade_date, strategy_id);
  END IF;
END
$$;

ALTER TABLE strategy_state
    ADD COLUMN IF NOT EXISTS oms_id VARCHAR(20) NOT NULL DEFAULT 'primary';

DO $$
DECLARE
  current_pk TEXT;
BEGIN
  SELECT tc.constraint_name
    INTO current_pk
  FROM information_schema.table_constraints tc
  WHERE tc.table_schema = 'public'
    AND tc.table_name = 'strategy_state'
    AND tc.constraint_type = 'PRIMARY KEY';

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.key_column_usage
    WHERE table_schema = 'public'
      AND table_name = 'strategy_state'
      AND constraint_name = current_pk
      AND column_name = 'oms_id'
  ) THEN
    IF current_pk IS NOT NULL THEN
      EXECUTE format('ALTER TABLE strategy_state DROP CONSTRAINT %I', current_pk);
    END IF;
    ALTER TABLE strategy_state ADD PRIMARY KEY (oms_id, strategy_id);
  END IF;
END
$$;
