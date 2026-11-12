-- Retain this row across ordinary restarts. A rebuilt ledger needs a new UUID.
CREATE TABLE IF NOT EXISTS execution_generations (
    oms_id VARCHAR(20) PRIMARY KEY,
    generation UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO execution_generations(oms_id,generation)
SELECT oms_id,uuid_generate_v4() FROM (
    SELECT DISTINCT oms_id FROM execution_feedback UNION SELECT DISTINCT oms_id FROM allocations
) AS accounts ON CONFLICT(oms_id) DO NOTHING;
