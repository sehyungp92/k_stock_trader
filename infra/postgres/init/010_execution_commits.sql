-- One commit boundary for execution accounting and the consumer stream.
CREATE TABLE IF NOT EXISTS execution_commits (
    oms_id VARCHAR(20) NOT NULL,
    identity TEXT NOT NULL,
    signature TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (oms_id, identity)
);
CREATE TABLE IF NOT EXISTS execution_feedback (
    sequence BIGSERIAL PRIMARY KEY,
    oms_id VARCHAR(20) NOT NULL,
    identity TEXT NOT NULL,
    strategy_id VARCHAR(20) NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (oms_id, identity)
);
CREATE INDEX IF NOT EXISTS execution_feedback_consumer
    ON execution_feedback (oms_id, strategy_id, sequence);
CREATE TABLE IF NOT EXISTS intent_effects (
    oms_id VARCHAR(20) NOT NULL,
    identity TEXT NOT NULL,
    signature TEXT NOT NULL,
    PRIMARY KEY (oms_id, identity)
);
