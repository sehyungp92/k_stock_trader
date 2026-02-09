-- ==================================================
-- OMS Core Tables
-- ==================================================

-- intents: Strategy intent log (append-only audit trail)
CREATE TABLE intents (
    intent_id UUID PRIMARY KEY,
    idempotency_key VARCHAR(255) NOT NULL UNIQUE,
    strategy_id VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    intent_type VARCHAR(20) NOT NULL,
    desired_qty INTEGER,
    target_qty INTEGER,
    urgency VARCHAR(10) NOT NULL,
    time_horizon VARCHAR(10) NOT NULL,
    max_slippage_bps NUMERIC(8,2),
    max_spread_bps NUMERIC(8,2),
    limit_price NUMERIC(18,4),
    stop_price NUMERIC(18,4),
    expiry_ts TIMESTAMPTZ,
    entry_px NUMERIC(18,4),
    stop_px NUMERIC(18,4),
    hard_stop_px NUMERIC(18,4),
    rationale_code VARCHAR(50),
    confidence VARCHAR(10),
    signal_hash VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    result_message TEXT,
    modified_qty INTEGER,
    order_id VARCHAR(50),
    cooldown_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_intents_strategy_created ON intents(strategy_id, created_at DESC);
CREATE INDEX idx_intents_symbol_created ON intents(symbol, created_at DESC);
CREATE INDEX idx_intents_status_created ON intents(status, created_at DESC);

-- orders: Current order state
CREATE TABLE orders (
    oms_order_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    qty INTEGER NOT NULL,
    filled_qty INTEGER NOT NULL DEFAULT 0,
    limit_price NUMERIC(18,4),
    stop_price NUMERIC(18,4),
    avg_fill_price NUMERIC(18,4),
    status VARCHAR(20) NOT NULL,
    kis_order_id VARCHAR(50),
    kis_order_date VARCHAR(8),
    intent_id UUID REFERENCES intents(intent_id),
    cancel_after_sec INTEGER,
    max_chase_bps NUMERIC(8,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta JSONB
);

CREATE INDEX idx_orders_status_updated ON orders(status, last_update_at DESC);
CREATE INDEX idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX idx_orders_strategy_created ON orders(strategy_id, created_at DESC);
CREATE INDEX idx_orders_kis_order ON orders(kis_order_id) WHERE kis_order_id IS NOT NULL;

-- order_events: Append-only event log
CREATE TABLE order_events (
    event_id BIGSERIAL PRIMARY KEY,
    oms_order_id UUID REFERENCES orders(oms_order_id),
    intent_id UUID REFERENCES intents(intent_id),
    strategy_id VARCHAR(20),
    symbol VARCHAR(20),
    event_type VARCHAR(50) NOT NULL,
    event_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB,
    status_before VARCHAR(20),
    status_after VARCHAR(20)
);

CREATE INDEX idx_order_events_ts ON order_events(event_ts DESC);
CREATE INDEX idx_order_events_order_ts ON order_events(oms_order_id, event_ts DESC);
CREATE INDEX idx_order_events_type_ts ON order_events(event_type, event_ts DESC);
CREATE INDEX idx_order_events_strategy_ts ON order_events(strategy_id, event_ts DESC);

-- fills: Deduplicated fill records
CREATE TABLE fills (
    fill_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kis_exec_id VARCHAR(100) NOT NULL UNIQUE,
    oms_order_id UUID REFERENCES orders(oms_order_id),
    strategy_id VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    qty INTEGER NOT NULL,
    price NUMERIC(18,4) NOT NULL,
    commission NUMERIC(18,4),
    tax NUMERIC(18,4),
    fill_ts TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_fills_symbol_ts ON fills(symbol, fill_ts DESC);
CREATE INDEX idx_fills_strategy_ts ON fills(strategy_id, fill_ts DESC);
CREATE INDEX idx_fills_order ON fills(oms_order_id);

-- ==================================================
-- Position & Allocation Tables
-- ==================================================

-- positions: Real broker positions
CREATE TABLE positions (
    symbol VARCHAR(20) PRIMARY KEY,
    real_qty INTEGER NOT NULL DEFAULT 0,
    avg_price NUMERIC(18,4) NOT NULL DEFAULT 0,
    current_price NUMERIC(18,4),
    unrealized_pnl NUMERIC(18,4),
    hard_stop_px NUMERIC(18,4),
    entry_lock_owner VARCHAR(20),
    entry_lock_until TIMESTAMPTZ,
    cooldown_until TIMESTAMPTZ,
    vi_cooldown_until TIMESTAMPTZ,
    frozen BOOLEAN NOT NULL DEFAULT FALSE,
    last_broker_sync_at TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_positions_frozen ON positions(frozen) WHERE frozen = TRUE;
CREATE INDEX idx_positions_updated ON positions(last_update_at DESC);

-- allocations: Virtual strategy allocations
CREATE TABLE allocations (
    symbol VARCHAR(20) NOT NULL,
    strategy_id VARCHAR(20) NOT NULL,
    qty INTEGER NOT NULL DEFAULT 0,
    cost_basis NUMERIC(18,4),
    entry_ts TIMESTAMPTZ,
    soft_stop_px NUMERIC(18,4),
    time_stop_ts TIMESTAMPTZ,
    entry_intent_id UUID REFERENCES intents(intent_id),
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, strategy_id)
);

CREATE INDEX idx_allocations_strategy ON allocations(strategy_id) WHERE qty > 0;

-- ==================================================
-- Risk Tables
-- ==================================================

-- risk_daily_strategy: Per-strategy daily risk
CREATE TABLE risk_daily_strategy (
    trade_date DATE NOT NULL,
    strategy_id VARCHAR(20) NOT NULL,
    realized_pnl_krw NUMERIC(18,0),
    realized_pnl_r NUMERIC(10,4),
    unrealized_pnl_krw NUMERIC(18,0),
    trades_count INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    max_drawdown_pct NUMERIC(8,4),
    peak_exposure_krw NUMERIC(18,0),
    peak_exposure_pct NUMERIC(8,4),
    halted BOOLEAN NOT NULL DEFAULT FALSE,
    halt_reason TEXT,
    halt_ts TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (trade_date, strategy_id)
);

-- risk_daily_portfolio: Portfolio-level daily risk
CREATE TABLE risk_daily_portfolio (
    trade_date DATE PRIMARY KEY,
    equity_krw NUMERIC(18,0),
    buyable_cash_krw NUMERIC(18,0),
    realized_pnl_krw NUMERIC(18,0),
    unrealized_pnl_krw NUMERIC(18,0),
    daily_pnl_pct NUMERIC(8,4),
    gross_exposure_krw NUMERIC(18,0),
    gross_exposure_pct NUMERIC(8,4),
    net_exposure_krw NUMERIC(18,0),
    net_exposure_pct NUMERIC(8,4),
    positions_count INTEGER DEFAULT 0,
    halted BOOLEAN NOT NULL DEFAULT FALSE,
    halt_reason TEXT,
    halt_ts TIMESTAMPTZ,
    safe_mode BOOLEAN NOT NULL DEFAULT FALSE,
    flatten_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    regime VARCHAR(20),
    regime_set_at TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ==================================================
-- Trade Analytics Tables
-- ==================================================

-- trades: Completed trade records
CREATE TABLE trades (
    trade_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(5) NOT NULL,
    entry_qty INTEGER NOT NULL,
    entry_price NUMERIC(18,4) NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL,
    entry_intent_id UUID REFERENCES intents(intent_id),
    exit_qty INTEGER,
    exit_price NUMERIC(18,4),
    exit_ts TIMESTAMPTZ,
    exit_intent_id UUID REFERENCES intents(intent_id),
    exit_reason VARCHAR(30),
    realized_pnl_krw NUMERIC(18,0),
    realized_pnl_pct NUMERIC(8,4),
    realized_r NUMERIC(10,4),
    setup_type VARCHAR(30),
    confidence VARCHAR(10),
    vwap_depth_pct NUMERIC(8,4),
    investor_signal VARCHAR(20),
    micro_signal VARCHAR(20),
    program_signal VARCHAR(20),
    meta JSONB,
    status VARCHAR(10) NOT NULL DEFAULT 'OPEN',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

CREATE INDEX idx_trades_strategy_entry ON trades(strategy_id, entry_ts DESC);
CREATE INDEX idx_trades_symbol_entry ON trades(symbol, entry_ts DESC);
CREATE INDEX idx_trades_status ON trades(status) WHERE status = 'OPEN';
CREATE INDEX idx_trades_exit_reason ON trades(exit_reason, exit_ts DESC);

-- trade_marks: MAE/MFE analytics
CREATE TABLE trade_marks (
    trade_id UUID PRIMARY KEY REFERENCES trades(trade_id),
    duration_seconds INTEGER,
    bars_in_trade INTEGER,
    mae_pct NUMERIC(8,4),
    mfe_pct NUMERIC(8,4),
    mae_r NUMERIC(10,4),
    mfe_r NUMERIC(10,4),
    max_adverse_price NUMERIC(18,4),
    max_favorable_price NUMERIC(18,4),
    capture_ratio NUMERIC(8,4),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ==================================================
-- Service State Tables
-- ==================================================

-- strategy_state: Strategy heartbeat & status
CREATE TABLE strategy_state (
    strategy_id VARCHAR(20) PRIMARY KEY,
    last_heartbeat_ts TIMESTAMPTZ,
    heartbeat_interval_sec INTEGER DEFAULT 30,
    mode VARCHAR(20) NOT NULL DEFAULT 'STOPPED',
    pause_reason TEXT,
    symbols_hot INTEGER DEFAULT 0,
    symbols_warm INTEGER DEFAULT 0,
    symbols_cold INTEGER DEFAULT 0,
    positions_count INTEGER DEFAULT 0,
    last_intent_ts TIMESTAMPTZ,
    last_fill_ts TIMESTAMPTZ,
    last_error_ts TIMESTAMPTZ,
    last_error TEXT,
    version VARCHAR(20),
    started_at TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- oms_state: OMS service state
CREATE TABLE oms_state (
    oms_id VARCHAR(20) PRIMARY KEY DEFAULT 'primary',
    last_heartbeat_ts TIMESTAMPTZ,
    safe_mode BOOLEAN NOT NULL DEFAULT FALSE,
    halt_new_entries BOOLEAN NOT NULL DEFAULT FALSE,
    flatten_in_progress BOOLEAN NOT NULL DEFAULT FALSE,
    equity_krw NUMERIC(18,0),
    buyable_cash_krw NUMERIC(18,0),
    daily_pnl_krw NUMERIC(18,0),
    daily_pnl_pct NUMERIC(8,4),
    last_recon_ts TIMESTAMPTZ,
    recon_status VARCHAR(10),
    allocation_drift_count INTEGER DEFAULT 0,
    kis_connected BOOLEAN DEFAULT FALSE,
    kis_last_ping_ts TIMESTAMPTZ,
    kis_token_expires_at TIMESTAMPTZ,
    version VARCHAR(20),
    started_at TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO oms_state (oms_id) VALUES ('primary') ON CONFLICT DO NOTHING;

-- recon_log: Reconciliation events
CREATE TABLE recon_log (
    recon_id BIGSERIAL PRIMARY KEY,
    recon_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    recon_type VARCHAR(30) NOT NULL,
    symbol VARCHAR(20),
    strategy_id VARCHAR(20),
    before_value JSONB,
    after_value JSONB,
    action VARCHAR(30),
    details TEXT
);

CREATE INDEX idx_recon_log_ts ON recon_log(recon_ts DESC);
CREATE INDEX idx_recon_log_symbol ON recon_log(symbol, recon_ts DESC);
