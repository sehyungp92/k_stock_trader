-- ============================================================
-- 006: Recreate views with oms_id exposed
-- DROP first because adding/reordering columns is incompatible
-- with CREATE OR REPLACE on existing views from 003.
-- Safe to re-run (IF EXISTS).
-- ============================================================

-- Drop all views that gain or reorder columns in this migration.
-- CASCADE is not needed — none of these views depend on each other.
DROP VIEW IF EXISTS v_live_positions;
DROP VIEW IF EXISTS v_live_allocations;
DROP VIEW IF EXISTS v_working_orders;
DROP VIEW IF EXISTS v_today_intents;
DROP VIEW IF EXISTS v_today_risk;
DROP VIEW IF EXISTS v_active_halts;
DROP VIEW IF EXISTS v_recent_trades;
DROP VIEW IF EXISTS v_strategy_performance;
DROP VIEW IF EXISTS v_fill_quality;
DROP VIEW IF EXISTS v_service_health;

-- ==================================================
-- Live Operations Views
-- ==================================================

CREATE OR REPLACE VIEW v_live_positions AS
SELECT
    p.oms_id,
    p.symbol,
    p.real_qty,
    p.avg_price,
    p.hard_stop_px,
    p.frozen,
    p.entry_lock_owner,
    p.vi_cooldown_until,
    COALESCE(a.pcim_qty, 0) AS pcim_qty,
    p.real_qty - COALESCE(a.total_alloc, 0) AS drift,
    p.last_update_at
FROM positions p
LEFT JOIN (
    SELECT
        oms_id,
        symbol,
        SUM(qty) AS total_alloc,
        SUM(CASE WHEN strategy_id = 'PCIM' THEN qty ELSE 0 END) AS pcim_qty
    FROM allocations
    GROUP BY oms_id, symbol
) a ON p.oms_id = a.oms_id AND p.symbol = a.symbol
WHERE p.real_qty != 0 OR p.frozen = TRUE
ORDER BY p.oms_id, p.symbol;

CREATE OR REPLACE VIEW v_live_allocations AS
SELECT
    p.oms_id,
    p.symbol,
    a.strategy_id,
    a.qty,
    COALESCE(a.cost_basis, p.avg_price) AS avg_price,
    a.entry_ts,
    a.soft_stop_px,
    p.hard_stop_px,
    p.frozen,
    p.real_qty - totals.total_alloc AS drift,
    p.last_update_at
FROM allocations a
JOIN positions p
  ON p.oms_id = a.oms_id
 AND p.symbol = a.symbol
JOIN (
    SELECT
        oms_id,
        symbol,
        SUM(qty) AS total_alloc
    FROM allocations
    GROUP BY oms_id, symbol
) totals
  ON totals.oms_id = p.oms_id
 AND totals.symbol = p.symbol
WHERE a.qty > 0
ORDER BY p.oms_id, p.symbol, a.strategy_id;

CREATE OR REPLACE VIEW v_working_orders AS
SELECT
    o.oms_id,
    o.oms_order_id,
    o.strategy_id,
    o.symbol,
    o.side,
    o.order_type,
    o.qty,
    o.filled_qty,
    o.qty - o.filled_qty AS remaining_qty,
    o.limit_price,
    o.stop_price,
    o.status,
    o.kis_order_id,
    EXTRACT(EPOCH FROM (NOW() - o.created_at))::INTEGER AS age_seconds,
    o.cancel_after_sec,
    o.created_at,
    o.last_update_at
FROM orders o
WHERE o.status IN ('CREATED', 'SUBMITTING', 'WORKING', 'PARTIAL')
ORDER BY o.created_at DESC;

CREATE OR REPLACE VIEW v_today_intents AS
SELECT
    i.oms_id,
    i.strategy_id,
    i.intent_type,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE i.status = 'EXECUTED') AS executed,
    COUNT(*) FILTER (WHERE i.status = 'REJECTED') AS rejected,
    COUNT(*) FILTER (WHERE i.status = 'DEFERRED') AS deferred
FROM intents i
WHERE i.created_at >= CURRENT_DATE AT TIME ZONE 'Asia/Seoul'
GROUP BY i.oms_id, i.strategy_id, i.intent_type
ORDER BY i.oms_id, i.strategy_id, i.intent_type;

-- ==================================================
-- Risk Views
-- ==================================================

CREATE OR REPLACE VIEW v_today_risk AS
SELECT
    p.oms_id,
    'PORTFOLIO' AS entity,
    NULL::VARCHAR(20) AS strategy_id,
    p.equity_krw,
    p.realized_pnl_krw,
    p.unrealized_pnl_krw,
    p.daily_pnl_pct,
    p.gross_exposure_pct,
    p.positions_count,
    p.halted,
    p.halt_reason,
    p.regime,
    p.safe_mode,
    p.last_update_at
FROM risk_daily_portfolio p
WHERE p.trade_date = CURRENT_DATE

UNION ALL

SELECT
    s.oms_id,
    'STRATEGY' AS entity,
    s.strategy_id,
    NULL AS equity_krw,
    s.realized_pnl_krw,
    s.unrealized_pnl_krw,
    NULL AS daily_pnl_pct,
    NULL::NUMERIC(8,4) AS gross_exposure_pct,
    s.trades_count AS positions_count,
    s.halted,
    s.halt_reason,
    NULL AS regime,
    FALSE AS safe_mode,
    s.last_update_at
FROM risk_daily_strategy s
WHERE s.trade_date = CURRENT_DATE
ORDER BY entity DESC, strategy_id;

CREATE OR REPLACE VIEW v_active_halts AS
SELECT
    p.oms_id,
    'PORTFOLIO' AS entity,
    NULL::VARCHAR(20) AS strategy_id,
    p.halt_reason,
    p.halt_ts,
    p.safe_mode,
    p.flatten_triggered
FROM risk_daily_portfolio p
WHERE p.trade_date = CURRENT_DATE AND (p.halted OR p.safe_mode OR p.flatten_triggered)

UNION ALL

SELECT
    s.oms_id,
    'STRATEGY' AS entity,
    s.strategy_id,
    s.halt_reason,
    s.halt_ts,
    FALSE AS safe_mode,
    FALSE AS flatten_triggered
FROM risk_daily_strategy s
WHERE s.trade_date = CURRENT_DATE AND s.halted;

-- ==================================================
-- Analytics Views
-- ==================================================

CREATE OR REPLACE VIEW v_recent_trades AS
SELECT
    t.oms_id,
    t.trade_id,
    t.strategy_id,
    t.symbol,
    t.direction,
    t.entry_qty,
    t.entry_price,
    t.entry_ts,
    t.exit_price,
    t.exit_ts,
    t.exit_reason,
    t.realized_pnl_krw,
    t.realized_r,
    t.setup_type,
    t.confidence,
    tm.duration_seconds,
    tm.mae_pct,
    tm.mfe_pct,
    tm.capture_ratio,
    t.status
FROM trades t
LEFT JOIN trade_marks tm ON t.trade_id = tm.trade_id
WHERE t.entry_ts >= NOW() - INTERVAL '7 days'
ORDER BY t.entry_ts DESC;

CREATE OR REPLACE VIEW v_strategy_performance AS
SELECT
    oms_id,
    strategy_id,
    COUNT(*) AS total_trades,
    COUNT(*) FILTER (WHERE status = 'CLOSED') AS closed_trades,
    SUM(realized_pnl_krw) FILTER (WHERE status = 'CLOSED') AS total_pnl_krw,
    AVG(realized_r) FILTER (WHERE status = 'CLOSED') AS avg_r,
    COUNT(*) FILTER (WHERE realized_pnl_krw > 0 AND status = 'CLOSED') AS wins,
    COUNT(*) FILTER (WHERE realized_pnl_krw <= 0 AND status = 'CLOSED') AS losses,
    ROUND(100.0 * COUNT(*) FILTER (WHERE realized_pnl_krw > 0 AND status = 'CLOSED') /
          NULLIF(COUNT(*) FILTER (WHERE status = 'CLOSED'), 0), 1) AS win_rate_pct
FROM trades
WHERE entry_ts >= NOW() - INTERVAL '30 days'
GROUP BY oms_id, strategy_id;

CREATE OR REPLACE VIEW v_fill_quality AS
SELECT
    f.oms_id,
    f.strategy_id,
    f.symbol,
    o.order_type,
    o.side,
    COUNT(*) AS fill_count,
    AVG(CASE
        WHEN o.side = 'BUY' THEN (f.price - o.limit_price) / o.limit_price * 10000
        ELSE (o.limit_price - f.price) / o.limit_price * 10000
    END) AS avg_slippage_bps,
    MAX(CASE
        WHEN o.side = 'BUY' THEN (f.price - o.limit_price) / o.limit_price * 10000
        ELSE (o.limit_price - f.price) / o.limit_price * 10000
    END) AS max_slippage_bps
FROM fills f
JOIN orders o ON f.oms_order_id = o.oms_order_id
WHERE f.fill_ts >= NOW() - INTERVAL '7 days'
  AND o.limit_price IS NOT NULL
  AND o.limit_price > 0
GROUP BY f.oms_id, f.strategy_id, f.symbol, o.order_type, o.side;

-- ==================================================
-- Service Health Views
-- ==================================================

CREATE OR REPLACE VIEW v_service_health AS
SELECT
    'OMS' AS service,
    o.oms_id,
    o.oms_id AS instance,
    o.last_heartbeat_ts,
    EXTRACT(EPOCH FROM (NOW() - o.last_heartbeat_ts))::INTEGER AS seconds_since_heartbeat,
    CASE
        WHEN o.last_heartbeat_ts > NOW() - INTERVAL '60 seconds' THEN 'HEALTHY'
        WHEN o.last_heartbeat_ts > NOW() - INTERVAL '300 seconds' THEN 'WARNING'
        ELSE 'CRITICAL'
    END AS health,
    o.safe_mode,
    o.kis_connected,
    o.recon_status,
    o.version
FROM oms_state o

UNION ALL

SELECT
    'STRATEGY' AS service,
    s.oms_id,
    s.strategy_id AS instance,
    s.last_heartbeat_ts,
    EXTRACT(EPOCH FROM (NOW() - s.last_heartbeat_ts))::INTEGER AS seconds_since_heartbeat,
    CASE
        WHEN s.mode = 'STOPPED' THEN 'STOPPED'
        WHEN s.last_heartbeat_ts > NOW() - INTERVAL '60 seconds' THEN 'HEALTHY'
        WHEN s.last_heartbeat_ts > NOW() - INTERVAL '300 seconds' THEN 'WARNING'
        ELSE 'CRITICAL'
    END AS health,
    s.mode = 'PAUSED' AS safe_mode,
    TRUE AS kis_connected,
    NULL AS recon_status,
    s.version
FROM strategy_state s
ORDER BY service, instance;
