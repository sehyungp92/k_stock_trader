-- Daily data retention cleanup job
-- Run via cron or scheduled task
-- Order: child tables first to avoid FK violations

-- Delete old order events (60 days)
DELETE FROM order_events WHERE event_ts < NOW() - INTERVAL '60 days';

-- Delete old recon logs (30 days)
DELETE FROM recon_log WHERE recon_ts < NOW() - INTERVAL '30 days';

-- Delete old fills (365 days) - must delete before orders due to FK
DELETE FROM fills WHERE fill_ts < NOW() - INTERVAL '365 days';

-- Delete old completed orders (365 days)
DELETE FROM orders WHERE status IN ('FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'FAILED')
    AND last_update_at < NOW() - INTERVAL '365 days';

-- Clear FK references before deleting intents (orders/trades may reference longer)
UPDATE orders SET intent_id = NULL
WHERE intent_id IN (SELECT intent_id FROM intents WHERE created_at < NOW() - INTERVAL '90 days');

UPDATE trades SET entry_intent_id = NULL
WHERE entry_intent_id IN (SELECT intent_id FROM intents WHERE created_at < NOW() - INTERVAL '90 days');

UPDATE trades SET exit_intent_id = NULL
WHERE exit_intent_id IN (SELECT intent_id FROM intents WHERE created_at < NOW() - INTERVAL '90 days');

-- Delete old intents (90 days) - now safe after FK nullification
DELETE FROM intents WHERE created_at < NOW() - INTERVAL '90 days';

-- Vacuum after large deletes (run during low activity)
VACUUM ANALYZE order_events;
VACUUM ANALYZE recon_log;
VACUUM ANALYZE fills;
VACUUM ANALYZE orders;
VACUUM ANALYZE intents;
