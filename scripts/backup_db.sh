#!/bin/bash
# Database backup script — run on VPS 2 only (hosts PostgreSQL)
# Scheduled via cron: 0 4 * * * /opt/k_stock_trader/scripts/backup_db.sh

BACKUP_DIR="/opt/k_stock_trader/backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

docker exec trading_db pg_dump -U postgres -d trading --format=custom \
  > "$BACKUP_DIR/trading_${TIMESTAMP}.dump"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "trading_*.dump" -mtime +30 -delete

echo "$(date): Backup complete — trading_${TIMESTAMP}.dump" >> /var/log/db_backup.log
