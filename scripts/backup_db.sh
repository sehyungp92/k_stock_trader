#!/bin/bash
# Single-VPS Postgres backup script.
# Scheduled via cron: 0 4 * * * /opt/k_stock_trader/scripts/backup_db.sh

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
DB_SERVICE="${DB_SERVICE:-postgres}"
DB_NAME="${DB_NAME:-trading}"
DB_USER="${DB_USER:-postgres}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups}"

mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_DIR"

docker compose -f "$COMPOSE_FILE" exec -T "$DB_SERVICE" \
  pg_dump -U "$DB_USER" -d "$DB_NAME" --format=custom \
  > "$BACKUP_DIR/trading_${TIMESTAMP}.dump"

# Keep only last 30 days of backups.
find "$BACKUP_DIR" -name "trading_*.dump" -mtime +30 -delete

echo "$(date): Backup complete - trading_${TIMESTAMP}.dump" >> /var/log/db_backup.log
