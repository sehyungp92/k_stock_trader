#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
DB_SERVICE="${DB_SERVICE:-postgres}"
DB_NAME="${DB_NAME:-trading}"
DB_USER="${DB_USER:-postgres}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is required to apply Postgres migrations" >&2
  exit 1
}

docker compose version >/dev/null 2>&1 || {
  echo "docker compose is required to apply Postgres migrations" >&2
  exit 1
}

cd "$PROJECT_DIR"

for sql in \
  "infra/postgres/init/005_oms_scoping.sql" \
  "infra/postgres/init/006_views_oms_scoped.sql" \
  "infra/postgres/init/007_oms_scope_finalize.sql"
do
  [ -f "$sql" ] || {
    echo "Missing migration file: $sql" >&2
    exit 1
  }
  echo "Applying $sql via $COMPOSE_FILE..."
  docker compose -f "$COMPOSE_FILE" exec -T "$DB_SERVICE" \
    psql -U "$DB_USER" -d "$DB_NAME" < "$sql"
done

echo "Scoped Postgres migrations applied successfully."
