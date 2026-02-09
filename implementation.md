# 2-VPS Deployment Guide: k_stock_trader Paper Trading

## Overview

Deploy the k_stock_trader system across 2 VPS instances with 2 KIS accounts for paper trading, maximizing WebSocket capacity and minimizing strategy interference.

### Architecture Summary

```
VPS 1 (Account 1)                    VPS 2/3 (Account 2)
┌─────────────────────┐              ┌─────────────────────┐
│  KMP Strategy       │              │  KPR Strategy       │
│  (40 WS 09:15-10:00)│              │  (40 WS all day)    │
├─────────────────────┤              ├─────────────────────┤
│  Nulrimok Strategy  │              │  PCIM Strategy      │
│  (40 WS after 10:00)│              │  (0 WS, REST-only)  │
├─────────────────────┤              ├─────────────────────┤
│  OMS Instance 1     │──── TCP ────►│  OMS Instance 2     │
│  (PostgreSQL remote)│   :5432      │  (PostgreSQL local) │
├─────────────────────┤              ├─────────────────────┤
│  SQLite (Nulrimok)  │              │  PostgreSQL         │◄── Shared DB
│  lrs.db (auto-pop)  │              │  + Metabase         │◄── Shared UI
└─────────────────────┘              └─────────────────────┘
         │                                      │
         └──────────────────┬───────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │   KIS Paper API     │
                 │   (Per Account)     │
                 │   40 WS + REST each │
                 └─────────────────────┘
```

### Why This Split?

| Factor | Benefit |
|--------|---------|
| **Zero WS time-sharing** | KMP->Nulrimok natural handoff at 10:00; PCIM uses 0 WS |
| **Shared PostgreSQL** | Both OMS instances log to VPS 2's Postgres; single Metabase dashboard for all 4 strategies |
| **Nulrimok LRS isolated** | SQLite stays local, no network dependency for strategy data |
| **KPR exclusive WS** | Full 40 slots all day without contention |

### Credential Model

Both VPSes use **paper trading by default** (`KIS_IS_PAPER=true`). Each VPS has:

- **Separate paper credentials** (`KIS_PAPER_APP_KEY`, `KIS_PAPER_APP_SECRET`, `KIS_PAPER_ACCOUNT_NO`) — isolates order execution so VPS-1 trades don't interfere with VPS-2
- **Shared real credentials** (`KIS_APP_KEY`, `KIS_APP_SECRET`) — used only as a read-only fallback for data endpoints the paper server doesn't support (e.g., program trading `FHPPG04650101`)

This is safe because the real credentials are only used for market data reads, never for orders.

---

## Phase 1: Prerequisites

### 1.1 Obtain Second KIS Paper Trading Account

1. **Log into KIS Developer Portal:** https://apiportal.koreainvestment.com
2. **Request second app credentials:**
   - Navigate to "마이페이지" -> "앱 관리"
   - Create new application for paper trading
   - Note: `APP_KEY_2`, `APP_SECRET_2`, `ACCOUNT_NO_2`
3. **Verify both accounts have paper trading enabled**

### 1.2 Provision VPS Instances

**Recommended Specifications:**

| Component | VPS 1 (KMP+Nulrimok) | VPS 2 (KPR+PCIM+DB) |
|-----------|----------------------|---------------------|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Storage | 40 GB SSD | 80 GB SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Location | Korea (Seoul) | Korea (Seoul) |

### 1.3 Install Dependencies on Both VPS

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for group to take effect

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Install git
sudo apt install git -y

# Verify
docker --version
docker compose version
```

**Status: COMPLETED**

---

## Phase 2: VPS 2 Setup (KPR + PCIM + Database)

**Start with VPS 2 because it hosts the database.**

### 2.1 Clone Repository

```bash
cd /opt
sudo git clone https://github.com/sehyungp92/k_stock_trader.git
sudo chown -R $USER:$USER k_stock_trader
cd k_stock_trader
```

### 2.2 Configure Environment Variables

```bash
cp .env.example .env
nano .env
```

**VPS 2 `.env` Configuration:**

```bash
# ============================================
# VPS 2: KPR + PCIM (Account 2)
# ============================================

# Real API credentials (shared across VPSes, used only for data fallback)
KIS_APP_KEY=your_real_app_key
KIS_APP_SECRET=your_real_app_secret

# Primary account info
KIS_ACCOUNT_NO=your_account2_number
KIS_ACCOUNT_PROD_CODE=01
KIS_HTS_ID=your_hts_id_2
KIS_IS_PAPER=true
KIS_MY_AGENT=Mozilla/5.0

# Paper trading credentials (Account 2 — separate from VPS 1)
KIS_PAPER_APP_KEY=your_paper_app_key_2
KIS_PAPER_APP_SECRET=your_paper_app_secret_2
KIS_PAPER_ACCOUNT_NO=your_paper_account_2

# Database
POSTGRES_PASSWORD=secure_admin_password_here
POSTGRES_WRITER_PASSWORD=secure_writer_password_here
POSTGRES_READER_PASSWORD=secure_reader_password_here

# PCIM
GEMINI_API_KEY=your_gemini_api_key
```

### 2.3 Start VPS 2 Services

```bash
sudo docker compose -f docker-compose.vps2.yml up -d --build

# Verify
sudo docker compose -f docker-compose.vps2.yml ps
curl http://localhost:8000/health
```

**Expected startup behavior:**
- OMS starts first, acquires KIS token, begins reconciliation loop
- KPR and PCIM start after OMS health check passes
- Token rate-limiting (`EGW00133`) is normal — KIS limits to 1 token/minute, so the second and third containers retry after 65s
- KPR runs universe filter on startup (~2 min of rate-limited API calls, then settles)
- OMS reconciliation loop shows `inquire-balance` rate-limit warnings after hours — this is normal and does not affect trading

### 2.4 Configure Firewall (VPS 2)

```bash
sudo ufw allow 22/tcp
# Allow Metabase (restrict to your IP in production)
sudo ufw allow 3000/tcp
# Allow VPS 1 OMS to connect to Postgres for logging
sudo ufw allow from VPS1_IP to any port 5432
sudo ufw enable
```

> **Important:** Replace `VPS1_IP` with VPS 1's actual IP address. This allows VPS 1's OMS to write intents/trades to the shared PostgreSQL database.

**Status: COMPLETED**

---

## Phase 3: VPS 1 Setup (KMP + Nulrimok)

### 3.1 Clone Repository

```bash
cd /opt
sudo git clone https://github.com/sehyungp92/k_stock_trader.git
sudo chown -R $USER:$USER k_stock_trader
cd k_stock_trader
```

### 3.2 Configure Environment Variables

```bash
cp .env.example .env
nano .env
```

**VPS 1 `.env` Configuration:**

```bash
# ============================================
# VPS 1: KMP + Nulrimok (Account 1)
# ============================================

# Real API credentials (shared across VPSes, used only for data fallback)
KIS_APP_KEY=your_real_app_key
KIS_APP_SECRET=your_real_app_secret

# Primary account info
KIS_ACCOUNT_NO=your_account1_number
KIS_ACCOUNT_PROD_CODE=01
KIS_HTS_ID=your_hts_id_1
KIS_IS_PAPER=true
KIS_MY_AGENT=Mozilla/5.0

# Paper trading credentials (Account 1 — separate from VPS 2)
KIS_PAPER_APP_KEY=your_paper_app_key_1
KIS_PAPER_APP_SECRET=your_paper_app_secret_1
KIS_PAPER_ACCOUNT_NO=your_paper_account_1

# Database — connect to VPS 2's Postgres for shared logging
DATABASE_URL=postgresql://trading_writer:writer_password@VPS2_IP:5432/trading
```

> **Important:** Replace `VPS2_IP` with VPS 2's actual IP address, and `writer_password` with the `POSTGRES_WRITER_PASSWORD` value from VPS 2's `.env`. This allows VPS 1's OMS to log intents and trades to the same database used by VPS 2, enabling a unified Metabase dashboard across both VPSes.

### 3.3 Start VPS 1 Services

```bash
sudo docker compose -f docker-compose.vps1.yml up -d --build

# Verify
sudo docker compose -f docker-compose.vps1.yml ps
curl http://localhost:8000/health
```

**Expected output:**
```json
{"status":"ok","uptime_sec":157.4,"positions_count":0,"kis_circuit_breaker":"CLOSED","recon_status":"OK"}
```

**Expected startup behavior:**
- OMS starts and connects to VPS 2's PostgreSQL over the network. If the connection fails (e.g., firewall not open, VPS 2 down), OMS falls back to in-memory state and logs `Postgres connection failed (will retry)`.
- KMP and Nulrimok start after OMS health check passes
- Token rate-limiting on startup is normal (same as VPS 2)

### 3.4 Nulrimok LRS Auto-Population

The Nulrimok LRS (Local Research Store) SQLite database is **populated automatically** from the KIS API. No manual seeding is required.

**How it works:**

1. On container start, `populate_lrs()` runs before the DSE (Daily Selection Engine)
2. It checks `MAX(date)` in the `daily_ohlcv` table:
   - **Empty DB (first boot):** Fetches all data — KOSPI index (280 days), daily OHLCV (60 days x 413 tickers), investor flow (20 days x 413 tickers). Takes ~7 minutes due to KIS rate limits.
   - **Stale DB (next-day restart):** Same full re-fetch (~7 minutes). The 60-day window overlaps with existing data; `INSERT OR REPLACE` deduplicates.
   - **Fresh DB (same-day restart):** Detects `MAX(date) >= today`, skips instantly.
3. Additionally, inside the daily loop, `populate_lrs()` runs again before each DSE phase. If the container runs continuously across days, LRS gets refreshed at the start of each trading day without needing a restart.
4. Sector map (413 tickers -> 18 sectors) is loaded from `config/nulrimok.yaml` into the `sector_map` table on every startup (instant, no API calls).

**LRS data is stored in a Docker volume** (`nulrimok_data:/data`) so it persists across container rebuilds. Only `docker volume rm` would wipe it.

**Status: COMPLETED**

---

## Phase 4: Paper Trading Verification

### 4.1 Pre-Market Checks (Before 09:00 KST)

**On both VPS:**

```bash
# Check all containers are running
sudo docker compose -f docker-compose.vpsX.yml ps

# Check OMS health
curl http://localhost:8000/health

# Check KIS API connectivity
sudo docker compose -f docker-compose.vpsX.yml logs --tail=20 oms
```

### 4.2 Market Hours Monitoring

**VPS 1 (KMP + Nulrimok) Timeline:**

| Time (KST) | Event | Log Check |
|------------|-------|---------|
| 07:30-08:00 | Nulrimok LRS refresh (if stale) | `docker logs strategy_nulrimok 2>&1 \| grep "LRS"` |
| 08:00-08:30 | Nulrimok DSE runs | `docker logs strategy_nulrimok 2>&1 \| grep "DSE"` |
| 09:00-09:15 | KMP OR building | `docker logs strategy_kmp 2>&1 \| grep "OR"` |
| 09:15-09:30 | KMP scanning | `docker logs strategy_kmp 2>&1 \| grep "scan"` |
| 09:30-10:00 | KMP entries | `docker logs strategy_kmp 2>&1 \| grep "ENTER"` |
| 10:00+ | Nulrimok IEPE (entry/exit) | `docker logs strategy_nulrimok 2>&1 \| grep "IEPE"` |
| 14:30 | KMP flatten | `docker logs strategy_kmp 2>&1 \| grep "flatten"` |

**VPS 2 (KPR + PCIM) Timeline:**

| Time (KST) | Event | Log Check |
|------------|-------|---------|
| 06:00 | PCIM daily stats | `docker logs strategy_pcim 2>&1 \| grep "daily_stats"` |
| 08:00-08:30 | PCIM approval | `docker logs strategy_pcim 2>&1 \| grep "approve"` |
| 09:03-10:00 | PCIM bucket triggers | `docker logs strategy_pcim 2>&1 \| grep "bucket"` |
| 09:10-14:00 | KPR setups | `docker logs strategy_kpr 2>&1 \| grep "setup"` |
| 20:00-23:59 | PCIM YouTube fetch | `docker logs strategy_pcim 2>&1 \| grep "video"` |

### 4.3 Position Verification

```bash
# VPS 1
curl http://localhost:8000/api/v1/positions | jq
curl http://localhost:8000/api/v1/allocations/KMP | jq
curl http://localhost:8000/api/v1/allocations/NULRIMOK | jq

# VPS 2
curl http://localhost:8000/api/v1/positions | jq
curl http://localhost:8000/api/v1/allocations/KPR | jq
curl http://localhost:8000/api/v1/allocations/PCIM | jq
```

### 4.4 Database Verification (VPS 2 — Contains All Data)

Both VPS 1 and VPS 2 OMS instances write to the same PostgreSQL database on VPS 2. All strategies' intents and trades appear here.

```bash
# Connect to PostgreSQL (on VPS 2)
sudo docker exec -it trading_db psql -U postgres -d trading

# Check intent history (should show KMP, Nulrimok from VPS 1 + KPR, PCIM from VPS 2)
SELECT strategy_id, intent_type, status, created_at
FROM intents
ORDER BY created_at DESC
LIMIT 20;

# Check today's trades
SELECT * FROM trades WHERE DATE(created_at) = CURRENT_DATE;

# Exit
\q
```

### 4.5 Paper Trading Schedule

| Day | Focus | Checks |
|-----|-------|--------|
| **Day 1** | Infrastructure | VPS provisioned, Docker running, repos cloned |
| **Day 2** | VPS 2 | Database up, OMS healthy, KPR+PCIM containers running |
| **Day 3** | VPS 1 | OMS healthy, KMP+Nulrimok containers running, LRS auto-populated |
| **Day 4** | First Trade Day | Monitor all timelines, verify intent submission |
| **Day 5-6** | Validation | Check fills, P&L, no errors in logs |
| **Day 7-10** | Optimization | Tune parameters based on signal counts |
| **Day 11-14** | Stability | Run without intervention, gather metrics |

### 4.6 Success Criteria

| Metric | Target |
|--------|--------|
| Uptime | > 99% during market hours |
| Intent submission latency | < 500ms |
| Fill acknowledgment | Within 5 seconds |
| Signal count (KMP) | 5-15 per day |
| Signal count (KPR) | 3-10 per day |
| Signal count (Nulrimok) | 1-5 per day |
| Signal count (PCIM) | 2-8 per day |
| Error rate | < 1% of intents rejected |

---

## Phase 5: Monitoring & Alerts Setup

### 5.1 Create Monitoring Script

**File:** `/opt/k_stock_trader/scripts/health_check.sh`

```bash
#!/bin/bash

VPS_NAME=${1:-"unknown"}
ALERT_WEBHOOK=${2:-""}  # Slack/Discord webhook URL

check_service() {
    local service=$1
    local url=$2

    if curl -sf "$url" > /dev/null 2>&1; then
        echo "[OK] $service is healthy"
        return 0
    else
        echo "[FAIL] $service is DOWN"
        if [ -n "$ALERT_WEBHOOK" ]; then
            curl -X POST -H "Content-Type: application/json" \
                -d "{\"text\":\"ALERT $VPS_NAME: $service is DOWN\"}" \
                "$ALERT_WEBHOOK"
        fi
        return 1
    fi
}

check_container() {
    local container=$1

    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "[OK] Container $container is running"
        return 0
    else
        echo "[FAIL] Container $container is NOT running"
        if [ -n "$ALERT_WEBHOOK" ]; then
            curl -X POST -H "Content-Type: application/json" \
                -d "{\"text\":\"ALERT $VPS_NAME: Container $container is NOT running\"}" \
                "$ALERT_WEBHOOK"
        fi
        return 1
    fi
}

echo "=== Health Check: $VPS_NAME ==="
echo "Time: $(date)"

check_service "OMS" "http://localhost:8000/health"

if [ "$VPS_NAME" == "VPS1" ]; then
    check_container "strategy_kmp"
    check_container "strategy_nulrimok"
elif [ "$VPS_NAME" == "VPS2" ]; then
    check_container "strategy_kpr"
    check_container "strategy_pcim"
    check_container "trading_db"
fi

echo "=== End Health Check ==="
```

```bash
chmod +x /opt/k_stock_trader/scripts/health_check.sh
```

### 5.2 Setup Cron Jobs

```bash
crontab -e
```

Add:

```cron
# Health check every 5 minutes during market hours (09:00-15:30 KST)
*/5 9-15 * * 1-5 /opt/k_stock_trader/scripts/health_check.sh VPS1 >> /var/log/health_check.log 2>&1

# Daily restart at 06:00 KST to refresh tokens
0 6 * * 1-5 cd /opt/k_stock_trader && sudo docker compose -f docker-compose.vps1.yml restart >> /var/log/restart.log 2>&1
```

### 5.3 Setup Metabase Dashboard (Shared — VPS 2)

**Status: NOT YET DONE**

Metabase runs on VPS 2 and serves as the **unified dashboard for both VPSes**. Since both OMS instances (VPS 1 remote, VPS 2 local) write to the same PostgreSQL database, all strategies' intents, trades, and positions are visible in a single Metabase instance.

1. Access Metabase: `http://VPS2_IP:3000`
2. Complete initial setup
3. Add database connection:
   - Type: PostgreSQL
   - Host: `postgres`
   - Port: `5432`
   - Database: `trading`
   - User: `trading_reader`
   - Password: `${POSTGRES_READER_PASSWORD}`

4. Create dashboards for:
   - Daily P&L by strategy (all 4: KMP, Nulrimok, KPR, PCIM)
   - Intent success/rejection rates
   - Position heat map
   - Fill latency metrics

---

## Remaining Items

### TODO

| Item | Priority | Notes |
|------|----------|-------|
| **Metabase setup** | Medium | VPS 2 container is running, needs initial config via web UI. Once configured, serves as unified dashboard for all 4 strategies across both VPSes. |
| **Cron health checks** | Medium | Scripts exist but cron jobs need to be installed on both VPSes |
| **Alert webhooks** | Low | Health check script supports Slack/Discord webhooks, needs URL configured |

### Verify VPS 1 → VPS 2 Postgres Connection

After completing Phase 3 setup (with `DATABASE_URL` in VPS 1's `.env` and VPS 2 firewall open on port 5432):

```bash
# On VPS 1: check OMS logs for successful Postgres connection
sudo docker compose -f docker-compose.vps1.yml logs oms 2>&1 | grep -i postgres

# Expected: no "connection failed" messages. If you see connection errors, verify:
# 1. VPS 2 firewall allows VPS 1's IP on port 5432
# 2. DATABASE_URL in VPS 1's .env has the correct VPS 2 IP and password
# 3. VPS 2's Postgres container is running and healthy
```

If the connection fails, OMS falls back to in-memory state gracefully — trading still works, but intent/trade history won't appear in Metabase for VPS 1 strategies.

---

## Deployment Updates

### Updating Code on a VPS

After pushing changes to git:

```bash
cd /opt/k_stock_trader
git pull

# Rebuild and restart only changed services
sudo docker compose -f docker-compose.vpsX.yml build <service_names>
sudo docker compose -f docker-compose.vpsX.yml up -d <service_names>

# Example: update kpr and oms on VPS 2
sudo docker compose -f docker-compose.vps2.yml build kpr oms
sudo docker compose -f docker-compose.vps2.yml up -d kpr oms
```

Notes:
- `up -d` automatically stops old containers and starts new ones if the image changed. No need for `docker down` or `docker stop`.
- If Docker uses cached layers and doesn't pick up code changes, force with `--no-cache`:
  ```bash
  sudo docker compose -f docker-compose.vpsX.yml build --no-cache <service>
  ```
- `config/` is bind-mounted (`:ro`), so YAML config changes take effect on container restart without rebuilding.
- Docker volumes (`nulrimok_data`, `postgres_data`, `pcim_data`) persist across rebuilds. Only `docker volume rm` wipes them.

---

## Troubleshooting

### Token Rate-Limiting on Startup (`EGW00133`)

```
Token rate-limited (attempt 1/5), retrying in 65.0s...
```

**Normal.** KIS limits token refreshes to 1/minute. When OMS, KMP, and Nulrimok (or KPR/PCIM) all start simultaneously, only the first gets a token; the others retry after 65s. Each container will acquire its token within 1-3 minutes.

### Universe Filter `NOT_EQUITY` Rejections

The universe filter calls `get_current_price()` for each ticker and checks the `rprs_mrkt_kor_name` field. KIS returns different market names for index constituents:

| KIS Field Value | Meaning | Filter Action |
|----------------|---------|---------------|
| `KOSPI` | Regular KOSPI stock | Pass |
| `KOSPI200` | KOSPI 200 constituent | Pass |
| `KOSDAQ` | Regular KOSDAQ stock | Pass |
| `KSQ150` | KOSDAQ 150 constituent | Pass |
| `ETF`, `ETN`, etc. | Not common equity | Reject |

If legitimate stocks are rejected, check what `rprs_mrkt_kor_name` the API returns:

```bash
sudo docker compose -f docker-compose.vpsX.yml exec <strategy> python -c "
from kis_core import KoreaInvestEnv, KoreaInvestAPI, build_kis_config_from_env
api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
d = api.get_current_price('TICKER')
print(d.get('rprs_mrkt_kor_name'))
"
```

### OMS Balance Rate-Limiting After Hours

```
KIS rate-limited on .../inquire-balance, attempt 1
```

**Normal after market close.** OMS runs a reconciliation loop every 5 seconds that queries account balance. After hours, combined with other API calls, this often exceeds KIS rate limits. The balance query fails gracefully (keeps last known values) and does not affect order execution during trading hours.

### Postgres Connection Warning on VPS 1

```
Postgres connection failed (will retry): ...
```

VPS 1's OMS connects to VPS 2's Postgres over the network. If this message appears, check:
1. VPS 2's Postgres container is running (`sudo docker compose -f docker-compose.vps2.yml ps postgres`)
2. VPS 2 firewall allows VPS 1's IP on port 5432 (`sudo ufw status` on VPS 2)
3. `DATABASE_URL` in VPS 1's `.env` has the correct VPS 2 IP and `POSTGRES_WRITER_PASSWORD`

OMS retries automatically and falls back to in-memory state if the connection cannot be established. Trading is unaffected, but intent/trade history won't be persisted to Postgres (and won't appear in Metabase) until the connection is restored.

### KIS API Rate-Limiting During Universe Filter

On startup, each strategy runs `filter_universe()` which makes ~2 API calls per ticker (price check + ADTV check). For large universes this causes a burst of rate-limited retries. The filter completes within 2-5 minutes and the strategy enters its normal loop.

---

## Appendix A: File Reference

| File | VPS | Purpose |
|------|-----|---------|
| `.env` | Both | Environment variables (different per VPS) |
| `docker-compose.vps1.yml` | VPS 1 | KMP + Nulrimok orchestration |
| `docker-compose.vps2.yml` | VPS 2 | KPR + PCIM + DB orchestration |
| `config/nulrimok.yaml` | VPS 1 | Nulrimok universe + sector_map (413 tickers, 18 sectors) |
| `config/kmp.yaml` | VPS 1 | KMP universe + parameters |
| `config/kpr.yaml` | VPS 2 | KPR universe + parameters |
| `config/pcim.yaml` | VPS 2 | PCIM parameters |
| `kis_core/universe_filter.py` | Both | Shared universe pre-filter (KOSPI/KOSDAQ/KSQ market check) |
| `strategy_nulrimok/lrs/loader.py` | VPS 1 | LRS auto-population from KIS API |
| `strategy_nulrimok/lrs/db.py` | VPS 1 | LRS SQLite database (schema + read/write) |
| `infra/postgres/init/*.sql` | VPS 2 | Database schema |

## Appendix B: Network Diagram

```
                     Internet
                         |
         +---------------+---------------+
         |                               |
    +----v----+                    +-----v-----+
    |  VPS 1  |                    |   VPS 2   |
    | Account1|                    |  Account2 |
    +----+----+                    +-----+-----+
         |                               |
    +----v------------------+    +-------v-------------------+
    | Docker Network        |    | Docker Network            |
    | +-------------------+ |    | +-------------------+     |
    | | OMS (8000)        |---TCP:5432-->| PostgreSQL    |     |
    | | (remote Postgres) | |    | |                   |     |
    | +--------+----------+ |    | +-------------------+     |
    |          |            |    | +-------------------+     |
    | +--------v----------+ |    | | OMS (8000)        |     |
    | | KMP Container     | |    | | (local Postgres)  |     |
    | +-------------------+ |    | +--------+----------+     |
    | +-------------------+ |    |          |                 |
    | | Nulrimok          | |    | +--------v----------+     |
    | | + SQLite LRS      | |    | | KPR Container     |     |
    | | (auto-populated)  | |    | +-------------------+     |
    | +-------------------+ |    | +-------------------+     |
    +-----------------------+    | | PCIM Container    |     |
                                 | +-------------------+     |
         Metabase UI             | +-------------------+     |
    http://VPS2_IP:3000  ◄──────── | Metabase (3000)   |     |
    (both VPSes' data)           | +-------------------+     |
                                 +---------------------------+
```

## Appendix C: Quick Command Reference

```bash
# === VPS 1 ===
cd /opt/k_stock_trader
sudo docker compose -f docker-compose.vps1.yml up -d           # Start all
sudo docker compose -f docker-compose.vps1.yml logs -f kmp     # KMP logs
sudo docker compose -f docker-compose.vps1.yml logs -f nulrimok # Nulrimok logs
curl http://localhost:8000/health                               # OMS health

# === VPS 2 ===
cd /opt/k_stock_trader
sudo docker compose -f docker-compose.vps2.yml up -d           # Start all
sudo docker compose -f docker-compose.vps2.yml logs -f kpr     # KPR logs
sudo docker compose -f docker-compose.vps2.yml logs -f pcim    # PCIM logs
sudo docker compose -f docker-compose.vps2.yml exec postgres psql -U postgres -d trading  # DB shell

# === Both VPS: Update deployment ===
git pull
sudo docker compose -f docker-compose.vpsX.yml build <services>
sudo docker compose -f docker-compose.vpsX.yml up -d <services>

# === Both VPS: OMS API ===
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/positions | jq
curl -X POST http://localhost:8000/api/v1/risk/safe-mode -d '{"enabled":true}'
```
