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
│  lrs.db (auto-pop)  │              │  + Dashboard        │◄── Monitoring UI
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
| **Shared PostgreSQL** | Both OMS instances log to VPS 2's Postgres; single monitoring dashboard for all 4 strategies |
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
# Allow dashboard (restrict to your IP in production)
sudo ufw allow 3000/tcp
# Allow VPS 1 OMS to connect to Postgres for logging
sudo ufw allow from VPS1_IP to any port 5432
sudo ufw enable
```

> **Important:** Replace `VPS1_IP` with VPS 1's actual IP address. This allows VPS 1's OMS to write intents/trades to the shared PostgreSQL database, and makes all 4 strategies visible in the dashboard on VPS 2.

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

> **Important:** Replace `VPS2_IP` with VPS 2's actual IP address, and `writer_password` with the `POSTGRES_WRITER_PASSWORD` value from VPS 2's `.env`. This allows VPS 1's OMS to log intents and trades to the same database used by VPS 2, making all 4 strategies visible in the dashboard on VPS 2.

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

### 5.3 Monitoring Dashboard (VPS 2)

**Status: DONE** — Replaced Metabase with a lightweight Next.js dashboard (`infra/dashboard/`).

The dashboard is built into the VPS 2 compose stack and starts automatically with the other services. It proxies OMS API calls server-side (no CORS issues, no internal URL exposed to browser) and auto-refreshes every 10 seconds.

**Access:** `http://VPS2_IP:3000`

**What it shows:**
- OMS status badge, PAPER/LIVE mode indicator, uptime, KST clock
- Safe mode / halt entries / flatten-in-progress alert banners
- Equity, daily P&L (+%), cash, open positions count
- Session P&L sparkline (in-memory, resets on page reload)
- Per-strategy position cards (KMP, NULRIMOK, KPR, PCIM)
- Full positions table with avg price, entry time, soft stop, hard stop
- KIS circuit breaker and reconciliation alerts

**Deploy / rebuild:**
```bash
# On VPS 2
cd /opt/k_stock_trader
git pull
sudo docker compose -f docker-compose.vps2.yml up -d --build dashboard
```

**Resource footprint:** ~100–250 MB RAM, ~130 MB image (vs Metabase's ~1.5 GB RAM, ~1.6 GB image).

---

### 5.4 Removing Metabase to Free Disk Space (VPS 2)

If Metabase was previously running, remove it to reclaim ~1.6 GB of disk and ~1.5 GB of RAM:

```bash
# Stop and remove the container (if still running)
docker stop metabase 2>/dev/null; docker rm metabase 2>/dev/null

# Remove the image (~1.6 GB)
docker rmi metabase/metabase:latest

# Remove any dangling layers left behind
docker image prune -f

# Verify space recovered
df -h /
docker system df
```

If Metabase had already written data to the `metabase` schema in Postgres and you want to clean that up too:

```bash
sudo docker exec -it trading_db psql -U postgres -d trading -c "DROP SCHEMA IF EXISTS metabase CASCADE;"
```

> This is safe — the `trading` schema (where intents/trades live) is unaffected. The `metabase` schema only held Metabase's own configuration.

---

## Debug Pipeline: Why No Trades?

If a strategy produced zero trades on a trading day, work through its pipeline top-to-bottom. The first step that shows nothing (or an error) is where trades are getting blocked.

> **Note:** Log timestamps are in **UTC**. Add 9 hours for KST (e.g., `00:16 UTC` = `09:16 KST`).

### Quick Health Check (All Strategies)

```bash
# Is the strategy alive or crash-looping?
docker logs strategy_kmp 2>&1 | grep -E "Starting KMP|Traceback|Error" | tail -20
docker logs strategy_nulrimok 2>&1 | grep -E "Starting Nulrimok|Traceback|Error" | tail -20
docker logs strategy_kpr 2>&1 | grep -E "Starting KPR|Traceback|Error" | tail -20
docker logs strategy_pcim 2>&1 | grep -E "Starting PCIM|Traceback|Error" | tail -20

# Are heartbeats flowing? (proves strategy is alive and looping)
docker logs strategy_kmp 2>&1 | grep "heartbeat" | tail -5

# Any OMS rejections?
docker logs oms_vps1 2>&1 | grep -i "reject\|halt\|frozen\|scaled\|MODIFY" | tail -20
docker logs oms_vps2 2>&1 | grep -i "reject\|halt\|frozen\|scaled\|MODIFY" | tail -20

# Current positions across all strategies
curl -s http://localhost:8000/positions | python3 -m json.tool
```

---

### KMP Debug Pipeline

KMP trades 09:15–14:30 KST. It scans at 09:15, then enters an FSM loop looking for breakout setups.

```bash
# Step 1: Did it reach market open?
docker logs strategy_kmp 2>&1 | grep "Waiting for market open\|Entering main loop"
# Expected: "Waiting..." then "Entering main loop" once (not repeatedly — repeats = crash loop)

# Step 2: Did the 09:15 scan find candidates?
docker logs strategy_kmp 2>&1 | grep "Scan complete"
# Expected: "Scan complete. 15 candidates" — if 0, trend anchor or surge filter killed everything

# Step 3: How many tickers passed trend anchor?
docker logs strategy_kmp 2>&1 | grep "Trend anchor"
# Expected: "Trend anchor applied. 85 tickers OK" — low number means weak market

# Step 4: Is regime/breadth blocking entries?
docker logs strategy_kmp 2>&1 | grep -i "breadth\|regime\|chop\|risk_off"
# If breadth stays < 8 all day, regime gate never opens → no entries possible

# Step 5: Did any symbol reach ARMED (ready to submit)?
docker logs strategy_kmp 2>&1 | grep -E "WATCH_BREAK|ARMED|WAIT_ACCEPTANCE"
# If nothing, setups never triggered (market didn't break out)

# Step 6: Were intents submitted to OMS?
docker logs strategy_kmp 2>&1 | grep "submit_intent\|Intent"

# Step 7: Did OMS reject them?
docker logs oms_vps1 2>&1 | grep -i "KMP.*reject\|KMP.*modify\|KMP.*scaled"
```

**Common KMP blockers:**
| Symptom | Cause |
|---------|-------|
| Multiple "Starting KMP" entries | Crash loop — check `grep Traceback` for root cause |
| Scan complete. 0 candidates | Trend anchor filtered everything (weak/range-bound market) |
| No WATCH_BREAK/ARMED lines | No breakout setups detected (market too quiet) |
| breadth < 8 all day | Regime gate blocked — not enough leaders surging |

---

### KPR Debug Pipeline

KPR trades 09:10–14:00 KST. It watches for VWAP pullback setups across tiered universe (HOT/WARM/COLD).

```bash
# Step 1: Did it start and connect?
docker logs strategy_kpr 2>&1 | grep "Starting KPR\|WebSocket\|universe"
# Expected: "KPR WebSocket connected for HOT tier"

# Step 2: Universe filter results
docker logs strategy_kpr 2>&1 | grep "Universe filter"
# Expected: "58 passed, 0 rejected"

# Step 3: Are symbols entering VWAP band?
docker logs strategy_kpr 2>&1 | grep -i "setup detected\|accepting\|entry\|order\|exit"
# If nothing, no stocks pulled back 2-5% below VWAP (market trending up or flat)

# Step 4: Did any reach ACCEPTING state?
docker logs strategy_kpr 2>&1 | grep "ACCEPTING"
# If nothing, setups detected but confirmation bars didn't follow through

# Step 5: Were intents submitted?
docker logs strategy_kpr 2>&1 | grep "submit_intent\|Intent"

# Step 6: Did OMS reject?
docker logs oms_vps2 2>&1 | grep -i "KPR.*reject\|KPR.*modify"

# Step 7: Check investor flow availability (key signal for KPR)
docker logs strategy_kpr 2>&1 | grep -i "investor\|flow\|stale"
```

**Common KPR blockers:**
| Symptom | Cause |
|---------|-------|
| No SETUP_DETECTED | Market didn't pull back to VWAP band (trending day) |
| SETUP_DETECTED but no ACCEPTING | Investor flow signal was stale/conflicting |
| halt_new_entries | OMS daily loss breaker tripped |

---

### Nulrimok Debug Pipeline

Nulrimok is a swing/multi-day strategy. DSE runs at 08:00–08:30 KST, IEPE entries happen on 30-minute bars throughout the day.

```bash
# Step 1: Did DSE run and produce a watchlist?
docker logs strategy_nulrimok 2>&1 | grep -i "DSE\|watchlist\|active_set"
# Expected: "DSE" output showing active set of tickers to watch

# Step 2: What regime tier was assigned?
docker logs strategy_nulrimok 2>&1 | grep -i "regime\|tier"
# Tier C with allow_tier_c_reduced=False → IEPE completely blocked

# Step 3: Did IEPE phase start? (requires active_set and correct time window)
docker logs strategy_nulrimok 2>&1 | grep -i "IEPE\|process_entry\|PENDING_FILL"

# Step 4: Were any tickers near the AVWAP band?
docker logs strategy_nulrimok 2>&1 | grep -i "near_band\|avwap\|band"
# If nothing, prices never reached entry zones

# Step 5: Were intents submitted?
docker logs strategy_nulrimok 2>&1 | grep "submit_intent\|Intent"

# Step 6: Did OMS reject?
docker logs oms_vps1 2>&1 | grep -i "NULRIMOK.*reject"

# Step 7: Check daily risk budget (may cap entries)
docker logs strategy_nulrimok 2>&1 | grep -i "risk_budget\|budget"

# Step 8: Check for position recovery on startup
docker logs strategy_nulrimok 2>&1 | grep "Recovered\|recovered\|Startup"

# Step 9: Check flow reversal exits (pre-market)
docker logs strategy_nulrimok 2>&1 | grep "flow reversal"
```

**Common Nulrimok blockers:**
| Symptom | Cause |
|---------|-------|
| DSE didn't run | Strategy started after 08:30 KST window |
| Tier C blocked | `allow_tier_c_reduced=False` (conservative) in weak market |
| No near_band | Prices never reached AVWAP entry zones |
| active_set empty | DSE filtering too strict, or LRS data stale |

---

### PCIM Debug Pipeline

PCIM is an influencer-signal strategy. Night pipeline (20:00–06:00) fetches YouTube videos, premarket (08:40–09:00) classifies, execution (09:01–10:30) trades.

```bash
# Step 1: Did the night pipeline find videos?
docker logs strategy_pcim 2>&1 | grep "Night pipeline\|YOUTUBE_FETCH"
# Expected: "Night pipeline: Checking for new videos" + channel fetches

# Step 2: Were signals extracted from transcripts?
docker logs strategy_pcim 2>&1 | grep "RECOMMENDATION\|extract_signals\|conviction"
# If nothing, influencers didn't mention any stocks (or transcript download failed)

# Step 3: How many candidates survived filters?
docker logs strategy_pcim 2>&1 | grep -E "consolidation|TREND_GATE|INSUFFICIENT|reject_reason|candidates"
# Check for: "Night pipeline: 5 candidates" then "After consolidation: 3 unique symbols"

# Step 4: Were candidates approved?
docker logs strategy_pcim 2>&1 | grep "Auto-approved\|Approval\|approval"
# Expected: "Auto-approved 3 candidates"

# Step 5: Did premarket selection pass? (regime + exposure caps)
docker logs strategy_pcim 2>&1 | grep "PREMARKET_SELECT"
# ACCEPTED = will trade, REJECTED = hit max_positions or exposure_cap

# Step 6: Did execution triggers fire?
docker logs strategy_pcim 2>&1 | grep "ENTRY_DECISION\|EXECUTION_VETO\|bucket"
# ENTRY_DECISION = order submitted, EXECUTION_VETO = spread/VI/upper-limit blocked

# Step 7: Were orders filled?
docker logs strategy_pcim 2>&1 | grep "Fill confirmed\|position created\|Partial fill"

# Step 8: Did OMS reject?
docker logs oms_vps2 2>&1 | grep -i "PCIM.*reject"

# Step 9: Check regime (affects exposure cap)
docker logs strategy_pcim 2>&1 | grep -i "regime"
# CRISIS/WEAK = severely limited exposure
```

**Common PCIM blockers:**
| Symptom | Cause |
|---------|-------|
| Night pipeline: 0 candidates | No new YouTube videos from configured influencers |
| RECOMMENDATION lines but 0 after filters | Stocks failed trend gate, hard filters, or gap reversal check |
| PREMARKET_SELECT: all REJECTED | CRISIS/WEAK regime capped exposure, or max positions reached |
| EXECUTION_VETO | Spread too wide, VI active, or price at upper limit |
| No Fill confirmed | Orders submitted but didn't fill (paper trading limitation or thin liquidity) |

---

### OMS Debug (Cross-Strategy)

The OMS is the final gatekeeper. If strategies are submitting intents but nothing trades, check here.

```bash
# All rejections (both VPS)
docker logs oms_vps1 2>&1 | grep -i "reject" | tail -20
docker logs oms_vps2 2>&1 | grep -i "reject" | tail -20

# Daily loss circuit breaker (halts ALL new entries)
docker logs oms_vps1 2>&1 | grep -i "daily_loss\|halt\|circuit"
docker logs oms_vps2 2>&1 | grep -i "daily_loss\|halt\|circuit"

# Exposure cap hits
docker logs oms_vps1 2>&1 | grep -i "exposure\|regime.*cap"
docker logs oms_vps2 2>&1 | grep -i "exposure\|regime.*cap"

# Strategy paused or frozen symbols
docker logs oms_vps1 2>&1 | grep -i "paused\|frozen\|safe.mode"
docker logs oms_vps2 2>&1 | grep -i "paused\|frozen\|safe.mode"

# Qty scaled down (trade went through but smaller)
docker logs oms_vps1 2>&1 | grep -i "scaled\|MODIFY"
docker logs oms_vps2 2>&1 | grep -i "scaled\|MODIFY"
```

**OMS rejection reasons:**
| Reason | Meaning |
|--------|---------|
| `Daily loss exceeds halt limit` | PnL dropped > 3% (or 5%) → all entries blocked |
| `Max positions reached` | Hit global (15) or per-strategy (4/3/5/8) position limit |
| `Gross exposure would exceed 80%` | Portfolio too concentrated |
| `Regime CRISIS/WEAK cap exceeded` | Market regime limits total exposure |
| `Sector exposure exceeded` | Too much in one sector (> 30%) |
| `Strategy paused` | Manually paused via API |
| `Symbol frozen` | Allocation drift unresolved |
| `VI cooldown` | Volatility Interruption, 10-min cooldown |

---

### Using Persistent Log Files

If log persistence is enabled (logs written to `data/*/logs/`), you can search across days without `docker logs`:

```bash
# Today's KMP log
cat data/kmp/logs/kmp_2026-02-11.log

# Search across multiple days
grep "Fill detected" data/kmp/logs/kmp_2026-02-*.log

# Compressed old logs
zcat data/kmp/logs/kmp_2026-02-01.log.gz | grep "ENTRY"

# Tail live
tail -f data/kmp/logs/kmp_$(date -u +%Y-%m-%d).log
```

---

## Remaining Items

### TODO

| Item | Priority | Notes |
|------|----------|-------|
| **Verify KPR drift fix** | High | On VPS-3, run `docker logs strategy_kpr 2>&1 \| grep -i "orphan\|trade_block"` — should return nothing. The old `ORDER_ORPHAN_LOCAL` spam that blocked all trades should be gone. |
| **Run LRS backfill on VPS-1** | High | Run `docker exec strategy_nulrimok python /app/scripts/backfill_lrs.py` to add 600 days of KOSPI/KOSDAQ history. Logs already show regime tier=A, but backfill provides more robust regime calculations. |
| **Config audit fixes** | Medium | Pending plan in `.claude/plans/`: (1) Fix `t3_bucket_a_allowed` default mismatch in PCIM switches (`False` should be `True`), (2) Wire `conservative.yaml` loading — currently dead code (no `main.py` ever calls `load_from_yaml()`), (3) Add config validation so bad YAML fails at startup, not mid-trading. |
| **Remove Metabase from VPS 2** | Medium | Run `docker stop metabase; docker rm metabase; docker rmi metabase/metabase:latest; docker image prune -f` to reclaim ~1.6 GB disk. See §5.4. |
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

OMS retries automatically and falls back to in-memory state if the connection cannot be established. Trading is unaffected, but intent/trade history won't be persisted to Postgres (and won't appear in the dashboard) until the connection is restored.

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
         Dashboard UI             | +-------------------+     |
    http://VPS2_IP:3000  ◄──────── | Dashboard (3000)  |     |
    (both VPSes' data)           | | Next.js, ~150MB   |     |
                                 | +-------------------+     |
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
sudo docker compose -f docker-compose.vps2.yml logs -f dashboard                          # Dashboard logs

# === Both VPS: Update deployment ===
git pull
sudo docker compose -f docker-compose.vpsX.yml build <services>
sudo docker compose -f docker-compose.vpsX.yml up -d <services>

# === Both VPS: OMS API ===
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/positions | jq
curl -X POST http://localhost:8000/api/v1/risk/safe-mode -d '{"enabled":true}'
```
