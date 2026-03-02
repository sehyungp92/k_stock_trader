# VPS Operations Runbook

## Architecture

```
VPS1: KMP + Nulrimok → OMS (oms_vps1) ──┐
      docker-compose.vps1.yml            ├──→ VPS2 PostgreSQL (optional)
                                         │
VPS2: KPR + PCIM → OMS (oms_vps2) ──────┤
      PostgreSQL (trading_db)  ──────────┘
      Dashboard :3000
      docker-compose.vps2.yml
```

**Deploy VPS2 first** — it hosts Postgres, which VPS1's OMS may connect to remotely.

### Expected Containers Per VPS

| VPS | Container Name | Service |
|-----|---------------|---------|
| VPS1 | `oms_vps1` | OMS (port 8000) |
| VPS1 | `strategy_kmp` | KMP momentum breakout |
| VPS1 | `strategy_nulrimok` | Nulrimok swing dip-buying |
| VPS2 | `trading_db` | PostgreSQL 16 (port 5432) |
| VPS2 | `oms_vps2` | OMS (port 8000) |
| VPS2 | `strategy_kpr` | KPR mean-reversion |
| VPS2 | `strategy_pcim` | PCIM influencer signal |
| VPS2 | `dashboard` | Next.js dashboard (port 3000) |

### Shared Dependencies — What Forces a Full Rebuild

| Module | Used By | Change Here → Rebuild |
|--------|---------|----------------------|
| `kis_core/` | ALL containers | Everything on both VPS |
| `oms/` | OMS containers | OMS on both VPS |
| `oms_client/` | All strategies | All strategy containers |
| `strategy_kmp/` | KMP only | KMP on VPS1 |
| `strategy_kpr/` | KPR only | KPR on VPS2 |
| `strategy_pcim/` | PCIM only | PCIM on VPS2 |
| `strategy_nulrimok/` | Nulrimok only | Nulrimok on VPS1 |
| `instrumentation/` | All strategies | All strategy containers |
| `requirements.txt` | ALL containers | Everything on both VPS |
| `infra/postgres/init/` | Postgres (first init only) | Manual `ALTER` on VPS2 |

---

## 1. Full Deploy Sequence (After Code Changes)

### Step 0: Commit and Push from Local Machine

```bash
cd /path/to/k_stock_trader

# Stage changed and new files
git add kis_core/ oms/ oms_client/ instrumentation/ \
       strategy_kmp/ strategy_kpr/ strategy_pcim/ strategy_nulrimok/ \
       requirements.txt docker-compose*.yml .dockerignore \
       infra/ tests/
       # ... add other changed files as needed

git commit -m "Describe your changes"
git push origin main
```

### Step 1: Deploy VPS2 (Postgres + OMS + KPR + PCIM + Dashboard)

```bash
ssh your-vps2
cd /path/to/k_stock_trader
```

**1a. Pull latest code**
```bash
git pull origin main
```

**1b. Apply Postgres schema changes (if `infra/postgres/init/` was modified)**

Init scripts in `docker-entrypoint-initdb.d/` only run on first database creation. For an existing database, apply changes manually:

```bash
# Check what changed
git diff HEAD~1 -- infra/postgres/init/

# If 004_roles.sql changed, apply manually (CREATE ROLE will error if role exists — use ALTER)
docker exec trading_db psql -U postgres -d trading -c \
  "ALTER ROLE trading_writer WITH PASSWORD '$(grep POSTGRES_WRITER_PASSWORD .env | cut -d= -f2)';"

# If new tables/views were added in other init scripts, apply them:
# docker exec trading_db psql -U postgres -d trading -f /docker-entrypoint-initdb.d/005_new_script.sql
```

**1c. Stop all services**
```bash
docker compose -f docker-compose.vps2.yml down
```

> `down` stops and removes containers but **preserves volumes** (postgres_data, pcim_data). Data is safe.

**1d. Rebuild all images**
```bash
# Full clean rebuild (use when requirements.txt or system deps changed)
docker compose -f docker-compose.vps2.yml build --no-cache

# Fast rebuild (use when only Python source code changed)
# docker compose -f docker-compose.vps2.yml build
```

**1e. Start everything**
```bash
docker compose -f docker-compose.vps2.yml up -d
```

> **Token startup delay:** KIS limits token requests to 1/minute. When OMS + KPR + PCIM start together, only the first gets a token immediately; others retry after ~65s. Expect `EGW00133` rate-limit warnings in logs for the first 1-3 minutes. This is normal.

**1f. Verify health**
```bash
sleep 20

# All containers running?
docker compose -f docker-compose.vps2.yml ps

# Postgres connected?
docker logs oms_vps2 --tail 30 2>&1 | grep -E "Postgres connection|DATABASE_URL"
# Expected: "Postgres connection pool established"

# OMS healthy?
curl -s http://localhost:8000/health

# Equity loaded?
curl -s http://localhost:8000/api/v1/state/account | python3 -m json.tool

# Strategy startup OK?
docker logs strategy_kpr --tail 20
docker logs strategy_pcim --tail 20
```

**1g. Verify instrumentation data directories were created**
```bash
# Each strategy writes JSONL to its own instrumentation data dir
ls -la data/kpr/instrumentation/ data/pcim/instrumentation/
# Expected: trades/ missed/ scores/ snapshots/ daily/ errors/ subdirectories
```

**1h. Resolve frozen positions (if any — see Section 3)**
```bash
curl -s http://localhost:8000/api/v1/positions | python3 -c "
import sys, json
data = json.load(sys.stdin)
frozen = [s for s, p in data.items() if p.get('frozen')]
print(f'Frozen: {frozen if frozen else \"NONE\"}')"
```

### Step 2: Deploy VPS1 (OMS + KMP + Nulrimok)

> Only proceed after VPS2 is fully up (especially if VPS1's OMS connects to VPS2's Postgres).

```bash
ssh your-vps1
cd /path/to/k_stock_trader
```

**2a. Pull latest code**
```bash
git pull origin main
```

**2b. Stop all services**
```bash
docker compose -f docker-compose.vps1.yml down
```

**2c. Rebuild all images**
```bash
# Full clean rebuild
docker compose -f docker-compose.vps1.yml build --no-cache

# Fast rebuild (only Python source changes)
# docker compose -f docker-compose.vps1.yml build
```

**2d. Start everything**
```bash
docker compose -f docker-compose.vps1.yml up -d
```

> **Token startup delay:** Same 1-3 minute `EGW00133` rate-limit delay for OMS + KMP + Nulrimok token acquisition.

**2e. Verify health**
```bash
sleep 20

# All containers running?
docker compose -f docker-compose.vps1.yml ps

# OMS healthy?
curl -s http://localhost:8000/health

# Postgres connection (if DATABASE_URL is set in .env)?
docker logs oms_vps1 --tail 30 2>&1 | grep -E "Postgres|in-memory"

# Equity loaded?
curl -s http://localhost:8000/api/v1/state/account | python3 -m json.tool

# Strategy startup OK?
docker logs strategy_kmp --tail 20
docker logs strategy_nulrimok --tail 20
```

**2f. Verify instrumentation data directories were created**
```bash
ls -la data/kmp/instrumentation/ data/nulrimok/instrumentation/
# Expected: trades/ missed/ scores/ snapshots/ daily/ errors/ subdirectories
```

**2g. Verify Nulrimok LRS database**

Nulrimok requires 280+ KOSPI bars in its local regime database for regime classification to work:

```bash
docker exec strategy_nulrimok python -c "
import sqlite3, os
db = os.environ.get('LRS_DB_PATH', '/data/lrs.db')
if os.path.exists(db):
    conn = sqlite3.connect(db)
    rows = conn.execute('SELECT COUNT(*) FROM kospi_ohlcv').fetchone()[0]
    print(f'LRS DB: {rows} KOSPI rows (need 280+)')
    conn.close()
else:
    print('WARNING: LRS DB not found — run backfill')
"

# If LRS is empty or missing:
docker exec strategy_nulrimok python /app/scripts/backfill_lrs.py
```

### Step 3: Post-Deploy Verification (Next Trading Day)

After KRX market opens (09:00 KST), verify strategies are actively processing:

**VPS2:**
```bash
# KPR — entry decisions after 09:10
docker logs strategy_kpr --since "8h" 2>&1 | grep -E "Entry ACCEPTED|Entry REJECTED|VWAP"

# PCIM — premarket scan after 09:01
docker logs strategy_pcim --since "8h" 2>&1 | grep -E "ENTRY_DECISION|signal|OMS"

# OMS — intent processing
docker logs oms_vps2 --since "8h" 2>&1 | grep -E "Intent.*ENTER|REJECTED|DEFERRED|fill"
```

**VPS1:**
```bash
# KMP — breakout scanning after 09:15
docker logs strategy_kmp --since "8h" 2>&1 | grep -E "breakout|ENTRY|gate|regime"

# Nulrimok — dip screening after 09:00
docker logs strategy_nulrimok --since "8h" 2>&1 | grep -E "scan|ENTRY|artifact|regime"

# OMS — intent processing
docker logs oms_vps1 --since "8h" 2>&1 | grep -E "Intent.*ENTER|REJECTED|fill"
```

---

## 2. Fix Postgres Authentication (Post-Deploy)

### Problem
`docker-compose.vps2.yml` injects `DATABASE_URL` with `${POSTGRES_WRITER_PASSWORD}` from `.env`, but `infra/postgres/init/004_roles.sql` created the `trading_writer` role with a different password on first init. If the env var is unset or mismatched, OMS silently degrades to in-memory mode — losing all persistence.

### Diagnosis
```bash
# Check if OMS has Postgres connection
docker logs oms_vps2 2>&1 | grep -E "Postgres connection|DATABASE_URL not set"

# Expected on success: "Postgres connection pool established"
# Expected on failure: "Postgres connection failed" or "DATABASE_URL not set"

# Verify the role password directly
docker exec trading_db psql -U postgres -d trading -c \
  "SELECT rolname FROM pg_roles WHERE rolname = 'trading_writer';"
```

### Fix Steps

**Step 1: Set the password in Postgres to match your `.env`**
```bash
# Choose a secure password
export NEW_PW='your_secure_password_here'

# Update the Postgres role
docker exec trading_db psql -U postgres -d trading -c \
  "ALTER ROLE trading_writer WITH PASSWORD '$NEW_PW';"

# Verify it works
docker exec trading_db psql -U trading_writer -d trading -h localhost -c "SELECT 1;" <<< "$NEW_PW"
```

**Step 2: Update `.env` on the VPS**
```bash
# Edit .env (use your preferred editor)
nano .env

# Set or update this line:
POSTGRES_WRITER_PASSWORD=your_secure_password_here

# Verify it's set
grep POSTGRES_WRITER_PASSWORD .env
```

**Step 3: Restart OMS to pick up the connection**
```bash
docker compose -f docker-compose.vps2.yml restart oms

# Wait 10 seconds, then verify
sleep 10
docker logs oms_vps2 --tail 20 2>&1 | grep "Postgres"
# Should see: "Postgres connection pool established"
```

**Step 4: Verify end-to-end**
```bash
# Check OMS can write to database
curl -s http://localhost:8000/api/v1/state/account | python3 -m json.tool

# Check intents table has data (may be empty if no trades yet)
docker exec trading_db psql -U postgres -d trading -c \
  "SELECT strategy_id, status, count(*) FROM intents GROUP BY strategy_id, status ORDER BY strategy_id;"
```

---

## 3. Resolve Frozen `_UNKNOWN_` Positions (Post-Deploy)

### Problem
When OMS detects positions on the broker that don't match any strategy allocation, it assigns them to `_UNKNOWN_` and freezes the symbol. Frozen symbols block new entries and consume gross exposure.

### API Reference
```
POST /api/v1/admin/resolve-drift
Content-Type: application/json

Body:
{
  "symbol": "006400",
  "action": "acknowledge" | "reassign",
  "target_strategy_id": "PCIM"     // required only for "reassign"
}

Response:
{
  "status": "ok",
  "symbol": "006400",
  "frozen": false
}
```

**Actions:**
- `acknowledge`: Clears the `_UNKNOWN_` allocation and unfreezes the symbol. The broker position still exists but OMS no longer tracks it. Use for positions that belong to no strategy (manual trades, pre-existing holdings).
- `reassign`: Moves the `_UNKNOWN_` allocation to the specified strategy. Use when a strategy placed the trade but lost tracking (e.g., due to Postgres failure).

### Fix Steps

**Prerequisite: Postgres must be connected first (Section 2 above).**

**Step 1: Verify current frozen positions**
```bash
curl -s http://localhost:8000/api/v1/positions | python3 -c "
import sys, json
data = json.load(sys.stdin)
frozen = [(s, p) for s, p in data.items() if p.get('frozen')]
if not frozen:
    print('No frozen positions.')
else:
    for sym, p in frozen:
        allocs = p.get('allocations', {})
        unknown_qty = allocs.get('_UNKNOWN_', {}).get('qty', 0)
        print(f'  {sym}: real_qty={p[\"real_qty\"]} unknown_qty={unknown_qty} allocations={list(allocs.keys())}')
"
```

**Step 2: Acknowledge positions that belong to no strategy**
```bash
# Replace with actual frozen symbols from Step 1
for sym in 006400 009540 042700 105560 259960 323410; do
  echo "Resolving $sym..."
  curl -s -X POST http://localhost:8000/api/v1/admin/resolve-drift \
    -H 'Content-Type: application/json' \
    -d "{\"symbol\":\"$sym\",\"action\":\"acknowledge\"}" | python3 -m json.tool
  echo ""
done
```

**Step 3: Reassign positions that belong to a strategy**
```bash
# Replace symbol and strategy_id with actual values from Step 1
curl -s -X POST http://localhost:8000/api/v1/admin/resolve-drift \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"034220","action":"reassign","target_strategy_id":"PCIM"}' \
  | python3 -m json.tool
```

**Step 4: Verify all positions unfrozen**
```bash
curl -s http://localhost:8000/api/v1/positions | python3 -c "
import sys, json
data = json.load(sys.stdin)
frozen = [s for s, p in data.items() if p.get('frozen')]
print(f'Frozen symbols: {frozen if frozen else \"NONE (all clear)\"}')
"
```

**Step 5: Verify exposure is freed**
```bash
curl -s http://localhost:8000/api/v1/state/account | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Equity:     {d.get(\"equity\", 0):>15,.0f} KRW')
print(f'Buyable:    {d.get(\"buyable_cash\", 0):>15,.0f} KRW')
"
```

### Step 6: Sell acknowledged positions to free broker exposure

After acknowledging, the OMS stops tracking these positions — but the shares are still on the broker and consume gross exposure. Use `flatten-all` or sell them individually.

**Option A: Flatten all (sells everything — use with caution)**
```bash
# This sells ALL positions on this OMS, including strategy-owned ones
curl -s -X POST http://localhost:8000/api/v1/admin/flatten-all | python3 -m json.tool
```

**Option B: Sell specific symbols directly via KIS (recommended)**

> **Use Option B.** It only targets the pre-existing holdings you specify, leaving strategy-owned positions untouched. Option A is an emergency kill-switch that liquidates *everything* — including active strategy positions — which would disrupt any in-progress trades.

Run from inside the OMS container, which has KIS credentials:
```bash
docker exec oms_vpsX python3 -c "
import os, sys
sys.path.insert(0, '/app')
from kis_core import KoreaInvestAPI

api = KoreaInvestAPI()

# Pre-existing paper account holdings to sell
# Replace with actual symbols and quantities from Step 1
holdings = {
    '006400': 10,   # Samsung SDI — replace qty
    '009540': 10,   # HD Korea Shipbuilding — replace qty
    '042700': 10,   # Hanmi Semiconductor — replace qty
    '105560': 10,   # KB Financial Group — replace qty
    '259960': 10,   # KRAFTON — replace qty
    '323410': 10,   # KakaoBank — replace qty
}

for sym, qty in holdings.items():
    result = api.place_market_sell(sym, qty)
    print(f'{sym}: order_id={result}')
"
```

> Replace `oms_vpsX` with `oms_vps1` or `oms_vps2` depending on which account holds the positions. Replace quantities with actual holdings from Step 1 output.

**Step 7: Verify positions are gone (next reconciliation cycle or after restart)**
```bash
curl -s http://localhost:8000/api/v1/state/account | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Equity:     {d.get(\"equity\", 0):>15,.0f} KRW')
print(f'Buyable:    {d.get(\"buyable_cash\", 0):>15,.0f} KRW')
"
```

### Important Notes

- **Acknowledged positions still hold real shares on the broker.** They continue to count toward gross exposure because the broker reports them. To truly free the exposure, sell them using Step 6 above.
- **Reassigned positions become owned by the strategy.** The strategy will manage exits (stop-loss, time-exit, take-profit) according to its rules.
- **This operation is idempotent.** Running it again on an already-resolved symbol returns 404 (no `_UNKNOWN_` allocation found).

---

## 4. Quick Reference

### Full Rebuild (Both VPS)

```bash
# === LOCAL ===
git add -A && git commit -m "Describe changes" && git push origin main

# === VPS2 first ===
ssh vps2
cd /path/to/k_stock_trader
git pull origin main
docker exec trading_db psql -U postgres -d trading -c \
  "ALTER ROLE trading_writer WITH PASSWORD '$(grep POSTGRES_WRITER_PASSWORD .env | cut -d= -f2)';"
docker compose -f docker-compose.vps2.yml down
docker compose -f docker-compose.vps2.yml build --no-cache
docker compose -f docker-compose.vps2.yml up -d
sleep 20 && docker compose -f docker-compose.vps2.yml ps
curl -s http://localhost:8000/health

# === VPS1 second ===
ssh vps1
cd /path/to/k_stock_trader
git pull origin main
docker compose -f docker-compose.vps1.yml down
docker compose -f docker-compose.vps1.yml build --no-cache
docker compose -f docker-compose.vps1.yml up -d
sleep 20 && docker compose -f docker-compose.vps1.yml ps
curl -s http://localhost:8000/health
```

### Single-VPS Restart (No Code Changes)

```bash
# Restart without rebuilding (e.g., after .env change)
docker compose -f docker-compose.vpsX.yml restart

# Restart a single service
docker compose -f docker-compose.vpsX.yml restart kpr
```

### Rollback

```bash
# On either VPS — revert to previous commit and rebuild
git log --oneline -5                          # find previous commit hash
git checkout <previous-hash> -- .
docker compose -f docker-compose.vpsX.yml build
docker compose -f docker-compose.vpsX.yml up -d
```

### View Logs

```bash
# Tail live logs for a service
docker logs -f strategy_kmp

# Last 100 lines
docker logs --tail 100 oms_vps2

# Logs since a time
docker logs --since "2h" strategy_pcim
```

### Disk Cleanup

```bash
# Remove unused Docker images (after rebuilds)
docker image prune -f

# Remove all stopped containers + unused images + build cache
docker system prune -f

# Check disk usage
docker system df
df -h /
```

---

## 5. Environment Variable Reference

Each VPS needs a `.env` file. Copy `.env.example` as a starting point.

| Variable | Required | VPS | Description |
|----------|----------|-----|-------------|
| `KIS_APP_KEY` | Yes | Both | Real API app key (used for data fallback on paper) |
| `KIS_APP_SECRET` | Yes | Both | Real API app secret |
| `KIS_ACCOUNT_NO` | Yes | Both | Primary account number |
| `KIS_ACCOUNT_PROD_CODE` | No | Both | Account product code (default: `01`) |
| `KIS_IS_PAPER` | Yes | Both | `true` for paper trading, `false` for live |
| `KIS_HTS_ID` | Yes | Both | HTS user ID |
| `KIS_MY_AGENT` | No | Both | User-Agent string (default: `Mozilla/5.0`) |
| `KIS_PAPER_APP_KEY` | Yes* | Both | Paper trading app key (separate per VPS) |
| `KIS_PAPER_APP_SECRET` | Yes* | Both | Paper trading app secret |
| `KIS_PAPER_ACCOUNT_NO` | Yes* | Both | Paper trading account number |
| `POSTGRES_PASSWORD` | Yes | VPS2 | Postgres superuser password |
| `POSTGRES_WRITER_PASSWORD` | Yes | VPS2 | `trading_writer` role password |
| `POSTGRES_READER_PASSWORD` | No | VPS2 | `trading_reader` role password |
| `DATABASE_URL` | No | VPS1 | Full Postgres DSN pointing to VPS2 (empty = in-memory) |
| `GEMINI_API_KEY` | Yes** | VPS2 | Google Gemini API key (PCIM only) |
| `CONSERVATIVE_MODE` | No | Both | `true` for tighter entry filters (default: `false`) |
| `INSTRUMENTATION_HMAC_SECRET` | No | Both | HMAC secret for sidecar event relay (leave empty if unused) |

\* Required when `KIS_IS_PAPER=true`. Falls back to `KIS_APP_KEY`/`KIS_APP_SECRET`/`KIS_ACCOUNT_NO` if not set.
\** Required on VPS2 where PCIM runs.

**Key differences between VPS1 and VPS2 `.env`:**
- VPS1 uses `DATABASE_URL=postgresql://trading_writer:<password>@<VPS2_IP>:5432/trading` to connect to VPS2's Postgres. Leave empty to fall back to in-memory mode.
- VPS2 uses `POSTGRES_PASSWORD` and `POSTGRES_WRITER_PASSWORD` for local Postgres.
- Each VPS should have **different** `KIS_PAPER_APP_KEY`/`KIS_PAPER_APP_SECRET`/`KIS_PAPER_ACCOUNT_NO` to isolate paper trading accounts.

---

## 6. Troubleshooting Quick Reference

For detailed debug pipelines (per-strategy log grep patterns, common blockers, and resolution steps), see `implementation.md` sections:
- **Debug Pipeline: Why No Trades?** — strategy-by-strategy pipeline analysis
- **Troubleshooting** — token rate-limiting, universe filter rejections, Postgres warnings, paper trading rate limits, KRX tick sizes

| Symptom | Quick Check |
|---------|-------------|
| OMS says "in-memory" | `POSTGRES_WRITER_PASSWORD` mismatch — see Section 2 |
| All entries rejected | Frozen `_UNKNOWN_` positions — see Section 3 |
| "EGW00133" on startup | Normal token rate-limiting — wait 1-3 minutes |
| Strategy crash-looping | `docker logs <container> 2>&1 \| grep Traceback` |
| 0 trades all day | See `implementation.md` debug pipeline for that strategy |
| "KIS order rejected" | Paper API rate limit (5 req/sec) — retries automatically |
| OMS equity=0 | Check `docker logs oms_vpsX 2>&1 \| grep EQUITY_ZERO` |
