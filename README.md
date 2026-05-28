# k_stock_trader

Algorithmic trading system for the Korean stock market (KRX), built on the Korea Investment & Securities (KIS) API.

PCIM runs through the centralised Order Management System (OMS) with pre-trade risk checks. KALCB and OLR remain available for research, backtests, artifact generation, and deployment readiness workflows.

## Architecture

```
Single-VPS Docker runtime
+---------------------------+
|  PCIM Strategy            |
|  OLR/KALCB Runtime        |
+---------------------------+
|  OMS                      |
|  PostgreSQL               |
|  Dashboard                |
+---------------------------+
              |
    +---------v---------+
    |    KIS API        |
    |  REST + WebSocket |
    +-------------------+
```

## Strategies

| Strategy | Style | Trading Hours (KST) | Signal Source |
|----------|-------|---------------------|---------------|
| **KALCB** | Intraday breakout research | Research/backtest | Completed-bar KRX/KIS replay |
| **OLR** | Overnight leader rotation | Research/backtest | Daily flow + afternoon execution |
| **PCIM** | Premarket catalyst | 09:01 - 10:30 | YouTube influencers + Gemini AI |

### KALCB

KALCB research and backtest modules evaluate completed-bar breakout candidates with artifact-backed replay paths and promotion checks.

### OLR

OLR research and backtest modules build overnight leader selections from daily/flow data, then evaluate afternoon execution plans with explicit holdout and parity artifacts.

### PCIM - AI Premarket Intelligence

Overnight pipeline fetches YouTube videos from configured influencer channels, extracts trading signals via Google Gemini, then scores/filters candidates through gap-reversal checks and trend gates. Two execution buckets (A: early trigger, B: normal) stage entries at market open.

PCIM front-runs the retail attention wave by processing influencer signals overnight before the market opens. AI extraction and scoring allows systematic participation in the opening momentum that retail viewers generate, with trend and gap-reversal filters to avoid crowded or exhausted setups.

## Project Structure

```
k_stock_trader/
|-- kis_core/            # KIS API wrapper: REST, WebSocket, auth, rate limiting
|-- oms/                 # Order Management System: risk gateway, state, persistence
|-- oms_client/          # Strategy-side OMS client library
|-- strategy_kalcb/      # KALCB strategy/research support
|-- strategy_olr/        # OLR strategy/research support
|-- strategy_pcim/       # PCIM strategy
|-- config/              # YAML configs per strategy + OMS
|-- deployment/          # Runtime entrypoints and Docker images
|-- infra/               # PostgreSQL init scripts, dashboard config, cron wrappers
|-- scripts/             # Utility scripts (artifacts, migrations, backups, health checks)
|-- tests/               # Unit + integration tests
|-- docker-compose.yml   # Canonical single-VPS compose stack
```

## Core Components

### OMS (Order Management System)

FastAPI service that sits between strategies and the KIS broker API.

- **Intent-based ordering**: strategies submit `Intent` objects (enter/exit/scale), OMS decides execution
- **Pre-trade risk gateway**: global limits, daily P&L, exposure caps, per-strategy budgets, sector caps
- **Position allocation**: virtual per-strategy allocations on top of real broker positions
- **Reconciliation**: periodic sync with KIS to detect external fills/cancels
- **Persistence**: full audit trail in PostgreSQL (intents, orders, fills, trades)

### kis_core

Shared library wrapping the KIS Open API.

- REST client with rate limiting and exponential backoff
- WebSocket client for real-time tick/ask-bid data (40 slots per account)
- VWAP computation, bar aggregation, technical indicators
- Universe filtering (market cap, ADTV, listing status)
- Sector exposure tracking

### Database

PostgreSQL stores the full trade lifecycle:

- `intents` / `orders` / `fills` / `trades` - order flow audit trail
- `positions` / `allocations` - real and virtual position state
- `risk_daily_strategy` / `risk_daily_portfolio` - daily risk snapshots
- Dashboard views: `v_live_positions`, `v_live_allocations`, `v_today_risk`, `v_strategy_performance`, `v_service_health`

## Setup

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- KIS Open API credentials (paper or production)
- Google Gemini API key (PCIM strategy only)

### Environment

Copy `.env.example` to `.env` and fill in:

```bash
# KIS API credentials
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_ACCOUNT_NO=...
KIS_IS_PAPER=true

# Paper trading credentials (separate from real)
KIS_PAPER_APP_KEY=...
KIS_PAPER_APP_SECRET=...
KIS_PAPER_ACCOUNT_NO=...

# Database
POSTGRES_PASSWORD=...
POSTGRES_WRITER_PASSWORD=...

# PCIM only
GEMINI_API_KEY=...
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start base services locally (Postgres + OMS)
docker compose up --build

# Start optional strategy/dashboard services
docker compose --profile pcim up --build pcim
docker compose --profile dashboard up --build dashboard

# Start the OLR/KALCB runtime after artifacts and runtime env are ready
docker compose --profile olr-kalcb up --build runtime
```

### Production Deployment

```bash
# Build the canonical single-VPS stack, including profiled services
docker compose --profile all build

# Start shared infrastructure first
docker compose up -d postgres oms

# Apply existing-database migrations when upgrading an installed VPS
scripts/apply_db_migrations.sh

# Start optional services for the portfolio
docker compose --profile dashboard up -d dashboard
docker compose --profile pcim up -d pcim

# Generate date-sensitive OLR/KALCB artifacts explicitly, then restart runtime
infra/cron/olr_kalcb_premarket_restart.sh
infra/cron/olr_kalcb_afternoon_restart.sh
```

The premarket wrapper defaults to the approved OLR deployment universe manifest at `config/olr_kalcb/olr_deployment_universe_103.yaml`; replacing it should be a reviewed config change, not an ad hoc runtime fallback.

### Configuration

Each strategy has a YAML config in `config/`:

| File | Purpose |
|------|---------|
| `kalcb.yaml` | KALCB runtime/research settings |
| `olr.yaml` | OLR runtime/research settings |
| `pcim.yaml` | YouTube channels, AI settings, filters |
| `oms_config.yaml` | Risk limits, exposure caps, reconciliation |
| `conservative.yaml` | Tighter thresholds for cautious mode |

Set `CONSERVATIVE_MODE=true` in `.env` to activate tighter entry filters across all strategies.

## Testing

```bash
# All tests
pytest tests/ -v

# By component
pytest tests/oms/ -v
pytest tests/strategy_kalcb/ -v
pytest tests/strategy_olr/ -v
pytest tests/strategy_pcim/ -v
pytest tests/kis_core/ -v
```

## Monitoring

### Dashboard

Available on the single VPS at port 3000 after starting the `dashboard` profile. Lightweight Next.js dashboard (`infra/dashboard/`) reads shared Postgres views server-side and shows positions, risk, and service health.

### Log Diagnostics

Containers log structured diagnostics. Key grep patterns:

```bash
# OMS health
docker compose logs oms 2>&1 | grep "KIS order rejected\|Limit BUY failed"

# Strategy status
docker compose logs pcim 2>&1 | grep "Signal\|Entry rejected\|OMS returned"

# Risk events
docker compose logs oms 2>&1 | grep -i "reject\|halt\|breach"
```
