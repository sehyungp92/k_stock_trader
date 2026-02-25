# k_stock_trader

Multi-strategy algorithmic trading system for the Korean stock market (KRX), built on the Korea Investment & Securities (KIS) API.

Four independent strategies share a centralized Order Management System (OMS) with pre-trade risk checks, deployed across Docker containers on two VPS instances.

## Architecture

```
VPS 1 (Account 1)                    VPS 2 (Account 2)
+-----------------------+            +-----------------------+
|  KMP Strategy         |            |  KPR Strategy         |
|  (intraday momentum)  |            |  (flow mean-reversion)|
+-----------------------+            +-----------------------+
|  Nulrimok Strategy    |            |  PCIM Strategy        |
|  (swing / multi-day)  |            |  (AI premarket)       |
+-----------------------+            +-----------------------+
|  OMS Instance 1       |--- TCP --->|  OMS Instance 2       |
|  (PostgreSQL remote)  |   :5432    |  (PostgreSQL local)   |
+-----------------------+            +-----------------------+
                                     |  PostgreSQL + Metabase|
                                     +-----------------------+
            |                                   |
            +----------------+------------------+
                             |
                  +----------v----------+
                  |    KIS Paper API    |
                  |  40 WS + REST each  |
                  +---------------------+
```

## Strategies

| Strategy | Style | Trading Hours (KST) | Signal Source |
|----------|-------|---------------------|---------------|
| **KMP** | Intraday momentum | 09:15 - 14:30 | Price breakouts + program flow |
| **KPR** | Intraday mean-reversion | 09:10 - 14:00 | VWAP pullbacks + investor flow |
| **Nulrimok** | Swing (multi-day) | 09:00 - 15:00 | AVWAP band + smart money flow |
| **PCIM** | Premarket catalyst | 09:01 - 10:30 | YouTube influencers + Gemini AI |

### KMP - Momentum Breakout

Scans the universe at 09:15 for trend-aligned breakout candidates. An FSM steps each candidate through gates: trend anchor, relative volume surge (2x+), spread check, regime/breadth gate, acceptance timeout, then entry. Positions flatten by 14:30.

### KPR - VWAP Pullback Reversal

Watches for pullbacks to VWAP in a tiered universe (HOT/WARM/COLD). Combines investor flow, program flow, and micro-pressure signals to confirm entries. Drift monitoring detects regime shifts for exits.

### Nulrimok - Swing Flow Strategy

Pre-market DSE (Daily Selection Engine) ranks the universe by flow score, relative strength, sector weight, and AVWAP proximity. During the day, IEPE (Intraday Entry/Position Engine) watches 30-minute bars for band entry setups. Positions are held across sessions with flow-reversal exits.

### PCIM - AI Premarket Intelligence

Overnight pipeline fetches YouTube videos from configured influencer channels, extracts trading signals via Google Gemini, then scores/filters candidates through gap-reversal checks and trend gates. Two execution buckets (A: early trigger, B: normal) stage entries at market open.

## Project Structure

```
k_stock_trader/
|-- kis_core/            # KIS API wrapper: REST, WebSocket, auth, rate limiting
|-- oms/                 # Order Management System: risk gateway, state, persistence
|-- oms_client/          # Strategy-side OMS client library
|-- strategy_kmp/        # KMP strategy
|-- strategy_kpr/        # KPR strategy
|-- strategy_nulrimok/   # Nulrimok strategy
|-- strategy_pcim/       # PCIM strategy
|-- config/              # YAML configs per strategy + OMS
|-- infra/               # PostgreSQL init scripts, dashboard config
|-- scripts/             # Utility scripts (LRS backfill, health checks)
|-- tests/               # Unit + integration tests
|-- docker-compose.yml          # Local development (all services)
|-- docker-compose.vps1.yml     # Production VPS 1 (KMP + Nulrimok)
|-- docker-compose.vps2.yml     # Production VPS 2 (KPR + PCIM + DB)
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
- Dashboard views: `v_live_positions`, `v_today_risk`, `v_strategy_performance`, `v_service_health`

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

# Start all services locally
docker compose up --build

# Or start specific strategies using profiles
docker compose --profile kmp up
docker compose --profile nulrimok up
```

### Production Deployment

```bash
# VPS 1: KMP + Nulrimok
docker compose -f docker-compose.vps1.yml up -d

# VPS 2: KPR + PCIM + Database + Dashboard
docker compose -f docker-compose.vps2.yml up -d
```

### Configuration

Each strategy has a YAML config in `config/`:

| File | Purpose |
|------|---------|
| `kmp.yaml` | Universe, OR window, risk parameters |
| `kpr.yaml` | Universe tiers, signal weights |
| `nulrimok.yaml` | Universe, sector map, LRS path |
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
pytest tests/strategy_kmp/ -v
pytest tests/strategy_nulrimok/ -v
pytest tests/strategy_pcim/ -v
pytest tests/strategy_kpr/ -v
pytest tests/kis_core/ -v
```

## Monitoring

### Metabase Dashboard

Available on VPS 2 at port 3000 when the `dashboard` profile is active. Connects to PostgreSQL for live views of positions, risk, and strategy performance.

### Log Diagnostics

Each strategy container logs structured diagnostics. Key grep patterns:

```bash
# OMS health
docker logs oms_vps1 2>&1 | grep "KIS order rejected\|Limit BUY failed"

# Strategy status
docker logs strategy_kmp 2>&1 | grep "Scan complete\|blocked by\|Entry rejected"
docker logs strategy_nulrimok 2>&1 | grep "DSE:\|Entry conditions not met\|Armed\|OMS returned"

# Risk events
docker logs oms_vps1 2>&1 | grep -i "reject\|halt\|breach"
```

See `implementation.md` for detailed per-strategy debug pipelines.
