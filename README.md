# k_stock_trader

Algorithmic trading system for the Korean stock market (KRX), built on the Korea Investment & Securities (KIS) Open API.

The active portfolio is **KALCB** (intraday breakout) and **OLR** (overnight leader rotation with afternoon execution). Both run through a single artifact-gated runtime service that talks to the centralised Order Management System (OMS) for pre-trade risk checks, order routing, and reconciliation.

## Architecture

```
Single-VPS Docker stack
+--------------------------------------------+
|  runtime  (KALCB + OLR, artifact-gated)    |
+--------------------------------------------+
|  oms        (FastAPI, risk + KIS routing)  |
|  postgres   (trade lifecycle + state)      |
|  dashboard  (Next.js, optional profile)    |
+--------------------------------------------+
                    |
            +-------v--------+
            |    KIS API     |
            | REST + WS      |
            +----------------+
```

## Strategies

| Strategy | Style | Selection Cutoff (KST) | Signal Source |
|----------|-------|------------------------|---------------|
| **KALCB** | Intraday breakout | Premarket daily artifact | KRX daily + KIS completed-bar replay |
| **OLR** | Overnight leader rotation | Stage1 premarket + final at 14:30 | Daily flow (foreign/inst) + afternoon execution context |

### KALCB

Structural-campaign breakout on a finalized daily candidate set. The premarket artifact pipeline scores universe symbols on relative strength, daily trend, and compression, then freezes the day's candidates into `data/strategy/kalcb/`. Runtime consumes completed bars and routes intents through OMS.

### OLR

Two-stage artifact pipeline. **Stage 1** runs premarket to build the overnight leader snapshot from daily OHLCV, foreign/institutional flow, and sector panels. **Final** runs after the 14:30 KST cutoff to layer the afternoon execution context (sector intraday panel, ranked entry plan) on top. The runtime is restarted between stages so it loads each artifact at the right time.

Both strategies share a portfolio arbitration layer (`deployment/olr_kalcb/portfolio.py`) and a single readiness manifest at `data/live_readiness/olr_kalcb/<DATE>/baseline_manifest.json`.

## Project Structure

```
k_stock_trader/
|-- kis_core/                # KIS API wrapper: REST, WS, auth, rate limiting
|-- oms/                     # OMS service: risk gateway, allocation, reconciliation
|-- oms_client/              # Strategy-side OMS HTTP client
|-- strategy_kalcb/          # KALCB research, signals, models, artifact store
|-- strategy_olr/            # OLR research, models, artifact store
|-- strategy_common/         # Shared clock, sector maps, daily/intraday panels, parquet loaders
|-- strategy_pcim/           # PCIM (research mode, not part of the deployed portfolio)
|-- deployment/olr_kalcb/    # Unified runtime: coordinator, readiness, replay, session driver
|-- config/                  # YAML/JSON configs (see below)
|-- scripts/                 # Artifact generation, preflight, optimization, promotion, parity
|-- infra/cron/              # Premarket + afternoon restart wrappers
|-- infra/postgres/          # DB init + retention SQL
|-- infra/dashboard/         # Next.js dashboard
|-- data/                    # Artifacts, parquet, paper sessions, readiness manifests
|-- tests/                   # Unit + integration tests
|-- docker-compose.yml       # Canonical single-VPS stack
```

## Core Components

### OMS

FastAPI service between strategies and KIS.

- Intent-based ordering (`enter` / `exit` / `scale`) with deterministic idempotency keys
- Pre-trade risk gateway: gross exposure, daily P&L, per-strategy budgets, sector caps, frozen symbols
- Virtual per-strategy allocations on top of real broker positions, updated on fill detection
- Reconciliation against KIS with `_UNKNOWN_` allocation on drift
- Full audit trail in Postgres (intents, orders, fills, trades, allocations)

### OLR/KALCB Runtime (`deployment/olr_kalcb/`)

One container, both strategies. Key pieces:

- `runtime.py` — preflight, mode gating, session preparation
- `coordinator.py` / `session_driver.py` — completed-bar dispatch into each strategy
- `action_router.py` — routes strategy actions through `OMSClient` or the dry-run recorder
- `portfolio.py` / `portfolio_context.py` — cross-strategy arbitration
- `readiness.py` — artifact validation + health-check gating
- `market_data_coordinator.py` — KIS WS completed-bar source with offline replay fallback
- `replay.py` / `session_capture.py` — paper session capture and replay for parity audits

Runtime modes form an escalating gate: `artifact_only_stage1` → `artifact_only` → `dry_run` → `paper` → `live`, each requiring the previous mode's gate to have passed plus its own health checks.

### Database

PostgreSQL stores the full trade lifecycle and readiness state:

- `intents` / `orders` / `order_events` / `fills` / `trades`
- `positions` / `allocations`
- `risk_daily_strategy` / `risk_daily_portfolio` / `recon_log` / `oms_state` / `strategy_state`
- Views: `v_live_positions`, `v_live_allocations`, `v_today_risk`, `v_strategy_performance`, `v_service_health`

Retention: see `infra/postgres/retention.sql` (order_events 60d, intents 90d, recon_log 30d).

## Setup

### Prerequisites

- Python 3.12+
- Docker + Docker Compose
- KIS Open API credentials (paper or production)
- KRX daily parquet + KIS intraday parquet mounted at `data/krx_daily_parquet/` and `data/kis_intraday_parquet/`

### Environment

Copy `.env.example` to `.env` and fill in:

```bash
# KIS API credentials
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_ACCOUNT_NO=...
KIS_ACCOUNT_PROD_CODE=01
KIS_IS_PAPER=true
KIS_HTS_ID=...

# Paper trading credentials (optional, if separate)
KIS_PAPER_APP_KEY=...
KIS_PAPER_APP_SECRET=...
KIS_PAPER_ACCOUNT_NO=...

# Postgres
POSTGRES_PASSWORD=...
POSTGRES_WRITER_PASSWORD=...
POSTGRES_READER_PASSWORD=...

# OLR/KALCB runtime
OLR_KALCB_RUNTIME_MODE=dry_run
OLR_KALCB_DAILY_UNIVERSE_FILE=config/olr_kalcb/olr_deployment_universe_103.yaml
OLR_KALCB_BASELINE_MANIFEST=/app/data/live_readiness/olr_kalcb/<DATE>/baseline_manifest.json
OLR_KALCB_BARS_PARQUET=    # required before starting runtime container
OLR_KALCB_START_RUNTIME_AFTER_PREMARKET=false
OLR_KALCB_RESTART_RUNTIME_AFTER_AFTERNOON=true
```

### Configuration

| File | Purpose |
|------|---------|
| `config/kalcb.yaml` | KALCB strategy + research settings |
| `config/olr/sector_map.yaml` | Canonical OLR sector mapping |
| `config/olr_kalcb/olr_deployment_universe_103.yaml` | Approved 103-symbol deployment universe |
| `config/olr_kalcb/portfolio_policy.conservative.json` | Cross-strategy arbitration policy |
| `config/backtests/kalcb.yaml` | KALCB backtest configuration |
| `config/optimization/kalcb.yaml`, `config/optimization/olr.yaml` | Optimization search configs |
| `config/optimization/portfolio_synergy.yaml` | Joint portfolio search |
| `config/oms_config.yaml` | Risk limits, exposure caps, reconciliation |
| `trading_assistant_backtest/contracts/k_stock_olr_kalcb/strategy_plugin_contract.json` | Assistant bridge contract for OLR/KALCB parameters, event joins, and deployment evidence |
| `config/conservative.yaml` | Tighter thresholds (`CONSERVATIVE_MODE=true`) |
| `config/universe_103.yaml` | Base universe definition |

Replacing the approved deployment universe should be a reviewed config change, not an ad-hoc runtime override.

### Assistant bridge contract

Refresh the checked-in OLR/KALCB assistant contract after changing strategy config fields, OMS risk fields, the approved universe, sector map, or portfolio policy:

```bash
python scripts/generate_olr_kalcb_bridge_contract.py
```

For paper/live approval runs, emit runtime deployment metadata against that contract:

```bash
python scripts/run_olr_kalcb_runtime_session.py watch-bars \
  --trade-date 2026-05-28 \
  --mode paper \
  --deployment-metadata-json data/paper_live/olr_kalcb/2026-05-28/deployment_metadata.json
```

The metadata writer records the strategy plugin contract hash, clean source-control provenance, strategy/config/resource-plan hashes, staged artifact hashes, KIS resource-plan hash, and runtime instance details.

## Daily Operations

### Premarket gate

Run after KRX/KIS daily data is refreshed, before market open:

```bash
infra/cron/olr_kalcb_premarket_restart.sh
```

This:
1. Brings up `postgres` and `oms`.
2. Stops the current `runtime` container.
3. Generates the KALCB daily artifact and the OLR stage1 artifact.
4. Runs the `artifact_only_stage1` preflight gate.
5. Optionally starts the runtime if `OLR_KALCB_START_RUNTIME_AFTER_PREMARKET=true` (requires `OLR_KALCB_BARS_PARQUET`).

### Afternoon gate

Run after the 14:30 KST completed-bar cutoff:

```bash
infra/cron/olr_kalcb_afternoon_restart.sh
```

This generates the OLR final afternoon artifact, runs the `artifact_only` gate, then recreates the `runtime` container so it loads the final OLR snapshot.

### Manual artifact + preflight commands

```bash
# KALCB daily + OLR stage1 (premarket)
docker compose run --rm runtime \
  python scripts/generate_olr_kalcb_artifacts.py daily \
    --trade-date 2026-05-28 \
    --daily-universe-file config/olr_kalcb/olr_deployment_universe_103.yaml

# OLR final (afternoon, after 14:30 KST)
docker compose run --rm runtime \
  python scripts/generate_olr_kalcb_artifacts.py afternoon \
    --trade-date 2026-05-28

# Preflight (modes: artifact_only_stage1 | artifact_only | dry_run | paper | live)
docker compose run --rm runtime \
  python scripts/run_olr_kalcb_runtime_session.py preflight \
    --trade-date 2026-05-28 --mode dry_run

# Replay completed bars from parquet without starting the runtime loop
docker compose run --rm runtime \
  python scripts/run_olr_kalcb_runtime_session.py dry-run-bars \
    --trade-date 2026-05-28 --bars-parquet /app/data/kis_intraday_parquet/...
```

### Local development

```bash
pip install -r requirements.txt
pytest tests/ -v

# Base services
docker compose up -d postgres oms

# Runtime (after artifacts + readiness manifest exist for the trade date)
docker compose --profile olr-kalcb up --build runtime

# Dashboard (optional)
docker compose --profile dashboard up -d dashboard

# Apply DB migrations on an existing VPS
scripts/apply_db_migrations.sh
```

## Research, Backtests, and Promotion

Research and optimization live in `scripts/` and the strategy packages. Key entry points:

- `strategy_kalcb/research.py`, `strategy_kalcb/research_generator.py` — KALCB candidate generation
- `strategy_olr/research.py`, `strategy_olr/research_generator.py` — OLR research pipeline
- `scripts/kalcb_*.py` — KALCB optimization sweeps, OOS ablations, route conversion, structural reruns
- `scripts/olr_*.py` — OLR score-band searches, sector data sweeps, reject filter analysis
- `scripts/promote_kalcb_*.py`, `scripts/promote_olr_*.py` — promotion scripts that snapshot champion configs into `data/strategy/<strategy>/`
- `scripts/replay_paper_session.py`, `scripts/summarize_paper_parity.py` — paper/live parity audits

Backtest configs are under `config/backtests/`; optimization configs under `config/optimization/`.

## Monitoring

### Dashboard

`docker compose --profile dashboard up -d dashboard` exposes the Next.js dashboard on port 3000. It reads shared Postgres views server-side and shows positions, allocations, risk, and service health.

### Log diagnostics

```bash
# Runtime artifact + preflight events
docker compose logs runtime 2>&1 | grep -E "artifact|preflight|gate"

# OMS health and rejections
docker compose logs oms 2>&1 | grep -iE "reject|halt|breach|KIS order"

# Strategy actions routed through OMS
docker compose logs runtime 2>&1 | grep -E "KALCB|OLR" | grep -iE "intent|fill|exit"
```

## Testing

```bash
pytest tests/ -v

pytest tests/oms/ -v
pytest tests/strategy_kalcb/ -v
pytest tests/strategy_olr/ -v
pytest tests/kis_core/ -v
pytest tests/deployment/olr_kalcb/ -v
```
