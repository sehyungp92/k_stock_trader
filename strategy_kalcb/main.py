from __future__ import annotations

import json

from .config import KALCBConfig
from .data import WebSocketRegistrationBudget


def health_status(config: KALCBConfig | None = None) -> dict:
    cfg = config or KALCBConfig()
    budget = WebSocketRegistrationBudget(
        max_registrations=cfg.ws_max_registrations,
        reserved_execution_regs=cfg.ws_reserved_execution_regs,
        hot_regs_per_symbol=cfg.ws_hot_regs_per_symbol,
        strategy_symbol_budget=cfg.ws_budget,
        rest_egw00201_cooldown_s=cfg.rest_egw00201_cooldown_s,
    )
    return {
        "strategy_id": cfg.strategy_id,
        "status": "ready",
        "timeframe": cfg.timeframe,
        "live_parity_fill_timing": cfg.live_parity_fill_timing,
        "auction_mode": cfg.auction_mode,
        "carry_mode": cfg.carry_mode.value,
        "ws_budget": budget.snapshot(),
        "rest_min_interval_paper_s": cfg.rest_min_interval_paper_s,
        "rest_min_interval_live_s": cfg.rest_min_interval_live_s,
    }


def main() -> int:
    print(json.dumps(health_status(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
