from __future__ import annotations

from typing import Any

from strategy_common.market import MarketBar


def market_regime_allows(mutations: dict[str, Any], bar: MarketBar) -> bool:
    mode = str(mutations.get("market_regime_filter", "all") or "all").lower()
    if mode in {"", "all", "none"}:
        return True
    regime = str(bar.metadata.get("market_regime", "") or "").lower()
    if not regime:
        return True
    if mode == "hot_only":
        return regime in {"hot", "red_hot", "risk_on"}
    if mode == "avoid_weak":
        return regime not in {"weak", "risk_off", "chop", "choppy"}
    return True
