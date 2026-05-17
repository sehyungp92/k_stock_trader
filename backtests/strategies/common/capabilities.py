from __future__ import annotations

KMP_OFFICIAL_REQUIREMENTS = ("tick", "bid_ask", "spread", "vi", "tick_imbalance")
KPR_OFFICIAL_REQUIREMENTS = ("investor", "program", "micro_pressure")
NULRIMOK_OFFICIAL_REQUIREMENTS = ("historical_lrs", "dse_artifacts", "30m_ohlcv")


def require_capabilities(strategy: str, requested_level: str, available: set[str], requirements: tuple[str, ...]) -> None:
    level = (requested_level or "synthetic").lower()
    if level not in {"official", "feature_complete"}:
        return
    missing = [item for item in requirements if item not in available]
    if missing:
        raise ValueError(
            f"{strategy} {requested_level} replay requires missing feature bundle(s): {', '.join(missing)}"
        )

