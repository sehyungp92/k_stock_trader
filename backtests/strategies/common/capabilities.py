from __future__ import annotations

KMP_OFFICIAL_REQUIREMENTS = ("ohlcv",)
KPR_OFFICIAL_REQUIREMENTS = ("investor", "program", "micro_pressure")
NULRIMOK_OFFICIAL_REQUIREMENTS = (
    "historical_lrs",
    "dse_artifacts",
    "watchlist_artifacts",
    "30m_ohlcv",
    "daily_ohlcv",
    "daily_flow",
    "benchmark_index",
    "sector_map",
)
KIARIC_OFFICIAL_REQUIREMENTS = (
    "ohlcv_5m",
    "daily_ohlcv",
    "benchmark_index",
    "sector_map",
    "kiaric_daily_candidate_artifact",
    "session_vwap",
    "intraday_volume_profile",
    "krx_tick_table",
    "halt_vi_state",
)
KALCB_OFFICIAL_REQUIREMENTS = (
    "completed_5m_signal_bars",
    "prior_completed_daily_ohlcv",
    "intraday_rvol_curve",
    "session_vwap",
    "opening_range",
    "candidate_artifact",
    "market_regime_inputs",
    "krx_tick_table",
)


def require_capabilities(strategy: str, requested_level: str, available: set[str], requirements: tuple[str, ...]) -> None:
    level = (requested_level or "synthetic").lower()
    if level not in {"official", "feature_complete"}:
        return
    missing = [item for item in requirements if item not in available]
    if missing:
        raise ValueError(
            f"{strategy} {requested_level} replay requires missing feature bundle(s): {', '.join(missing)}"
        )
