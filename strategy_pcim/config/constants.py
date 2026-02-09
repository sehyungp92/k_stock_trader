"""
PCIM-Alpha v1.3.1 Configuration Constants

DO NOT MODIFY without backtesting.
All thresholds, tolerances, and bands in one place.
"""

STRATEGY_ID = "PCIM"

# =============================================================================
# TIMING (KST)
# =============================================================================
TIMING = {
    "VIDEO_POLL_START": (20, 0),
    "VIDEO_POLL_END": (23, 59),
    "MARKET_STATS_REFRESH": (6, 0),
    "HUMAN_APPROVAL_START": (8, 0),
    "HUMAN_APPROVAL_END": (8, 30),
    "PREMARKET_CLASSIFY_START": (8, 40),
    "PREMARKET_CLASSIFY_END": (9, 0),
    "NO_TRADE_FIRST_SECONDS": 60,
    "BUCKET_A_SIGNAL_TIME": (9, 3, 5),
    "BUCKET_B_START": (9, 10),
    "CANCEL_ENTRIES_AT": (10, 0),
    "TRAILING_UPDATE_AFTER_CLOSE": (15, 35),
}

# =============================================================================
# GAP BUCKETING
# =============================================================================
BUCKETS = {
    "A": {"min": 0.00, "max": 0.03},
    "B": {"min": 0.03, "max": 0.07},
    "D": {"min": -1.0, "max": 1.0},
}

# =============================================================================
# EXECUTION VETOES
# =============================================================================
VETOES = {
    "MAX_SPREAD_PCT": 0.006,
    "NEAR_UPPER_LIMIT_TICKS": 2,
}

# =============================================================================
# BUCKET A: OPENING RANGE BAR
# =============================================================================
BUCKET_A = {
    "VOL_BASELINE_LOOKBACK_DAYS": 20,
    "VOL_RATIO_THRESHOLD": 1.20,
    "ORB_TOP_RANGE_PCT": 0.30,
    "FILL_TIMEOUT_SEC": 30,
}

# =============================================================================
# BUCKET B: VWAP TOUCH + RECLAIM
# =============================================================================
BUCKET_B = {
    "VWAP_TOUCH_TOL": 0.0010,
    "VWAP_RECLAIM_BUFFER": 0.0005,
    "TOUCH_RECLAIM_WINDOW_MINS": 2,
    "MAX_SIZE_PCT_OF_COMPUTED": 0.80,
}

# =============================================================================
# GAP REVERSAL FILTER
# =============================================================================
GAP_REVERSAL = {
    "LOOKBACK_DAYS": 60,
    "GAP_EVENT_MIN_PCT": 0.01,
    "THRESHOLD": 0.60,
    "MIN_EVENTS": 10,
}

# =============================================================================
# HARD FILTERS
# =============================================================================
HARD_FILTERS = {
    "ADTV_MIN": 5e9,
    "MCAP_MIN": 30e9,
    "MCAP_MAX": 15e12,
    "EARNINGS_WINDOW_DAYS": 5,
}

# =============================================================================
# SOFT FILTERS (MULTIPLIERS)
# =============================================================================
SOFT_FILTERS = {
    "ADTV_SOFT_LOW": 10e9,
    "ADTV_SOFT_HIGH": 15e9,
    "ADTV_SOFT_MULT": 0.5,
    "FIVEDAY_UP_PCT": 0.20,
    "FIVEDAY_MULT": 0.5,
}

# =============================================================================
# SIGNAL EXTRACTION
# =============================================================================
SIGNAL_EXTRACTION = {
    "CONVICTION_THRESHOLD": 0.7,  # Minimum conviction to accept signal
    "CONSOLIDATION_BOOST": 0.05,  # Boost per additional influencer
    "HUMAN_APPROVAL_REQUIRED": False,  # If False, auto-approve all eligible candidates
}

# =============================================================================
# TRADABILITY TIERS
# =============================================================================
TIERS = {
    "T1": {
        "adtv_min": 30e9,
        "size_mult": 1.0,
        "slip_band": 0.0020,
        "tv5m_participation": 0.15,
        "bucket_a_allowed": True,
        "order_type": "marketable_limit",
    },
    "T2": {
        "adtv_min": 15e9,
        "size_mult": 0.8,
        "slip_band": 0.0012,
        "tv5m_participation": 0.12,
        "bucket_a_allowed": True,
        "order_type": "marketable_limit",
    },
    "T3": {
        "adtv_min": 10e9,
        "size_mult": 0.5,
        "slip_band": None,
        "tv5m_participation": 0.08,
        "bucket_a_allowed": False,
        "order_type": "limit",
    },
}

TV5M_PROXY_DIVISOR = 78

# =============================================================================
# RISK / SIZING
# =============================================================================
SIZING = {
    "TARGET_RISK_PCT": 0.005,
    "STOP_ATR_MULT": 1.5,
    "SIZE_FLOOR_PCT": 0.20,
    "SINGLE_NAME_CAP_PCT": 0.15,
}

# =============================================================================
# PORTFOLIO CONTROLS
# =============================================================================
PORTFOLIO = {
    "MAX_OPEN_POSITIONS": 10,
    "KEEP_PARTIAL_FILL_PCT": 0.30,
}

# =============================================================================
# REGIME (KOSPI-based exposure caps)
# =============================================================================
REGIME = {
    "CRISIS": {"lt": -2.0, "max_exposure": 0.20, "disable_bucket_a": True},
    "WEAK":  {"lt":  0.0, "max_exposure": 0.50, "disable_bucket_a": True},
    "NORMAL":{"lt":  2.0, "max_exposure": 0.80, "disable_bucket_a": False},
    "STRONG":{"lt":  1e9, "max_exposure": 1.00, "disable_bucket_a": False},
}

INTRADAY_HALT_KOSPI_DD_PCT = -0.015

# =============================================================================
# EXIT PARAMETERS
# =============================================================================
EXITS = {
    "TAKE_PROFIT_ATR": 2.5,
    "TAKE_PROFIT_PCT": 0.60,
    "TRAIL_ATR": 1.5,
    "TIME_EXIT_DAY": 15,
}

# =============================================================================
# GEMINI LLM
# =============================================================================
GEMINI = {
    "MODEL_FLASH_25": "gemini-2.5-flash",
    "MODEL_FLASH_3": "gemini-3-flash-preview",
    "MAX_RETRIES": 5,
    "THINKING_BUDGET": {
        "LOW": 1024,
        "MEDIUM": 8192,
        "HIGH": 24576,
    },
}

# =============================================================================
# YOUTUBE
# =============================================================================
YOUTUBE = {
    "MAX_VIDEOS_PER_CHANNEL": 15,
    "VIDEO_CUTOFF_HOUR": 15,
    "VIDEO_MORNING_CUTOFF_HOUR": 8,
    "VIDEO_MORNING_CUTOFF_MIN": 30,
    "STATE_FILE": "pcim_youtube_state.json",
}
