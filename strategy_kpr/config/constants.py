"""KPR v4.3 Pinned Constants"""

STRATEGY_ID = "KPR"

# Timing
ENTRY_START, ENTRY_END = (9, 10), (14, 0)
LUNCH_START, LUNCH_END = (11, 20), (13, 10)

# Setup Detection
VWAP_DEPTH_MIN, VWAP_DEPTH_MAX = 0.02, 0.05
PANIC_DROP_PCT, PANIC_MAX_AGE_MIN = 0.03, 15
DRIFT_DROP_PCT, DRIFT_MIN_AGE_MIN = 0.02, 60

# Acceptance
BASE_ACCEPT_CLOSES = 2
RECLAIM_OFFSET_PCT, STOP_BUFFER_PCT = 0.003, 0.003

# Sizing
BASE_RISK_PCT = 0.005
GREEN_SIZE_MULT, YELLOW_SIZE_MULT = 1.0, 0.65
# Time-of-day sizing: 09:30-10:30=1.0, 10:30-11:20=0.8, 13:10-14:00=0.9, 14:00+=0.5
TOD_BRACKETS = [
    ((9, 30), (10, 30), 1.0),
    ((10, 30), (11, 20), 0.8),
    ((13, 10), (14, 0), 0.9),
    ((14, 0), (15, 30), 0.5),
]
TOD_DEFAULT_MULT = 0.8  # fallback (e.g., during lunch block)

# Exit
TIME_STOP_MINUTES = 45
PARTIAL_R_TARGET = 1.5
# Adaptive full R target by volatility: high=2.5, normal=2.0, low=1.5
FULL_R_TARGET_HIGH_VOL = 2.5
FULL_R_TARGET_NORMAL = 2.0
FULL_R_TARGET_LOW_VOL = 1.5
TRAIL_START_R = 1.0
TRAIL_FACTOR = 0.5
FLOW_DETERIORATE_TRAIL = 0.7

# Universe Tiers
HOT_MAX, WARM_MAX = 40, 25
WARM_POLL_DEFAULT, WARM_POLL_MICRO = 30, 15
COLD_POLL_SEC = 180
FLOW_STALE_DEFAULT, FLOW_STALE_MICRO = 300, 120
STALE_SIZE_PENALTY = 0.85  # Size multiplier when investor flow is stale

# Drift / Order timeout
ORDER_TIMEOUT_SEC = 30  # Cancel pending orders after this many seconds
DRIFT_CHECK_INTERVAL = 2.0  # Seconds between drift checks
MAX_SECTOR_POSITIONS = 2  # Maximum positions per sector

# Micro Windows
MICRO_WINDOWS = [((9, 10), (9, 30)), ((14, 0), (14, 30))]

# Micro Pressure (bar-strength proxy)
VOL_SURGE_THRESHOLD = 1.5
BAR_STRENGTH_BULL = 0.6
BAR_STRENGTH_BEAR = 0.3
MICRO_LOOKBACK_BARS = 20

# Program cache TTL
PROGRAM_CACHE_TTL = 120
