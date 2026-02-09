"""
KMP v2.3.3 Pinned Constants

DO NOT MODIFY without backtesting.
"""

# Timing
ENTRY_CUTOFF = (10, 0)          # 10:00 KST - no new entries after
FLATTEN_TIME = (14, 30)         # 14:30 KST - flatten all
OR_LOCK_TIME = (9, 15)          # Lock opening range at 09:15
SCAN_TIME = (9, 15)             # Value surge scan time

# Opening Range
OR_RANGE_MIN = 0.012            # 1.2% minimum OR range
OR_RANGE_MAX = 0.055            # 5.5% maximum OR range

# Gates
RVOL_MIN = 2.0                  # Minimum relative volume
SPREAD_MAX_PCT = 0.004          # 0.40% max spread
GAP_SKIP = 0.05                 # Skip 5%+ gap stocks

# VI (Volatility Interruption)
VI_WALL_TICKS = 10              # Ticks from VI trigger to block
VI_COOLDOWN_MIN = 10            # Minutes cooldown after VI

# Acceptance
ACCEPT_TIMEOUT_MIN = 5          # Minutes to wait for acceptance

# Sizing
BASE_RISK_PCT = 0.005           # 0.5% NAV risk per trade
LIQ_CAP_PCT_5M_VALUE = 0.05    # Max 5% of 5-min traded value

# Time Decay (linear)
MIN_SURGE_BASE = 3.0            # Base surge requirement
MIN_SURGE_SLOPE = 0.04          # Surge decay per minute
SIZE_DECAY_SLOPE = 0.012        # Size decay per minute
SIZE_DECAY_FLOOR = 0.45         # Minimum size multiplier

# Exit
STALL_MIN_MINUTES = 8           # Minutes before stall check
STALL_R_MIN = 0.5               # Minimum R for stall exit
HARD_STOP_ATR_MULT = 1.2        # Hard stop = entry - 1.2*ATR

# WebSocket Budget
# KIS allows 41 total registrations per session; use 40 to leave room for notifications
WS_MAX_REGS = 40                # Max combined WS registrations
FOCUS_MAX = 10                  # Max symbols with H0STASP0 (increased from 6)

# Program Regime
PROGRAM_POLL_SEC = 60           # REST poll interval
EWMA_ALPHA = 0.35               # EWMA smoothing factor

# Quality Score Thresholds
QUALITY_THRESHOLD_LOW = 40      # Below = skip
QUALITY_THRESHOLD_MED = 60      # Below = 0.5x size
QUALITY_THRESHOLD_HIGH = 80     # Above = 1.5x size

# Strategy ID
STRATEGY_ID = "KMP"
