"""
KIS - Korea Investment & Securities API Client

Shared library for interacting with KIS API:
- Authentication and token management
- REST API client for trading operations
- WebSocket support for real-time data
- Paper trading support with TR_ID mapping
"""

from .kis_auth import KoreaInvestEnv
from .kis_client import (
    KoreaInvestAPI,
    CircuitBreaker,
    get_paper_tr_id,
    PAPER_TR_ID_MAP,
    PAPER_TR_ID_PASSTHROUGH,
)
from .kis_decorators import (
    rate_limit,
    rate_limit_async,
    RateLimiter,
    get_global_limiter_stats,
)
from .kis_responses import APIResponse, create_error_response
from .vwap import VWAPLedger, compute_anchored_daily_vwap, vwap_band
from .bar_aggregator import Bar, BarAggregator, aggregate_bars
from .rate_budget import TokenBucket, RateBudget, RateLimitedError
from .shared_rate_budget import (
    SharedRateBudget,
    SharedRateBudgetClient,
    PriorityTokenBucket,
    PRIORITY_WINDOWS,
    create_strategy_client,
    get_shared_budget,
)
from .indicators import (
    sma, ema, atr, zscore, percentile_rank,
    RollingSMA, RollingATR,
)
from .ws_client import (
    KISWebSocketClient,
    BaseSubscriptionManager,
    TickMessage,
    AskBidMessage,
    parse_tick_message,
    parse_askbid_message,
    WS_MAX_REGS_DEFAULT,
)
from .trading_calendar import KRXTradingCalendar, get_trading_calendar
from .sector_exposure import SectorExposure, SectorExposureConfig
from .universe_filter import UniverseFilterConfig, filter_universe

__all__ = [
    # Auth
    'KoreaInvestEnv',
    # Client
    'KoreaInvestAPI',
    'CircuitBreaker',
    'get_paper_tr_id',
    'PAPER_TR_ID_MAP',
    'PAPER_TR_ID_PASSTHROUGH',
    # Decorators
    'rate_limit',
    'rate_limit_async',
    'RateLimiter',
    'get_global_limiter_stats',
    # Response
    'APIResponse',
    'create_error_response',
    # VWAP
    'VWAPLedger',
    'compute_anchored_daily_vwap',
    'vwap_band',
    # Bar Aggregation
    'Bar',
    'BarAggregator',
    'aggregate_bars',
    # Rate Budget
    'TokenBucket',
    'RateBudget',
    'RateLimitedError',
    # Shared Rate Budget (Priority-aware, multi-process)
    'SharedRateBudget',
    'SharedRateBudgetClient',
    'PriorityTokenBucket',
    'PRIORITY_WINDOWS',
    'create_strategy_client',
    'get_shared_budget',
    # Indicators
    'sma',
    'ema',
    'atr',
    'zscore',
    'percentile_rank',
    'RollingSMA',
    'RollingATR',
    # WebSocket
    'KISWebSocketClient',
    'BaseSubscriptionManager',
    'TickMessage',
    'AskBidMessage',
    'parse_tick_message',
    'parse_askbid_message',
    'WS_MAX_REGS_DEFAULT',
    # Trading Calendar
    'KRXTradingCalendar',
    'get_trading_calendar',
    # Sector Exposure
    'SectorExposure',
    'SectorExposureConfig',
    # Universe Filter
    'UniverseFilterConfig',
    'filter_universe',
]

__version__ = '2.1.0'
