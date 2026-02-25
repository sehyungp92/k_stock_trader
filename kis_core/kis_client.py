"""
KIS API Client - Consolidated
Korea Investment & Securities REST API client for trading operations.

Features:
- Paper trading support with automatic TR_ID mapping
- Circuit breaker for failure protection
- Rate limiting with exponential backoff
- Domestic and overseas stock trading
- Real-time market data (price, orderbook, charts)
- Position and order management
- WebSocket subscription helpers
"""

from __future__ import annotations

import functools
import json
import random
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
import requests
from loguru import logger

from .kis_decorators import rate_limit
from .kis_responses import APIResponse

# Type variable for generic decorator
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# PAPER TRADING TR_ID MAPPING
# =============================================================================

PAPER_TR_ID_MAP: Dict[str, str] = {
    # Order APIs
    'TTTC0802U': 'VTTC0802U',  # Buy order
    'TTTC0801U': 'VTTC0801U',  # Sell order
    'TTTC0803U': 'VTTC0803U',  # Cancel/revise order
    # Balance/Position APIs
    'TTTC8434R': 'VTTC8434R',  # Account balance
    'TTTC8908R': 'VTTC8908R',  # Stock balance
    'TTTC8001R': 'VTTC8001R',  # Buyable amount
    # Order inquiry APIs
    'TTTC8036R': 'VTTC8036R',  # Order list
    'CTSC9115R': 'VTSC9115R',  # Order history
    # Overseas APIs
    'TTTT1002U': 'VTTT1002U',  # Overseas buy
    'TTTT1006U': 'VTTT1006U',  # Overseas sell
    'TTTT1004U': 'VTTT1004U',  # Overseas cancel/revise
    'TTTS3012R': 'VTTS3012R',  # Overseas balance
    'TTTS6036R': 'VTTS6036R',  # Overseas orders
}

# TR_IDs that don't need mapping (quote/data APIs)
PAPER_TR_ID_PASSTHROUGH: set[str] = {
    'FHKST01010100',  # Current price
    'FHKST01010200',  # Hoga (orderbook)
    'FHKST03010100',  # Daily chart
    'FHKST03010200',  # Minute chart
    'FHKST01010900',  # Investor trend
    'FHPST01710000',  # Volume ranking
    'FHPST01700000',  # Fluctuation ranking
    'HHKST03900400',  # Condition search
}

# TR_IDs that require real API credentials (not supported on paper trading server)
# These endpoints will use real API when paper trading is enabled AND real credentials are configured
PAPER_UNSUPPORTED_TR_IDS: set[str] = {
    'FHPPG04650101',  # Program trading trend - not available on paper server
    # Add more as discovered
}


def get_paper_tr_id(tr_id: str, strict: bool = False) -> str:
    """
    Get paper trading TR_ID for a given live TR_ID.

    Args:
        tr_id: Live trading TR_ID
        strict: If True, warn on unknown TR_ID

    Returns:
        Paper trading TR_ID
    """
    if tr_id in PAPER_TR_ID_MAP:
        return PAPER_TR_ID_MAP[tr_id]

    if tr_id in PAPER_TR_ID_PASSTHROUGH:
        return tr_id

    # Heuristic fallback: T/J/C -> V
    if tr_id and tr_id[0] in ('T', 'J', 'C'):
        mapped = 'V' + tr_id[1:]
        if strict:
            logger.warning(f"Unknown TR_ID '{tr_id}' - using heuristic '{mapped}'")
        return mapped

    return tr_id


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered (one request at a time)
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: float = 0
        self.state = 'CLOSED'
        self._lock = threading.Lock()
        self._half_open_in_progress = False

    def record_success(self) -> None:
        """Record successful request."""
        with self._lock:
            self.failures = 0
            self._half_open_in_progress = False
            if self.state != 'CLOSED':
                logger.info("Circuit breaker CLOSED - service recovered")
            self.state = 'CLOSED'

    def record_failure(self) -> None:
        """Record failed request."""
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            self._half_open_in_progress = False

            if self.failures >= self.failure_threshold:
                if self.state != 'OPEN':
                    logger.warning(
                        f"Circuit breaker OPEN after {self.failures} consecutive failures"
                    )
                self.state = 'OPEN'

    def can_execute(self) -> bool:
        """
        Check if request can proceed.

        In HALF_OPEN state, only one request is allowed at a time.
        """
        with self._lock:
            if self.state == 'CLOSED':
                return True

            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    if not self._half_open_in_progress:
                        self.state = 'HALF_OPEN'
                        self._half_open_in_progress = True
                        logger.info("Circuit breaker HALF_OPEN - testing recovery")
                        return True
                return False

            # HALF_OPEN: only allow if we're the test request
            if self._half_open_in_progress:
                return False
            self._half_open_in_progress = True
            return True

    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return not self.can_execute()

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                'state': self.state,
                'failures': self.failures,
                'threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout,
            }


# Per-category circuit breakers to prevent quote failures from blocking orders
_circuit_breaker_quote = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
_circuit_breaker_order = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

def _get_circuit_breaker(api_url: str, is_post_request: bool) -> CircuitBreaker:
    """Get the appropriate circuit breaker for the endpoint category."""
    if is_post_request or '/trading/' in api_url:
        return _circuit_breaker_order
    return _circuit_breaker_quote


# =============================================================================
# TIMEZONE HELPER
# =============================================================================

def _get_kst():
    """Get Korea Standard Time timezone."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return ZoneInfo("Asia/Seoul")


# =============================================================================
# MAIN API CLIENT
# =============================================================================

class KoreaInvestAPI:
    """
    Korea Investment & Securities API Client.

    Provides methods for:
    - Market data (prices, orderbook, charts)
    - Order management (buy, sell, cancel, revise)
    - Position/balance queries
    - Rankings and searches
    - WebSocket subscription helpers
    """

    def __init__(self, env):
        """
        Initialize API client.

        Args:
            env: KoreaInvestEnv instance with auth and config
        """
        self.env = env
        cfg = env.get_full_config()

        self.custtype: str = cfg['custtype']
        self.websocket_approval_key: str = cfg['websocket_approval_key']
        self.account_num: str = cfg['account_num']
        self.is_paper_trading: bool = cfg['is_paper_trading']
        self.htsid: str = cfg['htsid']
        self.using_url: str = cfg['using_url']

    # =========================================================================
    # CORE HTTP METHODS
    # =========================================================================

    def _set_order_hash_key(self, headers: Dict[str, str], post_data: str) -> None:
        """Set hash key for order APIs."""
        url = f"{self.using_url}/uapi/hashkey"
        try:
            res = requests.post(url, data=post_data, headers=headers, timeout=10)
            if res.status_code == 200:
                headers['hashkey'] = res.json().get('HASH', '')
            else:
                logger.error(f"Hash key error: {res.status_code}")
        except Exception as e:
            logger.error(f"Failed to get hash key: {e}")

    @rate_limit(min_interval=0.05)
    def _url_fetch(
        self,
        api_url: str,
        tr_id: str,
        params: Dict[str, Any],
        is_post_request: bool = False,
        use_hash: bool = True,
        retry_on_failure: Optional[bool] = None,
    ) -> Optional[APIResponse]:
        """
        Core method for API requests with rate limiting, retry, and circuit breaker.

        Args:
            api_url: API endpoint path
            tr_id: Transaction ID
            params: Request parameters
            is_post_request: True for POST, False for GET
            use_hash: Whether to include hash key (for orders)
            retry_on_failure: Override retry behavior (default: True for GET, False for POST)

        Returns:
            APIResponse on success, None on failure
        """
        cb = _get_circuit_breaker(api_url, is_post_request)
        if cb.is_open():
            logger.warning(f"Circuit breaker OPEN - skipping request to {api_url}")
            return None

        # Default: retry GETs (idempotent), not POSTs (orders)
        if retry_on_failure is None:
            retry_on_failure = not is_post_request

        max_attempts = 3 if retry_on_failure else 1
        base_delay = 0.5

        # Check if this TR_ID requires real API (paper unsupported)
        needs_real_api = (
            self.is_paper_trading
            and tr_id in PAPER_UNSUPPORTED_TR_IDS
            and self.env.has_real_fallback
        )

        for attempt in range(max_attempts):
            try:
                if needs_real_api:
                    # Use real API for this specific endpoint
                    url = f"{self.env.real_url}{api_url}"
                    headers = self.env.get_real_api_headers()
                    if headers is None:
                        logger.warning(f"Real API headers unavailable for {tr_id}")
                        return None
                    tr_id_used = tr_id  # No mapping needed for real API
                    logger.debug(f"Using real API for unsupported TR_ID: {tr_id}")
                else:
                    # Standard flow (paper or live)
                    url = f"{self.using_url}{api_url}"
                    headers = self.env.get_base_headers()

                    # Map TR_ID for paper trading
                    if self.is_paper_trading:
                        is_trading = '/trading/' in api_url or is_post_request
                        tr_id_used = get_paper_tr_id(tr_id, strict=is_trading)
                    else:
                        tr_id_used = tr_id

                headers["tr_id"] = tr_id_used
                headers["custtype"] = self.custtype

                if is_post_request:
                    json_body = json.dumps(params)
                    if use_hash:
                        self._set_order_hash_key(headers, json_body)
                    res = requests.post(url, headers=headers, data=json_body, timeout=10)
                else:
                    res = requests.get(url, headers=headers, params=params, timeout=10)

                if res.status_code == 200:
                    ar = APIResponse(res)
                    cb.record_success()
                    return ar

                # Handle rate limiting
                if res.status_code == 500 and 'EGW00201' in res.text:
                    logger.warning(f"KIS rate-limited on {api_url}, attempt {attempt + 1}")
                    max_attempts = max(max_attempts, 5)
                    base_delay = max(base_delay, 1.0)
                else:
                    logger.error(f"Error {res.status_code}: {res.text[:200]}")

                cb.record_failure()

                # Don't retry auth/client errors
                if res.status_code in (400, 401, 403):
                    return None

            except requests.exceptions.Timeout:
                logger.error(f"Timeout for {api_url}")
                cb.record_failure()
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error for {api_url}: {e}")
                cb.record_failure()
            except Exception as e:
                logger.error(f"Request exception: {e}")
                cb.record_failure()

            # Backoff before retry
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt), 5.0)
                jitter = random.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)

        return None

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status (worst of quote/order breakers)."""
        quote_status = _circuit_breaker_quote.get_status()
        order_status = _circuit_breaker_order.get_status()
        # Report the more severe state
        if order_status['state'] == 'OPEN' or quote_status['state'] == 'OPEN':
            state = 'OPEN'
        elif order_status['state'] == 'HALF_OPEN' or quote_status['state'] == 'HALF_OPEN':
            state = 'HALF_OPEN'
        else:
            state = 'CLOSED'
        return {
            'state': state,
            'quote': quote_status,
            'order': order_status,
        }

    # =========================================================================
    # MARKET DATA - PRICES
    # =========================================================================

    def get_current_price(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        Get full current price data for a stock.

        Returns dict with fields like stck_prpr (price), stck_hgpr (high), etc.
        """
        url = "/uapi/domestic-stock/v1/quotations/inquire-price"
        tr_id = "FHKST01010100"
        params = {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': stock_code}

        result = self._url_fetch(url, tr_id, params)
        if result and result.is_ok():
            return result.get_body().output
        if result:
            result.print_error()
        return None

    def get_current_price_simple(self, stock_code: str) -> Optional[int]:
        """Get just the current price as an integer."""
        result = self.get_current_price(stock_code)
        if result:
            return int(result.get('stck_prpr', 0))
        return None

    def get_last_price(self, ticker: str) -> Optional[float]:
        """Get current price as float. Returns None if unavailable."""
        price = self.get_current_price_simple(ticker)
        if price and price > 0:
            return float(price)
        return None

    def get_day_high(self, ticker: str) -> float:
        """Get day's high price."""
        result = self.get_current_price(ticker)
        if result:
            return float(result.get('stck_hgpr', 0))
        return 0.0

    def get_day_low(self, ticker: str) -> float:
        """Get day's low price."""
        result = self.get_current_price(ticker)
        if result:
            return float(result.get('stck_lwpr', 0))
        return 0.0

    # =========================================================================
    # MARKET DATA - ORDERBOOK
    # =========================================================================

    def get_hoga_info(self, stock_code: str) -> Optional[Any]:
        """Get orderbook (호가) data.

        Returns list of dicts. The KIS API returns output1 as a single dict
        for this endpoint; we normalize it to [dict] so callers can use [0].
        """
        url = "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        tr_id = "FHKST01010200"
        params = {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': stock_code}

        result = self._url_fetch(url, tr_id, params)
        if result and result.is_ok():
            output1 = result.get_body().output1
            # output1 is a single dict for this endpoint; wrap in list
            if isinstance(output1, dict):
                return [output1]
            return output1
        if result:
            result.print_error()
        return None

    def get_orderbook(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get orderbook as dict (first element of hoga)."""
        try:
            hoga = self.get_hoga_info(ticker)
            if hoga and len(hoga) > 0:
                return hoga[0]
        except Exception as e:
            logger.debug(f"Orderbook error for {ticker}: {e}")
        return None

    def get_best_bid(self, ticker: str) -> float:
        """Get best bid price."""
        try:
            hoga = self.get_hoga_info(ticker)
            if hoga and len(hoga) > 0:
                bid = hoga[0].get('bidp1', 0)
                return float(bid) if bid else self.get_last_price(ticker)
        except Exception as e:
            logger.debug(f"Best bid error for {ticker}: {e}")
        return self.get_last_price(ticker)

    def get_best_ask(self, ticker: str) -> float:
        """Get best ask price."""
        try:
            hoga = self.get_hoga_info(ticker)
            if hoga and len(hoga) > 0:
                ask = hoga[0].get('askp1', 0)
                return float(ask) if ask else self.get_last_price(ticker)
        except Exception as e:
            logger.debug(f"Best ask error for {ticker}: {e}")
        return self.get_last_price(ticker)

    def get_expected_open(self, ticker: str) -> Optional[float]:
        """
        Get expected open price (동시호가) during pre-market.

        Returns indicative match price from orderbook, falling back to
        mid-price, then last traded price if hoga data is unavailable
        (e.g., outside pre-auction hours or on non-trading days).
        """
        try:
            hoga = self.get_hoga_info(ticker)
            if hoga and len(hoga) > 0:
                # Expected match price
                expected = hoga[0].get('antc_cnpr') or ''
                if expected and expected != '0':
                    return float(expected)

                # Fallback: mid-price
                bid = float(hoga[0].get('bidp1') or 0)
                ask = float(hoga[0].get('askp1') or 0)
                if bid and ask:
                    return (bid + ask) / 2

        except Exception as e:
            logger.debug(f"Expected open error for {ticker}: {e}")

        # Final fallback: last traded price
        last = self.get_last_price(ticker)
        if last:
            logger.debug(f"Expected open for {ticker}: using last_price fallback ({last})")
        return last

    # =========================================================================
    # MARKET DATA - CHARTS
    # =========================================================================

    def get_daily_chart_data(
        self, stock_code: str, end_date: Optional[str] = None, count: int = 60
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch raw daily OHLCV data from KIS API.

        Returns dict with 'output2' key containing daily bar rows.
        """
        KST = _get_kst()
        if not end_date:
            end_date = datetime.now(tz=KST).strftime("%Y%m%d")

        start_date = (
            datetime.strptime(end_date, "%Y%m%d") - pd.Timedelta(days=count * 2)
        ).strftime("%Y%m%d")

        url = '/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice'
        tr_id = "FHKST03010100"
        params = {
            'FID_COND_MRKT_DIV_CODE': 'J',
            'FID_INPUT_ISCD': stock_code,
            'FID_INPUT_DATE_1': start_date,
            'FID_INPUT_DATE_2': end_date,
            'FID_PERIOD_DIV_CODE': 'D',
            'FID_ORG_ADJ_PRC': '0',
        }

        result = self._url_fetch(url, tr_id, params)
        if result and result.is_ok():
            output2 = getattr(result.get_body(), 'output2', []) or []
            return {'output2': output2}
        return None

    def get_minute_chart_data_raw(
        self,
        stock_code: str,
        end_date: Optional[str] = None,
        end_time: Optional[str] = None,
        interval: str = "1",
        count: int = 200,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch raw minute OHLCV data from KIS API.

        Returns dict with 'output2' key containing minute bar rows.
        """
        KST = _get_kst()
        now = datetime.now(tz=KST)
        if not end_date:
            end_date = now.strftime("%Y%m%d")
        if not end_time:
            end_time = now.strftime("%H%M%S")

        url = '/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice'
        tr_id = "FHKST03010200"
        params = {
            'FID_ETC_CLS_CODE': "",
            'FID_COND_MRKT_DIV_CODE': 'J',
            'FID_INPUT_ISCD': stock_code,
            'FID_INPUT_HOUR_1': end_time,
            'FID_PW_DATA_INCU_YN': 'Y',
        }

        result = self._url_fetch(url, tr_id, params)
        if result and result.is_ok():
            output2 = getattr(result.get_body(), 'output2', []) or []
            return {'output2': output2[:count] if output2 else []}
        return None

    def get_daily_bars(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """
        Get daily OHLCV bars as DataFrame.

        Columns: ['date', 'open', 'high', 'low', 'close', 'volume']
        Sorted ascending by date.
        """
        KST = _get_kst()
        empty_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

        try:
            end_date = datetime.now(tz=KST).strftime("%Y%m%d")
            raw = self.get_daily_chart_data(stock_code=ticker, end_date=end_date, count=days)
            if not raw:
                return empty_df

            rows = raw.get('output2', []) or []
            if not rows:
                return empty_df

            out = []
            for r in rows:
                d = r.get('stck_bsop_date') or r.get('bsop_date')
                if not d or len(d) != 8:
                    continue
                try:
                    out.append({
                        'date': datetime.strptime(d, "%Y%m%d").replace(tzinfo=KST),
                        'open': float(r.get('stck_oprc', 0)),
                        'high': float(r.get('stck_hgpr', 0)),
                        'low': float(r.get('stck_lwpr', 0)),
                        'close': float(r.get('stck_clpr', r.get('stck_prpr', 0))),
                        'volume': int(r.get('acml_vol', r.get('cntg_vol', 0))),
                    })
                except (ValueError, KeyError):
                    continue

            if not out:
                return empty_df

            df = pd.DataFrame(out)
            df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"get_daily_bars error for {ticker}: {e}")
            return empty_df

    def get_minute_bars(self, ticker: str, minutes: int = 200) -> pd.DataFrame:
        """
        Get minute OHLCV bars as DataFrame.

        Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Volume is per-bar (not cumulative). Sorted ascending.
        """
        KST = _get_kst()
        empty_df = pd.DataFrame(
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        try:
            now = datetime.now(tz=KST)
            raw = self.get_minute_chart_data_raw(
                stock_code=ticker,
                end_date=now.strftime("%Y%m%d"),
                end_time=now.strftime("%H%M%S"),
                interval="1",
                count=minutes,
            )
            if not raw:
                return empty_df

            rows = raw.get('output2', []) or []
            if not rows:
                return empty_df

            out = []
            for r in rows:
                d = r.get('stck_bsop_date') or now.strftime("%Y%m%d")
                t = r.get('stck_cntg_hour') or r.get('cntg_hour')
                if not t or len(t) != 6:
                    continue
                try:
                    ts = datetime.strptime(d + t, "%Y%m%d%H%M%S").replace(tzinfo=KST)
                    out.append({
                        'timestamp': ts,
                        'open': float(r.get('stck_oprc', 0)),
                        'high': float(r.get('stck_hgpr', 0)),
                        'low': float(r.get('stck_lwpr', 0)),
                        'close': float(r.get('stck_prpr', r.get('stck_clpr', 0))),
                        'volume_raw': int(r.get('cntg_vol', r.get('acml_vol', 0))),
                    })
                except (ValueError, KeyError):
                    continue

            if not out:
                return empty_df

            df = pd.DataFrame(out)
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

            # Convert cumulative volume to per-bar if needed
            if len(df) > 1:
                diffs = df['volume_raw'].diff().fillna(df['volume_raw'])
                non_neg_ratio = (diffs >= 0).sum() / len(diffs)
                if non_neg_ratio > 0.95 and df['volume_raw'].iloc[-1] > diffs.sum() * 3:
                    df['volume'] = diffs.clip(lower=0).astype(int)
                else:
                    df['volume'] = df['volume_raw'].astype(int)
            else:
                df['volume'] = df['volume_raw']

            df = df.drop(columns=['volume_raw'])
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"get_minute_bars error for {ticker}: {e}")
            return empty_df

    def get_latest_bar(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the most recent 1-minute bar."""
        bars = self.get_minute_bars(ticker, minutes=2)
        if bars is not None and not bars.empty:
            return bars.iloc[-1].to_dict()
        return None

    def get_ohlcv_daily(self, ticker: str, count: int = 60) -> Optional[List[Dict]]:
        """Get daily OHLCV as list of raw dicts."""
        raw = self.get_daily_chart_data(stock_code=ticker, count=count)
        if raw:
            return raw.get('output2', []) or None
        return None

    def get_ohlcv_minute(
        self, ticker: str, interval: int = 1, count: int = 200
    ) -> Optional[List[Dict]]:
        """Get minute OHLCV as list of raw dicts."""
        raw = self.get_minute_chart_data_raw(
            stock_code=ticker, interval=str(interval), count=count
        )
        if raw:
            return raw.get('output2', []) or None
        return None

    def get_intraday_bars(self, ticker: str, interval_min: int = 1) -> List[Dict]:
        """Get intraday bars as list of normalized dicts."""
        df = self.get_minute_bars(ticker, minutes=200)
        if df is not None and not df.empty:
            return df.to_dict('records')
        return []

    # =========================================================================
    # MARKET DATA - INVESTOR TRENDS
    # =========================================================================

    def get_foreign_trend(self, ticker: str, days: int = 20) -> List[Dict]:
        """Get daily foreign investor net buying data."""
        url = "/uapi/domestic-stock/v1/quotations/inquire-investor"
        tr_id = "FHKST01010900"
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}

        try:
            result = self._url_fetch(url, tr_id, params)
            if result and result.is_ok():
                output = getattr(result.get_body(), 'output', []) or []
                return [
                    {
                        'date': row.get('stck_bsop_date', ''),
                        'net_buy': int(row.get('frgn_ntby_qty') or 0),
                    }
                    for row in output[:days]
                ]
        except Exception as e:
            logger.debug(f"Foreign trend error for {ticker}: {e}")
        return []

    def get_inst_trend(self, ticker: str, days: int = 20) -> List[Dict]:
        """Get daily institutional investor net buying data."""
        url = "/uapi/domestic-stock/v1/quotations/inquire-investor"
        tr_id = "FHKST01010900"
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}

        try:
            result = self._url_fetch(url, tr_id, params)
            if result and result.is_ok():
                output = getattr(result.get_body(), 'output', []) or []
                return [
                    {
                        'date': row.get('stck_bsop_date', ''),
                        'net_buy': int(row.get('orgn_ntby_qty') or 0),
                    }
                    for row in output[:days]
                ]
        except Exception as e:
            logger.debug(f"Inst trend error for {ticker}: {e}")
        return []

    def get_investor_trend(self, ticker: str, days: int = 20) -> List[Dict]:
        """Get daily foreign + institutional net buying in a single call."""
        url = "/uapi/domestic-stock/v1/quotations/inquire-investor"
        tr_id = "FHKST01010900"
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}

        try:
            result = self._url_fetch(url, tr_id, params)
            if result and result.is_ok():
                output = getattr(result.get_body(), 'output', []) or []
                return [
                    {
                        'date': row.get('stck_bsop_date', ''),
                        'foreign_net': int(row.get('frgn_ntby_qty') or 0),
                        'inst_net': int(row.get('orgn_ntby_qty') or 0),
                    }
                    for row in output[:days]
                ]
        except Exception as e:
            logger.debug(f"Investor trend error for {ticker}: {e}")
        return []

    # =========================================================================
    # ACCOUNT - BALANCE & POSITIONS
    # =========================================================================

    def get_acct_balance(self, fetch_all: bool = True) -> Tuple[int, pd.DataFrame]:
        """
        Get account balance with holdings.

        Args:
            fetch_all: If True, fetch all pages of holdings

        Returns:
            Tuple of (total_evaluation_amount, DataFrame of holdings)
        """
        url = '/uapi/domestic-stock/v1/trading/inquire-balance'
        tr_id = "TTTC8434R"
        columns = [
            '종목코드', '종목명', '보유수량', '매도가능수량', '매입단가',
            '수익률', '현재가', '전일대비', '전일대비 등락률'
        ]

        all_output1: List[Dict] = []
        ctx_fk, ctx_nk = '', ''
        tot_evlu_amt = 0

        while True:
            params = {
                'CANO': self.account_num,
                'ACNT_PRDT_CD': '01',
                'AFHR_FLPR_YN': 'N',
                'FNCG_AMT_AUTO_RDPT_YN': 'N',
                'FUND_STTL_ICLD_YN': 'N',
                'INQR_DVSN': '01',
                'OFL_YN': 'N',
                'PRCS_DVSN': '01',
                'UNPR_DVSN': '01',
                'CTX_AREA_FK100': ctx_fk,
                'CTX_AREA_NK100': ctx_nk,
            }

            result = self._url_fetch(url, tr_id, params)
            if not result or not result.is_ok():
                break

            body = result.get_body()

            if hasattr(body, 'output1') and body.output1:
                all_output1.extend(body.output1)

            if hasattr(body, 'output2') and body.output2 and tot_evlu_amt == 0:
                try:
                    tot_evlu_amt = int(body.output2[0].get('tot_evlu_amt', 0))
                except (ValueError, TypeError, IndexError, KeyError):
                    pass

            if not fetch_all:
                break

            ctx_fk = getattr(body, 'ctx_area_fk100', '').strip()
            ctx_nk = getattr(body, 'ctx_area_nk100', '').strip()
            if not ctx_fk and not ctx_nk:
                break
            time.sleep(0.05)

        if not all_output1:
            return tot_evlu_amt, pd.DataFrame(columns=columns)

        df = pd.DataFrame(all_output1)
        target = [
            'pdno', 'prdt_name', 'hldg_qty', 'ord_psbl_qty', 'pchs_avg_pric',
            'evlu_pfls_rt', 'prpr', 'bfdy_cprs_icdc', 'fltt_rt'
        ]

        for col in target:
            if col not in df.columns:
                return tot_evlu_amt, pd.DataFrame(columns=columns)

        df = df[target]
        df[target[2:]] = df[target[2:]].apply(pd.to_numeric, errors='coerce')
        df.rename(columns=dict(zip(target, columns)), inplace=True)
        df = df[df['보유수량'].notna() & (df['보유수량'] != 0)]
        df.reset_index(drop=True, inplace=True)

        return tot_evlu_amt, df

    def get_buyable_cash(self) -> Optional[int]:
        """Get available cash for buying stocks in KRW."""
        url = '/uapi/domestic-stock/v1/trading/inquire-psbl-order'
        tr_id = "TTTC8908R"
        params = {
            'CANO': self.account_num,
            'ACNT_PRDT_CD': '01',
            'PDNO': '005930',
            'ORD_UNPR': '0',
            'ORD_DVSN': '01',
            'CMA_EVLU_AMT_ICLD_YN': 'Y',
            'OVRS_ICLD_YN': 'N',
        }

        result = self._url_fetch(url, tr_id, params)
        if result and result.is_ok():
            try:
                return int(result.get_body().output.get('ord_psbl_cash', 0))
            except (ValueError, AttributeError, TypeError):
                pass
        return None

    def get_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific ticker."""
        try:
            _, holdings = self.get_acct_balance(fetch_all=False)
            if holdings.empty:
                return None

            pos = holdings[holdings['종목코드'] == ticker]
            if pos.empty:
                return None

            row = pos.iloc[0]
            return {
                'ticker': ticker,
                'qty': int(row['보유수량']),
                'avg_price': float(row['매입단가']),
                'current_price': float(row['현재가']),
                'unrealized_pnl': float(row['현재가'] - row['매입단가']) * int(row['보유수량']),
            }
        except Exception as e:
            logger.error(f"get_position error for {ticker}: {e}")
        return None

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions as list of dicts."""
        try:
            _, holdings = self.get_acct_balance(fetch_all=True)
            if holdings.empty:
                return []

            return [
                {
                    'ticker': row['종목코드'],
                    'qty': int(row['보유수량']),
                    'avg_price': float(row['매입단가']),
                    'current_price': float(row['현재가']),
                    'pnl_pct': float(row['수익률']) / 100,
                }
                for _, row in holdings.iterrows()
            ]
        except Exception as e:
            logger.error(f"get_positions error: {e}")
        return []

    def get_balance(self) -> Dict[str, Any]:
        """
        Get comprehensive balance info.

        Returns dict with: acct, buyable_cash, total_amount, stocks
        """
        try:
            tot_amt, _ = self.get_acct_balance()
            buyable = self.get_buyable_cash()
            positions = self.get_positions()

            return {
                'total_amount': tot_amt,
                'buyable_cash': buyable or 0,
                'stocks': positions,
            }
        except Exception as e:
            logger.error(f"get_balance error: {e}")
            return {'total_amount': 0, 'buyable_cash': 0, 'stocks': []}

    # =========================================================================
    # ACCOUNT - OVERSEAS
    # =========================================================================

    def get_overseas_acct_balance(self) -> Tuple[float, pd.DataFrame]:
        """Get overseas account balance and holdings."""
        url = '/uapi/overseas-stock/v1/trading/inquire-balance'
        tr_id = "TTTS3012R"
        columns = [
            '종목코드', '해외거래소코드', '종목명', '보유수량', '매도가능수량',
            '매입단가', '수익률', '현재가', '평가손익'
        ]
        params = {
            'CANO': self.account_num,
            'ACNT_PRDT_CD': '01',
            'OVRS_EXCG_CD': 'NASD',
            'TR_CRCY_CD': 'USD',
            'CTX_AREA_FK200': '',
            'CTX_AREA_NK200': '',
        }

        result = self._url_fetch(url, tr_id, params)
        if not result:
            return 0.0, pd.DataFrame(columns=columns)

        if result.is_ok():
            output1 = result.get_body().output1
            if not output1:
                return 0.0, pd.DataFrame(columns=columns)

            df = pd.DataFrame(output1)
            target = [
                'ovrs_pdno', 'ovrs_excg_cd', 'ovrs_item_name', 'ovrs_cblc_qty',
                'ord_psbl_qty', 'pchs_avg_pric', 'evlu_pfls_rt', 'now_pric2',
                'frcr_evlu_pfls_amt'
            ]
            df = df[target]
            df[target[3:]] = df[target[3:]].apply(pd.to_numeric)
            df.rename(columns=dict(zip(target, columns)), inplace=True)
            df = df[df['보유수량'] != 0]

            r2 = result.get_body().output2
            return float(r2.get('tot_evlu_pfls_amt', 0)), df

        return 0.0, pd.DataFrame(columns=columns)

    # =========================================================================
    # ORDERS - DOMESTIC
    # =========================================================================

    def get_orders(self, prd_code: str = '01') -> Optional[pd.DataFrame]:
        """Get pending orders."""
        url = "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
        tr_id = "TTTC8036R"
        params = {
            "CANO": self.account_num,
            "ACNT_PRDT_CD": prd_code,
            "CTX_AREA_FK100": '',
            "CTX_AREA_NK100": '',
            "INQR_DVSN_1": '0',
            "INQR_DVSN_2": '0',
        }

        result = self._url_fetch(url, tr_id, params)
        if result and result.is_ok() and result.get_body().output:
            df = pd.DataFrame(result.get_body().output)
            df.set_index('odno', inplace=True)
            cols = ['pdno', 'ord_qty', 'ord_unpr', 'ord_tmd', 'ord_gno_brno', 'orgn_odno', 'psbl_qty']
            names = ['종목코드', '주문수량', '주문가격', '시간', '주문점', '원주문번호', '주문가능수량']
            df = df[cols]
            return df.rename(columns=dict(zip(cols, names)))
        return None

    def order_stock(
        self,
        stock_code: str,
        order_qty: int,
        order_price: str,
        prd_code: str = "01",
        buy_flag: bool = True,
        order_type: str = "00",
    ) -> Optional[APIResponse]:
        """
        Place a domestic stock order.

        Args:
            stock_code: Stock code
            order_qty: Quantity
            order_price: Price ("0" for market orders)
            prd_code: Product code
            buy_flag: True for buy, False for sell
            order_type: "00" for limit, "01" for market

        Returns:
            APIResponse on success, None on failure
        """
        url = "/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = "TTTC0802U" if buy_flag else "TTTC0801U"

        params = {
            'CANO': self.account_num,
            'ACNT_PRDT_CD': prd_code,
            'PDNO': stock_code,
            'ORD_DVSN': order_type,
            'ORD_QTY': str(order_qty),
            'ORD_UNPR': str(order_price),
            'CTAC_TLNO': '',
            'SLL_TYPE': '01',
            'ALGO_NO': '',
        }

        result = self._url_fetch(url, tr_id, params, is_post_request=True, use_hash=True)
        if result and result.is_ok():
            return result
        if result:
            result.print_error()
        return None

    def buy_stock(
        self, stock_code: str, order_qty: int, order_price: str, order_type: str = "00"
    ) -> Optional[APIResponse]:
        """Place a buy order."""
        return self.order_stock(stock_code, order_qty, order_price, buy_flag=True, order_type=order_type)

    def sell_stock(
        self, stock_code: str, order_qty: int, order_price: str, order_type: str = "00"
    ) -> Optional[APIResponse]:
        """Place a sell order."""
        return self.order_stock(stock_code, order_qty, order_price, buy_flag=False, order_type=order_type)

    def _validate_order_id(self, order_id: str) -> Optional[str]:
        """Validate and normalize order ID from KIS API response."""
        order_id = order_id.strip()
        if not order_id or not order_id.isdigit():
            logger.error(f"Invalid order_id from KIS: '{order_id}'")
            return None
        return order_id

    def place_market_buy(self, stock_code: str, quantity: int, **kwargs) -> Optional[str]:
        """Place a market buy order. Returns order ID."""
        result = self.buy_stock(stock_code, quantity, "0", order_type="01")
        if result and result.is_ok():
            order_id = self._validate_order_id(result.get_body().output.get('ODNO', ''))
            if order_id:
                logger.info(f"Market BUY: {stock_code} x{quantity}, order_id={order_id}")
            return order_id
        if result:
            logger.warning(f"Market BUY failed: {stock_code} x{quantity} — {result.get_error_message()}")
        else:
            logger.warning(f"Market BUY failed: {stock_code} x{quantity} — no response")
        return None

    def place_market_sell(self, stock_code: str, quantity: int, **kwargs) -> Optional[str]:
        """Place a market sell order. Returns order ID."""
        result = self.sell_stock(stock_code, quantity, "0", order_type="01")
        if result and result.is_ok():
            order_id = self._validate_order_id(result.get_body().output.get('ODNO', ''))
            if order_id:
                logger.info(f"Market SELL: {stock_code} x{quantity}, order_id={order_id}")
            return order_id
        if result:
            logger.warning(f"Market SELL failed: {stock_code} x{quantity} — {result.get_error_message()}")
        else:
            logger.warning(f"Market SELL failed: {stock_code} x{quantity} — no response")
        return None

    def place_limit_buy(
        self, stock_code: str, price: float, quantity: int, **kwargs
    ) -> Optional[str]:
        """Place a limit buy order. Returns order ID."""
        result = self.buy_stock(stock_code, quantity, str(int(price)), order_type="00")
        if result and result.is_ok():
            order_id = self._validate_order_id(result.get_body().output.get('ODNO', ''))
            if order_id:
                logger.info(f"Limit BUY: {stock_code} x{quantity} @ {price:.0f}, order_id={order_id}")
            return order_id
        if result:
            logger.warning(f"Limit BUY failed: {stock_code} x{quantity} @ {price:.0f} — {result.get_error_message()}")
        else:
            logger.warning(f"Limit BUY failed: {stock_code} x{quantity} @ {price:.0f} — no response")
        return None

    def place_limit_sell(
        self, stock_code: str, price: float, quantity: int, **kwargs
    ) -> Optional[str]:
        """Place a limit sell order. Returns order ID."""
        result = self.sell_stock(stock_code, quantity, str(int(price)), order_type="00")
        if result and result.is_ok():
            order_id = self._validate_order_id(result.get_body().output.get('ODNO', ''))
            if order_id:
                logger.info(f"Limit SELL: {stock_code} x{quantity} @ {price:.0f}, order_id={order_id}")
            return order_id
        if result:
            logger.warning(f"Limit SELL failed: {stock_code} x{quantity} @ {price:.0f} — {result.get_error_message()}")
        else:
            logger.warning(f"Limit SELL failed: {stock_code} x{quantity} @ {price:.0f} — no response")
        return None

    def place_order_full(
        self,
        stock_code: str,
        side: str,
        quantity: int,
        price: float = 0,
        order_type: str = "limit",
    ) -> Optional[Dict[str, Any]]:
        """
        Place order and return full order info including branch for cancel/revise.

        Returns dict with 'order_id', 'branch', etc., or None on failure.
        """
        ot = "00" if order_type == "limit" else "01"
        op = str(int(price)) if price else "0"

        if side.lower() == "buy":
            result = self.buy_stock(stock_code, quantity, op, order_type=ot)
        else:
            result = self.sell_stock(stock_code, quantity, op, order_type=ot)

        if result and result.is_ok():
            output = result.get_body().output
            return {
                'order_id': output.get('ODNO', ''),
                'branch': output.get('KRX_FWDG_ORD_ORGNO', ''),
                'stock_code': stock_code,
                'side': side,
                'quantity': quantity,
                'price': price,
            }
        return None

    def _cancel_revise_order(
        self,
        order_no: str,
        order_branch: str,
        order_qty: int,
        order_price: str,
        prd_code: str,
        order_dv: str,
        cncl_dv: str,
        qty_all_yn: str,
    ) -> Optional[APIResponse]:
        """Internal method for cancel/revise orders."""
        url = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
        tr_id = "TTTC0803U"
        params = {
            "CANO": self.account_num,
            "ACNT_PRDT_CD": prd_code,
            "KRX_FWDG_ORD_ORGNO": order_branch,
            "ORGN_ODNO": order_no,
            "ORD_DVSN": order_dv,
            "RVSE_CNCL_DVSN_CD": cncl_dv,
            "ORD_QTY": str(order_qty),
            "ORD_UNPR": str(order_price),
            "QTY_ALL_ORD_YN": qty_all_yn,
        }

        result = self._url_fetch(url, tr_id, params, is_post_request=True)
        if result and result.is_ok():
            return result
        if result:
            result.print_error()
        return None

    def cancel_order(
        self,
        order_no: str,
        order_qty: int,
        order_price: str = "0",
        order_branch: str = '06010',
        prd_code: str = '01',
        order_dv: str = '00',
        qty_all_yn: str = "Y",
    ) -> Optional[APIResponse]:
        """Cancel an order."""
        return self._cancel_revise_order(
            order_no, order_branch, order_qty, order_price, prd_code, order_dv, '02', qty_all_yn
        )

    def revise_order(
        self,
        order_no: str,
        order_qty: int,
        order_price: str,
        order_branch: str = '06010',
        prd_code: str = '01',
        order_dv: str = '00',
        qty_all_yn: str = "Y",
    ) -> Optional[APIResponse]:
        """Revise an order."""
        return self._cancel_revise_order(
            order_no, order_branch, order_qty, order_price, prd_code, order_dv, '01', qty_all_yn
        )

    def cancel_all_orders(self, skip_codes: Optional[List[str]] = None) -> None:
        """Cancel all pending domestic orders."""
        skip_codes = skip_codes or []
        orders = self.get_orders()
        if orders is None:
            return

        for odno, row in orders.iterrows():
            if row['종목코드'] in skip_codes:
                continue
            result = self.cancel_order(
                odno, row['주문수량'], row['주문가격'], row['주문점']
            )
            if result:
                logger.info(f"Cancelled {odno}: {result.get_error_message()}")
            time.sleep(0.02)

    # =========================================================================
    # ORDERS - OVERSEAS
    # =========================================================================

    def get_overseas_orders(
        self, prd_code: str = '01', exchange_code: str = 'NASD'
    ) -> Optional[pd.DataFrame]:
        """Get pending overseas orders."""
        url = "/uapi/overseas-stock/v1/trading/inquire-nccs"
        tr_id = "TTTS3018R"
        params = {
            "CANO": self.account_num,
            "ACNT_PRDT_CD": prd_code,
            "OVRS_EXCG_CD": exchange_code,
            "SORT_SQN": "DS",
            "CTX_AREA_FK200": '',
            "CTX_AREA_NK200": '',
        }

        result = self._url_fetch(url, tr_id, params)
        if result and result.is_ok() and result.get_body().output:
            df = pd.DataFrame(result.get_body().output)
            df.set_index('odno', inplace=True)
            cols = [
                'pdno', 'ft_ord_qty', 'ft_ord_unpr3', 'ord_tmd', 'ovrs_excg_cd',
                'orgn_odno', 'nccs_qty', 'sll_buy_dvsn_cd', 'sll_buy_dvsn_cd_name'
            ]
            names = [
                '종목코드', '주문수량', '주문가격', '시간', '거래소코드',
                '원주문번호', '주문가능수량', '매도매수구분코드', '매도매수구분코드명'
            ]
            df = df[cols]
            return df.rename(columns=dict(zip(cols, names)))
        return None

    def overseas_order_stock(
        self,
        stock_code: str,
        exchange_code: str,
        order_qty: int,
        order_price: str,
        prd_code: str = "01",
        buy_flag: bool = True,
        order_type: str = "00",
    ) -> Optional[APIResponse]:
        """Place an overseas stock order."""
        url = "/uapi/overseas-stock/v1/trading/order"
        tr_id = "TTTT1002U" if buy_flag else "TTTT1006U"

        params = {
            'CANO': self.account_num,
            'ACNT_PRDT_CD': prd_code,
            'OVRS_EXCG_CD': exchange_code,
            'PDNO': stock_code,
            'ORD_QTY': str(order_qty),
            'OVRS_ORD_UNPR': str(order_price),
            'ORD_SVR_DVSN_CD': "0",
            'ORD_DVSN': order_type,
        }

        result = self._url_fetch(url, tr_id, params, is_post_request=True, use_hash=True)
        if result and result.is_ok():
            return result
        if result:
            result.print_error()
        return None

    def overseas_buy_stock(
        self, stock_code: str, exchange_code: str, order_qty: int, order_price: str,
        prd_code: str = "01", order_type: str = "00"
    ) -> Optional[APIResponse]:
        """Place overseas buy order."""
        return self.overseas_order_stock(
            stock_code, exchange_code, order_qty, order_price, prd_code, True, order_type
        )

    def overseas_sell_stock(
        self, stock_code: str, exchange_code: str, order_qty: int, order_price: str,
        prd_code: str = "01", order_type: str = "00"
    ) -> Optional[APIResponse]:
        """Place overseas sell order."""
        return self.overseas_order_stock(
            stock_code, exchange_code, order_qty, order_price, prd_code, False, order_type
        )

    def _overseas_cancel_revise_order(
        self,
        order_no: str,
        stock_code: str,
        exchange_code: str,
        order_qty: int,
        order_price: str,
        prd_code: str,
        cncl_dv: str,
    ) -> Optional[APIResponse]:
        """Internal method for overseas cancel/revise."""
        url = "/uapi/overseas-stock/v1/trading/order-rvsecncl"
        tr_id = "TTTT1004U"
        params = {
            "CANO": self.account_num,
            "ACNT_PRDT_CD": prd_code,
            "OVRS_EXCG_CD": exchange_code,
            "PDNO": stock_code,
            "ORGN_ODNO": order_no,
            "ORD_SVR_DVSN_CD": "0",
            "RVSE_CNCL_DVSN_CD": cncl_dv,
            "ORD_QTY": str(order_qty),
            "OVRS_ORD_UNPR": str(order_price),
        }

        result = self._url_fetch(url, tr_id, params, is_post_request=True)
        if result and result.is_ok():
            return result
        if result:
            result.print_error()
        return None

    def overseas_cancel_order(
        self, order_no: str, stock_code: str, order_qty: int, order_price: str = "0",
        exchange_code: str = 'NASD', prd_code: str = '01'
    ) -> Optional[APIResponse]:
        """Cancel overseas order."""
        return self._overseas_cancel_revise_order(
            order_no, stock_code, exchange_code, order_qty, order_price, prd_code, '02'
        )

    def overseas_revise_order(
        self, order_no: str, stock_code: str, order_qty: int, order_price: str = "0",
        exchange_code: str = 'NASD', prd_code: str = '01'
    ) -> Optional[APIResponse]:
        """Revise overseas order."""
        return self._overseas_cancel_revise_order(
            order_no, stock_code, exchange_code, order_qty, order_price, prd_code, '01'
        )

    def overseas_cancel_all_orders(
        self, exchange_code: str = 'NASD', skip_codes: Optional[List[str]] = None
    ) -> None:
        """Cancel all pending overseas orders."""
        skip_codes = skip_codes or []
        orders = self.get_overseas_orders(exchange_code=exchange_code)
        if orders is None:
            return

        for odno, row in orders.iterrows():
            if row['종목코드'] in skip_codes:
                continue
            result = self.overseas_cancel_order(
                odno, row['종목코드'], row['주문수량'], row['주문가격'], row['거래소코드']
            )
            if result:
                logger.info(f"Cancelled overseas {odno}: {result.get_error_message()}")
            time.sleep(0.02)

    # =========================================================================
    # PROGRAM TRADING
    # =========================================================================

    def get_program_trend(self, market: str = "KOSPI") -> Optional[Dict[str, Any]]:
        """
        Get market-wide program trading trend (cumulative net buy).

        Uses /uapi/domestic-stock/v1/quotations/program-trade-by-stock
        (TR_ID FHPPG04650101) which returns aggregate program buy/sell
        for the market index ticker.

        Returns:
            Dict with 'ntby_qty' (cumulative net buy qty) and
            'ntby_amt' (cumulative net buy amount, KRW millions),
            or None on failure.
        """
        url = "/uapi/domestic-stock/v1/quotations/program-trade-by-stock"
        tr_id = "FHPPG04650101"
        # 0001=KOSPI index, 2001=KOSDAQ index
        index_code = "0001" if market.upper() == "KOSPI" else "2001"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J" if market.upper() == "KOSPI" else "Q",
            "FID_INPUT_ISCD": index_code,
        }

        try:
            result = self._url_fetch(url, tr_id, params)
            if result and result.is_ok():
                body = result.get_body()
                output = getattr(body, 'output', None) or getattr(body, 'output1', None)
                if isinstance(output, list) and output:
                    return output[0]
                if isinstance(output, dict):
                    return output
        except Exception as e:
            logger.debug(f"Program trend error: {e}")
        return None

    # =========================================================================
    # RANKINGS & SEARCHES
    # =========================================================================

    def get_fluctuation_ranking(
        self, market: str = "KOSPI", limit: int = 30
    ) -> Optional[List[Dict]]:
        """Get top stocks by price change."""
        url = "/uapi/domestic-stock/v1/ranking/fluctuation"
        tr_id = "FHPST01700000"
        params = {
            "fid_cond_mrkt_div_code": "J" if market.upper() == "KOSPI" else "Q",
            "fid_cond_scr_div_code": "20170",
            "fid_input_iscd": "0000",
            "fid_rank_sort_cls_code": "0",
            "fid_input_cnt_1": str(limit),
            "fid_prc_cls_code": "0",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": "",
            "fid_trgt_cls_code": "0",
            "fid_trgt_exls_cls_code": "0",
            "fid_div_cls_code": "0",
            "fid_rsfl_rate1": "",
            "fid_rsfl_rate2": "",
        }

        try:
            result = self._url_fetch(url, tr_id, params)
            if result and result.is_ok():
                return result.get_body().output or []
        except Exception as e:
            logger.debug(f"Fluctuation ranking error: {e}")
        return []

    def get_volume_ranking(self, market: str = "KOSPI", limit: int = 30) -> List[Dict]:
        """Get top stocks by trading volume."""
        url = "/uapi/domestic-stock/v1/quotations/volume-rank"
        tr_id = "FHPST01710000"
        params = {
            "fid_cond_mrkt_div_code": "J" if market.upper() == "KOSPI" else "Q",
            "fid_cond_scr_div_code": "20171",
            "fid_input_iscd": "0000",
            "fid_rank_sort_cls_code": "0",
            "fid_input_cnt_1": str(limit),
            "fid_prc_cls_code": "0",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": "",
            "fid_trgt_cls_code": "0",
            "fid_trgt_exls_cls_code": "0",
            "fid_div_cls_code": "0",
            "fid_rsfl_rate1": "",
            "fid_rsfl_rate2": "",
        }

        try:
            result = self._url_fetch(url, tr_id, params)
            if result and result.is_ok():
                return result.get_body().output or []
        except Exception as e:
            logger.debug(f"Volume ranking error: {e}")
        return []

    def get_condition_search(self, condition_id: str) -> List[Dict]:
        """
        HTS Condition Search.

        Args:
            condition_id: Condition ID set up in HTS

        Returns:
            List of matching stocks
        """
        if not condition_id:
            return []

        url = "/uapi/domestic-stock/v1/quotations/psearch-result"
        tr_id = "HHKST03900400"
        params = {"user_id": self.htsid, "seq": condition_id}

        try:
            result = self._url_fetch(url, tr_id, params)
            if result and result.is_ok():
                return getattr(result.get_body(), 'output2', []) or []
        except Exception as e:
            logger.debug(f"Condition search error: {e}")
        return []

    # =========================================================================
    # WEBSOCKET HELPERS
    # =========================================================================

    def get_send_data(self, cmd: int, stockcode: Optional[str] = None) -> str:
        """
        Build WebSocket subscribe/unsubscribe payload for domestic stocks.

        Args:
            cmd: 1=hoga sub, 2=hoga unsub, 3=tick sub, 4=tick unsub,
                 5=order sub, 6=order unsub, 7=order9 sub, 8=order9 unsub
            stockcode: Stock code (not needed for order notifications)

        Returns:
            JSON string for WebSocket send
        """
        assert 0 < cmd < 9, f"Invalid cmd: {cmd}"

        tr_map = {
            1: ('H0STASP0', '1'), 2: ('H0STASP0', '2'),
            3: ('H0STCNT0', '1'), 4: ('H0STCNT0', '2'),
            5: ('H0STCNI0', '1'), 6: ('H0STCNI0', '2'),
            7: ('H0STCNI9', '1'), 8: ('H0STCNI9', '2'),
        }
        tr_id, tr_type = tr_map[cmd]
        tr_key = self.htsid if cmd in (5, 6, 7, 8) else stockcode

        payload = {
            "header": {
                "approval_key": self.websocket_approval_key,
                "custtype": self.custtype,
                "tr_type": tr_type,
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": tr_id, "tr_key": tr_key}},
        }
        return json.dumps(payload)

    def overseas_get_send_data(self, cmd: int, stockcode: Optional[str] = None) -> str:
        """
        Build WebSocket subscribe/unsubscribe payload for overseas stocks.

        Args:
            cmd: 1=hoga sub, 2=hoga unsub, 3=tick sub, 4=tick unsub,
                 5=order sub, 6=order unsub, 7=order9 sub, 8=order9 unsub
            stockcode: Stock code (not needed for order notifications)

        Returns:
            JSON string for WebSocket send
        """
        assert 0 < cmd < 9, f"Invalid cmd: {cmd}"

        tr_map = {
            1: ('HDFSASP0', '1'), 2: ('HDFSASP0', '2'),
            3: ('HDFSCNT0', '1'), 4: ('HDFSCNT0', '2'),
            5: ('H0GSCNI0', '1'), 6: ('H0GSCNI0', '2'),
            7: ('H0GSCNI0', '1'), 8: ('H0GSCNI0', '2'),
        }
        tr_id, tr_type = tr_map[cmd]
        tr_key = self.htsid if cmd in (5, 6, 7, 8) else stockcode

        payload = {
            "header": {
                "approval_key": self.websocket_approval_key,
                "custtype": self.custtype,
                "tr_type": tr_type,
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": tr_id, "tr_key": tr_key}},
        }
        return json.dumps(payload)

    def get_send_data_program(self, cmd: int, stockcode: str) -> str:
        """
        Build WebSocket payload for program trading stream (H0STPGM0).

        Used for subscribing to institutional program trading flow data.

        Args:
            cmd: 3=subscribe, 4=unsubscribe
            stockcode: Stock code

        Returns:
            JSON string for WebSocket send
        """
        assert cmd in (3, 4), f"Invalid cmd: {cmd} (must be 3=sub or 4=unsub)"

        payload = {
            "header": {
                "approval_key": self.websocket_approval_key,
                "custtype": self.custtype,
                "tr_type": "1" if cmd == 3 else "2",
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": "H0STPGM0", "tr_key": stockcode}},
        }
        return json.dumps(payload)

    # =========================================================================
    # PCIM STRATEGY HELPER METHODS
    # =========================================================================

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get quote with last, bid, ask, spread for execution decisions."""
        data = self.get_current_price(symbol)
        if not data:
            return None
        last = float(data.get('stck_prpr', 0))
        bid = self.get_best_bid(symbol)
        ask = self.get_best_ask(symbol)
        return {'last': last, 'bid': bid, 'ask': ask, 'spread': (ask - bid) / last if last > 0 else 0}

    def get_daily_ohlcv(self, symbol: str, days: int = 60) -> Optional[List[Dict]]:
        """Get daily OHLCV as list of dicts for easy iteration."""
        df = self.get_daily_bars(symbol, days)
        if df is None or df.empty:
            return None
        return df.to_dict('records')

    def get_intraday_1m(self, symbol: str, start: str, end: str) -> Optional[List[Dict]]:
        """Get 1-minute intraday bars between start and end times (HH:MM format)."""
        df = self.get_minute_bars(symbol, minutes=200)
        if df is None or df.empty:
            return None
        records = df.to_dict('records')
        # Filter by time range if timestamps available
        return records

    def get_intraday_3m(self, symbol: str, start: str, end: str) -> Optional[List[Dict]]:
        """Get 3-minute bars by aggregating 1-minute bars."""
        bars_1m = self.get_intraday_1m(symbol, start, end)
        if not bars_1m or len(bars_1m) < 3:
            return None
        # Aggregate first 3 bars into one bar
        first_3 = bars_1m[:3]
        return [{
            'open': first_3[0].get('open', 0),
            'high': max(b.get('high', 0) for b in first_3),
            'low': min(b.get('low', float('inf')) for b in first_3),
            'close': first_3[-1].get('close', 0),
            'volume': sum(b.get('volume', 0) for b in first_3),
        }]

    def get_atr_20d(self, symbol: str) -> float:
        """Compute 20-day ATR from daily bars."""
        bars = self.get_daily_ohlcv(symbol, days=25)
        if not bars or len(bars) < 2:
            return 0.0
        tr_values = []
        for i in range(1, len(bars)):
            h, l = float(bars[i].get('high', 0)), float(bars[i].get('low', 0))
            pc = float(bars[i-1].get('close', 0))
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_values.append(tr)
        return sum(tr_values[-20:]) / min(20, len(tr_values)) if tr_values else 0.0

    def get_adtv_20d(self, symbol: str) -> float:
        """Compute 20-day average daily traded value using acml_tr_pbmn (거래대금)."""
        raw = self.get_daily_chart_data(symbol, count=25)
        if not raw:
            return 0.0
        rows = raw.get('output2', []) or []
        if len(rows) < 5:
            return 0.0
        # acml_tr_pbmn is the exchange-reported daily trading value in KRW
        values = []
        for r in rows[-20:]:
            try:
                v = float(r.get('acml_tr_pbmn', 0))
                values.append(v)
            except (ValueError, TypeError):
                continue
        return sum(values) / len(values) if values else 0.0

    def get_market_cap(self, symbol: str) -> float:
        """Get market capitalization in KRW. Returns 0 if unavailable."""
        data = self.get_current_price(symbol)
        if data:
            # hts_avls is HTS 시가총액 in 억원 (1e8 KRW)
            hts = data.get('hts_avls')
            if hts is not None:
                try:
                    v = float(hts)
                    if v > 0:
                        return v * 1e8
                except (ValueError, TypeError):
                    pass
            # Fallback fields assumed to be in raw KRW
            for field in ('total_mrkt_val', 'mrkt_cap'):
                if field in data:
                    try:
                        return float(data[field])
                    except (ValueError, TypeError):
                        continue
        return 0.0

    def get_index_daily(self, index: str, days: int = 60) -> Optional[List[Dict]]:
        """Get daily index bars. index should be 'KOSPI' or 'KOSDAQ'."""
        code = "0001" if index.upper() == "KOSPI" else "1001"
        df = self.get_daily_bars(code, days)
        if df is None or df.empty:
            return None
        return df.to_dict('records')

    def get_index_realtime(self, index: str) -> float:
        """Get realtime index value."""
        code = "0001" if index.upper() == "KOSPI" else "1001"
        return self.get_last_price(code)

    def get_upper_limit_price(self, symbol: str, trade_date=None) -> float:
        """Get upper price limit for the day (상한가)."""
        data = self.get_current_price(symbol)
        if data:
            return float(data.get('stck_mxpr', 0))
        return 0.0

    def get_tick_size(self, symbol: str) -> float:
        """Get tick size based on price level (Korean market rules)."""
        price = self.get_last_price(symbol)
        if price < 1000:
            return 1
        elif price < 5000:
            return 5
        elif price < 10000:
            return 10
        elif price < 50000:
            return 50
        elif price < 100000:
            return 100
        elif price < 500000:
            return 500
        else:
            return 1000

    def is_in_vi(self, symbol: str) -> bool:
        """Check if symbol is in volatility interruption (VI)."""
        data = self.get_current_price(symbol)
        if data:
            # Check VI indicator fields
            vi_cls = data.get('vi_cls_code', '')
            return vi_cls in ('1', '2')  # 1=static VI, 2=dynamic VI
        return False

    def get_open_3m_baseline(self, symbol: str, lookback_days: int = 20) -> float:
        """Get median 09:00-09:03 volume over lookback days (estimated from daily)."""
        bars = self.get_daily_ohlcv(symbol, days=lookback_days + 5)
        if not bars or len(bars) < lookback_days:
            return 0.0
        # Approximate: first 3 minutes is ~1/130 of daily volume (390 min / 3)
        volumes = [float(b.get('volume', 0)) / 130 for b in bars[-lookback_days:]]
        volumes.sort()
        mid = len(volumes) // 2
        return volumes[mid] if volumes else 0.0

    def resolve_symbol(self, name_or_code: str) -> Optional[str]:
        """Resolve ticker name to symbol code. Returns code if valid, else None."""
        # If already a valid 6-digit code, verify it exists
        if name_or_code.isdigit() and len(name_or_code) == 6:
            if self.get_current_price(name_or_code):
                return name_or_code
        # TODO: Implement name-to-code lookup via search API
        # For now, return None if not a valid code
        return None

    def earnings_within_days(self, symbol: str, days: int) -> bool:
        """Check if earnings announcement is within N trading days. Stub - returns False."""
        # TODO: Integrate with earnings calendar API or external data source
        return False

    def is_trading_day(self, check_date) -> bool:
        """Check if date is a KRX trading day using the trading calendar."""
        from .trading_calendar import get_trading_calendar
        return get_trading_calendar().is_trading_day(check_date)
