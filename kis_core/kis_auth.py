"""
KIS Authentication and Environment Configuration

Handles:
- API credentials management (live vs paper trading)
- OAuth2 token lifecycle with automatic refresh
- WebSocket approval key acquisition
- Base header construction for API requests
"""

from __future__ import annotations

import copy
import json
import threading
import time
from typing import Any, Dict, Optional

import requests
from loguru import logger


class KoreaInvestEnv:
    """
    KIS API Environment and Authentication Manager.

    Manages credentials, tokens, and provides authenticated headers
    for API requests. Supports both live and paper trading modes.

    Args:
        cfg: Configuration dictionary with required keys:
            - custtype: Customer type (e.g., 'P' for personal)
            - my_agent: User-Agent string
            - is_paper_trading: True for paper trading mode
            - htsid: HTS user ID

            For paper trading:
            - paper_url, paper_api_key, paper_api_secret_key, paper_stock_account_number

            For live trading:
            - url, api_key, api_secret_key, stock_account_number

            Optional (for paper trading with real API fallback):
            - url, api_key, api_secret_key: Real API credentials for endpoints
              not supported by paper trading server (e.g., program trading trend)

    Example:
        >>> cfg = {
        ...     'custtype': 'P',
        ...     'my_agent': 'MyTradingBot/1.0',
        ...     'is_paper_trading': True,
        ...     'htsid': 'user123',
        ...     'paper_url': 'https://openapivts.koreainvestment.com:29443',
        ...     'paper_api_key': 'xxx',
        ...     'paper_api_secret_key': 'yyy',
        ...     'paper_stock_account_number': '50000000-01',
        ...     # Optional: real API fallback for unsupported endpoints
        ...     'url': 'https://openapi.koreainvestment.com:9443',
        ...     'api_key': 'real_xxx',
        ...     'api_secret_key': 'real_yyy',
        ... }
        >>> env = KoreaInvestEnv(cfg)
        >>> headers = env.get_base_headers()
    """
    
    # Required configuration keys
    REQUIRED_KEYS = frozenset({'custtype', 'my_agent', 'is_paper_trading', 'htsid'})
    
    # Token validity duration (24 hours minus 5 minute buffer)
    TOKEN_VALIDITY_SECONDS = 86400
    TOKEN_REFRESH_BUFFER_SECONDS = 300
    
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._validate_config(cfg)
        
        self.cfg = cfg.copy()
        self.custtype: str = cfg['custtype']
        self.htsid: str = cfg['htsid']
        self._is_paper_trading: bool = cfg['is_paper_trading']
        
        # Base headers (without auth token)
        self._base_headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
            "charset": "UTF-8",
            'User-Agent': cfg['my_agent'],
        }
        
        # Select credentials based on trading mode
        if self._is_paper_trading:
            self.using_url: str = cfg['paper_url']
            self.api_key: str = cfg['paper_api_key']
            self.api_secret_key: str = cfg['paper_api_secret_key']
            account_num: str = cfg['paper_stock_account_number']
        else:
            self.using_url: str = cfg['url']
            self.api_key: str = cfg['api_key']
            self.api_secret_key: str = cfg['api_secret_key']
            account_num: str = cfg['stock_account_number']
        
        # Token management
        self._token: Optional[str] = None
        self._token_expires_at: float = 0
        self._token_lock = threading.Lock()

        # Initialize token
        self._refresh_token_if_needed()

        # Get WebSocket approval key
        websocket_approval_key = self._get_websocket_approval_key()

        # Optional: Real API credentials for fallback (paper trading only)
        # Used for endpoints not supported by paper trading server
        self._has_real_fallback = False
        self._real_url: Optional[str] = None
        self._real_api_key: Optional[str] = None
        self._real_api_secret_key: Optional[str] = None
        self._real_token: Optional[str] = None
        self._real_token_expires_at: float = 0
        self._real_token_lock = threading.Lock()

        if self._is_paper_trading:
            # Check if real API credentials are provided for fallback
            real_keys = {'url', 'api_key', 'api_secret_key'}
            if real_keys.issubset(set(cfg.keys())):
                self._real_url = cfg['url']
                self._real_api_key = cfg['api_key']
                self._real_api_secret_key = cfg['api_secret_key']
                self._has_real_fallback = True
                logger.info("Real API fallback enabled for unsupported paper trading endpoints")
        
        # Update headers with credentials
        self._base_headers["appkey"] = self.api_key
        self._base_headers["appsecret"] = self.api_secret_key
        
        # Store derived config
        self.cfg['websocket_approval_key'] = websocket_approval_key
        self.cfg['account_num'] = account_num
        self.cfg['using_url'] = self.using_url
    
    def _validate_config(self, cfg: Dict[str, Any]) -> None:
        """Validate required configuration keys exist."""
        missing = self.REQUIRED_KEYS - set(cfg.keys())
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
        
        # Validate mode-specific keys
        if cfg['is_paper_trading']:
            paper_keys = {'paper_url', 'paper_api_key', 'paper_api_secret_key', 'paper_stock_account_number'}
            missing_paper = paper_keys - set(cfg.keys())
            if missing_paper:
                raise ValueError(f"Paper trading requires: {missing_paper}")
        else:
            live_keys = {'url', 'api_key', 'api_secret_key', 'stock_account_number'}
            missing_live = live_keys - set(cfg.keys())
            if missing_live:
                raise ValueError(f"Live trading requires: {missing_live}")
    
    def _refresh_token_if_needed(self) -> None:
        """
        Check and refresh token if expired or expiring soon.
        
        Thread-safe with double-checked locking pattern.
        """
        current_time = time.time()
        refresh_threshold = self._token_expires_at - self.TOKEN_REFRESH_BUFFER_SECONDS
        
        if current_time < refresh_threshold:
            return
        
        with self._token_lock:
            # Double-check after acquiring lock
            if current_time >= self._token_expires_at - self.TOKEN_REFRESH_BUFFER_SECONDS:
                logger.info("Refreshing access token...")
                try:
                    self._token = self._fetch_access_token()
                    self._token_expires_at = time.time() + self.TOKEN_VALIDITY_SECONDS
                    logger.info("Access token refreshed successfully")
                except Exception as e:
                    logger.error(f"Failed to refresh token: {e}")
                    raise
    
    def _fetch_access_token(
        self,
        max_retries: int = 5,
        base_delay: float = 65.0,
    ) -> str:
        """
        Fetch OAuth2 access token from KIS API.
        
        KIS rate limits token requests to 1 per minute per key,
        so we retry with exponential backoff on 403 errors.
        
        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries (seconds)
        
        Returns:
            Bearer token string
        
        Raises:
            requests.exceptions.RequestException: On network failure
            KeyError: On unexpected response format
        """
        url = f'{self.using_url}/oauth2/tokenP'
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.api_key,
            "appsecret": self.api_secret_key,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
        }
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, max_retries + 1):
            try:
                res = requests.post(
                    url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=10,
                )
                
                # Handle rate limiting
                if res.status_code == 403 and attempt < max_retries:
                    logger.warning(
                        f"Token rate-limited (attempt {attempt}/{max_retries}), "
                        f"retrying in {base_delay}s... | {res.text[:100]}"
                    )
                    time.sleep(base_delay)
                    continue
                
                res.raise_for_status()
                
                access_token = res.json()['access_token']
                return f"Bearer {access_token}"
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Token request failed (attempt {attempt}/{max_retries}): {e}, "
                        f"retrying in {base_delay}s..."
                    )
                    time.sleep(base_delay)
                    continue
                raise
            
            except KeyError as e:
                logger.error(f"Unexpected token response format: {e}")
                raise
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Token fetch failed unexpectedly")
    
    def _get_websocket_approval_key(self) -> str:
        """
        Get WebSocket approval key for real-time data subscriptions.
        
        Returns:
            Approval key string
        
        Raises:
            requests.exceptions.RequestException: On network failure
        """
        url = f"{self.using_url}/oauth2/Approval"
        headers = {"Content-Type": "application/json"}
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.api_key,
            "secretkey": self.api_secret_key,
        }
        
        try:
            res = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10,
            )
            res.raise_for_status()
            return res.json()["approval_key"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get WebSocket approval key: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected approval response format: {e}")
            raise
    
    def get_base_headers(self) -> Dict[str, str]:
        """
        Get base headers with fresh authorization token.

        Automatically refreshes token if needed.

        Returns:
            Copy of headers dict with current auth token
        """
        self._refresh_token_if_needed()

        headers = copy.deepcopy(self._base_headers)
        headers["authorization"] = self._token
        return headers

    def _refresh_real_token_if_needed(self) -> None:
        """
        Check and refresh real API token if expired or expiring soon.

        Thread-safe with double-checked locking pattern.
        Only used when real API fallback is enabled.
        """
        if not self._has_real_fallback:
            return

        current_time = time.time()
        refresh_threshold = self._real_token_expires_at - self.TOKEN_REFRESH_BUFFER_SECONDS

        if current_time < refresh_threshold:
            return

        with self._real_token_lock:
            # Double-check after acquiring lock
            if current_time >= self._real_token_expires_at - self.TOKEN_REFRESH_BUFFER_SECONDS:
                logger.info("Refreshing real API access token...")
                try:
                    self._real_token = self._fetch_real_access_token()
                    self._real_token_expires_at = time.time() + self.TOKEN_VALIDITY_SECONDS
                    logger.info("Real API access token refreshed successfully")
                except Exception as e:
                    logger.error(f"Failed to refresh real API token: {e}")
                    raise

    def _fetch_real_access_token(
        self,
        max_retries: int = 5,
        base_delay: float = 65.0,
    ) -> str:
        """
        Fetch OAuth2 access token from real KIS API.

        Similar to _fetch_access_token but uses real API credentials.

        Returns:
            Bearer token string

        Raises:
            requests.exceptions.RequestException: On network failure
            RuntimeError: If real API fallback not configured
        """
        if not self._has_real_fallback:
            raise RuntimeError("Real API fallback not configured")

        url = f'{self._real_url}/oauth2/tokenP'
        payload = {
            "grant_type": "client_credentials",
            "appkey": self._real_api_key,
            "appsecret": self._real_api_secret_key,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                res = requests.post(
                    url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=10,
                )

                # Handle rate limiting
                if res.status_code == 403 and attempt < max_retries:
                    logger.warning(
                        f"Real API token rate-limited (attempt {attempt}/{max_retries}), "
                        f"retrying in {base_delay}s... | {res.text[:100]}"
                    )
                    time.sleep(base_delay)
                    continue

                res.raise_for_status()

                access_token = res.json()['access_token']
                return f"Bearer {access_token}"

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Real API token request failed (attempt {attempt}/{max_retries}): {e}, "
                        f"retrying in {base_delay}s..."
                    )
                    time.sleep(base_delay)
                    continue
                raise

            except KeyError as e:
                logger.error(f"Unexpected real API token response format: {e}")
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Real API token fetch failed unexpectedly")

    def get_real_api_headers(self) -> Optional[Dict[str, str]]:
        """
        Get headers using real API credentials (for unsupported paper endpoints).

        Returns headers with real API authorization token and credentials,
        or None if real API fallback is not configured.

        Returns:
            Copy of headers dict with real API auth, or None
        """
        if not self._has_real_fallback:
            return None

        self._refresh_real_token_if_needed()

        headers = copy.deepcopy(self._base_headers)
        headers["authorization"] = self._real_token
        headers["appkey"] = self._real_api_key
        headers["appsecret"] = self._real_api_secret_key
        return headers

    @property
    def has_real_fallback(self) -> bool:
        """Whether real API fallback is configured for unsupported endpoints."""
        return self._has_real_fallback

    @property
    def real_url(self) -> Optional[str]:
        """Real API URL for fallback (None if not configured)."""
        return self._real_url if self._has_real_fallback else None
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        Get full configuration dictionary.
        
        Returns:
            Copy of configuration with derived values
        """
        return copy.deepcopy(self.cfg)
    
    @property
    def is_paper_trading(self) -> bool:
        """Whether this environment is configured for paper trading."""
        return self._is_paper_trading
    
    @property
    def account_num(self) -> str:
        """The account number for this environment."""
        return self.cfg['account_num']
    
    @property
    def websocket_approval_key(self) -> str:
        """The WebSocket approval key."""
        return self.cfg['websocket_approval_key']
    
    def __repr__(self) -> str:
        mode = "paper" if self._is_paper_trading else "live"
        masked_account = self.account_num[:4] + "****" if self.account_num else "N/A"
        return f"KoreaInvestEnv(mode={mode}, account={masked_account})"
