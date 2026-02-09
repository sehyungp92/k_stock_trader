"""
KIS API Response Handling

Provides:
- Structured response parsing with namedtuple-based access
- Error extraction and logging
- Success/failure detection
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict, NamedTuple, Optional, Tuple

import requests
from loguru import logger


class APIResponse:
    """
    Wrapper for KIS API HTTP responses.
    
    Parses response headers and JSON body into namedtuple objects
    for convenient attribute-style access.
    
    Args:
        resp: requests.Response object from API call
    
    Example:
        >>> res = requests.get('https://api.example.com/price')
        >>> api_res = APIResponse(res)
        >>> if api_res.is_ok():
        ...     print(api_res.get_body().output)
        ... else:
        ...     api_res.print_error()
    
    Attributes:
        status_code: HTTP status code
        error_code: KIS error code (rt_cd field)
        error_message: KIS error message (msg1 field)
    """
    
    # KIS success codes
    SUCCESS_CODES = frozenset({'0', ''})
    
    # Default body for parse failures
    _DEFAULT_BODY = namedtuple('body', ['rt_cd', 'msg1'])('999', 'JSON Decode Error')
    
    def __init__(self, resp: requests.Response) -> None:
        self._response = resp
        self._status_code = resp.status_code
        self._header = self._parse_header()
        self._body = self._parse_body()
        
        # Cache error info
        self._error_code: str = getattr(self._body, 'rt_cd', '999')
        self._error_message: str = getattr(self._body, 'msg1', 'Unknown error')
    
    def _parse_header(self) -> NamedTuple:
        """
        Parse response headers into namedtuple.
        
        Only includes lowercase header keys (KIS-specific headers).
        """
        fields: Dict[str, str] = {}
        
        for key in self._response.headers.keys():
            if key.islower():
                fields[key] = self._response.headers.get(key, '')
        
        if not fields:
            # Return empty tuple if no lowercase headers
            empty_header = namedtuple('header', [])
            return empty_header()
        
        header_class = namedtuple('header', fields.keys())
        return header_class(**fields)
    
    def _parse_body(self) -> NamedTuple:
        """
        Parse response JSON body into namedtuple.
        
        Returns default error body on parse failure.
        """
        try:
            json_data = self._response.json()
            
            if not isinstance(json_data, dict):
                logger.warning(f"Unexpected JSON type: {type(json_data)}")
                return self._DEFAULT_BODY
            
            if not json_data:
                # Empty dict
                empty_body = namedtuple('body', ['rt_cd', 'msg1'])('0', 'Empty response')
                return empty_body
            
            # Sanitize keys for namedtuple (remove invalid characters)
            sanitized = {}
            for key, value in json_data.items():
                # Replace hyphens and spaces with underscores
                safe_key = str(key).replace('-', '_').replace(' ', '_')
                sanitized[safe_key] = value
            
            body_class = namedtuple('body', sanitized.keys())
            return body_class(**sanitized)
            
        except requests.exceptions.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            return self._DEFAULT_BODY
        except Exception as e:
            logger.debug(f"Body parse error: {e}")
            return self._DEFAULT_BODY
    
    # =========================================================================
    # Public Properties
    # =========================================================================
    
    @property
    def status_code(self) -> int:
        """HTTP status code."""
        return self._status_code
    
    @property
    def error_code(self) -> str:
        """KIS error code (rt_cd field)."""
        return self._error_code
    
    @property
    def error_message(self) -> str:
        """KIS error message (msg1 field)."""
        return self._error_message
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def get_result_code(self) -> int:
        """Get HTTP status code."""
        return self._status_code
    
    def get_header(self) -> NamedTuple:
        """Get parsed response headers."""
        return self._header
    
    def get_body(self) -> NamedTuple:
        """Get parsed response body."""
        return self._body
    
    def get_response(self) -> requests.Response:
        """Get original requests.Response object."""
        return self._response
    
    def get_error_code(self) -> str:
        """Get KIS error code."""
        return self._error_code
    
    def get_error_message(self) -> str:
        """Get KIS error message."""
        return self._error_message
    
    def is_ok(self) -> bool:
        """
        Check if the API call was successful.
        
        Returns:
            True if HTTP 200 and KIS rt_cd indicates success
        """
        if self._status_code != 200:
            return False
        
        return self._error_code in self.SUCCESS_CODES
    
    def is_error(self) -> bool:
        """Check if the API call failed."""
        return not self.is_ok()
    
    def get_output(self, key: str = 'output', default: Any = None) -> Any:
        """
        Get specific output field from body.
        
        Args:
            key: Field name to retrieve
            default: Default value if field not found
        
        Returns:
            Field value or default
        """
        return getattr(self._body, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary.
        
        Returns:
            Dict with status_code, is_ok, error_code, error_message, body
        """
        # Convert namedtuple body to dict
        body_dict = {}
        if hasattr(self._body, '_fields'):
            body_dict = {f: getattr(self._body, f) for f in self._body._fields}
        
        return {
            'status_code': self._status_code,
            'is_ok': self.is_ok(),
            'error_code': self._error_code,
            'error_message': self._error_message,
            'body': body_dict,
        }
    
    # =========================================================================
    # Logging Methods
    # =========================================================================
    
    def print_all(self) -> None:
        """Log all response headers and body fields."""
        logger.info("<Header>")
        if hasattr(self._header, '_fields'):
            for field in self._header._fields:
                logger.info(f"\t-{field}: {getattr(self._header, field)}")
        
        logger.info("<Body>")
        if hasattr(self._body, '_fields'):
            for field in self._body._fields:
                value = getattr(self._body, field)
                # Truncate long values
                if isinstance(value, (list, dict)) and len(str(value)) > 200:
                    value = f"{str(value)[:200]}..."
                logger.info(f"\t-{field}: {value}")
    
    def print_error(self) -> None:
        """Log error details."""
        logger.info("-" * 40)
        logger.info(f"Error in response: HTTP {self._status_code}")
        logger.info(f"  rt_cd: {self._error_code}")
        logger.info(f"  msg1: {self._error_message}")
        
        # Log additional error fields if present
        for field in ('msg_cd', 'msg2', 'msg3'):
            if hasattr(self._body, field):
                value = getattr(self._body, field)
                if value:
                    logger.info(f"  {field}: {value}")
        
        logger.info("-" * 40)
    
    # =========================================================================
    # Magic Methods
    # =========================================================================
    
    def __bool__(self) -> bool:
        """Allow truthy check: `if response:`"""
        return self.is_ok()
    
    def __repr__(self) -> str:
        status = "OK" if self.is_ok() else f"ERROR({self._error_code})"
        return f"APIResponse(status={self._status_code}, {status})"
    
    def __str__(self) -> str:
        if self.is_ok():
            return f"APIResponse OK (HTTP {self._status_code})"
        return f"APIResponse ERROR: [{self._error_code}] {self._error_message}"


def create_error_response(
    status_code: int = 500,
    error_code: str = '999',
    error_message: str = 'Internal error',
) -> APIResponse:
    """
    Create a synthetic error APIResponse.
    
    Useful for error handling when no actual HTTP response is available.
    
    Args:
        status_code: HTTP status code to simulate
        error_code: KIS error code
        error_message: Error message
    
    Returns:
        APIResponse with error state
    """
    # Create mock response
    mock_response = requests.models.Response()
    mock_response.status_code = status_code
    mock_response._content = f'{{"rt_cd": "{error_code}", "msg1": "{error_message}"}}'.encode()
    mock_response.headers['content-type'] = 'application/json'
    
    return APIResponse(mock_response)
