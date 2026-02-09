"""
KIS API Decorators and Utilities

Provides:
- Rate limiting with configurable intervals
- Thread-safe request throttling
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable, Optional, TypeVar, cast

from loguru import logger

# Type variable for preserving function signatures
F = TypeVar('F', bound=Callable[..., Any])


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.
    
    Ensures minimum time interval between calls. Useful for
    respecting API rate limits.
    
    Args:
        min_interval: Minimum seconds between calls (default: 0.05 = 50ms)
        name: Optional name for logging purposes
    
    Example:
        >>> limiter = RateLimiter(min_interval=0.1)  # 10 calls/sec max
        >>> for i in range(5):
        ...     limiter.wait()
        ...     make_api_call()
    
    Thread Safety:
        This class is thread-safe. Multiple threads can call wait()
        concurrently and will be properly throttled.
    """
    
    def __init__(self, min_interval: float = 0.05, name: Optional[str] = None) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        
        self._min_interval = min_interval
        self._name = name or "RateLimiter"
        self._last_call: float = 0.0
        self._lock = threading.Lock()
        self._total_waits: int = 0
        self._total_wait_time: float = 0.0
    
    @property
    def min_interval(self) -> float:
        """Minimum interval between calls in seconds."""
        return self._min_interval
    
    @property
    def max_rate(self) -> float:
        """Maximum calls per second."""
        if self._min_interval == 0:
            return float('inf')
        return 1.0 / self._min_interval
    
    def wait(self) -> float:
        """
        Wait if necessary to respect rate limit.
        
        Returns:
            Actual time waited in seconds (0 if no wait needed)
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                time.sleep(sleep_time)
                self._total_waits += 1
                self._total_wait_time += sleep_time
                self._last_call = time.time()
                return sleep_time
            
            self._last_call = now
            return 0.0
    
    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self._lock:
            self._last_call = 0.0
            self._total_waits = 0
            self._total_wait_time = 0.0
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dict with total_waits, total_wait_time, avg_wait_time
        """
        with self._lock:
            avg_wait = (
                self._total_wait_time / self._total_waits
                if self._total_waits > 0
                else 0.0
            )
            return {
                'name': self._name,
                'min_interval': self._min_interval,
                'total_waits': self._total_waits,
                'total_wait_time': round(self._total_wait_time, 3),
                'avg_wait_time': round(avg_wait, 4),
            }
    
    def __repr__(self) -> str:
        return f"RateLimiter(min_interval={self._min_interval}, name={self._name!r})"


# Global rate limiter instance for default decorator usage
_global_rate_limiter: Optional[RateLimiter] = None
_global_limiter_lock = threading.Lock()


def _get_global_limiter(min_interval: float) -> RateLimiter:
    """Get or create global rate limiter with specified interval."""
    global _global_rate_limiter
    
    with _global_limiter_lock:
        if _global_rate_limiter is None or _global_rate_limiter.min_interval != min_interval:
            _global_rate_limiter = RateLimiter(min_interval=min_interval, name="global")
        return _global_rate_limiter


def rate_limit(
    min_interval: float = 0.05,
    limiter: Optional[RateLimiter] = None,
) -> Callable[[F], F]:
    """
    Decorator to rate limit function calls.
    
    Ensures minimum time interval between successive calls to the
    decorated function. Useful for API rate limiting.
    
    Args:
        min_interval: Minimum seconds between calls (default: 50ms)
        limiter: Optional custom RateLimiter instance. If None, uses
                 a global shared limiter.
    
    Returns:
        Decorated function with rate limiting
    
    Example:
        >>> @rate_limit(min_interval=0.1)  # Max 10 calls/sec
        ... def call_api():
        ...     return requests.get('https://api.example.com')
        
        >>> # With custom limiter for per-endpoint limiting
        >>> order_limiter = RateLimiter(min_interval=0.5)
        >>> @rate_limit(limiter=order_limiter)
        ... def place_order():
        ...     pass
    
    Note:
        When using the global limiter (default), all decorated functions
        share the same rate limit. Use custom limiters for independent
        rate limiting per function or endpoint.
    """
    def decorator(func: F) -> F:
        # Use provided limiter or fall back to global
        _limiter = limiter or _get_global_limiter(min_interval)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _limiter.wait()
            return func(*args, **kwargs)
        
        # Attach limiter reference for testing/inspection
        wrapper._rate_limiter = _limiter  # type: ignore[attr-defined]
        
        return cast(F, wrapper)
    
    return decorator


def rate_limit_async(
    min_interval: float = 0.05,
    limiter: Optional[RateLimiter] = None,
) -> Callable[[F], F]:
    """
    Async-compatible rate limit decorator.
    
    Same as rate_limit but uses asyncio.sleep for async functions.
    
    Args:
        min_interval: Minimum seconds between calls
        limiter: Optional custom RateLimiter instance
    
    Example:
        >>> @rate_limit_async(min_interval=0.1)
        ... async def async_api_call():
        ...     async with aiohttp.ClientSession() as session:
        ...         return await session.get('https://api.example.com')
    """
    import asyncio
    
    def decorator(func: F) -> F:
        _limiter = limiter or _get_global_limiter(min_interval)
        _async_lock = asyncio.Lock()
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with _async_lock:
                # Calculate wait time (same logic as sync version)
                now = time.time()
                with _limiter._lock:
                    elapsed = now - _limiter._last_call
                    if elapsed < _limiter._min_interval:
                        sleep_time = _limiter._min_interval - elapsed
                        await asyncio.sleep(sleep_time)
                    _limiter._last_call = time.time()
            
            return await func(*args, **kwargs)
        
        wrapper._rate_limiter = _limiter  # type: ignore[attr-defined]
        return cast(F, wrapper)
    
    return decorator


def get_global_limiter_stats() -> dict[str, Any]:
    """
    Get statistics from the global rate limiter.
    
    Returns:
        Dict with limiter statistics, or empty dict if not initialized
    """
    if _global_rate_limiter is not None:
        return _global_rate_limiter.get_stats()
    return {}
