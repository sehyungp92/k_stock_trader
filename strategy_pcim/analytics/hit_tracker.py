"""Bucket A Hit Rate Tracking for Adaptive Volume Baseline.

Spec §3.2: Calibrate volume threshold based on hit-rate:
- hit-rate > 80% → raise to 140%
- hit-rate < 30% → lower to 110%
- otherwise → default 120%
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import json
import os
from loguru import logger


@dataclass
class BucketAHitTracker:
    """Tracks Bucket A trigger hit-rate for threshold calibration.

    Hit = Bucket A triggered AND entry was filled (position created).
    Miss = Bucket A triggered but entry was NOT filled (cancelled/unfilled).
    """
    triggers: int = 0
    fills: int = 0
    last_reset_date: Optional[date] = None
    _state_file: str = field(default="pcim_bucket_a_hits.json", repr=False)

    # Threshold calibration constants
    HIGH_HIT_RATE = 0.80
    LOW_HIT_RATE = 0.30
    THRESHOLD_HIGH = 1.40  # When hit-rate is high, raise threshold
    THRESHOLD_LOW = 1.10   # When hit-rate is low, lower threshold
    THRESHOLD_DEFAULT = 1.20

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate (0.0 to 1.0)."""
        if self.triggers <= 0:
            return 0.5  # Default to neutral when no data
        return self.fills / self.triggers

    def calibrated_threshold(self) -> float:
        """Get calibrated volume threshold based on hit-rate.

        - hit-rate > 80% → 1.40 (raise threshold, be more selective)
        - hit-rate < 30% → 1.10 (lower threshold, be more permissive)
        - otherwise → 1.20 (default)
        """
        rate = self.hit_rate
        if rate > self.HIGH_HIT_RATE:
            return self.THRESHOLD_HIGH
        elif rate < self.LOW_HIT_RATE:
            return self.THRESHOLD_LOW
        return self.THRESHOLD_DEFAULT

    def record_trigger(self, filled: bool):
        """Record a Bucket A trigger event.

        Args:
            filled: True if the entry order was filled, False otherwise
        """
        self.triggers += 1
        if filled:
            self.fills += 1
        logger.debug(f"Bucket A hit tracker: triggers={self.triggers} fills={self.fills} "
                    f"hit_rate={self.hit_rate:.2%} threshold={self.calibrated_threshold():.2f}")

    def reset_if_new_period(self, today: date, reset_period_days: int = 20):
        """Reset counters if we've passed the reset period.

        Default reset every 20 trading days (~1 month) to adapt to
        changing market conditions while maintaining statistical significance.
        """
        if self.last_reset_date is None:
            self.last_reset_date = today
            return

        days_since_reset = (today - self.last_reset_date).days
        if days_since_reset >= reset_period_days:
            old_rate = self.hit_rate
            self.triggers = 0
            self.fills = 0
            self.last_reset_date = today
            logger.info(f"Bucket A hit tracker reset (was {old_rate:.2%} over {days_since_reset} days)")

    def save(self, state_dir: str = "."):
        """Persist state to JSON file."""
        path = os.path.join(state_dir, self._state_file)
        data = {
            "triggers": self.triggers,
            "fills": self.fills,
            "last_reset_date": self.last_reset_date.isoformat() if self.last_reset_date else None,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save Bucket A hit tracker: {e}")

    @classmethod
    def load(cls, state_dir: str = ".") -> "BucketAHitTracker":
        """Load state from JSON file."""
        path = os.path.join(state_dir, "pcim_bucket_a_hits.json")
        try:
            with open(path) as f:
                data = json.load(f)
            tracker = cls(
                triggers=data.get("triggers", 0),
                fills=data.get("fills", 0),
            )
            if data.get("last_reset_date"):
                tracker.last_reset_date = date.fromisoformat(data["last_reset_date"])
            logger.info(f"Loaded Bucket A hit tracker: {tracker.triggers} triggers, "
                       f"{tracker.fills} fills, rate={tracker.hit_rate:.2%}")
            return tracker
        except FileNotFoundError:
            logger.info("No existing Bucket A hit tracker state, starting fresh")
            return cls()
        except Exception as e:
            logger.warning(f"Failed to load Bucket A hit tracker: {e}")
            return cls()
