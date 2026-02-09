"""KRX Trading Calendar with holiday awareness."""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Set

import yaml
from loguru import logger


class KRXTradingCalendar:
    """Korean Exchange trading calendar with holiday support."""

    _instance: Optional["KRXTradingCalendar"] = None

    def __init__(self, holidays: Optional[Set[date]] = None):
        """Initialize with optional holidays set. Loads from YAML if not provided."""
        if holidays is not None:
            self._holidays = holidays
        else:
            self._holidays = self._load_holidays()

    @classmethod
    def get_instance(cls) -> "KRXTradingCalendar":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_holidays(self) -> Set[date]:
        """Load holidays from YAML file."""
        holidays: Set[date] = set()
        yaml_path = Path(__file__).parent / "data" / "krx_holidays.yaml"

        if not yaml_path.exists():
            logger.warning(f"Holiday file not found: {yaml_path}. Using weekday-only calendar.")
            return holidays

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data:
                    for year, dates in data.items():
                        if isinstance(dates, list):
                            for date_str in dates:
                                try:
                                    holidays.add(date.fromisoformat(date_str))
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Invalid date in holidays: {date_str}: {e}")
            logger.info(f"Loaded {len(holidays)} KRX holidays from {yaml_path}")
        except Exception as e:
            logger.error(f"Error loading holiday file: {e}")

        return holidays

    def is_trading_day(self, check_date: date) -> bool:
        """Check if date is a KRX trading day (not weekend, not holiday)."""
        if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        if check_date in self._holidays:
            return False
        return True

    def previous_trading_day(self, from_date: date) -> date:
        """Get the previous trading day before from_date."""
        candidate = from_date - timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def next_trading_day(self, from_date: date) -> date:
        """Get the next trading day after from_date."""
        candidate = from_date + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate


def get_trading_calendar() -> KRXTradingCalendar:
    """Get the singleton trading calendar instance."""
    return KRXTradingCalendar.get_instance()
