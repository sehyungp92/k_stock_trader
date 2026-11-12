"""KRX Trading Calendar with holiday awareness."""

from datetime import date, time, timedelta
from pathlib import Path
from typing import Optional, Set

import yaml
from loguru import logger


class KRXTradingCalendar:
    """Korean Exchange trading calendar with holiday support."""

    _instance: Optional["KRXTradingCalendar"] = None

    def __init__(self, holidays: Optional[Set[date]] = None):
        """Initialize with optional holidays set. Loads from YAML if not provided."""
        self.covered_years: set[int] = set()
        self.special_sessions: dict[date, tuple[time, time]] = {}
        self.session_hours_verified = False
        if holidays is not None:
            self._holidays = holidays
            self.covered_years = {day.year for day in holidays}
        else:
            self._holidays = self._load_holidays()
        self.session_coverage = None
        self.session_schedule_version = "unverified"
        self.load_session_schedule(Path(__file__).parent / "data" / "krx_sessions.yaml")
        self.load_intraday_halts(Path(__file__).parent / "data" / "krx_intraday_halts.yaml")

    def load_intraday_halts(self, path: Path) -> None:
        """Load dated market halts with retained exchange and membership sources."""
        self.intraday_halts = []
        self.periodic_auctions = []
        self.closing_auctions = {}
        if not path.exists():
            return
        import hashlib
        from kis_core.input_manifest import read_input_bytes
        content = read_input_bytes(path)
        data = yaml.safe_load(content)
        root = Path(__file__).resolve().parents[1]
        for name, digest in data["evidence_files"].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or hashlib.sha256(read_input_bytes(source)).hexdigest() != digest:
                raise ValueError("Intraday halt source identity mismatch")
        for row in data["halts"]:
            day = date.fromisoformat(row["date"])
            first, collection, last = (time.fromisoformat(row[k]) for k in ("halt_start", "order_collection_start", "matching_resumes"))
            symbols = row.get("symbols") or data["markets"][row["market"]]
            if (not first < collection < last or not symbols or not row.get("sources")
                    or any(not isinstance(s, str) or len(s) != 6 or not s.isdigit() for s in symbols)):
                raise ValueError("Invalid sourced intraday halt")
            self.intraday_halts.append((day, frozenset(symbols), first, last, tuple(row["sources"])))
        for row in data.get("periodic_auctions", []):
            first, last = (date.fromisoformat(row[k]) for k in ("start", "end"))
            symbols = row["symbols"]
            if (first > last or not symbols or row["interval_minutes"] != 30
                    or any(not isinstance(s, str) or len(s) != 6 or not s.isdigit() for s in symbols)
                    or not row.get("sources")):
                raise ValueError("Invalid sourced periodic auction")
            self.periodic_auctions.append((first, last, frozenset(symbols), tuple(row["sources"])))
        for row in data.get("closing_auctions", []):
            import json
            day, symbol = date.fromisoformat(row["date"]), row["symbol"]
            match = time.fromisoformat(row["matching_time"])
            closing = self.session_hours(day)[1]
            if (not closing < match < time(closing.hour, 40) or not row.get("sources")
                    or any(p not in data["evidence_files"] for p in row["sources"])
                    or (symbol, day) in self.closing_auctions):
                raise ValueError("Invalid observed closing auction")
            event = json.loads(read_input_bytes(root / row["sources"][0])).get("closing_event")
            if event != {key: row[key] for key in ("symbol", "date", "matching_time")}:
                raise ValueError("Closing auction source event mismatch")
            self.closing_auctions[(symbol, day)] = match
        self.session_schedule_hash = hashlib.sha256((self.session_schedule_hash + hashlib.sha256(content).hexdigest()).encode()).hexdigest()

    def closing_match_time(self, symbol: str | None, day: date):
        """Observed matching time; use only for execution, never early decisions."""
        return self.closing_auctions.get((str(symbol).zfill(6), day), self.session_hours(day)[1])

    def intraday_halts_for(self, symbol: str | None, day: date):
        return [(first, last, sources) for on, symbols, first, last, sources in self.intraday_halts
                if on == day and str(symbol).zfill(6) in symbols]

    def periodic_auction_sources(self, symbol: str | None, day: date):
        return tuple(source for first, last, symbols, sources in self.periodic_auctions
                     if first <= day <= last and str(symbol).zfill(6) in symbols for source in sources)

    def load_session_schedule(self, path: Path) -> None:
        if not path.exists():
            return
        import hashlib
        from kis_core.input_manifest import read_input_bytes
        content = read_input_bytes(path)
        data = yaml.safe_load(content)
        start, end = date.fromisoformat(data["coverage_start"]), date.fromisoformat(data["coverage_end"])
        if start > end or not data.get("sources", {}).get("regular"):
            raise ValueError("Invalid sourced session coverage")
        regular = data["regular"]
        self.regular_hours = (time.fromisoformat(regular["open"]), time.fromisoformat(regular["close"]))
        self.closing_auction_minutes = int(regular["closing_auction_minutes"])
        if self.regular_hours[0] >= self.regular_hours[1] or not 0 < self.closing_auction_minutes < 60:
            raise ValueError("Invalid regular session hours")
        self.special_sessions.clear()
        for raw, row in data.get("exceptions", {}).items():
            day = date.fromisoformat(str(raw))
            hours = (time.fromisoformat(row["open"]), time.fromisoformat(row["close"]))
            if not start <= day <= end or hours[0] >= hours[1] or row["source"] not in data["sources"]:
                raise ValueError("Invalid exceptional session evidence")
            self.special_sessions[day] = hours
        self.symbol_suspensions = []
        for row in data.get("symbol_suspensions", []):
            first, last = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
            symbols = row["symbols"]
            if (not start <= first <= last <= end or not symbols
                    or any(not isinstance(s, str) or len(s) != 6 or not s.isdigit() for s in symbols)
                    or row["source"] not in data["sources"] or not row.get("reason")):
                raise ValueError("Invalid sourced symbol suspension")
            self.symbol_suspensions.append((first, last, frozenset(symbols), row["source"]))
        self.session_coverage = (start, end)
        self.session_schedule_version = str(data["version"])
        self.session_schedule_hash = hashlib.sha256(content).hexdigest()
        self.session_hours_verified = True

    def has_session_coverage(self, day: date) -> bool:
        return bool(self.session_coverage and self.session_coverage[0] <= day <= self.session_coverage[1])

    def suspension_source(self, symbol: str | None, day: date) -> str | None:
        """A dated source, never absence of observations, establishes a suspension."""
        if symbol is None:
            return None
        for first, last, symbols, source in getattr(self, "symbol_suspensions", ()):
            if str(symbol).zfill(6) in symbols and first <= day <= last:
                return source
        return None

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
            from kis_core.input_manifest import read_input_bytes
            data = yaml.safe_load(read_input_bytes(yaml_path))
            if data:
                for year, dates in data.items():
                    if isinstance(dates, list):
                        self.covered_years.add(int(year))
                        for date_str in dates:
                            try:
                                holidays.add(date.fromisoformat(date_str))
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Invalid date in holidays: {date_str}: {e}")
            logger.info(f"Loaded {len(holidays)} KRX holidays from {yaml_path}")
        except Exception as e:
            logger.error(f"Error loading holiday file: {e}")

        return holidays

    def has_coverage(self, day: date) -> bool:
        return day.year in self.covered_years

    def session_hours(self, day: date) -> tuple[time, time]:
        # Holiday coverage does not establish exceptional opening/closing hours.
        return self.special_sessions.get(day, getattr(self, "regular_hours", (time(9), time(15, 30))))

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
    from kis_core.input_manifest import active_manifest
    generation = active_manifest()
    if generation is not None:
        if not hasattr(generation, "calendar"):
            generation.calendar = KRXTradingCalendar()
        return generation.calendar
    return KRXTradingCalendar.get_instance()
