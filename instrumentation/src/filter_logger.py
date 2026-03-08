"""Filter decision logger — standalone filter evaluation events."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.filter_logger")


@dataclass
class FilterDecisionEvent:
    """One filter gate evaluation result, emitted independently of TradeEvent."""
    bot_id: str
    pair: str
    timestamp: str                   # ISO 8601
    filter_name: str                 # e.g. "volume_min", "regime_gate", "spread_max"
    passed: bool
    threshold: float
    actual_value: float
    signal_name: str = ""            # signal being evaluated when filter ran
    signal_strength: float = 0.0
    strategy_type: str = ""
    event_id: str = ""
    bar_id: Optional[str] = None

    def __post_init__(self):
        if not self.event_id:
            raw = f"{self.bot_id}|{self.timestamp}|filter_decision|{self.pair}:{self.filter_name}"
            self.event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def margin_pct(self) -> float | None:
        """How far inside/outside the threshold, as percentage.
        Positive = passed with margin, negative = blocked below threshold.
        Returns None for boolean filters (threshold == 0).
        """
        if self.threshold == 0.0:
            return None
        return round((self.actual_value - self.threshold) / abs(self.threshold) * 100, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["margin_pct"] = self.margin_pct
        return d


class FilterLogger:
    """Writes filter decision events to daily JSONL files."""

    def __init__(self, data_dir: str | Path, bot_id: str) -> None:
        self._data_dir = Path(data_dir) / "filter_decisions"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id

    def log_decision(
        self,
        pair: str,
        filter_name: str,
        passed: bool,
        threshold: float,
        actual_value: float,
        signal_name: str = "",
        signal_strength: float = 0.0,
        strategy_type: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> FilterDecisionEvent:
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        event = FilterDecisionEvent(
            bot_id=self._bot_id,
            pair=pair,
            timestamp=ts_str,
            filter_name=filter_name,
            passed=passed,
            threshold=threshold,
            actual_value=actual_value,
            signal_name=signal_name,
            signal_strength=signal_strength,
            strategy_type=strategy_type,
            bar_id=bar_id,
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._data_dir / f"filter_decisions_{today}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write filter decision: %s", e)

        return event
