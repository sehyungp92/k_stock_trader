"""Indicator snapshot logger — captures indicator state at signal evaluation."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.indicator_logger")


@dataclass
class IndicatorSnapshot:
    """Snapshot of all indicator values at a signal evaluation point."""
    bot_id: str
    pair: str
    timestamp: str                    # ISO 8601
    indicators: dict[str, float]      # {"sma_20": 45000.0, "atr_14": 1200.0, ...}
    signal_name: str                  # e.g. "kmp_value_surge", "kpr_vwap_pullback"
    signal_strength: float            # 0.0-1.0
    decision: str                     # "enter", "skip", or "exit"
    strategy_type: str                # "kmp", "kpr", "pcim", "nulrimok"
    event_id: str = ""
    bar_id: Optional[str] = None
    context: dict = field(default_factory=dict)  # strategy-specific extra context

    def __post_init__(self):
        if not self.event_id:
            raw = f"{self.bot_id}|{self.timestamp}|indicator_snapshot|{self.pair}:{self.signal_name}"
            self.event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


class IndicatorLogger:
    """Writes indicator snapshots to daily JSONL files."""

    def __init__(self, data_dir: str | Path, bot_id: str) -> None:
        self._data_dir = Path(data_dir) / "indicators"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id

    def log_snapshot(
        self,
        pair: str,
        indicators: dict[str, float],
        signal_name: str,
        signal_strength: float,
        decision: str,
        strategy_type: str,
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> IndicatorSnapshot:
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        snapshot = IndicatorSnapshot(
            bot_id=self._bot_id,
            pair=pair,
            timestamp=ts_str,
            indicators=indicators,
            signal_name=signal_name,
            signal_strength=signal_strength,
            decision=decision,
            strategy_type=strategy_type,
            bar_id=bar_id,
            context=context or {},
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._data_dir / f"indicators_{today}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write indicator snapshot: %s", e)

        return snapshot
