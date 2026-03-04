"""Heartbeat emitter — periodic liveness signal for orchestrator."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class HeartbeatEmitter:
    """Writes periodic heartbeat events to JSONL for sidecar pickup."""

    def __init__(self, bot_id: str, strategy_type: str,
                 data_dir: str = "instrumentation/data"):
        self._bot_id = bot_id
        self._strategy_type = strategy_type
        self._data_dir = Path(data_dir)
        (self._data_dir / "heartbeats").mkdir(parents=True, exist_ok=True)

    def emit(self, active_positions: int = 0, open_orders: int = 0,
             uptime_s: float = 0, error_count_1h: int = 0,
             extra: Optional[dict] = None) -> None:
        """Write a single heartbeat record."""
        now = datetime.now(timezone.utc)
        record = {
            "bot_id": self._bot_id,
            "strategy_type": self._strategy_type,
            "timestamp": now.isoformat(),
            "status": "alive",
            "active_positions": active_positions,
            "open_orders": open_orders,
            "uptime_s": round(uptime_s, 1),
            "error_count_1h": error_count_1h,
        }
        if extra:
            record["extra"] = extra

        today = now.strftime("%Y-%m-%d")
        filepath = self._data_dir / "heartbeats" / f"heartbeats_{today}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
