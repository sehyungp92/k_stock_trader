"""Indicator snapshot logger — captures indicator state at signal evaluation."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .event_metadata import create_event_metadata
from .lineage import LineageContext

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
    strategy_type: str
    event_id: str = ""
    bar_id: Optional[str] = None
    event_metadata: dict[str, Any] | None = None
    schema_version: str = "indicator_snapshot_v1"
    strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    decision_id: str = ""
    event_ref: str = ""
    trace_id: str = ""
    context: dict = field(default_factory=dict)  # strategy-specific extra context
    account_alias: str = ""
    strategy_version: str = ""
    config_version: str = ""
    portfolio_config_version: str = ""
    risk_config_version: str = ""
    allocation_version: str = ""
    strategy_registry_version: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    code_sha: str = ""

    def __post_init__(self):
        if not self.event_id:
            key = f"{self.strategy_id or self.strategy_type}:{self.pair}:{self.bar_id or self.timestamp}:{self.signal_name}"
            raw = f"{self.bot_id}|{self.timestamp}|indicator_snapshot|{key}"
            self.event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


class IndicatorLogger:
    """Writes indicator snapshots to daily JSONL files."""

    def __init__(self, data_dir: str | Path, bot_id: str, *, lineage: LineageContext | Mapping[str, Any] | None = None) -> None:
        self._data_dir = Path(data_dir) / "indicators"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id
        self._lineage = lineage

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
        decision_id: str = "",
        event_ref: str = "",
        lineage: LineageContext | Mapping[str, Any] | None = None,
    ) -> IndicatorSnapshot:
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        strategy_id = str(strategy_type or "").upper().strip()
        metadata = create_event_metadata(
            bot_id=self._bot_id,
            event_type="indicator_snapshot",
            payload_key=f"{strategy_id}:{pair}:{bar_id or ts_str}:{signal_name}",
            exchange_timestamp=ts,
            data_source_id="runtime_session",
            bar_id=bar_id,
            schema_version="indicator_snapshot_v1",
            lineage=lineage or self._lineage,
            strategy_id=strategy_id,
            family_id="krx_equity" if strategy_id in {"KALCB", "OLR"} else "",
            portfolio_id="olr_kalcb" if strategy_id in {"KALCB", "OLR"} else "",
            decision_id=decision_id,
            trace_id=event_ref,
            scope="strategy",
        ).to_dict()

        snapshot = IndicatorSnapshot(
            bot_id=self._bot_id,
            pair=pair,
            timestamp=ts_str,
            indicators=indicators,
            signal_name=signal_name,
            signal_strength=signal_strength,
            decision=decision,
            strategy_type=strategy_type,
            strategy_id=strategy_id,
            bar_id=bar_id,
            event_id=metadata["event_id"],
            event_metadata=metadata,
            family_id=metadata.get("family_id") or "",
            portfolio_id=metadata.get("portfolio_id") or "",
            decision_id=decision_id,
            event_ref=event_ref,
            trace_id=event_ref,
            context=context or {},
            **_lineage_fields(metadata),
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._data_dir / f"indicators_{today}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write indicator snapshot: %s", e)

        return snapshot


def _lineage_fields(metadata: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "account_alias",
        "strategy_version",
        "config_version",
        "portfolio_config_version",
        "risk_config_version",
        "allocation_version",
        "strategy_registry_version",
        "deployment_id",
        "parameter_set_id",
        "code_sha",
    )
    return {key: str(metadata.get(key) or "") for key in keys}
