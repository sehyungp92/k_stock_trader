"""Filter decision logger — standalone filter evaluation events."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .event_metadata import create_event_metadata
from .lineage import LineageContext

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
    event_metadata: dict[str, Any] | None = None
    schema_version: str = "filter_decision_v1"
    strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    decision_id: str = ""
    event_ref: str = ""
    trace_id: str = ""
    filter_group: str = "entry"
    threshold_operator: str = ">="
    distance_to_threshold: Optional[float] = None
    input_refs: list[str] | None = None
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
            key = f"{self.strategy_id or self.strategy_type}:{self.pair}:{self.bar_id or self.timestamp}:{self.signal_name}:{self.filter_name}"
            raw = f"{self.bot_id}|{self.timestamp}|filter_decision|{key}"
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

    def __init__(self, data_dir: str | Path, bot_id: str, *, lineage: LineageContext | Mapping[str, Any] | None = None) -> None:
        self._data_dir = Path(data_dir) / "filter_decisions"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id
        self._lineage = lineage

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
        decision_id: str = "",
        event_ref: str = "",
        filter_group: str = "entry",
        threshold_operator: str = ">=",
        input_refs: Optional[list[str]] = None,
        lineage: LineageContext | Mapping[str, Any] | None = None,
    ) -> FilterDecisionEvent:
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        strategy_id = str(strategy_type or "").upper().strip()
        metadata = create_event_metadata(
            bot_id=self._bot_id,
            event_type="filter_decision",
            payload_key=f"{strategy_id}:{pair}:{bar_id or ts_str}:{signal_name}:{filter_name}",
            exchange_timestamp=ts,
            data_source_id="runtime_session",
            bar_id=bar_id,
            schema_version="filter_decision_v1",
            lineage=lineage or self._lineage,
            strategy_id=strategy_id,
            family_id="krx_equity" if strategy_id in {"KALCB", "OLR"} else "",
            portfolio_id="olr_kalcb" if strategy_id in {"KALCB", "OLR"} else "",
            decision_id=decision_id,
            trace_id=event_ref,
            scope="strategy",
        ).to_dict()
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
            strategy_id=strategy_id,
            bar_id=bar_id,
            event_metadata=metadata,
            event_id=metadata["event_id"],
            family_id=metadata.get("family_id") or "",
            portfolio_id=metadata.get("portfolio_id") or "",
            decision_id=decision_id,
            event_ref=event_ref,
            trace_id=event_ref,
            filter_group=filter_group,
            threshold_operator=threshold_operator,
            distance_to_threshold=None if threshold == 0.0 else actual_value - threshold,
            input_refs=input_refs or [],
            **_lineage_fields(metadata),
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._data_dir / f"filter_decisions_{today}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write filter decision: %s", e)

        return event


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
