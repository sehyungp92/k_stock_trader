"""
KPR Strategy Switches: Configurable parameters for tuning trade frequency.

Defaults are set to MAXIMIZE trade frequency (permissive).
Use conservative.yaml to restore strict settings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger


@dataclass
class KPRSwitches:
    """
    KPR strategy configuration switches.

    Defaults maximize trade frequency. Conservative values in comments.
    """

    # HIGH PRIORITY: Lunch block
    # False = trade through lunch (more trades)
    # True = block entries during 11:20-13:10 (conservative)
    enable_lunch_block: bool = False  # Conservative: True

    # HIGH PRIORITY: CONFLICT signal handling
    # False = CONFLICT -> YELLOW (allow trade with reduced size)
    # True = CONFLICT -> RED (block trade, conservative)
    conflict_is_red: bool = False  # Conservative: True

    # MEDIUM PRIORITY: VWAP depth range
    # 0.015 = allow shallower setups (more trades)
    # 0.02 = stricter depth requirement (conservative)
    vwap_depth_min: float = 0.015  # Conservative: 0.02

    # MEDIUM PRIORITY: VWAP depth max
    # 0.06 = allow deeper setups (more trades)
    # 0.05 = stricter max depth (conservative)
    vwap_depth_max: float = 0.06  # Conservative: 0.05

    # MEDIUM PRIORITY: Late session size multiplier
    # 0.65 = higher size late in day (more trades)
    # 0.5 = reduced late session size (conservative)
    tod_late_mult: float = 0.65  # Conservative: 0.5

    # CRITICAL: Stale flow acceptance adder (redundant with size penalty)
    # False = no acceptance adder for stale flow (keep 0.85x size penalty only)
    # True = add +1 acceptance close for stale flow (double-penalty, conservative)
    enable_stale_flow_acceptance_adder: bool = False  # Conservative: True

    # CRITICAL: Late session acceptance adder (redundant with TOD sizing)
    # False = no acceptance adder for late session (keep TOD sizing only)
    # True = add +1 acceptance close for late (double-penalty, conservative)
    enable_late_acceptance_adder: bool = False  # Conservative: True

    # Tracking fields (not user-configurable)
    would_block_count: int = field(default=0, init=False, repr=False)
    would_block_log: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def log_would_block(
        self,
        symbol: str,
        reason: str,
        actual: Any,
        strict_threshold: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a case where permissive settings allowed what strict would block.

        Args:
            symbol: Stock code
            reason: Reason code (e.g., "LUNCH_BLOCK", "CONFLICT_SIGNAL")
            actual: Actual value that passed
            strict_threshold: The strict threshold that would have blocked
            extra: Additional context
        """
        self.would_block_count += 1
        entry = {
            "symbol": symbol,
            "reason": reason,
            "actual": actual,
            "strict_threshold": strict_threshold,
            "timestamp": datetime.now().isoformat(),
            "extra": extra or {},
        }
        self.would_block_log.append(entry)
        logger.info(
            f"{symbol}: WOULD_BLOCK_{reason} "
            f"(actual={actual}, strict={strict_threshold})"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of would-block events.

        Returns:
            Dict with total count and breakdown by reason.
        """
        by_reason: Dict[str, int] = {}
        for entry in self.would_block_log:
            reason = entry["reason"]
            by_reason[reason] = by_reason.get(reason, 0) + 1

        return {
            "total": self.would_block_count,
            "by_reason": by_reason,
            "log": self.would_block_log,
        }

    def reset_stats(self) -> None:
        """Reset would-block tracking for new session."""
        self.would_block_count = 0
        self.would_block_log = []

    def log_session_summary(self) -> None:
        """Log end-of-session summary."""
        stats = self.get_stats()
        if stats["total"] > 0:
            logger.info(
                f"KPR session would-block stats: "
                f"total={stats['total']}, by_reason={stats['by_reason']}"
            )

    def update_from_yaml(self, path: str) -> None:
        """Load switches from YAML and update this instance in-place."""
        import yaml
        from dataclasses import fields as dc_fields
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        section = data.get("kpr", {})
        configurable = {
            f.name for f in dc_fields(self)
            if f.name not in ("would_block_count", "would_block_log")
        }
        for key, value in section.items():
            if key in configurable:
                if isinstance(getattr(self, key, None), tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(self, key, value)
        logger.info(f"Switches updated from {path}")

    def log_active_config(self) -> None:
        """Log all active switch values at startup."""
        from dataclasses import fields as dc_fields
        active = {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name not in ("would_block_count", "would_block_log")
        }
        logger.info(f"Active switches: {active}")

    @classmethod
    def load_from_yaml(cls, path: str) -> "KPRSwitches":
        """
        Load switches from YAML config file.

        Args:
            path: Path to YAML file

        Returns:
            KPRSwitches instance with loaded values
        """
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        kpr_data = data.get("kpr", {})
        return cls(
            enable_lunch_block=kpr_data.get("enable_lunch_block", False),
            conflict_is_red=kpr_data.get("conflict_is_red", False),
            vwap_depth_min=kpr_data.get("vwap_depth_min", 0.015),
            vwap_depth_max=kpr_data.get("vwap_depth_max", 0.06),
            tod_late_mult=kpr_data.get("tod_late_mult", 0.65),
            enable_stale_flow_acceptance_adder=kpr_data.get("enable_stale_flow_acceptance_adder", False),
            enable_late_acceptance_adder=kpr_data.get("enable_late_acceptance_adder", False),
        )

    @classmethod
    def conservative(cls) -> "KPRSwitches":
        """Create switches with conservative (strict) settings."""
        return cls(
            enable_lunch_block=True,
            conflict_is_red=True,
            vwap_depth_min=0.02,
            vwap_depth_max=0.05,
            tod_late_mult=0.5,
            enable_stale_flow_acceptance_adder=True,
            enable_late_acceptance_adder=True,
        )


# Global instance with max-frequency defaults
kpr_switches = KPRSwitches()
