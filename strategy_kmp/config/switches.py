"""
KMP Strategy Switches: Configurable parameters for tuning trade frequency.

Defaults are set to MAXIMIZE trade frequency (permissive).
Use conservative.yaml to restore strict settings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger


@dataclass
class KMPSwitches:
    """
    KMP strategy configuration switches.

    Defaults maximize trade frequency. Conservative values in comments.
    """

    # HIGH PRIORITY: Acceptance pattern
    # False = accept without held support (more trades)
    # True = require price to hold support on retest (conservative)
    require_held_support: bool = False  # Conservative: True

    # HIGH PRIORITY: Quality threshold for size multiplier
    # 30 = allow lower quality setups (more trades)
    # 40 = stricter quality filtering (conservative)
    quality_min_threshold: int = 30  # Conservative: 40

    # MEDIUM PRIORITY: OR range maximum
    # 0.07 = allow 7% ranges (more trades)
    # 0.055 = stricter 5.5% max (conservative)
    or_range_max: float = 0.07  # Conservative: 0.055

    # MEDIUM PRIORITY: Surge decay slope
    # 0.02 = slowest decay, easiest to qualify later (more trades)
    # 0.04 = faster decay, harder to qualify (conservative)
    min_surge_slope: float = 0.02  # Conservative: 0.04

    # CRITICAL: RVOL hard gate (redundant with quality score)
    # False = no hard gate, let quality score weight RVOL (more trades)
    # True = hard gate blocks if RVOL < 2.0 (conservative, double-filters)
    enable_rvol_hard_gate: bool = False  # Conservative: True

    # REGIME GATE: Minimum leader breadth count for regime_ok
    # 1 = single leader sufficient (more trades, paper trading)
    # 2 = require 2+ leaders (conservative)
    regime_breadth_min: int = 1  # Conservative: 2

    # ENTRY CUTOFF: (hour, minute) after which new entries are blocked
    # (10, 30) = extended window until 10:30 (more trades)
    # (10, 0)  = strict 10:00 cutoff (conservative)
    entry_cutoff: tuple = (10, 30)  # Conservative: (10, 0)

    # REGIME GATE SCOPE: Whether regime gate blocks FSM progression
    # False = only block at intent submission, allow break detection (more trades)
    # True  = block all FSM states including WATCH_BREAK (conservative)
    regime_blocks_progression: bool = False  # Conservative: True

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
            reason: Reason code (e.g., "HELD_SUPPORT", "QUALITY_SCORE")
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
                f"KMP session would-block stats: "
                f"total={stats['total']}, by_reason={stats['by_reason']}"
            )

    def update_from_yaml(self, path: str) -> None:
        """Load switches from YAML and update this instance in-place."""
        import yaml
        from dataclasses import fields as dc_fields
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        section = data.get("kmp", {})
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

    def to_params_dict(self) -> Dict[str, Any]:
        from dataclasses import fields as dc_fields
        return {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name not in ("would_block_count", "would_block_log")
        }

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
    def load_from_yaml(cls, path: str) -> "KMPSwitches":
        """
        Load switches from YAML config file.

        Args:
            path: Path to YAML file

        Returns:
            KMPSwitches instance with loaded values
        """
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        kmp_data = data.get("kmp", {})
        entry_cutoff = kmp_data.get("entry_cutoff", [10, 30])
        if isinstance(entry_cutoff, list):
            entry_cutoff = tuple(entry_cutoff)
        return cls(
            require_held_support=kmp_data.get("require_held_support", False),
            quality_min_threshold=kmp_data.get("quality_min_threshold", 30),
            or_range_max=kmp_data.get("or_range_max", 0.07),
            min_surge_slope=kmp_data.get("min_surge_slope", 0.02),
            enable_rvol_hard_gate=kmp_data.get("enable_rvol_hard_gate", False),
            regime_breadth_min=kmp_data.get("regime_breadth_min", 1),
            entry_cutoff=entry_cutoff,
            regime_blocks_progression=kmp_data.get("regime_blocks_progression", False),
        )

    @classmethod
    def conservative(cls) -> "KMPSwitches":
        """Create switches with conservative (strict) settings."""
        return cls(
            require_held_support=True,
            quality_min_threshold=40,
            or_range_max=0.055,
            min_surge_slope=0.04,
            enable_rvol_hard_gate=True,
            regime_breadth_min=2,
            entry_cutoff=(10, 0),
            regime_blocks_progression=True,
        )


# Global instance with max-frequency defaults
kmp_switches = KMPSwitches()
