"""
NULRIMOK Strategy Switches: Configurable parameters for tuning trade frequency.

Defaults are set to MAXIMIZE trade frequency (permissive).
Use conservative.yaml to restore strict settings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger


@dataclass
class NulrimokSwitches:
    """
    NULRIMOK strategy configuration switches.

    Defaults maximize trade frequency. Conservative values in comments.
    """

    # HIGH PRIORITY: Tier C reduced sizing
    # True = allow Tier C with 0.25x sizing (more trades)
    # False = block Tier C completely (conservative, 0.0x)
    allow_tier_c_reduced: bool = True  # Conservative: False

    # HIGH PRIORITY: RS percentile thresholds for leader filter
    # 50/60 = lower thresholds (more trades)
    # 60/70 = stricter thresholds (conservative)
    leader_tier_a_pct: int = 50  # Conservative: 60
    leader_tier_b_pct: int = 60  # Conservative: 70

    # MEDIUM PRIORITY: Flow persistence minimum
    # 0.55 = looser flow requirement (more trades)
    # 0.60 = stricter flow persistence (conservative)
    flow_persistence_min: float = 0.55  # Conservative: 0.60

    # MEDIUM PRIORITY: Confirmation bars for entry
    # 3 = more confirmation required (allows more time for setup)
    # 2 = faster confirmation (conservative, fewer false signals)
    confirm_bars: int = 3  # Conservative: 2

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
            reason: Reason code (e.g., "TIER_C_REDUCED", "RS_PERCENTILE")
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
                f"NULRIMOK session would-block stats: "
                f"total={stats['total']}, by_reason={stats['by_reason']}"
            )

    @classmethod
    def load_from_yaml(cls, path: str) -> "NulrimokSwitches":
        """
        Load switches from YAML config file.

        Args:
            path: Path to YAML file

        Returns:
            NulrimokSwitches instance with loaded values
        """
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        nulrimok_data = data.get("nulrimok", {})
        return cls(
            allow_tier_c_reduced=nulrimok_data.get("allow_tier_c_reduced", True),
            leader_tier_a_pct=nulrimok_data.get("leader_tier_a_pct", 50),
            leader_tier_b_pct=nulrimok_data.get("leader_tier_b_pct", 60),
            flow_persistence_min=nulrimok_data.get("flow_persistence_min", 0.55),
            confirm_bars=nulrimok_data.get("confirm_bars", 3),
        )

    @classmethod
    def conservative(cls) -> "NulrimokSwitches":
        """Create switches with conservative (strict) settings."""
        return cls(
            allow_tier_c_reduced=False,
            leader_tier_a_pct=60,
            leader_tier_b_pct=70,
            flow_persistence_min=0.60,
            confirm_bars=2,
        )


# Global instance with max-frequency defaults
nulrimok_switches = NulrimokSwitches()
