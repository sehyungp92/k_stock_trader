"""
PCIM Strategy Switches: Configurable parameters for tuning trade frequency.

Defaults are set to MAXIMIZE trade frequency (permissive).
Use conservative.yaml to restore strict settings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger


@dataclass
class PCIMSwitches:
    """
    PCIM strategy configuration switches.

    Defaults maximize trade frequency. Conservative values in comments.
    """

    # HIGH PRIORITY: T3 Bucket A trading
    # True = allow T3 tier Bucket A trades (more trades)
    # False = block T3 tier from Bucket A (conservative)
    t3_bucket_a_allowed: bool = False  # Conservative: False

    # HIGH PRIORITY: Entry cutoff time (hour, minute)
    # (10, 30) = later cutoff (more trades)
    # (10, 0) = earlier cutoff (conservative)
    entry_cutoff: Tuple[int, int] = (10, 30)  # Conservative: (10, 0)

    # MEDIUM PRIORITY: Gap reversal threshold
    # 0.65 = allow stocks with lower gap fill rate (more trades)
    # 0.60 = stricter gap reversal requirement (conservative)
    gap_reversal_threshold: float = 0.65  # Conservative: 0.60

    # MEDIUM PRIORITY: Spread veto percentage
    # 0.007 = allow wider spreads (more trades)
    # 0.006 = stricter spread requirement (conservative)
    spread_veto_pct: float = 0.007  # Conservative: 0.006

    # CRITICAL: ADTV soft penalty (redundant with tier sizing)
    # False = no soft ADTV penalty (rely on tier sizing only)
    # True = apply 0.5x for ADTV 10-15B (double-penalty with T3 tier, conservative)
    enable_adtv_soft_penalty: bool = False  # Conservative: True

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
            reason: Reason code (e.g., "T3_BUCKET_A", "SPREAD_VETO")
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
                f"PCIM session would-block stats: "
                f"total={stats['total']}, by_reason={stats['by_reason']}"
            )

    @classmethod
    def load_from_yaml(cls, path: str) -> "PCIMSwitches":
        """
        Load switches from YAML config file.

        Args:
            path: Path to YAML file

        Returns:
            PCIMSwitches instance with loaded values
        """
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        pcim_data = data.get("pcim", {})
        entry_cutoff = pcim_data.get("entry_cutoff", [10, 30])
        if isinstance(entry_cutoff, list):
            entry_cutoff = tuple(entry_cutoff)

        return cls(
            t3_bucket_a_allowed=pcim_data.get("t3_bucket_a_allowed", True),
            entry_cutoff=entry_cutoff,
            gap_reversal_threshold=pcim_data.get("gap_reversal_threshold", 0.65),
            spread_veto_pct=pcim_data.get("spread_veto_pct", 0.007),
            enable_adtv_soft_penalty=pcim_data.get("enable_adtv_soft_penalty", False),
        )

    @classmethod
    def conservative(cls) -> "PCIMSwitches":
        """Create switches with conservative (strict) settings."""
        return cls(
            t3_bucket_a_allowed=False,
            entry_cutoff=(10, 0),
            gap_reversal_threshold=0.60,
            spread_veto_pct=0.006,
            enable_adtv_soft_penalty=True,
        )


# Global instance with max-frequency defaults
pcim_switches = PCIMSwitches()
