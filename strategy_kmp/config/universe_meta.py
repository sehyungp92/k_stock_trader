"""Universe metadata for KMP strategy.

Provides ticker→sector mapping since KIS API does not reliably
provide sector taxonomy. Load from config premarket.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TickerMeta:
    """Metadata for a ticker in the universe."""

    ticker: str
    sector: str
    theme: str = ""


def load_universe_meta(cfg: dict) -> Dict[str, TickerMeta]:
    """Load ticker→sector mapping from config.

    Args:
        cfg: Configuration dict with 'sector_map' key.

    Returns:
        Dict mapping ticker to TickerMeta.
    """
    sector_map = cfg.get("sector_map", {})
    return {t: TickerMeta(ticker=t, sector=s) for t, s in sector_map.items()}
