"""Tests for SectorExposure module."""

import pytest
from kis_core.sector_exposure import SectorExposure, SectorExposureConfig


@pytest.fixture
def sector_map():
    """Sample symbol-to-sector mapping."""
    return {
        "005930": "IT",       # Samsung Electronics
        "000660": "IT",       # SK Hynix
        "035420": "IT",       # NAVER
        "051910": "Chemicals", # LG Chem
        "006400": "Chemicals", # Samsung SDI
        "035720": "Internet",  # Kakao
        "105560": "Auto",      # KB Financial
    }


class TestSectorExposureConfig:
    """Tests for SectorExposureConfig dataclass."""

    def test_default_config(self):
        cfg = SectorExposureConfig()
        assert cfg.mode == "both"
        assert cfg.max_positions_per_sector == 2
        assert cfg.max_sector_pct == 0.30
        assert cfg.unknown_sector_policy == "allow"

    def test_custom_config(self):
        cfg = SectorExposureConfig(
            mode="count",
            max_positions_per_sector=1,
            unknown_sector_policy="block",
        )
        assert cfg.mode == "count"
        assert cfg.max_positions_per_sector == 1
        assert cfg.unknown_sector_policy == "block"


class TestCountMode:
    """Tests for count-based sector limits."""

    def test_can_enter_empty(self, sector_map):
        """Can enter when sector has no positions."""
        cfg = SectorExposureConfig(mode="count", max_positions_per_sector=2)
        exp = SectorExposure(sector_map, cfg)

        assert exp.can_enter("005930", 100, 60000, 100_000_000)
        assert exp.can_enter("000660", 100, 80000, 100_000_000)

    def test_can_enter_at_limit(self, sector_map):
        """Cannot enter when sector is at max positions."""
        cfg = SectorExposureConfig(mode="count", max_positions_per_sector=1)
        exp = SectorExposure(sector_map, cfg)

        # First IT position OK
        assert exp.can_enter("005930", 100, 60000, 100_000_000)
        exp.reserve("005930", 100, 60000)
        exp.on_fill("005930", 100, 60000)

        # Second IT position blocked
        assert not exp.can_enter("000660", 100, 80000, 100_000_000)

        # Different sector OK
        assert exp.can_enter("051910", 100, 500000, 100_000_000)

    def test_reserve_prevents_race(self, sector_map):
        """Reserve blocks concurrent entries."""
        cfg = SectorExposureConfig(mode="count", max_positions_per_sector=1)
        exp = SectorExposure(sector_map, cfg)

        # Reserve slot
        exp.reserve("005930", 100, 60000)

        # Second entry blocked even before fill
        assert not exp.can_enter("000660", 100, 80000, 100_000_000)

    def test_unreserve_releases_slot(self, sector_map):
        """Unreserve allows new entry after order failure."""
        cfg = SectorExposureConfig(mode="count", max_positions_per_sector=1)
        exp = SectorExposure(sector_map, cfg)

        # Reserve then unreserve (simulating order rejection)
        exp.reserve("005930", 100, 60000)
        exp.unreserve("005930", 100, 60000)

        # Now entry allowed
        assert exp.can_enter("000660", 100, 80000, 100_000_000)

    def test_on_close_frees_slot(self, sector_map):
        """Closing position allows new entry."""
        cfg = SectorExposureConfig(mode="count", max_positions_per_sector=1)
        exp = SectorExposure(sector_map, cfg)

        # Fill position
        exp.reserve("005930", 100, 60000)
        exp.on_fill("005930", 100, 60000)

        # Cannot add second IT
        assert not exp.can_enter("000660", 100, 80000, 100_000_000)

        # Close position
        exp.on_close("005930", 100, 60000)

        # Now can enter IT
        assert exp.can_enter("000660", 100, 80000, 100_000_000)


class TestPctMode:
    """Tests for percentage-based sector limits."""

    def test_can_enter_below_cap(self, sector_map):
        """Can enter when sector exposure is below cap."""
        cfg = SectorExposureConfig(mode="pct", max_sector_pct=0.30)
        exp = SectorExposure(sector_map, cfg)
        equity = 100_000_000

        # 20% position in IT OK
        assert exp.can_enter("005930", 333, 60000, equity)  # ~20M / 100M = 20%

    def test_cannot_enter_above_cap(self, sector_map):
        """Cannot enter when sector would exceed cap."""
        cfg = SectorExposureConfig(mode="pct", max_sector_pct=0.30)
        exp = SectorExposure(sector_map, cfg)
        equity = 100_000_000

        # Add 25% IT position
        exp.reserve("005930", 416, 60000)  # ~25M
        exp.on_fill("005930", 416, 60000)

        # Adding another 10% would exceed 30%
        assert not exp.can_enter("000660", 125, 80000, equity)  # ~10M

    def test_can_enter_small_after_large(self, sector_map):
        """Can add small position if still under cap."""
        cfg = SectorExposureConfig(mode="pct", max_sector_pct=0.30)
        exp = SectorExposure(sector_map, cfg)
        equity = 100_000_000

        # Add 20% IT position
        exp.reserve("005930", 333, 60000)  # ~20M
        exp.on_fill("005930", 333, 60000)

        # Adding 5% still under 30%
        assert exp.can_enter("000660", 62, 80000, equity)  # ~5M


class TestBothMode:
    """Tests for dual mode (count + pct)."""

    def test_blocked_by_count(self, sector_map):
        """Entry blocked by count even if under pct cap."""
        cfg = SectorExposureConfig(
            mode="both",
            max_positions_per_sector=1,
            max_sector_pct=0.50,
        )
        exp = SectorExposure(sector_map, cfg)
        equity = 100_000_000

        # Add small IT position (only 5%)
        exp.reserve("005930", 83, 60000)
        exp.on_fill("005930", 83, 60000)

        # Blocked by count=1, even though pct is only 5%
        assert not exp.can_enter("000660", 62, 80000, equity)

    def test_blocked_by_pct(self, sector_map):
        """Entry blocked by pct even if under count limit."""
        cfg = SectorExposureConfig(
            mode="both",
            max_positions_per_sector=5,
            max_sector_pct=0.20,
        )
        exp = SectorExposure(sector_map, cfg)
        equity = 100_000_000

        # Add 15% IT position
        exp.reserve("005930", 250, 60000)
        exp.on_fill("005930", 250, 60000)

        # Blocked by pct, even though count=1 < 5
        assert not exp.can_enter("000660", 125, 80000, equity)


class TestUnknownSector:
    """Tests for unknown sector handling."""

    def test_allow_unknown(self, sector_map):
        """Unknown sector allowed when policy is 'allow'."""
        cfg = SectorExposureConfig(unknown_sector_policy="allow")
        exp = SectorExposure(sector_map, cfg)

        # Symbol not in sector map
        assert exp.can_enter("UNKNOWN_SYMBOL", 100, 50000, 100_000_000)
        assert exp.get_sector("UNKNOWN_SYMBOL") == "UNKNOWN"

    def test_block_unknown(self, sector_map):
        """Unknown sector blocked when policy is 'block'."""
        cfg = SectorExposureConfig(unknown_sector_policy="block")
        exp = SectorExposure(sector_map, cfg)

        # Symbol not in sector map
        assert not exp.can_enter("UNKNOWN_SYMBOL", 100, 50000, 100_000_000)


class TestReconciliation:
    """Tests for reconciliation from OMS truth."""

    def test_reconcile_positions(self, sector_map):
        """Reconcile rebuilds state from positions."""
        cfg = SectorExposureConfig(mode="count", max_positions_per_sector=2)
        exp = SectorExposure(sector_map, cfg)

        # Simulated dirty state
        exp.sector_open_count["IT"] = 99
        exp.sector_working_count["IT"] = 99

        # Reconcile from truth
        positions = {
            "005930": (100, 60000),  # Samsung
            "051910": (50, 500000),   # LG Chem
        }
        exp.reconcile(positions)

        assert exp.sector_open_count.get("IT", 0) == 1
        assert exp.sector_open_count.get("Chemicals", 0) == 1
        assert exp.sector_working_count.get("IT", 0) == 0

    def test_reconcile_with_working_orders(self, sector_map):
        """Reconcile includes working orders."""
        cfg = SectorExposureConfig(mode="count", max_positions_per_sector=2)
        exp = SectorExposure(sector_map, cfg)

        positions = {"005930": (100, 60000)}
        working_orders = {"000660"}  # SK Hynix pending

        exp.reconcile(positions, working_orders)

        # 1 open + 1 working = 2 in IT sector
        assert exp.count_in_sector("IT") == 2
        assert not exp.can_enter("035420", 100, 200000, 100_000_000)

    def test_reset_clears_all(self, sector_map):
        """Reset clears all tracking state."""
        cfg = SectorExposureConfig(mode="both")
        exp = SectorExposure(sector_map, cfg)

        exp.reserve("005930", 100, 60000)
        exp.on_fill("005930", 100, 60000)

        exp.reset()

        assert exp.sector_open_count == {}
        assert exp.sector_working_count == {}
        assert exp.sector_open_notional == {}
        assert exp.sector_working_notional == {}


class TestHelperMethods:
    """Tests for helper/query methods."""

    def test_get_sector(self, sector_map):
        """get_sector returns sector or UNKNOWN."""
        exp = SectorExposure(sector_map)

        assert exp.get_sector("005930") == "IT"
        assert exp.get_sector("051910") == "Chemicals"
        assert exp.get_sector("INVALID") == "UNKNOWN"

    def test_count_in_sector(self, sector_map):
        """count_in_sector returns position count."""
        cfg = SectorExposureConfig(mode="count")
        exp = SectorExposure(sector_map, cfg)

        exp.reserve("005930", 100, 60000)
        exp.on_fill("005930", 100, 60000)
        exp.reserve("000660", 100, 80000)

        # 1 open + 1 working
        assert exp.count_in_sector("IT", include_working=True) == 2
        assert exp.count_in_sector("IT", include_working=False) == 1

    def test_notional_in_sector(self, sector_map):
        """notional_in_sector returns notional exposure."""
        cfg = SectorExposureConfig(mode="pct")
        exp = SectorExposure(sector_map, cfg)

        exp.reserve("005930", 100, 60000)  # 6M
        exp.on_fill("005930", 100, 60000)
        exp.reserve("000660", 50, 80000)   # 4M working

        assert exp.notional_in_sector("IT", include_working=True) == 10_000_000
        assert exp.notional_in_sector("IT", include_working=False) == 6_000_000

    def test_sector_pct(self, sector_map):
        """sector_pct returns exposure as fraction of equity."""
        cfg = SectorExposureConfig(mode="pct")
        exp = SectorExposure(sector_map, cfg)
        equity = 100_000_000

        exp.reserve("005930", 100, 60000)
        exp.on_fill("005930", 100, 60000)

        assert exp.sector_pct("IT", equity) == pytest.approx(0.06, abs=0.001)

    def test_sector_pct_zero_equity(self, sector_map):
        """sector_pct returns 0 when equity is zero."""
        exp = SectorExposure(sector_map)

        exp.reserve("005930", 100, 60000)
        exp.on_fill("005930", 100, 60000)

        assert exp.sector_pct("IT", 0) == 0.0


class TestEdgeCases:
    """Edge case tests."""

    def test_unreserve_never_negative(self, sector_map):
        """Unreserve doesn't go below zero."""
        exp = SectorExposure(sector_map)

        # Unreserve without prior reserve
        exp.unreserve("005930", 100, 60000)

        assert exp.sector_working_count.get("IT", 0) == 0
        assert exp.sector_working_notional.get("IT", 0.0) == 0.0

    def test_on_close_never_negative(self, sector_map):
        """on_close doesn't go below zero."""
        exp = SectorExposure(sector_map)

        # Close without prior fill
        exp.on_close("005930", 100, 60000)

        assert exp.sector_open_count.get("IT", 0) == 0
        assert exp.sector_open_notional.get("IT", 0.0) == 0.0

    def test_empty_sector_map(self):
        """Works with empty sector map."""
        cfg = SectorExposureConfig(unknown_sector_policy="allow")
        exp = SectorExposure({}, cfg)

        assert exp.can_enter("ANY_SYMBOL", 100, 50000, 100_000_000)
        assert exp.get_sector("ANY_SYMBOL") == "UNKNOWN"

    def test_zero_qty_or_price(self, sector_map):
        """Handles zero qty or price gracefully."""
        exp = SectorExposure(sector_map)

        # Zero notional shouldn't affect anything
        exp.reserve("005930", 0, 60000)
        exp.reserve("005930", 100, 0)

        assert exp.sector_working_notional.get("IT", 0.0) == 0.0
        # Count still increments
        assert exp.sector_working_count.get("IT", 0) == 2
