import pytest
from strategy_kpr.universe.tier_manager import UniverseManager, FeatureSet
from strategy_kpr.core.state import SymbolState, FSMState, Tier

class TestClassifyTicker:
    def test_in_position_is_hot(self):
        um = UniverseManager()
        feat = FeatureSet()
        assert um.classify_ticker("005930", feat, FSMState.IDLE, in_pos=True) == Tier.HOT

    def test_setup_detected_is_hot(self):
        um = UniverseManager()
        feat = FeatureSet()
        assert um.classify_ticker("005930", feat, FSMState.SETUP_DETECTED, False) == Tier.HOT

    def test_accepting_is_hot(self):
        um = UniverseManager()
        feat = FeatureSet()
        assert um.classify_ticker("005930", feat, FSMState.ACCEPTING, False) == Tier.HOT

    def test_deep_drop_is_hot(self):
        um = UniverseManager()
        feat = FeatureSet(drop_from_open=-0.02)
        assert um.classify_ticker("005930", feat, FSMState.IDLE, False) == Tier.HOT

    def test_in_vwap_band_is_hot(self):
        um = UniverseManager()
        feat = FeatureSet(in_vwap_band=True)
        assert um.classify_ticker("005930", feat, FSMState.IDLE, False) == Tier.HOT

    def test_moderate_drop_is_warm(self):
        um = UniverseManager()
        feat = FeatureSet(drop_from_open=-0.01)
        assert um.classify_ticker("005930", feat, FSMState.IDLE, False) == Tier.WARM

    def test_range_expand_is_warm(self):
        um = UniverseManager()
        feat = FeatureSet(range_expand=True)
        assert um.classify_ticker("005930", feat, FSMState.IDLE, False) == Tier.WARM

    def test_neutral_is_cold(self):
        um = UniverseManager()
        feat = FeatureSet()
        assert um.classify_ticker("005930", feat, FSMState.IDLE, False) == Tier.COLD

class TestGetTier:
    def test_hot_membership(self):
        um = UniverseManager()
        um.hot = {"005930"}
        assert um.get_tier("005930") == Tier.HOT

    def test_warm_membership(self):
        um = UniverseManager()
        um.warm = {"005930"}
        assert um.get_tier("005930") == Tier.WARM

    def test_default_cold(self):
        um = UniverseManager()
        assert um.get_tier("005930") == Tier.COLD

class TestRebalance:
    def test_basic_rebalance(self):
        um = UniverseManager(hot_max=2, warm_max=2)
        universe = ["A", "B", "C", "D"]
        states = {s: SymbolState(code=s) for s in universe}
        features = {
            "A": FeatureSet(drop_from_open=-0.02),  # HOT
            "B": FeatureSet(drop_from_open=-0.01),  # WARM
            "C": FeatureSet(range_expand=True),       # WARM
            "D": FeatureSet(),                         # COLD
        }
        hot, warm, cold = um.rebalance(universe, states, features, set())
        assert "A" in hot
        assert "D" in cold
