"""KPR Universe Manager."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from ..core.state import SymbolState, FSMState, Tier
from ..config.constants import HOT_MAX, WARM_MAX, VWAP_DEPTH_MIN, VWAP_DEPTH_MAX


@dataclass
class FeatureSet:
    drop_from_open: float = 0.0
    in_vwap_band: bool = False
    range_expand: bool = False
    vol_score: float = 0.0
    dist_to_vwap_band: float = float('inf')


class UniverseManager:
    def __init__(self, hot_max: int = HOT_MAX, warm_max: int = WARM_MAX):
        self.hot_max, self.warm_max = hot_max, warm_max
        self.hot: Set[str] = set()
        self.warm: Set[str] = set()
        self.cold: Set[str] = set()

    def classify_ticker(self, sym: str, feat: FeatureSet, fsm: FSMState, in_pos: bool) -> Tier:
        if in_pos or fsm in (FSMState.SETUP_DETECTED, FSMState.ACCEPTING):
            return Tier.HOT
        if feat.drop_from_open <= -0.015 or feat.in_vwap_band:
            return Tier.HOT
        if feat.drop_from_open <= -0.007 or feat.range_expand:
            return Tier.WARM
        return Tier.COLD

    def rebalance(self, universe: List[str], states: Dict[str, SymbolState],
                  features: Dict[str, FeatureSet], positions: Set[str]) -> Tuple[Set, Set, Set]:
        new_hot, new_warm, new_cold = set(), set(), set()

        for sym in universe:
            state = states.get(sym)
            feat = features.get(sym, FeatureSet())
            tier = self.classify_ticker(sym, feat, state.fsm if state else FSMState.IDLE, sym in positions)
            (new_hot if tier == Tier.HOT else new_warm if tier == Tier.WARM else new_cold).add(sym)

        # Cap HOT (protect positions/setups)
        protected = {s for s in new_hot if s in positions or
                     (states.get(s) and states[s].fsm in (FSMState.SETUP_DETECTED, FSMState.ACCEPTING))}
        candidates = sorted(new_hot - protected, key=lambda s: features.get(s, FeatureSet()).vol_score)
        self.hot = protected | set(candidates[:max(0, self.hot_max - len(protected))])

        # Cap WARM
        warm_cands = sorted(new_warm - self.hot, key=lambda s: features.get(s, FeatureSet()).dist_to_vwap_band)
        self.warm = set(warm_cands[:self.warm_max])
        self.cold = (new_cold | (new_warm - self.warm) | (new_hot - self.hot))

        return self.hot, self.warm, self.cold

    def get_tier(self, sym: str) -> Tier:
        if sym in self.hot:
            return Tier.HOT
        if sym in self.warm:
            return Tier.WARM
        return Tier.COLD
