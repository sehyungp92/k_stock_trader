"""KPR Symbol State and FSM."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional
import math


class FSMState(Enum):
    IDLE = auto()
    SETUP_DETECTED = auto()
    ACCEPTING = auto()
    IN_POSITION = auto()
    PENDING_EXIT = auto()  # Exit order submitted, awaiting fill confirmation
    INVALIDATED = auto()
    DONE = auto()


class Tier(Enum):
    HOT = auto()
    WARM = auto()
    COLD = auto()


@dataclass
class SymbolState:
    code: str
    fsm: FSMState = FSMState.IDLE
    tier: Tier = Tier.COLD

    # Sector metadata (for sector cap enforcement)
    sector: str = ""

    hod: float = 0.0
    hod_time: Optional[datetime] = None
    lod: float = math.inf
    vwap: float = 0.0

    setup_low: Optional[float] = None
    reclaim_level: Optional[float] = None
    stop_level: Optional[float] = None
    setup_time: Optional[datetime] = None
    setup_type: Optional[str] = None  # "panic" or "drift"
    accept_closes: int = 0
    required_closes: int = 2

    entry_px: float = 0.0
    entry_ts: Optional[datetime] = None
    qty: int = 0
    remaining_qty: int = 0
    max_price: float = 0.0
    trail_stop: float = 0.0
    partial_filled: bool = False
    confidence: str = "YELLOW"

    # Order tracking (for timeout/drift detection)
    entry_order_id: Optional[str] = None
    order_submit_ts: float = 0.0  # Epoch time of order submission

    # Signals
    investor_signal: str = "NEUTRAL"
    micro_signal: str = "NEUTRAL"
    program_signal: str = "NEUTRAL"

    # Features
    drop_from_open: float = 0.0
    in_vwap_band: bool = False

    def reset_setup(self):
        self.setup_low = self.reclaim_level = self.stop_level = self.setup_time = self.setup_type = None
        self.accept_closes = 0

    def reset_for_new_day(self):
        """Reset state for a new trading day."""
        self.fsm = FSMState.IDLE
        self.hod = 0.0
        self.hod_time = None
        self.lod = math.inf
        self.vwap = 0.0
        self.reset_setup()
        self.entry_px = 0.0
        self.entry_ts = None
        self.qty = 0
        self.remaining_qty = 0
        self.max_price = 0.0
        self.trail_stop = 0.0
        self.partial_filled = False
        self.confidence = "YELLOW"
        self.entry_order_id = None
        self.order_submit_ts = 0.0
        self.investor_signal = "NEUTRAL"
        self.micro_signal = "NEUTRAL"
        self.program_signal = "NEUTRAL"
        self.drop_from_open = 0.0
        self.in_vwap_band = False
