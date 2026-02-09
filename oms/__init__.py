"""
OMS - Order Management System

Shared library for multi-strategy order management:
- Intent-based API for strategies
- Risk gateway with exposure limits
- Multi-strategy arbitration
- KIS execution adapter
"""

from .intent import (
    Intent,
    IntentType,
    IntentStatus,
    IntentResult,
    Urgency,
    TimeHorizon,
    IntentConstraints,
    RiskPayload,
)
from .state import (
    StateStore,
    SymbolPosition,
    StrategyAllocation,
    WorkingOrder,
    OrderStatus,
)
from .risk import RiskGateway, RiskConfig, RiskResult, RiskDecision
from .arbitration import ArbitrationEngine, ArbitrationDecision, ArbitrationResult
from .planner import OrderPlanner, OrderPlan, OrderType
from .adapter import KISExecutionAdapter, AdapterResult, AdapterError
from .oms_core import OMSCore, IdempotencyStore, InMemoryIdempotencyStore
from .persistence import OMSPersistence

__all__ = [
    # Intent
    'Intent', 'IntentType', 'IntentStatus', 'IntentResult',
    'Urgency', 'TimeHorizon', 'IntentConstraints', 'RiskPayload',
    # State
    'StateStore', 'SymbolPosition', 'StrategyAllocation', 'WorkingOrder', 'OrderStatus',
    # Risk
    'RiskGateway', 'RiskConfig', 'RiskResult', 'RiskDecision',
    # Arbitration
    'ArbitrationEngine', 'ArbitrationDecision', 'ArbitrationResult',
    # Planner
    'OrderPlanner', 'OrderPlan', 'OrderType',
    # Adapter
    'KISExecutionAdapter', 'AdapterResult', 'AdapterError',
    # Core
    'OMSCore', 'IdempotencyStore', 'InMemoryIdempotencyStore',
    # Persistence
    'OMSPersistence',
]

__version__ = '2.0.0'
