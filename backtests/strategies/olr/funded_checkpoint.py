"""Complete native continuation snapshots with explicit hash-state restoration.

Only locally generated, receipt-bound snapshots may be loaded. No subset of
orders, reservations or strategy history is substituted for a native account.
"""
import hashlib
import io
import json
import pickle
from types import MappingProxyType
from pathlib import Path


def _readonly_mapping(value):
    return MappingProxyType(value)


def _source_placeholder():
    return None


class _CheckpointPickler(pickle.Pickler):
    """Local reducers preserve immutable event metadata without global hooks."""
    def __init__(self, buffer, source_events):
        super().__init__(buffer, protocol=5)
        self.source_events = source_events

    def reducer_override(self, value):
        if value is self.source_events:
            return _source_placeholder, ()
        if isinstance(value, MappingProxyType):
            return _readonly_mapping, (dict(value),)
        return NotImplemented


def fork(driver):
    # A partial realization adds immutable TradeOutcome/DecisionEvent mappings.
    # A local native serialization round trip preserves their types and aliases;
    # deepcopy alone cannot handle MappingProxyType. The source hasher keeps its
    # exact accumulated state through hashlib.copy, not a reset digest.
    payload=pickle.loads(state_bytes(driver))
    detached=payload['driver']
    detached.broker.source_events=driver.broker.source_events.copy()
    return detached


def state_bytes(driver):
    stream=io.BytesIO()
    _CheckpointPickler(stream,driver.broker.source_events).dump(dict(
        version=2,source_prefix=driver.broker.source_events.hexdigest(),driver=driver))
    return stream.getvalue()


def fingerprint(driver):
    return hashlib.sha256(state_bytes(driver)).hexdigest()


def save(driver,path):
    path=Path(path);payload=state_bytes(driver)
    with path.open('xb') as f:f.write(payload)
    return dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),
                source_prefix=driver.broker.source_events.hexdigest(),
                source_events=driver.broker.source_event_count,
                pending_orders=len(driver.broker.orders),positions=len(driver.broker.positions),
                complete_account=True)


def load(path,expected_sha256):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha256:raise ValueError('Changed checkpoint bytes')
    payload=pickle.loads(data)
    if payload['version'] not in (1,2):raise ValueError('Unknown checkpoint contract')
    driver=payload['driver'];h=hashlib.sha256();count=0
    for batch in driver.broker.expected_batches:
        for raw in batch:h.update(json.dumps(raw,sort_keys=True,default=str).encode());count+=1
    if h.hexdigest()!=payload['source_prefix'] or count!=driver.broker.source_event_count:
        raise ValueError('Checkpoint source prefix mismatch')
    driver.broker.source_events=h
    return driver


def economic_state(driver):
    """Deterministic structural comparison independent of pickle memo layout."""
    from dataclasses import asdict
    from backtests.auto.shared.cache_keys import stable_signature
    from strategy_olr.core.serializers import snapshot_state
    b=driver.broker;a=driver.adapter
    return stable_signature(dict(cash=b.cash,marks=b.last_prices,
        positions={str(k):asdict(v) for k,v in b.positions.items()},orders=[asdict(o) for o in b.orders],
        fills=[asdict(f) for f in b.fills],trades=[t.to_json_dict() for t in b.trades],
        journal=b.journal,core=snapshot_state(a.state),source=b.source_events.hexdigest(),
        equity=b.equity_curve,timestamps=b.timestamps,order_sequence=b._order_sequence,
        cursors=[a._fill_cursor,a._expired_cursor,a._order_journal_cursor],
        capacity_used={str(k):v for k,v in b.capacity_used.items()},order_filled=b.order_filled))
