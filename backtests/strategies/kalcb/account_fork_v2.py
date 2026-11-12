"""Content-addressed complete-account forks with immutable copy-on-read inputs.

Every attribute remains in the state receipt. Only immutable serialized source
maps and append-only history bytes are shared; mutable trading state is copied.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import pickle
from types import MappingProxyType

from .account_history_v1 import AccountHistory, _encode
from .pending_entry_account_pilot_v1 import canonical, neutral
from .account_decision_v1 import AccountBoundary


class MerkleHistory(AccountHistory):
    def __init__(self, values=(), *, prefix=(), root=None):
        super().__init__(values, prefix=prefix)
        self.root = root or hashlib.sha256(b'kalcb-history-v2').hexdigest()

    def seal(self):
        if self._tail:
            encoded=[]
            for value in self._tail:
                payload=canonical(neutral(value)).encode()
                self.root=hashlib.sha256(bytes.fromhex(self.root)+hashlib.sha256(payload).digest()).hexdigest()
                encoded.append(_encode(value))
            self._prefix += tuple(encoded)
            self._tail=[]

    def __deepcopy__(self, memo):
        self.seal()
        result=type(self)(prefix=self._prefix,root=self.root)
        memo[id(self)]=result
        return result

    def commitment(self):
        self.seal()
        return {'kind':'append-history-v2','length':len(self),'sha256':self.root}


class FrozenValueMap(Mapping):
    """Private immutable bytes; every returned value is independently decoded."""
    def __init__(self, values):
        self._data=MappingProxyType({k:_encode(v) for k,v in values.items()})
        self.root=hashlib.sha256(canonical(neutral(values)).encode()).hexdigest()

    def __getitem__(self,key):return pickle.loads(self._data[key])
    def __iter__(self):return iter(self._data)
    def __len__(self):return len(self._data)
    def __deepcopy__(self,memo):memo[id(self)]=self;return self


HISTORIES=('journal','fills','trades','expected_batches','mark_coverage',
           'equity_curve','timestamps','rejected_orders','expired_orders',
           'qualification_snapshots')


def prepare(adapter,broker):
    if adapter.frontier_shadow is not None:raise ValueError('Unsupported shadow account')
    for key in ('snapshots','timed_snapshots'):
        if not isinstance(getattr(adapter,key),FrozenValueMap):
            setattr(adapter,key,FrozenValueMap(getattr(adapter,key)))
    for key in HISTORIES:
        if hasattr(broker,key) and not isinstance(getattr(broker,key),MerkleHistory):
            setattr(broker,key,MerkleHistory(getattr(broker,key)))
    if hasattr(broker,'qualification_snapshot_scopes') and not isinstance(broker.qualification_snapshot_scopes,FrozenValueMap):
        broker.qualification_snapshot_scopes=FrozenValueMap(broker.qualification_snapshot_scopes)


def receipt(adapter,broker):
    prepare(adapter,broker)
    def attributes(obj):
        return {k:(v.commitment() if isinstance(v,MerkleHistory) else
                   {'kind':'immutable-map-v2','length':len(v),'sha256':v.root} if isinstance(v,FrozenValueMap)
                   else neutral(v)) for k,v in vars(obj).items()}
    value={'adapter':attributes(adapter),'broker':attributes(broker)}
    return {'sha256':hashlib.sha256(canonical(value).encode()).hexdigest(),'state':value}


def capture(adapter,broker,available_at):
    r=receipt(adapter,broker)
    return AccountBoundary(r['sha256'],available_at.isoformat(),broker._portfolio_equity(),
        max(broker.equity_curve),len(broker.timestamps),len(broker.fills),len(broker.orders)),r


def fork(adapter,broker,boundary):
    if receipt(adapter,broker)['sha256']!=boundary.state_sha256:raise ValueError('Changed account boundary')
    memo={id(broker.source_events):broker.source_events.copy()}
    aa,bb=deepcopy(adapter),deepcopy(broker,memo)
    if receipt(aa,bb)['sha256']!=boundary.state_sha256:raise ValueError('Incomplete account fork')
    return aa,bb
