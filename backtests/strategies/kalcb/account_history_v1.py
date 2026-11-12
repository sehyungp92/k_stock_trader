"""Immutable history prefixes with private append tails for native account forks.

Record reads from a prefix return fresh objects. No branch can mutate another
branch's nested fill/order/source evidence. Prefix bytes are shared, avoiding
quadratic deepcopy of accumulated replay history at every decision.
"""
from copy import deepcopy
import io
import pickle
from types import MappingProxyType


class _HistoryPickler(pickle.Pickler):
    def reducer_override(self, value):
        if isinstance(value, MappingProxyType):
            return dict, (dict(value),)
        return NotImplemented


def _encode(value):
    stream=io.BytesIO()
    _HistoryPickler(stream,protocol=5).dump(value)
    return stream.getvalue()


class AccountHistory(list):
    """Append-only list interface. Sealed records are immutable serialized data."""
    def __init__(self, values=(), *, prefix=()):
        super().__init__()
        self._prefix=prefix
        self._tail=list(values)

    def seal(self):
        if self._tail:
            self._prefix += tuple(_encode(x) for x in self._tail)
            self._tail=[]

    def __deepcopy__(self,memo):
        self.seal()
        result=type(self)(prefix=self._prefix)
        memo[id(self)]=result
        return result

    def __len__(self):
        return len(self._prefix)+len(self._tail)

    def __iter__(self):
        for record in self._prefix:
            yield pickle.loads(record)
        yield from self._tail

    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:
            index+=len(self)
        if not 0<=index<len(self):
            raise IndexError('Account history index out of range')
        return pickle.loads(self._prefix[index]) if index<len(self._prefix) else self._tail[index-len(self._prefix)]

    def append(self,value):
        self._tail.append(value)

    def extend(self,values):
        self._tail.extend(values)

    def __eq__(self,other):
        return isinstance(other,(list,tuple)) and list(self)==list(other)

    def __ne__(self,other):
        return not self==other

    def __contains__(self,value):
        return any(x==value for x in self)

    def copy(self):
        return list(self)

    def __add__(self,other):
        return list(self)+list(other)

    def __radd__(self,other):
        return list(other)+list(self)

    def __iadd__(self,other):
        self.extend(other)
        return self

    def _reject_mutation(self,*args,**kwargs):
        raise TypeError('Native account history is append-only')

    __setitem__=__delitem__=insert=pop=remove=clear=sort=reverse=__imul__=_reject_mutation


def clone_complete_account(adapter,broker):
    from .mechanism_observer_v2 import clone_broker
    if adapter.frontier_shadow is not None:
        raise ValueError('Unsupported active frontier account')
    # Every mutable state field is still cloned. Only these immutable record
    # prefixes share bytes; orders, holdings, risk state and capacity remain private.
    for name in ('journal','fills','trades','expected_batches','mark_coverage'):
        if hasattr(broker,name) and not isinstance(getattr(broker,name),AccountHistory):
            setattr(broker,name,AccountHistory(getattr(broker,name)))
    return deepcopy(adapter),clone_broker(broker)
