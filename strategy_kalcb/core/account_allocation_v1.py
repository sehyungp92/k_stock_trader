"""Explicit, scoped research allocation at the integer entry decision boundary.

Disabled by default. A replay must declare the complete previewed opening book.
The core still owns eligibility, orders, metadata and all subsequent management.
No orders, quantities or fills are retrospectively edited.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_allocation = ContextVar('kalcb_account_allocation_v1', default=None)


@contextmanager
def entry_allocation(book):
    token = _allocation.set(book)
    try:
        yield
    finally:
        _allocation.reset(token)


def allocate_quantity(symbol, qty, metadata):
    book = _allocation.get()
    if book is None or metadata.get('entry_route') != 'first30_open_anchor':
        return qty
    if symbol not in book:
        raise ValueError('Opening intent absent from complete account preview: ' + symbol)
    record = book[symbol]
    if qty != record['parent_qty']:
        raise ValueError('Opening integer quantity changed after joint preview: ' + symbol)
    selected = record['qty']
    if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected <= qty:
        raise ValueError('Allocation must be a feasible integer reduction')
    if selected != qty:
        metadata['account_allocation_v1'] = dict(record)
    return selected
