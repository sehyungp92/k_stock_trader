"""C15 R1 audit-only disk storage; preserve exact source checks and values.

Only locally constructed tuples are pickled. No external pickle is consumed.
The historical verifier still authenticates every input and runs every check.
"""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import pickle
import sqlite3
import tempfile

_CURRENT = ContextVar('c15_audit_storage', default=None)


def packed(value):
    return pickle.dumps(value, protocol=5)


class AuditStore:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='c15-source-audit-')
        self.path = Path(self.temp.name) / 'observations.sqlite'
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA cache_size=-8192')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('CREATE TABLE events (key BLOB PRIMARY KEY, value BLOB NOT NULL) WITHOUT ROWID')
        self.db.execute('CREATE TABLE stamps (ordinal INTEGER PRIMARY KEY, key BLOB NOT NULL, value BLOB NOT NULL)')
        self.db.execute('CREATE INDEX stamp_key ON stamps(key, ordinal)')

    def close(self):
        try:
            self.db.close()
        finally:
            self.temp.cleanup()


def audit_storage(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        store = AuditStore()
        token = _CURRENT.set(store)
        try:
            return function(*args, **kwargs)
        finally:
            _CURRENT.reset(token)
            store.close()
    return wrapped


class DiskEventIndex:
    def __init__(self):
        self.store = _CURRENT.get()
        if self.store is None:
            raise RuntimeError('Audit storage scope missing')

    def __contains__(self, key):
        return self.store.db.execute('SELECT 1 FROM events WHERE key=?', (packed(key),)).fetchone() is not None

    def __setitem__(self, key, value):
        self.store.db.execute('INSERT INTO events VALUES (?,?)', (packed(key), packed(value)))

    def get(self, key, default=None):
        row = self.store.db.execute('SELECT value FROM events WHERE key=?', (packed(key),)).fetchone()
        return default if row is None else pickle.loads(row[0])


class StampAppender:
    def __init__(self, store, key):
        self.store, self.key = store, packed(key)

    def append(self, value):
        self.store.db.execute('INSERT INTO stamps(key,value) VALUES (?,?)', (self.key, packed(value)))


class DiskStampGroups:
    def __init__(self):
        self.store = _CURRENT.get()
        if self.store is None:
            raise RuntimeError('Audit storage scope missing')

    def __getitem__(self, key):
        return StampAppender(self.store, key)

    def get(self, key, default=None):
        rows = self.store.db.execute('SELECT value FROM stamps WHERE key=? ORDER BY ordinal', (packed(key),)).fetchall()
        return [pickle.loads(row[0]) for row in rows] if rows else default


def patched_verifier(source):
    """Fail closed unless the exact historical allocation points are present."""
    signature = 'def verify_qualification(broker, identity, *, capacity_report=None):'
    allocation = '        indexed, grouped = {}, defaultdict(list)'
    if source.count(signature) != 1 or source.count(allocation) != 1:
        raise ValueError('Unexpected historical qualification implementation')
    imported = ('from backtests.strategies.kalcb.qualification_storage_v1 import '
                'audit_storage, DiskEventIndex, DiskStampGroups\n\n\n@audit_storage\n')
    return source.replace(signature, imported + signature).replace(
        allocation, '        indexed, grouped = DiskEventIndex(), DiskStampGroups()')
