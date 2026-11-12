"""Bound rebuildable input caches without changing verified bytes or decisions."""
from collections import OrderedDict
import gc


class BoundedVerifiedBytes(OrderedDict):
    def __init__(self, maximum_bytes):
        super().__init__()
        self.maximum_bytes = maximum_bytes
        self.bytes = self.peak_bytes = self.evictions = self.hits = self.misses = 0

    def get(self, key, default=None):
        if key not in self:
            self.misses += 1
            return default
        self.hits += 1
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self:
            self.bytes -= len(super().pop(key))
        size = len(value)
        if size > self.maximum_bytes:
            return
        while self.bytes + size > self.maximum_bytes:
            _, removed = super().popitem(last=False)
            self.bytes -= len(removed)
            self.evictions += 1
        super().__setitem__(key, value)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def clear(self):
        super().clear()
        self.bytes = 0


def install_memory_controls(observe, maximum_cache_bytes=64 * 1024**2):
    """Keep the native reader, auditor, source files and integrity gates intact."""
    from backtests.core import input_manifest
    from backtests.engine import integrity
    import pyarrow as pa

    # Each cache hit and cache miss still passes through InputManifest.read's
    # original SHA256 verification. Eviction changes storage, not qualification.
    previous = input_manifest._VERIFIED_BYTES
    cache = BoundedVerifiedBytes(maximum_cache_bytes)
    for key, value in previous.items():
        cache[key] = value
    input_manifest._VERIFIED_BYTES = cache
    previous.clear()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    original_audit = integrity.audit_broker

    def trim():
        cache.clear()
        gc.collect()
        pa.default_memory_pool().release_unused()

    def observed_audit(broker, *args, **kwargs):
        observe('before_audit_trim')
        trim()
        observe('after_audit_trim')
        result = original_audit(broker, *args, **kwargs)
        observe('after_native_audit')
        trim()
        observe('after_audit_cleanup')
        return result

    integrity.audit_broker = observed_audit
    return cache
