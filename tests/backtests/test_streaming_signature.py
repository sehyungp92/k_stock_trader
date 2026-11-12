import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tracemalloc

import pytest

from backtests.auto.shared.cache_keys import _default, stable_signature
from backtests.core.input_manifest import signature


@pytest.mark.parametrize("sign,default", [(stable_signature, _default), (signature, str)])
@pytest.mark.parametrize("value", [
    None, {}, [], [True, False, None, -0.0, 1e-300, 1e300, float("inf"), float("nan")],
    {"unicode": "한글 😀", "escaped": "\n\t\"\\", "nested": [{"z": 3, "a": 2}]},
    {"path": Path("source/events"), "time": datetime(2026, 10, 5, tzinfo=timezone.utc)},
    [{"symbol": f"{i:06}", "volume": i, "price": i / 7} for i in range(1000)],
])
def test_streaming_signature_preserves_canonical_hash(value, sign, default):
    original = json.dumps(value, sort_keys=True, separators=(",", ":"), default=default).encode("utf-8")
    assert sign(value) == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("sign", [stable_signature, signature])
def test_large_event_signature_does_not_materialize_serialized_ledger(sign):
    row = {"symbol": "005930", "timestamp": "2026-10-05T09:05:00+09:00", "metadata": "x" * 512}
    events = [row] * 20_000
    tracemalloc.start()
    try:
        result = sign(events)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(result) == 64
    assert peak < 1_000_000
