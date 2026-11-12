"""Session indexing preserves source values and authentication after warmup."""
import json

import pandas as pd
import pytest

from backtests.engine.olr_minute_source_v5 import (
    _SessionFrameCache, normalize, origin_verifier, verify_minute_sources,
)
from tests.strategy_olr.test_minute_source_v5 import DAY, fixture


def archive():
    # UTC midnight is not KST midnight; deliberately unsorted and sparse.
    stamps = pd.to_datetime([
        '2026-06-25T00:01:00Z', '2026-06-23T15:00:00Z',
        '2026-06-24T14:59:00Z', '2026-06-23T14:59:00Z',
    ])
    return pd.DataFrame(dict(timestamp=stamps, open=[10., 20., 30., 40.],
        high=[11., 21., 31., 41.], low=[9., 19., 29., 39.],
        close=[10., 20., 30., 40.], volume=[1., 2., 0., 4.]))


@pytest.mark.parametrize('day', ['2026-06-23', '2026-06-24', '2026-06-25',
    '2026-06-26', '2026-6-24', 'not-a-day'])
def test_index_equals_original_literal_kst_predicate(day):
    raw = archive()
    expected = normalize(raw)
    expected = expected[expected.timestamp.dt.strftime('%Y-%m-%d') == day].reset_index(drop=True)
    actual = _SessionFrameCache(lambda path: raw).dayframe('archive', day)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_sources_and_returned_slices_are_isolated_and_clear_reloads():
    raw = archive()
    calls = []
    def load(path):
        calls.append(path)
        return raw.assign(volume=raw.volume + (100 if path == 'other' else 0))
    cache = _SessionFrameCache(load)
    original = cache.dayframe('first', '2026-06-24')
    changed = cache.dayframe('first', '2026-06-24')
    changed.loc[0, 'volume'] = -123
    pd.testing.assert_frame_equal(cache.dayframe('first', '2026-06-24'), original)
    other = cache.dayframe('other', '2026-06-24')
    assert list(other.volume) == list(original.volume + 100)
    assert calls == ['first', 'other']
    cache.clear()
    raw.loc[:, 'volume'] += 7
    assert list(cache.dayframe('first', '2026-06-24').volume) == list(original.volume + 7)
    assert calls == ['first', 'other', 'first']


def test_empty_and_duplicate_archive_contracts_are_preserved():
    empty = archive().iloc[:0]
    pd.testing.assert_frame_equal(_SessionFrameCache(lambda _: empty).dayframe('x', '2026-06-24'), normalize(empty))
    duplicate = pd.concat([archive(), archive().iloc[:1]], ignore_index=True)
    with pytest.raises(AssertionError, match='Duplicate minute'):
        _SessionFrameCache(lambda _: duplicate).dayframe('x', '2026-06-24')


def test_warm_index_does_not_bypass_changed_source_bytes(tmp_path):
    _, generation, _, _ = fixture(tmp_path)
    definition = json.loads(generation.read(generation.root / 'definition.json'))
    session, _, _ = origin_verifier(generation, definition)
    assert len(session('000100', str(DAY))[0]) == 5
    path = generation.root / 'raw.parquet'
    raw = pd.read_parquet(path)
    raw.loc[0, 'volume'] += 1
    raw.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Frozen input bytes changed'):
        session('000100', str(DAY))


def test_complete_source_verdict_equal_to_legacy_lookup(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    indexed = verify_minute_sources(*args)
    def legacy(self, path, day):
        frame = self.frame(path)
        return frame[frame.timestamp.dt.strftime('%Y-%m-%d') == day].reset_index(drop=True)
    monkeypatch.setattr(_SessionFrameCache, 'dayframe', legacy)
    assert verify_minute_sources(*args) == indexed
