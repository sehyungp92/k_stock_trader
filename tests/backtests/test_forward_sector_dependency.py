"""An omitted or incomplete sector dependency must fail before model features."""
import hashlib
import json
import sqlite3
from dataclasses import make_dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from backtests.core.input_manifest import signature
from scripts import kalcb_executed_value_study as study


@pytest.mark.parametrize('case', ['omitted_table', 'unknown_sector', 'missing_symbol', 'auxiliary_context'])
def test_forward_preparation_rejects_missing_sector_identity(tmp_path, monkeypatch, case):
    from backtests.strategies.kalcb import first30_signal_sweep as native

    root = tmp_path / 'generation'
    root.mkdir()
    files = {}
    if case != 'omitted_table':
        name = 'data/krx_daily_parquet/tables/sector_map.parquet'
        path = root / 'files' / name
        path.parent.mkdir(parents=True)
        pd.DataFrame([{'ticker': '000660' if case == 'missing_symbol' else '005930',
                       'sector': 'SEMICONDUCTORS' if case in {'missing_symbol', 'auxiliary_context'} else 'UNKNOWN'}]).to_parquet(path, index=False)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {'source_root': str(tmp_path), 'files': files, 'evidence': {}}
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({**payload, 'manifest_hash': signature(payload)}))
    database = tmp_path / 'market.sqlite'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE bars (day TEXT, symbol TEXT)')
        conn.execute("INSERT INTO bars VALUES ('2026-06-01','005930')")
        if case == 'auxiliary_context':
            conn.execute("INSERT INTO bars VALUES ('2026-06-01','000880')")
    universe = tmp_path / 'universe.yaml'
    universe.write_text("symbols: ['005930']")
    guards = tmp_path / 'guards.jsonl'
    guards.write_text('')
    monkeypatch.setattr(study, 'config_for', lambda plan: SimpleNamespace())
    for name in ('_load_daily_rows', '_load_flow_rows', '_load_foreign_flow_rows',
                 '_load_institutional_flow_rows', '_load_index_rows'):
        monkeypatch.setattr(native, name, lambda *args: {})
    plan = {'market_manifest': str(manifest), 'universe': str(universe),
            'guard_features': str(guards), 'database': str(database),
            'start': '2026-06-01', 'end': '2026-06-01', 'forward_feature_only': True}
    output = tmp_path / 'output'
    output.mkdir()
    if case == 'auxiliary_context':
        # Historical context can include stocks outside the traded universe;
        # preserving those original inputs must not weaken traded-stock checks.
        cfg = make_dataclass('FixtureConfig', [('unused', int)])(1)
        monkeypatch.setattr(study, 'config_for', lambda plan: cfg)
        monkeypatch.setattr(study, 'load_day', lambda *args: [])
        monkeypatch.setattr(native, 'build_contexts', lambda *args, **kwargs: {})
        policy = tmp_path / 'policy.json'
        policy.write_text(json.dumps({'mutations': {}}))
        plan.update(seed=str(policy), q3=str(policy), input_sha256={str(database): 'fixture'})
        study.prepare(plan, output)
        prepared = json.loads((output / 'prepared.json').read_text())
        assert prepared['context_symbols'] == ['000880', '005930']
        assert prepared['universe'] == ['005930']
        return
    message = 'Unmanifested research input' if case == 'omitted_table' else 'Incomplete frozen forward sector map'
    with pytest.raises(ValueError, match=message):
        study.prepare(plan, output)
    assert not (output / 'opportunities.jsonl').exists()
