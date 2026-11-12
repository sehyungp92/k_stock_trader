from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from scripts import olr_snapshot_acceleration as acceleration


def generation(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{"value": 1}')
    expected = acceleration.digest(source)
    return SimpleNamespace(root=tmp_path, loaded={'source.json'},
        digest=lambda path: expected if path.name == 'source.json' else None)


def test_cache_verifies_source_and_payload_before_deserializing(tmp_path, monkeypatch):
    gen = generation(tmp_path)
    path = tmp_path / 'cache.pickle'
    acceleration.save_cache(path, {'generation': 'a'}, gen, {'native': [1, 2]})
    assert acceleration.load_cache(path, {'generation': 'a'}, gen) == {'native': [1, 2]}
    monkeypatch.setattr(acceleration.pickle, 'load', lambda *args: pytest.fail('Unverified pickle loaded'))
    with pytest.raises(ValueError, match='authentication'):
        acceleration.load_cache(path, {'generation': 'b'}, gen)
    (tmp_path / 'source.json').write_text('{"value": 2}')
    with pytest.raises(ValueError, match='input bytes changed'):
        acceleration.load_cache(path, {'generation': 'a'}, gen)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='authentication'):
        acceleration.load_cache(path, {'generation': 'a'}, gen)


def test_partial_cache_is_rejected_not_silently_trusted(tmp_path):
    gen = generation(tmp_path)
    path = tmp_path / 'cache.pickle'
    assert acceleration.load_cache(path, {}, gen) is None
    path.write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='Incomplete'):
        acceleration.load_cache(path, {}, gen)


@pytest.mark.parametrize('field,value', [('manifest_hash','other'), ('source_hash','other'),
    ('database_hash','other'), ('days',[date(2026,3,31),date(2026,4,1)]),
    ('output_days',[date(2026,4,2)]), ('lookback',105), ('complete',False)])
def test_cache_identity_separates_every_feature_dependency(field, value):
    args = dict(manifest_hash='manifest', source_hash='source', database_hash='database',
                days=[date(2026,4,1)], output_days=[date(2026,4,1)], lookback=100, complete=True)
    original = acceleration.cache_identity(**args)
    args[field] = value
    assert acceleration.cache_identity(**args) != original


def test_native_research_is_invariant_to_selection_config_and_rebinds_hash():
    from strategy_olr.config import OLRConfig
    from strategy_olr.research import build_research_snapshot
    from backtests.strategies.olr.research_sweep import _research_snapshot_with_config
    day = date(2026,4,1)
    rows = [{'date':str(day-timedelta(days=120-i)), 'open':100+i, 'high':102+i,
             'low':99+i, 'close':101+i, 'volume':1000000} for i in range(120)]
    base = OLRConfig()
    changed = replace(base, research_top_long_count=1)
    options = dict(source_fingerprint='authenticated', generated_at=datetime(2026,4,1,tzinfo=timezone.utc),
                   metadata={'research_snapshot_daily_lookback':100, 'research_snapshot_flow_lookback':100})
    a = build_research_snapshot({'005930':rows}, day, base, **options)
    b = build_research_snapshot({'005930':rows}, day, changed, **options)
    assert a.symbols  # Exercise real feature construction, not an empty snapshot.
    assert asdict(_research_snapshot_with_config(a, changed)) == asdict(b)
    assert a.metadata['research_config_hash'] != b.metadata['research_config_hash']


def test_requested_research_dates_keep_full_previous_session_context(monkeypatch):
    from backtests.strategies.olr import research_sweep as native
    prior, scoring = date(2026,3,31), date(2026,4,1)
    seen = []
    monkeypatch.setattr(native, '_research_snapshot_with_config', lambda value, cfg: value)
    monkeypatch.setattr(native, 'daily_selection_from_snapshot', lambda value, cfg: value)
    def attach(value, dataset):
        seen.append(native._previous_trading_date(dataset.trading_dates, scoring))
        return value
    monkeypatch.setattr(native, '_with_prev_session_sector_intraday', attach)
    dataset = SimpleNamespace(config={}, trading_dates=(prior,scoring))
    assert native.snapshots_for_experiment(dataset, {}, research_snapshots={scoring:'snapshot'}) == {scoring:'snapshot'}
    assert seen == [prior]


@pytest.mark.parametrize('workers', [3,4])
def test_queue_uses_requested_slots_and_does_not_starve_long_window(tmp_path, monkeypatch, workers):
    import threading
    from scripts import olr_round1_mechanism_study as study
    barrier = threading.Barrier(workers, timeout=10)
    seen = []
    def evaluate(root, candidate, window):
        seen.append((candidate['id'], window))
        barrier.wait()
        return {'status':'completed', 'qualified':True}
    monkeypatch.setattr(study, 'evaluate', evaluate)
    jobs = [dict(candidate={'id':str(i),'label':str(i)},window='train',index=i+1) for i in range(workers)]
    study.run_evaluation_jobs(tmp_path, jobs, workers, 'test')
    assert len(seen) == workers
    assert json.loads((tmp_path/'progress_train.json').read_text())['completed_evaluations'] == workers


def test_queue_failure_stops_dispatch_but_finishes_active_job(tmp_path, monkeypatch):
    import threading
    from scripts import olr_round1_mechanism_study as study
    entered, failed = threading.Event(), threading.Event()
    seen, finished = [], []
    def evaluate(root, candidate, window):
        cid = candidate['id']
        seen.append(cid)
        if cid == '0':
            entered.wait(10)
            failed.set()
            raise ValueError('native audit failure')
        entered.set()
        assert failed.wait(10)
        finished.append(cid)
        return {'status':'completed','qualified':True}
    monkeypatch.setattr(study, 'evaluate', evaluate)
    jobs = [dict(candidate={'id':str(i),'label':str(i)},window='train',index=i+1) for i in range(5)]
    with pytest.raises((RuntimeError, ValueError)):
        study.run_evaluation_jobs(tmp_path, jobs, 2, 'test')
    assert set(seen) == {'0','1'}
    assert finished == ['1']


def test_queue_honors_hold_and_rejects_duplicate_cases(tmp_path, monkeypatch):
    from scripts import olr_round1_mechanism_study as study
    monkeypatch.setattr(study, 'evaluate', lambda *args,**kwargs: pytest.fail('Unexpected dispatch'))
    job = dict(candidate={'id':'a','label':'a'},window='train',index=1)
    with pytest.raises(ValueError, match='Duplicate'):
        study.run_evaluation_jobs(tmp_path, [job,job], 2, 'test')
    (tmp_path/'STOP_MECHANISM.json').write_text('{}')
    with pytest.raises(RuntimeError, match='stopped'):
        study.run_evaluation_jobs(tmp_path, [job], 2, 'test')


def test_native_panel_cache_has_exact_input_guard_and_restores_on_error():
    from scripts import olr_snapshot_streaming as streaming
    day, cutoff = date(2026,4,1), 'cutoff'
    daily_original = lambda symbols, day: 'uncached daily'
    intraday_original = lambda *args, **kwargs: 'uncached intraday'
    research = SimpleNamespace(_sector_daily_panel_from_symbols=daily_original,
                               build_sector_intraday_panel=intraday_original)
    symbols = {'005930':object()}
    snapshot = SimpleNamespace(symbols=symbols)
    with pytest.raises(RuntimeError):
        with streaming.native_panel_cache(research,day,snapshot,'cached daily','cached intraday',{'005930':'IT'},cutoff):
            assert research._sector_daily_panel_from_symbols(symbols,day) == 'cached daily'
            assert research._sector_daily_panel_from_symbols(dict(symbols),day) == 'uncached daily'
            assert research.build_sector_intraday_panel({}, {'005930':'IT'},trade_dates=(day,),cutoff=cutoff) == 'cached intraday'
            assert research.build_sector_intraday_panel({}, {'005930':'IT'},trade_dates=(day,),cutoff='other') == 'uncached intraday'
            assert research.build_sector_intraday_panel({}, {'005930':'OTHER'},trade_dates=(day,),cutoff=cutoff) == 'uncached intraday'
            raise RuntimeError('exception must restore original native functions')
    assert research._sector_daily_panel_from_symbols is daily_original
    assert research.build_sector_intraday_panel is intraday_original


def test_streaming_payload_tamper_rejected_before_pickle_load(tmp_path, monkeypatch):
    from scripts import olr_snapshot_streaming as streaming
    path = tmp_path/'2026-04-01.pickle'
    path.write_bytes(b'changed')
    monkeypatch.setattr(streaming.pickle, 'load', lambda *a: pytest.fail('Unverified pickle loaded'))
    with pytest.raises(ValueError, match='bytes changed'):
        streaming.read_payload(path, 'expected', acceleration.digest)


def test_prefix_memo_reuses_only_identical_native_timestamp_contract(tmp_path):
    from datetime import time
    from scripts import olr_prefix_memo as memo
    day = date(2026,4,1)
    bars = {(day,'005930'):(SimpleNamespace(timestamp=datetime(2026,4,1,9,0),price=100),),
            (day,'000660'):()}
    calls = []
    def native(values, cutoff):
        calls.append(1)
        return {k:v for k,v in values.items() if v}, ['000660:2026-04-01']
    counts = {}
    first = memo.memoized_prefix(native,bars,time(14,30),tmp_path,{'calendar':'a'},counts)
    # Price is irrelevant to timestamp completeness, but the caller's current
    # bars must be returned, never bar objects retained from a previous call.
    updated = {**bars,(day,'005930'):(SimpleNamespace(timestamp=datetime(2026,4,1,9,0),price=200),)}
    second = memo.memoized_prefix(native,updated,time(14,30),tmp_path,{'calendar':'a'},counts)
    assert len(calls)==1 and counts=={'hits':1,'misses':1}
    assert second[0][day,'005930'] is updated[day,'005930']
    changed = {**bars,(day,'005930'):(SimpleNamespace(timestamp=datetime(2026,4,1,9,5)),)}
    memo.memoized_prefix(native,changed,time(14,30),tmp_path,{'calendar':'a'},counts)
    memo.memoized_prefix(native,bars,time(14,30),tmp_path,{'calendar':'b'},counts)
    memo.memoized_prefix(native,bars,time(14,25),tmp_path,{'calendar':'a'},counts)
    assert len(calls)==4


def test_prefix_memo_rejects_changed_receipt(tmp_path):
    from datetime import time
    from scripts import olr_prefix_memo as memo
    bars = {(date(2026,4,1),'005930'):()}
    memo.memoized_prefix(lambda values,cutoff: ({},['005930:2026-04-01']),bars,time(14,30),tmp_path,{}, {})
    p = next(tmp_path.glob('*.json'))
    row = json.loads(p.read_text()); row['accepted_indices']=[0];p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='authentication'):
        memo.memoized_prefix(lambda *a: pytest.fail('Changed receipt was used'),bars,time(14,30),tmp_path,{}, {})
