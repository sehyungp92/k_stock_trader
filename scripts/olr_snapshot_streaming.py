"""Stream authenticated native features one session at a time.

Cached panels are outputs of the frozen native panel functions, not alternate
formulas. Their complete data/code/date identities are verified on each use.
Selection, configuration hashing, ranking and rejection remain native calls.
"""
from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace


def helper():
    path = Path(__file__).with_name('olr_snapshot_acceleration.py')
    spec = importlib.util.spec_from_file_location('olr_invariant_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_payload(path, expected_sha256, digest):
    if digest(path) != expected_sha256:
        raise ValueError('Streaming feature bytes changed: ' + str(path))
    with path.open('rb') as stream:
        return pickle.load(stream)


@contextmanager
def native_panel_cache(research_module, day, research_snapshot, daily_panel, intraday_panel, sector_map, cutoff):
    """Memoize only exact native calls made on the authenticated session inputs."""
    daily_original = research_module._sector_daily_panel_from_symbols
    intraday_original = research_module.build_sector_intraday_panel
    def daily(symbols, trade_date):
        if trade_date == day and symbols is research_snapshot.symbols:
            return daily_panel
        return daily_original(symbols, trade_date)
    def intraday(bars, sectors, **kwargs):
        if (dict(sectors) == dict(sector_map) and tuple(kwargs.get('trade_dates', ())) == (day,)
                and kwargs.get('cutoff') == cutoff and set(kwargs) == {'trade_dates','cutoff'}):
            return intraday_panel
        return intraday_original(bars, sectors, **kwargs)
    research_module._sector_daily_panel_from_symbols = daily
    research_module.build_sector_intraday_panel = intraday
    try:
        yield
    finally:
        research_module._sector_daily_panel_from_symbols = daily_original
        research_module.build_sector_intraday_panel = intraday_original


def build(manifest, database, request, days, *, output_days, cache_root, source_hash, profile):
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr import research
    from strategy_olr.config import OLRConfig
    from strategy_common.sector_intraday import build_sector_intraday_panel, AFTERNOON_CUTOFF
    from strategy_common.source_quality import complete_feature_bars, POLICY
    from scripts.kalcb_executed_value_study import connect, load_day
    h = helper()
    started = time.perf_counter()
    days, output_days = tuple(days), tuple(output_days)
    if (len(set(days)) != len(days) or sorted(days) != list(days)
            or len(set(output_days)) != len(output_days) or not set(output_days).issubset(days)):
        raise ValueError('Invalid reconstruction context/output dates')
    generation = InputManifest(manifest)
    cfg = OLRConfig.from_mapping(request['config'], request['mutations'])
    source = h.digest(database)
    if generation.digest(database) != source:
        raise ValueError('Frozen intraday database bytes changed')
    base_identity = h.cache_identity(generation.identity, source_hash, source, days, output_days,
                                   max(100,int(cfg.research_min_history_days)+5), cfg.require_complete_feature_prefix)
    identity = {'version':3, 'base':base_identity, 'streaming_sha256':h.digest(__file__)}
    directory = Path(cache_root) / ('stream_' + h.signature(identity))
    index_path = directory / 'index.json'
    profile.update(cache_hit=index_path.exists(), cache=str(directory),
                   context_sessions=len(days), output_sessions=len(output_days))

    def day_bars(conn, day, symbols):
        values = defaultdict(list)
        for bar in load_day(conn, str(day), source, set(symbols)):
            values[day,bar.symbol].append(bar)
        return {key:tuple(value) for key,value in values.items()}

    if not index_path.exists():
        preparation_start = time.perf_counter()
        base_path = Path(cache_root) / (h.signature(base_identity) + '.pickle')
        payload = h.load_cache(base_path, base_identity, generation)
        if payload is None:
            h.build(manifest, database, request, days, output_days=output_days, cache_root=cache_root,
                    source_hash=source_hash, profile={})
            payload = h.load_cache(base_path, base_identity, generation)
        base_receipt = json.loads(base_path.with_suffix('.receipt.json').read_text())
        directory.mkdir(parents=True, exist_ok=True)
        index = {'identity':identity, 'inputs':base_receipt['inputs'], 'files':{},
                 'parent_receipt':str(base_path.with_suffix('.receipt.json')),
                 'parent_receipt_sha256':h.digest(base_path.with_suffix('.receipt.json'))}
        def save(name, value):
            path = directory / name
            if path.exists():
                raise ValueError('Incomplete streaming cache retained; investigate before replacing: '+str(path))
            with path.open('xb') as stream:
                pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            index['files'][name] = h.digest(path)
        save('static.pickle', {k:payload[k] for k in ('symbols','sectors','panel','exclusions')})
        conn = connect({'database':str(database)})
        try:
            for day in output_days:
                snapshot = payload['research'].pop(day)
                daily_panel = research._sector_daily_panel_from_symbols(snapshot.symbols, day)
                bars = day_bars(conn, day, payload['symbols'])
                feature_bars = complete_feature_bars(
                    {(day,s):bars.get((day,s),()) for s in payload['sectors']}, AFTERNOON_CUTOFF)[0] if cfg.require_complete_feature_prefix else bars
                intraday_panel = build_sector_intraday_panel(feature_bars, payload['sectors'],
                                                           trade_dates=(day,), cutoff=AFTERNOON_CUTOFF)
                save(day.isoformat()+'.pickle', (snapshot, daily_panel, intraday_panel, bars))
        finally:
            conn.close()
        del payload, snapshot, daily_panel, intraday_panel, bars, feature_bars
        index['index_signature'] = h.signature(index)
        temporary = index_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(index, sort_keys=True, indent=2), encoding='utf-8')
        temporary.replace(index_path)
        profile['cache_preparation_seconds'] = time.perf_counter()-preparation_start
        print(json.dumps({'stage':'streaming_cache_prepared','seconds':profile['cache_preparation_seconds']}),flush=True)

    index = json.loads(index_path.read_text(encoding='utf-8'))
    signature = index.pop('index_signature')
    if signature != h.signature(index) or index['identity'] != identity:
        raise ValueError('Streaming cache index changed')
    expected_names = {'static.pickle', *(d.isoformat()+'.pickle' for d in output_days)}
    if set(index['files']) != expected_names:
        raise ValueError('Streaming cache changed session coverage')
    for name, expected in index['inputs'].items():
        path = generation.root / name
        if generation.digest(path) != expected or h.digest(path) != expected:
            raise ValueError('Streaming cache source bytes changed: '+name)
    static = read_payload(directory/'static.pickle', index['files']['static.pickle'], h.digest)
    profile['authentication_and_preparation_seconds'] = time.perf_counter()-started
    print(json.dumps({'stage':'streaming_inputs_ready','seconds':profile['authentication_and_preparation_seconds']}),flush=True)
    dataset = SimpleNamespace(config=request['config'], trading_dates=days, sector_map=static['sectors'],
                              sector_intraday_panel_1430=static['panel'])
    result = {}
    selection_start = time.perf_counter()
    for day in output_days:
        filename = day.isoformat()+'.pickle'
        snapshot, daily_panel, intraday_panel, bars = read_payload(directory/filename, index['files'][filename], h.digest)
        if snapshot.trade_date != day:
            raise ValueError('Streaming research date changed')
        with native_panel_cache(research, day, snapshot, daily_panel, intraday_panel, static['sectors'], AFTERNOON_CUTOFF):
            morning = native.snapshots_for_experiment(dataset, request['mutations'], research_snapshots={day:snapshot})
            result[day] = research.afternoon_selection_from_snapshot(morning[day], bars, cfg, static['sectors'])
        if cfg.require_complete_feature_prefix:
            excluded = [x.split(':')[0] for x in static['exclusions'] if x.endswith(':'+day.isoformat())]
            if set(excluded) & {c.symbol for c in result[day].candidates}:
                raise ValueError('Incomplete feature prefix admitted by frozen OLR rule')
            result[day].metadata.update(feature_prefix_policy=POLICY, feature_prefix_exclusions=excluded)
    profile['candidate_selection_seconds'] = time.perf_counter()-selection_start
    profile['total_seconds'] = time.perf_counter()-started
    print(json.dumps({'stage':'streaming_selection','seconds':profile['candidate_selection_seconds']}),flush=True)
    return result
