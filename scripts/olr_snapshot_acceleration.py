"""Authenticated invariant-feature cache for the OLR study's frozen builder.

This is orchestration, not a replacement strategy or performance evaluator.
All research, selection and sector calculations call the frozen native code.
Warm-up bars still build the full rolling panel. Only research/selection outputs
which the caller discards are omitted. Every trial still receives native replay.
"""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def cache_identity(manifest_hash, source_hash, database_hash, days, output_days, lookback, complete):
    return {'version': 1, 'implementation_sha256': digest(__file__),
            'manifest_hash': manifest_hash, 'source_hash': source_hash,
            'database_sha256': database_hash, 'context_dates': list(map(str, days)),
            'output_dates': list(map(str, output_days)), 'daily_lookback': lookback,
            'require_complete_feature_prefix': complete}


def load_cache(path, identity, generation):
    """Only load our local pickle after authenticating its bytes and source inputs."""
    receipt_path = path.with_suffix('.receipt.json')
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.exists() or not receipt_path.exists():
        raise ValueError('Incomplete invariant cache; retain and investigate: ' + str(path))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['identity'] != identity or receipt['sha256'] != digest(path):
        raise ValueError('Invariant cache authentication failed')
    for name, expected in receipt['inputs'].items():
        source = generation.root / name
        if generation.digest(source) != expected or digest(source) != expected:
            raise ValueError('Invariant cache input bytes changed: ' + name)
    with path.open('rb') as stream:
        return pickle.load(stream)


def save_cache(path, identity, generation, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with temporary.open('xb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    receipt = {'identity': identity, 'sha256': digest(path),
               'inputs': {name.replace('\\', '/'): generation.digest(generation.root / name)
                          for name in sorted(generation.loaded)}}
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding='utf-8')


def build(manifest, database, request, days, *, output_days, cache_root, source_hash, profile):
    import yaml
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr.config import OLRConfig
    from strategy_olr.research import afternoon_selection_from_snapshot
    from strategy_common.sector_intraday import build_sector_intraday_panel, AFTERNOON_CUTOFF, load_canonical_sector_map
    from strategy_common.source_quality import complete_feature_bars, POLICY
    from scripts.kalcb_executed_value_study import connect, load_day

    @contextmanager
    def stage(name):
        started = time.perf_counter()
        try:
            yield
        finally:
            profile[name + '_seconds'] = time.perf_counter() - started
            print(json.dumps({'stage': name, 'seconds': profile[name + '_seconds']}), flush=True)

    started = time.perf_counter()
    days, output_days = tuple(days), tuple(output_days)
    if len(set(days)) != len(days) or sorted(days) != list(days) or not set(output_days).issubset(days):
        raise ValueError('Invalid reconstruction context/output dates')
    generation = InputManifest(manifest)
    contract = generation.payload['evidence']['shared_market_inputs']
    cfg = OLRConfig.from_mapping(request['config'], request['mutations'])
    with stage('source_authentication'):
        source = digest(database)
        if generation.digest(database) != source:
            raise ValueError('Frozen intraday database bytes changed')
        identity = cache_identity(generation.identity, source_hash, source, days, output_days,
                                  max(100, int(cfg.research_min_history_days) + 5), cfg.require_complete_feature_prefix)
        path = Path(cache_root) / (signature(identity) + '.pickle')
        cached = load_cache(path, identity, generation)
    profile.update(cache_hit=cached is not None, cache=str(path), context_sessions=len(days),
                   output_sessions=len(output_days), daily_lookback=identity['daily_lookback'])

    if cached is None:
        with stage('daily_sources'):
            with generation.activate():
                symbols = tuple(str(s).zfill(6) for s in yaml.safe_load(generation.read(generation.root / contract['universe']))['symbols'])
                daily_root, end = generation.root / contract['daily_data_root'], max(days)
                daily = native._load_daily_rows(daily_root, symbols, end)
                flow = native._load_flow_rows(daily_root, symbols, end)
                foreign = native._load_foreign_flow_rows(daily_root, symbols, end)
                inst = native._load_institutional_flow_rows(daily_root, symbols, end)
                indices = native._load_index_rows(daily_root, end)
                sectors = load_canonical_sector_map({'sector_map_path': str(generation.root / contract['sector_map_path'])})
    else:
        symbols, sectors = cached['symbols'], cached['sectors']

    with stage('intraday_bars'):
        conn = connect({'database': str(database)})
        bars = defaultdict(list)
        try:
            # The cached panel already contains *all* warm-up observations.
            # Afternoon selection reads only the requested current-session bars.
            for day in (days if cached is None else ()):
                for bar in load_day(conn, str(day), source, set(symbols)):
                    bars[day, bar.symbol].append(bar)
        finally:
            conn.close()
        bars = {key: tuple(value) for key, value in bars.items()}
        if cached is not None:
            bars = None

    if cached is None:
        with stage('continuous_sector_panel'):
            feature_bars, exclusions = (complete_feature_bars(
                {(d, s): bars.get((d, s), ()) for d in days for s in symbols}, AFTERNOON_CUTOFF)
                if cfg.require_complete_feature_prefix else (bars, []))
            panel = build_sector_intraday_panel(feature_bars, sectors, trade_dates=days,
                                               cutoff=AFTERNOON_CUTOFF, symbols=symbols)
            del feature_bars
        with stage('research_features'):
            research_dataset = SimpleNamespace(config=request['config'], source_fingerprint=source,
                requested_symbols=symbols, symbols=symbols, excluded_symbols={}, daily_by_symbol=daily,
                flow_by_symbol=flow, foreign_flow_by_symbol=foreign, institutional_flow_by_symbol=inst,
                index_by_code=indices, sector_map=sectors, trading_dates=output_days)
            research = native.research_snapshots_for_dataset(research_dataset, request['mutations'])
            del research_dataset, daily, flow, foreign, inst, indices
        cached = {'symbols': symbols, 'sectors': sectors, 'panel': panel,
                  'exclusions': exclusions, 'research': research}
        with stage('cache_write'):
            save_cache(path, identity, generation, cached)

    with stage('candidate_selection'):
        # Keep the complete trading calendar for previous-session panel lookup.
        dataset = SimpleNamespace(config=request['config'], trading_dates=days, sector_map=sectors,
                                  sector_intraday_panel_1430=cached['panel'])
        morning = native.snapshots_for_experiment(dataset, request['mutations'], research_snapshots=cached['research'])
        result = {}
        # A warm cache supplies the full-history sector panel and daily research.
        # Native afternoon inference reads only its own date; stream those bars
        # instead of retaining every session's intraday objects simultaneously.
        conn = connect({'database': str(database)}) if bars is None else None
        try:
            for d, snap in morning.items():
                if conn is None:
                    day_bars = bars
                else:
                    day_bars = defaultdict(list)
                    for bar in load_day(conn, str(d), source, set(symbols)):
                        day_bars[d, bar.symbol].append(bar)
                    day_bars = {key: tuple(value) for key, value in day_bars.items()}
                result[d] = afternoon_selection_from_snapshot(snap, day_bars, cfg, sectors)
        finally:
            if conn is not None:
                conn.close()
        if cfg.require_complete_feature_prefix:
            for day, snapshot in result.items():
                excluded = [item.split(':')[0] for item in cached['exclusions'] if item.endswith(':' + day.isoformat())]
                if set(excluded) & {c.symbol for c in snapshot.candidates}:
                    raise ValueError('Incomplete feature prefix admitted by frozen OLR rule')
                snapshot.metadata.update(feature_prefix_policy=POLICY, feature_prefix_exclusions=excluded)
        if set(result) != set(output_days):
            raise ValueError('Accelerated builder changed output session coverage')
    profile['total_seconds'] = time.perf_counter() - started
    return result
