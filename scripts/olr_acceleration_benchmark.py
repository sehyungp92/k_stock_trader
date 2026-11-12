"""Compare accelerated reconstruction to preserved native outputs, without replay dispatch."""
import argparse
from datetime import date
import importlib.util
import os
from pathlib import Path
import sys
import time
import psutil


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--window', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--prefix-cache', action='store_true')
    args = parser.parse_args()
    n = module('olr_benchmark_native', Path(__file__).with_name('olr_round1_comprehensive_study.py'))
    acceleration = module('olr_benchmark_acceleration', Path(__file__).with_name(
        'olr_snapshot_streaming.py' if args.streaming else 'olr_snapshot_acceleration.py'))
    info = n.read(n.generation_folder(args.root, args.window) / 'ready.json')
    study, engine = info['study'], Path(info['files'])
    sys.path.insert(0, str(engine))
    os.chdir(engine)
    from backtests.auto.shared.cache_keys import source_generation_hash
    if source_generation_hash() != study['source_generation_hash']:
        raise ValueError('Frozen engine changed')
    registration = args.root / 'registrations' / args.window / (args.candidate + '_100000000')
    case_study = n.read(registration / 'study.json')
    request = n.read(engine / case_study['olr_request'])
    reference_path = engine / case_study['olr_snapshots']
    # The archived IS control has legacy provenance/outcome annotations. The
    # authenticated pre-existing causal builder bridge is the full-metadata
    # reference for reconstruction; native replay separately bridges economics
    # to the archived selected control. Never drop arbitrary metadata to pass.
    if args.candidate == n.read(args.root/'candidate_plan.json')['baseline_id']:
        bridge = args.root/'builder_bridge'/args.window
        receipt = n.read(bridge/'snapshots.receipt.json')
        candidate = n.read(registration/'candidate.json')
        if (receipt['snapshots_sha256'] != n.sha(bridge/'snapshots.json')
                or receipt['source_hash'] != study['source_generation_hash']
                or receipt['candidate_hash'] != candidate['mutations_hash']
                or not n.read(bridge/'bridge.json')['pass']):
            raise ValueError('Original native reconstruction bridge is not authenticated')
        reference_path = bridge/'snapshots.json'
    expected = n.read(reference_path)
    days = [date.fromisoformat(row['trade_date']) for row in expected]
    context = n.reconstruction_days(engine / study['database'], days)
    profile = {'candidate': args.candidate, 'window': args.window, 'label': args.label,
               'reference_path': str(reference_path), 'reference_sha256': n.sha(reference_path)}
    from backtests.strategies.olr import research_sweep
    from strategy_olr import research
    def timed(function, key):
        def wrapped(*a, **kw):
            began = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                profile[key] = profile.get(key, 0) + time.perf_counter() - began
        return wrapped
    research_sweep.snapshots_for_experiment = timed(research_sweep.snapshots_for_experiment, 'morning_selection_seconds')
    research.afternoon_selection_from_snapshot = timed(research.afternoon_selection_from_snapshot, 'afternoon_selection_seconds')
    research._sector_daily_panel_from_symbols = timed(research._sector_daily_panel_from_symbols, 'uncached_daily_panel_seconds')
    research.build_sector_intraday_panel = timed(research.build_sector_intraday_panel, 'uncached_afternoon_panel_seconds')
    start = time.perf_counter()
    from contextlib import nullcontext
    prefix_counts = {}
    prefix = module('olr_benchmark_prefix', Path(__file__).with_name('olr_prefix_memo.py')) if args.prefix_cache else None
    cache_root = (args.cache_root or args.root / 'invariant_cache') / args.window
    with (prefix.cached_prefix(cache_root/'prefix', study['source_generation_hash'],prefix_counts) if prefix else nullcontext()):
        result = acceleration.build(study['manifest'], engine / study['database'], request, context,
            output_days=days, cache_root=cache_root,
            source_hash=study['source_generation_hash'], profile=profile)
    profile['prefix_cache'] = prefix_counts
    actual = [result[d].to_json_dict() for d in sorted(result)]
    # Creation time is the sole nondeterministic field. Artifact hashes, full
    # metadata, candidate order and all rejection records must match exactly.
    normalize = lambda rows: [{k:v for k,v in row.items() if k != 'generated_at'} for row in rows]
    profile.update(elapsed_seconds=time.perf_counter() - start,
                   parity_pass=normalize(actual) == normalize(expected),
                   expected_semantic_hash=n.sig(normalize(expected)), actual_semantic_hash=n.sig(normalize(actual)),
                   sessions=len(actual), ignored_fields=['generated_at'],
                   process_memory=psutil.Process().memory_info()._asdict())
    out = args.root / 'revisions/revision_18_reconstruction_cache/benchmarks'
    n.write(out / (args.label + '.snapshots.json'), actual)
    n.write(out / (args.label + '.json'), profile)
    print(profile, flush=True)
    if not profile['parity_pass']:
        raise ValueError('Full reconstruction parity failed')


if __name__ == '__main__':
    main()
