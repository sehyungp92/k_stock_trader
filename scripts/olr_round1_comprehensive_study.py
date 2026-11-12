"""Resumable native OLR lineage ablation, perturbation and targeted research.

All formerly OOS observations used here are development data. Does not publish
canonical rounds or trade. Every candidate retains costs and source audits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import date, datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009')
HOME_RESEARCH = Path.home() / '.codex/research'
OOS_STUDY = HOME_RESEARCH / 'portfolio_source_repair_20261009/oos/validation/study.json'
TRAIN_STUDY = REPO / 'data/backtests/output/olr/round_1/native_study.json'
RULES = 'olr.afternoon.score_band_rules'
PROTECTED = ('olr.cost.', 'olr.source.', 'olr.universe.complete_size', 'olr.execution.auction_fill_time')
GENERATION_DIRECTORY = 'generations_v2'
PREPARATION_LOCKS = {window: threading.RLock() for window in ('train', 'oos')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sig(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def event(root, stage, **values):
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **values}
    with (root / 'progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, default=str) + '\n')
    write(root / 'status.json', row)
    print(json.dumps(row, default=str), flush=True)


def flatten(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            if key != 'name':
                yield from flatten(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, (*prefix, index))
    else:
        yield prefix, value


def set_leaf(value, path, replacement, remove=False):
    result = copy.deepcopy(value)
    node = result
    for part in path[:-1]:
        node = node[part]
    if remove:
        if isinstance(node, list):
            del node[path[-1]]
        else:
            node.pop(path[-1], None)
    else:
        node[path[-1]] = replacement
    return result


def protected(path):
    return str(path[0]).startswith(PROTECTED)


def mutation_edits(before, after, path=()):
    """Leaf edits for interactions; a changed list length is one structural edit."""
    if isinstance(before, dict) and isinstance(after, dict):
        edits = []
        for key in sorted(set(before) | set(after)):
            if key == 'name':
                continue
            if key not in after:
                edits.append(((*path, key), None, True))
            elif key not in before:
                edits.append(((*path, key), copy.deepcopy(after[key]), False))
            else:
                edits.extend(mutation_edits(before[key], after[key], (*path, key)))
        return edits
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [edit for i, (a, b) in enumerate(zip(before, after))
                for edit in mutation_edits(a, b, (*path, i))]
    return [] if before == after else [(path, copy.deepcopy(after), False)]


def combine_disjoint(base, left, right):
    a, b = mutation_edits(base, left), mutation_edits(base, right)
    if not a or not b:
        return None
    if any(x[:min(len(x),len(y))] == y[:min(len(x),len(y))]
           for x,_,_ in a for y,_,_ in b):
        return None
    result = copy.deepcopy(base)
    for path, value, remove in [*a, *b]:
        result = set_leaf(result, path, value, remove)
    return result


def register_plan(root):
    """Each leaf removal, each historical leaf value, all cumulative endpoints."""
    current_path = REPO / 'data/backtests/output/olr/round_1/optimized_config.json'
    baseline = read(current_path)['mutations']
    archive = REPO / 'data/backtests/output/olr/archive/2026-10-05-before-fully-repaired-control'
    locations = [(f'legacy_round_{i}', archive / f'round_{i}') for i in range(1, 6)]
    locations += [('repaired_control', REPO / 'data/backtests/output/olr/archive/2026-10-08-before-conditional-capital-recycling/round_1'),
                  ('low325', REPO / 'data/backtests/output/olr/round_2'), ('current', current_path.parent)]
    lineage, snapshots = [], []
    for name, directory in locations:
        p = directory / 'optimized_config.json'
        record = read(p)
        item = {'name': name, 'path': str(p.resolve()), 'sha256': sha(p), 'mutations': record['mutations']}
        sp = directory / 'phase_state.json'
        if sp.exists():
            state = read(sp)
            item['phase_state'] = {'path': str(sp.resolve()), 'sha256': sha(sp)}
            item['accepted_phases'] = []
            for phase, result in sorted(state.get('phase_results', {}).items(), key=lambda p: str(p[0])):
                applied = result.get('applied_phase_mutations', True)
                if applied is False:
                    continue
                changes = applied if isinstance(applied, dict) else result.get('new_mutations', {})
                item['accepted_phases'].append({'phase': phase, 'mutations': changes,
                    'adoption_reason': result.get('adoption_reason'), 'accepted_count': result.get('accepted_count')})
                final = result.get('final_mutations')
                if final:
                    snapshots.append((name + '_phase_' + str(phase), final))
        lineage.append(item)
        snapshots.append((name, record['mutations']))
        target = root / 'lineage' / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ('optimized_config.json', 'phase_state.json', 'round_evaluation.txt'):
            source = directory / filename
            if source.exists():
                shutil.copyfile(source, target / filename)
    candidates, by_hash = [], {}

    def add(label, kind, mutations, reason, paths=None):
        # Safety/execution correctness changes are controls, not alpha mutations.
        mutations = copy.deepcopy(mutations)
        for key, value in baseline.items():
            if protected((key,)):
                mutations[key] = copy.deepcopy(value)
        key = sig(mutations)
        alias = {'label': label, 'kind': kind, 'reason': reason, 'paths': paths or []}
        if key in by_hash:
            by_hash[key]['aliases'].append(alias)
            return
        row = {'id': 'c' + key[:16], 'mutations_hash': key, 'mutations': mutations,
               **alias, 'aliases': []}
        candidates.append(row)
        by_hash[key] = row

    add('current_round1', 'baseline', baseline, 'Exact selected conditional recycling recipe')
    # Preserve chronological accepted mutation states, including superseded states.
    for label, mutations in snapshots:
        add('cumulative_' + label, 'cumulative', mutations, 'Historical mutation state replayed with current audited execution')
    coverage = []
    for path, value in flatten(baseline):
        if protected(path):
            coverage.append({'path': path, 'value': value, 'status': 'integrity_or_cost_control_preserved'})
            continue
        label = '/'.join(map(str, path))
        add('remove_' + label, 'leaf_ablation', set_leaf(baseline, path, None, True), 'Remove exactly one accepted leaf', [path])
        coverage.append({'path': path, 'value': value, 'status': 'individual_removal_and_perturbation'})
        if isinstance(value, bool):
            add('flip_' + label, 'leaf_perturbation', set_leaf(baseline, path, not value), 'Boolean alternative', [path])
        elif isinstance(value, (int, float)) and value != 0 and abs(value) < 1e10:
            for factor in (.8, 1.2):
                changed = value * factor
                if isinstance(value, int):
                    changed = max(0, round(changed)) if value >= 0 else round(changed)
                add(f'perturb_{label}_x{factor}', 'leaf_perturbation', set_leaf(baseline, path, changed), 'Local scalar sensitivity', [path])
        # Every unique historical value of every surviving leaf, without clusters.
        for historical, mutations in snapshots:
            old = dict(flatten(mutations)).get(path, '__absent__')
            if old != '__absent__' and old != value:
                add('restore_' + historical + '_' + label, 'historical_leaf_ablation',
                    set_leaf(baseline, path, old), 'Restore single historical accepted value', [path])
    # Revisit keys no longer in the current final map individually.
    for historical, mutations in snapshots:
        for key, value in mutations.items():
            if key not in baseline and not protected((key,)):
                add('superseded_' + historical + '_' + key, 'superseded_mutation',
                    {**baseline, key: value}, 'Previously accepted but absent from current recipe', [[key]])
        baseline_leaves = dict(flatten(baseline))
        for path, value in flatten(mutations):
            if path not in baseline_leaves and path[0] in baseline and not protected(path):
                # Nested historical entry/exit settings may be omitted from the
                # current serialized map even though their parents survive.
                try:
                    restored = set_leaf(baseline, path, value)
                except (KeyError, IndexError, TypeError):
                    coverage.append({'path':path,'value':value,'historical':historical,
                        'status':'structural_historical_branch_requires_explicit_review'})
                    continue
                add('historical_leaf_' + historical + '_' + '/'.join(map(str,path)),
                    'historical_leaf_ablation', restored,
                    'Restore one historical nested leaf absent from current serialization', [path])
    for index, rule in enumerate(baseline.get(RULES, [])):
        add(f'remove_band_{index}', 'rule_ablation', set_leaf(baseline, (RULES, index), None, True),
            'Remove a single score-band branch; individual leaves also tested', [[RULES, index]])
        for key in ('allowed_sectors', 'sector_admission'):
            if key in rule:
                add(f'remove_band_{index}_{key}', 'rule_component_ablation', set_leaf(baseline, (RULES, index, key), None, True),
                    'Remove one admission component', [[RULES, index, key]])
    add('no_score_bands', 'rule_ablation', {**baseline, RULES: []}, 'Unrestricted score coverage with all other rules fixed', [[RULES]])
    # Dependency-aware removals accompany invalid single toggles, never replace them.
    add('disable_recycling_and_topups', 'dependency_ablation', {**baseline,
        'capacity_recycle_liquidity_only': False, 'capacity_redistribute': False},
        'Top-up disable requires disabling its dependent recycling rule', [['capacity_redistribute'],['capacity_recycle_liquidity_only']])
    add('disable_joint_allocation_and_dependents', 'dependency_ablation', {**baseline,
        'capacity_joint_allocation': False, 'capacity_recycle_liquidity_only': False, 'capacity_redistribute': False},
        'Joint allocation disable requires disabling dependent top-ups', [['capacity_joint_allocation'],['capacity_redistribute'],['capacity_recycle_liquidity_only']])
    # Explicitly include zero-weight tests (deleting a key may activate a default).
    for key in baseline:
        if 'weight' in key and isinstance(baseline[key], (int, float)):
            add('zero_' + key, 'leaf_ablation', {**baseline, key: 0.0}, 'Zero one weight', [[key]])
    plan = {'version': 1, 'baseline_id': candidates[0]['id'], 'baseline': baseline,
            'lineage': lineage, 'coverage': coverage, 'candidates': candidates,
            'counts': dict(Counter(x['kind'] for x in candidates))}
    write(root / 'candidate_plan.json', plan)
    write(root / 'protocol.json', {
        'version': 1, 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'harness_sha256': sha(__file__), 'plan_sha256': sha(root / 'candidate_plan.json'),
        'reference_capital': 100_000_000, 'secondary_capitals': [10_000_000, 25_000_000, 50_000_000],
        'windows': {'train': ['2025-08-07', '2026-03-31'], 'oos': ['2026-06-01', '2026-07-10']},
        'oos_scope': 'All June-July results are now consumed development; no independent validation claim.',
        'other_history': 'April-May was already consumed. Primary matched study uses saved 156-session training and 29-session June-July contracts; no claim of continuous April-July coverage.',
        'phases': ['baseline reproduction and loss/concentration diagnostics', 'all cumulative history and individual leaf ablations/perturbations on BOTH windows',
                   'diagnostic-driven targeted risk/activity changes and disjoint top-12 interactions',
                   'top-5 stress, capital and no-upward-compounding verification', 'paired block uncertainty and complete results report'],
        'selection': {'is_terminal_wealth_floor_ratio': .97, 'is_return_max_drop_pp': 3,
                      'oos_return_min_gain_pp': 2, 'trade_floor_ratio_each_window': 1.0,
                      'drawdown_max_increase_pp': 2,
                      'ranking': 'Pareto return and entry frequency; tie by worst of training/OOS return gain'},
        'timeout_seconds_per_native_process': 172800, 'overall_timeout': None,
        'fixed_controls': ['source and calendar audits', 'causal feature cutoffs', 'costs', 'actual mark-to-market NAV', 'empty terminal inventory'],
        'promotion': 'research only, original round untouched; do not promise maximum expected future returns',
        'parent_studies': {'train': {'path': str(TRAIN_STUDY.resolve()), 'sha256': sha(TRAIN_STUDY)},
                           'oos': {'path': str(OOS_STUDY), 'sha256': sha(OOS_STUDY)}}})
    event(root, 'registered', candidates=len(candidates), counts=plan['counts'], lineage=len(lineage))


def source_hash(root):
    directories = ('backtests', 'strategy_common', 'strategy_kalcb', 'strategy_olr', 'strategy_pcim',
                   'deployment', 'instrumentation', 'oms', 'oms_client', 'kis_core', 'scripts')
    paths = [p for name in directories for p in (root / name).rglob('*.py')]
    paths.extend((root / 'kis_core/data').glob('*.yaml'))
    return sig([{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)} for p in sorted(paths)])


def generated_bytecode(name):
    path = Path(name)
    return '__pycache__' in path.parts or path.suffix.lower() in ('.pyc', '.pyo')


def generation_folder(root, window):
    protocol = read(root / 'protocol.json')
    directory = protocol.get('generation_directories', {}).get(window, GENERATION_DIRECTORY)
    return root / directory / window


def verify_window_contract(root, window, study, output=None):
    """A candidate ID alone must never authenticate a different evaluation window."""
    protocol = read(root / 'protocol.json')
    expected = protocol['windows'][window]
    actual = study['configuration']['date_range']
    if [actual['start'], actual['end']] != expected:
        raise ValueError(f'Window contract mismatch for {window}: {actual} != {expected}')
    if output is not None:
        daily = read(Path(output) / 'daily.json')
        days = [row['day'] for row in daily]
        if not days or days != sorted(set(days)) or [days[0], days[-1]] != expected:
            raise ValueError('Native account marks do not cover the registered window')
        result = read(Path(output) / 'result.json')
        if result['sessions'] != len(days):
            raise ValueError('Native session count differs from complete account marks')


def verify_market_bindings(manifest):
    contract = manifest['evidence']['shared_market_inputs']
    if sig({k:v for k,v in contract.items() if k != 'market_input_hash'}) != contract['market_input_hash']:
        raise ValueError('Changed parent market contract')
    mismatch = [name for name, expected in contract['files'].items() if manifest['files'].get(name) != expected]
    if mismatch:
        raise ValueError('Market input binding mismatch: ' + ', '.join(mismatch))


def retained_parser_mismatches(manifest, files):
    qualification = read(files / manifest['evidence']['qualification_path'])
    report_path = qualification.get('basis', {}).get('acceptance_report')
    if not report_path:
        return []
    acceptance = read(files / report_path)
    return [name for name, expected in acceptance['parser_files'].items()
            if manifest['files'].get(name) != expected]


def acceptance_worker(input_path, output):
    """Recompute source evidence after a parser revision, without altering observations."""
    info = read(input_path)
    files = Path(info['files'])
    os.chdir(files)
    sys.path.insert(0, str(files))
    from backtests.engine.retained_source import build_evidence
    manifest = info['manifest']
    old = read(files / manifest['evidence']['qualification_path'])
    old_acceptance = read(files / old['basis']['acceptance_report'])
    destination = files / 'research/olr_comprehensive/source_acceptance_v4'
    qualification_path = build_evidence(files, manifest['evidence']['shared_market_inputs'], destination)
    new = read(qualification_path)
    acceptance = read(files / new['basis']['acceptance_report'])
    daily_equal = read(files / old['references']['daily_rows']) == read(files / new['references']['daily_rows'])
    failures_equal = old_acceptance['basis_failures'] == acceptance['basis_failures']
    if not daily_equal or not failures_equal:
        raise ValueError('Recomputed source evidence changed daily references or basis dispositions')
    for key, value in old.items():
        if key not in ('basis', 'references', 'version', 'source_files'):
            new[key] = value
    write(qualification_path, new)
    manifest['evidence']['qualification_path'] = qualification_path.relative_to(files).as_posix()
    manifest['evidence']['recomputed_retained_acceptance'] = {
        'old_acceptance_sha256': sha(files / old['basis']['acceptance_report']),
        'new_acceptance_sha256': sha(files / new['basis']['acceptance_report']),
        'daily_references_identical': daily_equal, 'basis_dispositions_identical': failures_equal,
        'observations_changed': False}
    for path in destination.glob('*.json'):
        manifest['files'][path.relative_to(files).as_posix()] = sha(path)
    if retained_parser_mismatches(manifest, files):
        raise ValueError('Recomputed source evidence remains unbound')
    write(output, manifest)


def prepare_window(root, window):
    destination = generation_folder(root, window)
    if (destination / 'ready.json').exists():
        info = read(destination / 'ready.json')
        verify_window_contract(root, window, info['study'])
        return info
    protocol = read(root / 'protocol.json')
    parent = protocol['parent_studies'][window]
    if sha(parent['path']) != parent['sha256']:
        raise ValueError('Changed parent study')
    study = read(parent['path'])
    verify_window_contract(root, window, study)
    manifest = read(study['manifest'])
    frozen = Path(study['code'])
    # Identical execution implementation for both periods. Historical files stay separate.
    source_reference = protocol.get('common_python_study', protocol['parent_studies']['oos'])
    if sha(source_reference['path']) != source_reference['sha256']:
        raise ValueError('Changed common Python source study')
    source_study = read(source_reference['path'])
    source_manifest = read(source_study['manifest'])
    source_root = Path(source_study['code'])
    files = destination / 'inputs/files'
    entries = dict(manifest['files'])
    code_names = {n for n in source_manifest['files'] if n.endswith('.py')}
    # Calendar and halt references are data: keep each period's registered bytes.
    # Only Python implementation is common between periods.
    names = ((set(entries) - {n for n in entries if n.endswith('.py')}) | code_names)
    # Bytecode is a mutable interpreter cache, not a source/data input. Do not
    # copy it: -B prevents writes but still permits reads from an existing cache.
    excluded = sorted(n for n in names if generated_bytecode(n))
    names -= set(excluded)
    actual = {}
    for index, name in enumerate(sorted(names)):
        src = (source_root if name in code_names else frozen) / name
        expected = (source_manifest if name in code_names else manifest)['files'][name]
        target = files / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # Never write through a hardlink; all derived files have unique paths.
            previous_directory = protocol.get('reusable_generation_directories', {}).get(window, 'generations_v2')
            previous = root / previous_directory / window / 'inputs/files' / name
            reusable = previous.exists() and not name.startswith('kis_core/data/')
            link_source = previous if reusable else src
            try:
                os.link(link_source, target)
            except OSError:
                shutil.copyfile(src, target)
        if sha(target) != expected:
            raise ValueError('Frozen file changed: ' + name)
        actual[name] = expected
        if index % 1000 == 0:
            event(root, 'freezing_inputs', window=window, files_done=index, files_total=len(names))
    common_hash = source_hash(files)
    if any(actual[name] != source_manifest['files'][name] for name in code_names):
        raise ValueError('Python execution implementation differs between periods')
    if window == 'oos' and 'common_python_study' not in protocol and common_hash != source_manifest['source_generation_hash']:
        raise ValueError('OOS frozen source does not reproduce its identity')
    manifest.update(files=actual, source_generation_hash=common_hash)
    manifest.pop('manifest_hash', None)
    manifest['evidence'] = {**manifest['evidence'], 'comprehensive_olr_study': {'protocol': sha(root / 'protocol.json'),
        'window': window, 'parent_manifest': sha(study['manifest']), 'current_engine': common_hash,
        'python_implementation_hash': sig({n:actual[n] for n in sorted(code_names)}),
        'reference_policy': 'Retain each period parent calendar and halt reference bytes; common Python implementation'}}
    verify_market_bindings(manifest)
    mismatches = retained_parser_mismatches(manifest, files)
    if mismatches:
        event(root, 'recomputing_source_acceptance', window=window, parser_mismatches=mismatches)
        acceptance_input = destination / 'acceptance_input.json'
        acceptance_output = destination / 'acceptance_manifest.json'
        write(acceptance_input, {'files': str(files), 'manifest': manifest})
        rc = run_child([sys.executable, '-B', '-u', str(Path(__file__)), 'acceptance',
                        '--input', str(acceptance_input), '--output', str(acceptance_output)],
                       files, destination / 'acceptance.log')
        if rc:
            raise ValueError('Source evidence recomputation failed: ' + str(destination / 'acceptance.log'))
        manifest = read(acceptance_output)
        verify_market_bindings(manifest)
    mp = destination / 'inputs/manifest.json'
    write(mp, {**manifest, 'manifest_hash': sig(manifest)})
    study.update(code=str(files), manifest=str(mp), source_generation_hash=common_hash)
    info = {'study': study, 'manifest': manifest, 'files': str(files), 'excluded_generated_bytecode': excluded}
    write(destination / 'ready.json', info)
    event(root, 'inputs_ready', window=window, source_hash=common_hash)
    return info


def run_child(command, cwd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    if logfile.exists():
        shutil.copy2(logfile, logfile.with_name(logfile.name + '.attempt_' + str(time.time_ns())))
    with logfile.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        try:
            return proc.wait(timeout=172800)
        except subprocess.TimeoutExpired:
            # Only terminate this child; the supervisor preserves failure and continues.
            proc.terminate()
            proc.wait(timeout=60)
            raise


def reconstruction_days(database, requested):
    """Carry all available pre-window history across the IS/OOS scoring boundary.

    Sector cutoff-volume normalization depends on prior complete observations,
    not just yesterday. The scoring boundary must not reset that feature state.
    Output entry dates remain restricted to requested by snapshot_worker.
    """
    import sqlite3
    with sqlite3.connect('file:' + Path(database).as_posix() + '?mode=ro', uri=True) as connection:
        previous = connection.execute('SELECT DISTINCT day FROM bars WHERE day < ? ORDER BY day',
                                      (min(requested).isoformat(),)).fetchall()
    return [date.fromisoformat(row[0]) for row in previous] + list(requested)


def snapshot_worker(root, window, candidate_file, output):
    info = read(generation_folder(root, window) / 'ready.json')
    engine = Path(info['files'])
    sys.path.insert(0, str(engine))
    os.chdir(engine)
    from scripts.portfolio_forward_olr import build
    from backtests.auto.shared.cache_keys import source_generation_hash
    study = info['study']
    if source_generation_hash() != study['source_generation_hash']:
        raise ValueError('Frozen builder changed')
    candidate = read(candidate_file)
    request = read(engine / study['olr_request'])
    request['mutations'] = {k: v for k, v in candidate['mutations'].items() if not k.startswith('capacity_')}
    request['config'].update(initial_mutations=request['mutations'], initial_equity=100_000_000,
        date_range=study['configuration']['date_range'])
    original = read(engine / study['olr_snapshots'])
    days = [date.fromisoformat(row['trade_date']) for row in original]
    context_days = reconstruction_days(engine / study['database'], days)
    acceleration = read(root / 'mechanism_plan.json').get('snapshot_acceleration') if (root / 'mechanism_plan.json').exists() else None
    profile = None
    if acceleration:
        import importlib.util
        if acceleration['validated_source_hashes'].get(window) != study['source_generation_hash']:
            raise ValueError('Snapshot acceleration was not validated for this frozen source contract')
        module_path = Path(__file__).with_name(acceleration.get('module', 'olr_snapshot_acceleration.py'))
        if sha(module_path) != acceleration['implementation_sha256']:
            raise ValueError('Registered snapshot acceleration changed')
        for name, expected in acceleration.get('dependencies', {}).items():
            if sha(Path(__file__).with_name(name)) != expected:
                raise ValueError('Registered snapshot acceleration dependency changed: ' + name)
        spec = importlib.util.spec_from_file_location('olr_snapshot_acceleration', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        profile = {}
        from contextlib import nullcontext
        prefix_counts = {}
        cache_root = Path(acceleration['cache_root']) / window
        prefix = None
        if acceleration.get('prefix_memo'):
            prefix_path = Path(__file__).with_name('olr_prefix_memo.py')
            if sha(prefix_path) != acceleration['dependencies'].get(prefix_path.name):
                raise ValueError('Registered native prefix memo changed')
            prefix_spec = importlib.util.spec_from_file_location('olr_prefix_memo', prefix_path)
            prefix = importlib.util.module_from_spec(prefix_spec)
            prefix_spec.loader.exec_module(prefix)
        with (prefix.cached_prefix(cache_root/'prefix', study['source_generation_hash'], prefix_counts) if prefix else nullcontext()):
            snapshots = module.build(study['manifest'], engine / study['database'], request, context_days,
                output_days=days, cache_root=cache_root,
                source_hash=study['source_generation_hash'], profile=profile)
        profile['prefix_cache'] = prefix_counts
    else:
        snapshots = build(study['manifest'], engine / study['database'], request, context_days)
    rows = [snapshots[d].to_json_dict() for d in sorted(snapshots) if d in set(days)]
    write(output, rows)
    # Daily no-entry dates are represented by empty snapshots, never dropped.
    if sorted(row['trade_date'] for row in rows) != sorted(row['trade_date'] for row in original):
        raise ValueError('Candidate changed declared session coverage')
    write(output.with_suffix('.receipt.json'), {'snapshots_sha256': sha(output), 'candidate_hash': candidate['mutations_hash'],
        'source_hash': source_generation_hash(), 'sessions': len(rows), 'candidate_count': sum(len(r['candidates']) for r in rows),
        'builder_harness_sha256':sha(__file__), 'context_only_dates':[str(d) for d in context_days if d not in set(days)],
        **({'acceleration_profile': profile, 'acceleration_sha256': acceleration['implementation_sha256']} if acceleration else {})})


def candidate_preflight(root, window, candidate):
    """Reject invalid parameter combinations before costly reconstruction."""
    from dataclasses import asdict, replace
    info = read(generation_folder(root, window) / 'ready.json')
    sys.path.insert(0, info['files'])
    from strategy_olr.config import OLRConfig
    request = read(Path(info['files']) / info['study']['olr_request'])
    pure = {k:v for k,v in candidate['mutations'].items() if not k.startswith('capacity_')}
    request['config']['initial_mutations'] = pure
    config = OLRConfig.from_mapping(request['config'], pure)
    overrides = {k:v for k,v in candidate['mutations'].items() if k.startswith('capacity_')}
    config = replace(config, **overrides)
    return asdict(config)


def capital_registration(qualification, capital, parent_sha256):
    """Declare new research account sizes without weakening capacity validation."""
    result = copy.deepcopy(qualification)
    registrations = result['capacity']['registered_execution_policies']
    templates = [r for r in registrations if r.get('account') == 'shared_current_20261007'
                 and r.get('initial_equity') == 100_000_000]
    if not templates:
        raise ValueError('Missing exact selected-account policy registration templates')
    added = []
    for template in templates:
        row = {**copy.deepcopy(template), 'initial_equity': capital,
               'account': 'olr_mechanism_research_' + str(capital)}
        identity = lambda r: (r['initial_equity'], r['buying_power_leverage'], sig(r['policy']))
        matches = [r for r in registrations if identity(r) == identity(row)]
        if len(matches) > 1:
            raise ValueError('Duplicate research account policy registrations')
        if not matches:
            registrations.append(row)
            added.append(sig(row))
    result['research_account_registration_extension'] = {
        'parent_sha256': parent_sha256, 'capital': capital, 'added_registration_hashes': added,
        'scope': 'Declared research capacity model only; original market basis, source evidence, costs, leverage and audit requirements preserved.'}
    return result


def prepare_case(root, window, candidate, capital):
    # One reconstruction per window avoids racing
    # writers when distinct capacity trials share the same pure snapshot key.
    # Native replay/auditing runs outside this lock and can proceed concurrently.
    with PREPARATION_LOCKS[window]:
        return _prepare_case(root, window, candidate, capital)


def _prepare_case(root, window, candidate, capital):
    info = read(generation_folder(root, window) / 'ready.json')
    study, manifest = copy.deepcopy(info['study']), copy.deepcopy(info['manifest'])
    engine = Path(info['files'])
    cid = candidate['id']
    key = cid + '_' + str(capital)
    case_root = root / 'registrations' / window / key
    registered = case_root / 'study.json'
    if registered.exists():
        verify_window_contract(root, window, read(registered))
        return registered
    case_root.mkdir(parents=True, exist_ok=True)
    write(case_root / 'candidate.json', candidate)
    write(case_root / 'effective_config.json', candidate_preflight(root, window, candidate))
    baseline = read(root / 'candidate_plan.json')['baseline']
    pure = lambda m: {k: v for k, v in m.items() if not k.startswith('capacity_')}
    source_key = sig(pure(candidate['mutations']))[:20]
    snapshots = engine / 'research/olr_comprehensive/snapshots' / (source_key + '.json')
    snapshots.parent.mkdir(parents=True, exist_ok=True)
    if snapshots.exists() and pure(candidate['mutations']) != pure(baseline):
        receipt_path = snapshots.with_suffix('.receipt.json')
        if not receipt_path.exists():
            # A killed writer can leave bytes without a completion receipt.
            # Preserve them and reconstruct instead of trusting partial input.
            snapshots.rename(snapshots.with_name(snapshots.name + '.incomplete_' + str(time.time_ns())))
        else:
            receipt = read(receipt_path)
            if receipt['snapshots_sha256'] != sha(snapshots) or receipt['source_hash'] != study['source_generation_hash']:
                raise ValueError('Candidate snapshot cache authentication failed: ' + str(snapshots))
    if not snapshots.exists():
        if pure(candidate['mutations']) == pure(baseline):
            shutil.copyfile(engine / study['olr_snapshots'], snapshots)
        else:
            rc = run_child([sys.executable, '-B', '-u', str(Path(__file__)), 'snapshots', '--root', str(root),
                '--window', window, '--candidate', str(case_root / 'candidate.json'), '--output', str(snapshots)],
                engine, case_root / 'snapshot.log')
            if rc:
                raise RuntimeError('Candidate reconstruction failed: ' + str(case_root / 'snapshot.log'))
    request = read(engine / study['olr_request'])
    request['mutations'] = pure(candidate['mutations'])
    request['config'].update(initial_mutations=request['mutations'], initial_equity=capital,
        date_range=study['configuration']['date_range'])
    prefix = 'research/olr_comprehensive/cases/' + key + '/'
    if capital != 100_000_000:
        parent_qualification = engine / manifest['evidence']['qualification_path']
        qualified = capital_registration(read(parent_qualification), capital, sha(parent_qualification))
        qualification_name = 'research/olr_comprehensive/capital_contracts/' + str(capital) + '/qualification.json'
        qualification_path = engine / qualification_name
        if qualification_path.exists() and read(qualification_path) != qualified:
            raise ValueError('Changed registered capital contract')
        write(qualification_path, qualified)
        manifest['evidence']['qualification_path'] = qualification_name
        manifest['files'][qualification_name] = sha(qualification_path)
        write(case_root/'capital_registration.json', qualified['research_account_registration_extension'])
    request_path = engine / (prefix + 'request.json')
    write(request_path, request)
    parent_path = read(root/'protocol.json')['parent_studies']['train']['path']
    recipe = copy.deepcopy(read(parent_path)['candidates']['native_entry_recycle2'])
    recipe['olr_capacity'] = {k: v for k, v in candidate['mutations'].items() if k.startswith('capacity_')}
    study.update(initial_equity=capital, olr_request=prefix + 'request.json',
        olr_snapshots=snapshots.relative_to(engine).as_posix(), candidates={'candidate': recipe})
    study['configuration'].update(initial_equity=capital)
    spec_path = prefix + 'spec.json'
    mp = engine.parent / (key + '.manifest.json')
    study.update(manifest=str(mp), spec_path=spec_path)
    study['configuration']['input_manifest_path'] = str(mp)
    spec = {k: v for k, v in study.items() if k not in ('manifest', 'code', 'spec_path', 'source_generation_hash')}
    write(engine / spec_path, spec)
    for name in (study['olr_request'], study['olr_snapshots'], spec_path):
        manifest['files'][name] = sha(engine / name)
    write(mp, {**manifest, 'manifest_hash': sig(manifest)})
    write(registered, study)
    return registered


def native_replay_registration(root, window, registered):
    """Versioned native replay acceleration, independent of snapshot caches."""
    plan = read(root/'mechanism_plan.json')
    acceleration = plan.get('native_replay_acceleration')
    if not acceleration:
        return registered
    from scripts import olr_native_acceleration as acceleration_module
    if sha(acceleration_module.__file__) != acceleration['implementation_sha256']:
        raise ValueError('Changed native replay acceleration helper')
    item = acceleration['generations'][window]
    if sha(item['path']) != item['sha256']:
        raise ValueError('Changed native replay generation registration')
    with PREPARATION_LOCKS[window]:
        return acceleration_module.prepare_replay(registered, item['path'],
                                                   registered.with_name('replay_study.json'))


def execution_registration(registered, source_hash):
    """Find the exact source-bound study for old or newly accelerated evidence."""
    original = read(registered)
    if original['source_generation_hash'] == source_hash:
        return original
    replay = registered.with_name('replay_study.json')
    if not replay.exists() or not registered.with_name('native_acceleration_receipt.json').exists():
        raise ValueError('Unregistered native replay generation')
    receipt = read(registered.with_name('native_acceleration_receipt.json'))
    if (receipt['original_study_sha256'] != sha(registered)
            or receipt['replay_study_sha256'] != sha(replay)
            or receipt['generation_sha256'] != sha(receipt['generation'])
            or receipt['replay_source_hash'] != source_hash):
        raise ValueError('Changed native replay registration receipt')
    study = read(replay)
    if study['source_generation_hash'] != source_hash:
        raise ValueError('Native replay source identity mismatch')
    return study


def evaluate(root, candidate, window, capital=100_000_000, scenario='baseline', capped=False):
    out = root / 'cases' / candidate['id'] / window / (f'{capital}_{scenario}' + ('_capped' if capped else ''))
    if (out / 'assessment.json').exists():
        assessment = read(out / 'assessment.json')
        if assessment.get('qualified'):
            registered = root / 'registrations' / window / (candidate['id'] + '_' + str(capital)) / 'study.json'
            verify_window_contract(root, window, read(registered), out)
        return assessment
    try:
        registered = prepare_case(root, window, candidate, capital)
        registered = native_replay_registration(root, window, registered)
        study = read(registered)
        if out.exists() and not (out / 'completion.json').exists():
            # Preserve failed attempts; scoped move within this study only.
            attempt = out.with_name(out.name + '_attempt_' + str(time.time_ns()))
            assert out.resolve().is_relative_to(root.resolve()) and attempt.resolve().is_relative_to(root.resolve())
            out.rename(attempt)
        if not (out / 'completion.json').exists():
            cmd = [sys.executable, '-B', '-u', str(Path(study['code']) / 'scripts/portfolio_synergy_study.py'),
                   'replay', '--study', str(registered), '--recipe', 'candidate', '--scenario', scenario,
                   '--standalone', 'OLR', '--output', str(out), '--full-audit']
            if capped:
                cmd.append('--no-upward-compounding')
            rc = run_child(cmd, study['code'], out.with_suffix('.log'))
            if rc:
                raise RuntimeError('Native replay failed; see ' + str(out.with_suffix('.log')))
        completion = read(out / 'completion.json')
        result, audit = read(out / 'result.json'), read(out / 'execution_audit.json')
        verify_window_contract(root, window, study, out)
        if sha(out / 'result.json') != completion['result_sha256'] or sha(out / 'execution_audit.json') != completion['audit_sha256']:
            raise ValueError('Changed native output')
        valid = bool(result['execution_validity_pass'] and audit.get('audit_pass') and
            result['audit_status'] == 'passed' and not result['terminal_inventory'] and result['terminal_pending'] == 0)
        result.update(candidate_id=candidate['id'], label=candidate['label'], window=window,
                      capital=capital, capped=capped, qualified=valid, output=str(out), status='completed')
        write(out / 'assessment.json', result)
    except Exception as exc:
        result = {'candidate_id': candidate['id'], 'label': candidate['label'], 'window': window,
                  'capital': capital, 'scenario': scenario, 'capped': capped, 'qualified': False,
                  'status': 'failed', 'error': str(exc), 'output': str(out), 'traceback': traceback.format_exc()}
        out.mkdir(parents=True, exist_ok=True)
        write(out / 'assessment.json', result)
    return result


def concentration(path):
    trades = read(Path(path) / 'trades.json')
    net = [t['net_pnl'] for t in trades]
    losses = sorted((t for t in trades if t['net_pnl'] < 0), key=lambda t: t['net_pnl'])
    wins = sorted((t for t in trades if t['net_pnl'] > 0), key=lambda t: -t['net_pnl'])
    cohorts = {}
    for dimension in ('symbol', 'month', 'rank', 'sector', 'band', 'exit_reason', 'holding_days'):
        groups = defaultdict(list)
        for t in trades:
            meta = t.get('route_metadata') or {}
            value = {'symbol': t['symbol'], 'month': t['entry_fill_time'][:7],
                     'rank': meta.get('candidate_rank'), 'sector': meta.get('sector'),
                     'band': meta.get('afternoon_score_band_rule'), 'exit_reason': t['exit_reason'],
                     'holding_days': (date.fromisoformat(t['exit_fill_time'][:10])-date.fromisoformat(t['entry_fill_time'][:10])).days}[dimension]
            groups[str(value)].append(t['net_pnl'])
        cohorts[dimension] = {k: {'trades': len(v), 'win_rate': sum(p > 0 for p in v)/len(v),
            'net_pnl': sum(v), 'expectancy': sum(v)/len(v)} for k, v in groups.items()}
    return {'trades': len(trades), 'win_rate': sum(p > 0 for p in net)/len(net) if net else 0,
            'net_pnl': sum(net), 'cohorts': cohorts,
            'tail_diagnostics': [{'k': k, 'loss_share': -sum(t['net_pnl'] for t in losses[:k]) / max(1, -sum(t['net_pnl'] for t in losses)),
                'net_without_k_worst': sum(net)-sum(t['net_pnl'] for t in losses[:k]),
                'net_without_k_best': sum(net)-sum(t['net_pnl'] for t in wins[:k])} for k in (1, 2, 3, 5, 10)],
            'worst_trades': losses[:10], 'best_trades': wins[:10],
            'interpretation': 'Arithmetic concentration, not a counterfactual replay. Symbol cohorts are descriptive and cannot become blacklist mutations.'}


def results(root):
    rows = []
    for path in (root / 'cases').glob('*/*/100000000_baseline/assessment.json'):
        rows.append(read(path))
    return rows


def rank_pairs(root):
    plan = read(root / 'candidate_plan.json')
    pairs = defaultdict(dict)
    for row in results(root):
        if row.get('qualified'):
            pairs[row['candidate_id']][row['window']] = row
    base = pairs[plan['baseline_id']]
    eligible = []
    if set(base) != {'train', 'oos'}:
        return [], pairs
    for cid, pair in pairs.items():
        if set(pair) != {'train', 'oos'} or cid == plan['baseline_id']:
            continue
        deltas = {w: pair[w]['return_pct']-base[w]['return_pct'] for w in base}
        gates = {'is_wealth': (100+pair['train']['return_pct']) >= .97*(100+base['train']['return_pct']),
                 'is_return': deltas['train'] >= -3, 'oos_uplift': deltas['oos'] >= 2,
                 'activity': all(pair[w]['trades'] >= base[w]['trades'] for w in base),
                 'drawdown': all(pair[w]['drawdown_pct'] <= base[w]['drawdown_pct']+2 for w in base)}
        if all(gates.values()):
            eligible.append({'id': cid, 'gates': gates, 'deltas_pp': deltas,
                             'score': min(deltas.values()), 'pair': pair})
    def objectives(row):
        return tuple(row['pair'][w][key] for w in ('train','oos') for key in ('return_pct','trades'))
    for row in eligible:
        value = objectives(row)
        row['pareto_frontier'] = not any(all(a>=b for a,b in zip(objectives(other),value))
            and any(a>b for a,b in zip(objectives(other),value)) for other in eligible)
    eligible.sort(key=lambda r: (r['pareto_frontier'],r['score'],r['deltas_pp']['oos'],r['pair']['oos']['trades']), reverse=True)
    return eligible, pairs


def targeted_plan(root):
    """Generate a second phase after the baseline and complete ablation evidence."""
    plan = read(root / 'candidate_plan.json')
    leaders, pairs = rank_pairs(root)
    known = {x['id']: x for x in plan['candidates']}
    bases = [known[plan['baseline_id']], *(known[x['id']] for x in leaders[:3])]
    out, seen = [], {x['mutations_hash'] for x in plan['candidates']}

    def add(label, mutations, reason):
        key = sig(mutations)
        if key in seen:
            return
        seen.add(key)
        out.append({'id': 'c'+key[:16], 'label': label, 'mutations': mutations, 'mutations_hash': key,
                    'kind': 'targeted', 'reason': reason, 'aliases': []})

    diagnostics = read(root / 'baseline_diagnostics.json')
    weak = diagnostics['oos']['cohorts']
    # This grid addresses observed loss tails, weak holding exits and thin activity.
    # No symbols, dates or outcome-known feature values enter the policy.
    for parent in bases:
        base = parent['mutations']
        def leaf(path, values, reason):
            for value in values:
                m = copy.deepcopy(base)
                node = m
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = value
                add(parent['id']+'_'+str(path)+'_'+str(value), m, reason)
        for key, values in {'hard_stop_enabled': [True], 'stop_pct': [.015, .025, .04],
                            'mfe_fade_start_r': [.5, 1.5, 2], 'mfe_fade_gap_r': [.5, .75, 1, 1.5, 2],
                            'mfe_fade_floor_r': [.25, .5], 'max_hold_bars': [78, 156, 234],
                            'vwap_fail_bars': [3, 6], 'failed_followthrough_bars': [6, 12]}.items():
            leaf(('olr.trade_plan.exit', key), values, 'Observed tail/holding/exit cohorts: '+str(sorted(weak['exit_reason'])))
        # Coherent activation companions for parameters that are otherwise dormant.
        for stop in (.015, .025, .04):
            m = copy.deepcopy(base)
            m['olr.trade_plan.exit'].update(hard_stop_enabled=True, stop_mode='fixed_pct', stop_pct=stop)
            add(parent['id']+'_hard_stop_pct_'+str(stop), m, 'Causal stop rule, native next executable fill; overnight gaps remain possible')
        for path, values in [(('olr.trade_plan.entry','max_signal_bars'), [3, 9, 12]),
                             (('olr.trade_plan.entry','min_close_location'), [.3,.4,.6]),
                             (('olr.trade_plan.entry','max_vwap_extension_pct'), [.015,.04,.06]),
                             (('olr.afternoon.top_n',), [5,10,12,16]),
                             (('olr.research.top_long_count',), [15,25,30,40]),
                             (('olr.allocation.rank_decay',), [0,.5,1,2]),
                             (('olr.allocation.max_position_pct',), [.25,.35,.5]),
                             (('olr.overnight.slot_count',), [3,4,6,8]),
                             (('capacity_redistribute_max_rank',), [3,5]),
                             (('capacity_redistribution_max_multiplier',), [0,2]),
                             (('capacity_stale_release',), [True])]:
            leaf(path, values, 'Target concentration and opportunity throughput while preserving native fills')
        # These bounded hypotheses are grounded in the saved opportunity funnel
        # and the observed post-entry loss paths, not symbol/date exclusions.
        mechanism_path = root/'mechanism_followup_plan.json'
        if mechanism_path.exists():
            for hypothesis in read(mechanism_path).get('bounded_parameter_hypotheses', []):
                leaf(tuple(hypothesis['path']), hypothesis['values'], hypothesis['reason'])
    # Include complementary components that fail a final gate individually.
    # Their combinations still face the original paired acceptance criteria.
    control = pairs[plan['baseline_id']]
    components = []
    for cid, pair in pairs.items():
        if cid == plan['baseline_id'] or set(pair) != {'train','oos'}:
            continue
        gain = {w:pair[w]['return_pct']-control[w]['return_pct'] for w in control}
        frequency = min(pair[w]['trades']/control[w]['trades'] for w in control)
        if gain['train'] >= -10 and gain['oos'] >= -2 and (gain['oos'] > 0 or frequency > 1):
            components.append({'id':cid,'gain':gain,'frequency':frequency})
    risk = sorted(components,key=lambda r:(r['gain']['oos'],r['gain']['train']),reverse=True)[:4]
    activity = sorted(components,key=lambda r:(r['frequency'],r['gain']['oos']),reverse=True)[:4]
    ids = list(dict.fromkeys([r['id'] for r in leaders[:6]] + [r['id'] for r in risk+activity]))
    top = [known[cid] for cid in ids]
    base = plan['baseline']
    for a,b in itertools.combinations(top,2):
        m = combine_disjoint(base,a['mutations'],b['mutations'])
        if m is None:
            continue
        add('interaction_'+a['id']+'_'+b['id'], m, 'Disjoint pair; parents '+a['id']+' '+b['id'])
    write(root / 'targeted_plan.json', {'baseline_diagnostics_sha256': sha(root/'baseline_diagnostics.json'),
        'ablation_results_sha256': sha(root/'screen_results.json'), 'candidates': out,
        'interaction_components':ids, 'complementary_component_screen':{'is_gain_floor_pp':-10,'oos_gain_floor_pp':-2,
            'final_acceptance_gates_unchanged':True,'selection':'top six gate passers plus four OOS and four frequency components'},
        'selection_bias': 'Adaptive OOS-informed development; requires later independent evaluation'})
    return out


def batch(root, candidates, workers, phase):
    # One worker per window avoids duplicate writes into a shared snapshot cache.
    def window_batch(window):
        for index, candidate in enumerate(candidates):
            value = evaluate(root, candidate, window)
            print(json.dumps({'stage':phase,'window':window,'index':index+1,'total':len(candidates),
                              'candidate':candidate['id'],'status':value['status'],'qualified':value['qualified']}, default=str), flush=True)
            write(root / ('progress_' + window + '.json'), {'phase':phase,'index':index+1,'total':len(candidates),
                'candidate':candidate['id'],'status':value['status'],'utc':datetime.now(timezone.utc).isoformat()})
    with ThreadPoolExecutor(max_workers=min(workers,2)) as pool:
        futures = [pool.submit(window_batch, window) for window in ('oos','train')]
        for future in as_completed(futures):
            future.result()
    write(root / (phase + '_results.json'), results(root))


def candidate_effects(root):
    """Distinguish serialized trials from real configuration, selection and fill changes."""
    plan = read(root/'candidate_plan.json')
    cache_path = root/'candidate_effect_fingerprints.json'
    cache = read(cache_path) if cache_path.exists() else {}
    rows = results(root)
    snapshots_cache = {}
    fingerprints = {}
    for row in rows:
        if row.get('status') != 'completed':
            continue
        cid, window = row['candidate_id'], row['window']
        case = root/'registrations'/window/(cid+'_100000000')
        output = Path(row['output'])
        if not (case/'effective_config.json').exists():
            continue
        key = cid+'/'+window
        identity = sig([sha(case/'study.json'), sha(case/'effective_config.json'), sha(output/'result.json')])
        if key not in cache or cache[key]['identity'] != identity:
            study = read(case/'study.json')
            daily = read(output/'daily.json')
            active_days = {r['day'] for r in daily}
            snapshot_path = str(Path(study['code'])/study['olr_snapshots'])
            if snapshot_path not in snapshots_cache:
                snapshots = [s for s in read(snapshot_path) if s['trade_date'] in active_days]
                snapshots_cache[snapshot_path] = {
                    'selection_hash': sig([(s['trade_date'],[(c['symbol'],c['rank'],c['selection_score'])
                        for c in s['candidates']]) for s in snapshots]),
                    'candidate_rows': sum(len(s['candidates']) for s in snapshots)}
            trades = read(output/'trades.json')
            economics = [{k:t.get(k) for k in ('symbol','qty','entry_decision_time','entry_fill_time',
                'exit_fill_time','entry_price','exit_price','gross_pnl','net_pnl','exit_reason')} for t in trades]
            cache[key] = {'identity': identity, 'configuration': read(case/'effective_config.json'),
                **snapshots_cache[snapshot_path], 'execution_hash':sig(economics),
                'daily_equity_hash':sig([(r['day'],r['equity']) for r in daily])}
        fingerprints[key] = cache[key]
    effects = []
    for row in rows:
        key = row['candidate_id']+'/'+row['window']
        reference = plan['baseline_id']+'/'+row['window']
        if key not in fingerprints or reference not in fingerprints:
            effects.append({'candidate_id':row['candidate_id'],'window':row['window'],
                'status':'not_comparable','error':row.get('error'),'qualified':row.get('qualified')})
            continue
        current, baseline = fingerprints[key], fingerprints[reference]
        edits = mutation_edits(baseline['configuration'],current['configuration'])
        effects.append({'candidate_id':row['candidate_id'],'window':row['window'],
            'qualified':row['qualified'],'effective_configuration_changed':bool(edits),
            'effective_configuration_paths':[list(path) for path,_,_ in edits],
            'selected_candidates_or_scores_changed':current['selection_hash']!=baseline['selection_hash'],
            'executions_changed':current['execution_hash']!=baseline['execution_hash'],
            'daily_equity_changed':current['daily_equity_hash']!=baseline['daily_equity_hash'],
            'candidate_rows':current['candidate_rows'], 'baseline_candidate_rows':baseline['candidate_rows']})
    write(cache_path,cache)
    write(root/'candidate_effects.json',{'cases':effects,
        'interpretation':'Unchanged effective configuration is an alias or ignored mapping to inspect. Changed configuration with identical executions is observed inactivity, not proven irrelevance in other markets. Failed audits cannot establish performance effects.'})


def report(root):
    candidate_effects(root)
    leaders, pairs = rank_pairs(root)
    write(root/'ranked_candidates.json', leaders)
    plan = read(root/'candidate_plan.json')
    names = {c['id']:c['label'] for c in plan['candidates']}
    if (root/'targeted_plan.json').exists():
        names.update({c['id']:c['label'] for c in read(root/'targeted_plan.json')['candidates']})
    if (root/'mechanism_plan.json').exists():
        names.update({c['id']:c['label'] for c in read(root/'mechanism_plan.json')['candidates']})
    for path in (root/'adaptive_plans').glob('*.json'):
        names.update({c['id']:c['label'] for c in read(path)['candidates']})
    import csv
    with (root/'all_candidate_comparisons.csv').open('w', newline='', encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['id','label','window','qualified','return_pct','drawdown_pct','trades','win_rate','expectancy','profit_factor','error'])
        for row in results(root):
            m=row.get('metrics',{})
            writer.writerow([row['candidate_id'], names.get(row['candidate_id']),row['window'],row['qualified'],
                row.get('return_pct'),row.get('drawdown_pct'),row.get('trades'),m.get('win_rate'),m.get('expectancy'),m.get('profit_factor'),row.get('error')])
    windows = read(root/'protocol.json')['windows']
    lines=['OLR round 1 comprehensive development study','',
        f"IS: {' to '.join(windows['train'])}; formerly OOS: {' to '.join(windows['oos'])}. Reference account KRW100m.",
        'The formerly OOS window is consumed by this study. Improvements are development evidence, not established expected returns.',
        'All cases use native execution and full audit; failed cases are retained and cannot be ranked. Original strategy remains unchanged.','',
        'Baseline:']
    for window,row in pairs.get(plan['baseline_id'],{}).items():
        lines.append(f"{window}: return {row['return_pct']:.4f}%, trades {row['trades']}, win rate {row['metrics']['win_rate']:.2%}, drawdown {row['drawdown_pct']:.4f}%")
    lines+=['',f'Eligible development candidates: {len(leaders)}','']
    for row in leaders[:20]:
        lines.append(f"{row['id']} {names.get(row['id'])}: IS/OOS gains {row['deltas_pp']}; trades {row['pair']['train']['trades']}/{row['pair']['oos']['trades']}")
    lines+=['','Full individual results: all_candidate_comparisons.csv. Tail and cohort attribution: baseline_diagnostics.json.',
            'This report is provisional unless completion.json exists. No selected mutation is automatically promoted.']
    (root/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def paired_uncertainty(control, candidate, draws=2000, block_sessions=5):
    """Paired circular five-session block bootstrap; diagnostic, not selection proof."""
    import numpy as np
    if block_sessions < 1:
        raise ValueError('Block length must be positive')
    left, right = read(Path(control['output'])/'daily.json'), read(Path(candidate['output'])/'daily.json')
    if [r['day'] for r in left] != [r['day'] for r in right]:
        raise ValueError('Paired uncertainty requires identical sessions')
    def returns(rows):
        equity = np.array([control['capital'], *[r['equity'] for r in rows]], dtype=float)
        return equity[1:]/equity[:-1]-1
    a,b=returns(left),returns(right)
    n=len(a)
    rng=np.random.default_rng(20261009)
    indices=(rng.integers(0,n,size=(draws,math.ceil(n/block_sessions),1))+np.arange(block_sessions)).reshape(draws,-1)[:,:n]%n
    gains=100*(np.prod(1+b[indices],axis=1)-np.prod(1+a[indices],axis=1))
    monthly={}
    for month in sorted({r['day'][:7] for r in left}):
        use=np.array([r['day'][:7]!=month for r in left])
        monthly[month]=float(100*(np.prod(1+b[use])-np.prod(1+a[use])))
    return {'sessions':n,'draws':draws,'block_sessions':block_sessions,'paired_return_gain_ci95_pp':np.quantile(gains,[.025,.975]).tolist(),
        'bootstrap_fraction_positive':float(np.mean(gains>0)), 'leave_one_month_out_gain_pp':monthly,
        'warning':'Descriptive resampling after adaptive search; neither independent validation nor multiplicity-corrected significance.'}


def finish_analytics(root, finalists):
    plan=read(root/'candidate_plan.json')
    _,pairs=rank_pairs(root)
    control=pairs[plan['baseline_id']]
    uncertainty={}
    robustness=[]
    for candidate in finalists:
        cid=candidate['id']
        if cid in pairs and set(pairs[cid])=={'train','oos'}:
            uncertainty[cid]={w:paired_uncertainty(control[w],pairs[cid][w]) for w in ('train','oos')}
        for path in (root/'cases'/cid).glob('*/*/assessment.json'):
            row=read(path)
            robustness.append({k:row.get(k) for k in ('candidate_id','window','capital','scenario','capped','qualified',
                'return_pct','drawdown_pct','trades','metrics','status','error')})
    write(root/'paired_uncertainty.json',uncertainty)
    write(root/'robustness_results.json',robustness)


def reconstruction_bridge(root, window, baseline):
    """Prove the new reconstruction path reproduces all saved baseline candidates."""
    destination=root/'builder_bridge'/window
    if (destination/'bridge.json').exists():
        if not read(destination/'bridge.json')['pass']:
            raise ValueError('Previously failed builder bridge requires a new revision')
        return
    destination.mkdir(parents=True,exist_ok=True)
    write(destination/'candidate.json',baseline)
    info=read(generation_folder(root,window)/'ready.json')
    output=destination/'snapshots.json'
    receipt_path=output.with_suffix('.receipt.json')
    if not (output.exists() and receipt_path.exists()):
        rc=run_child([sys.executable,'-B','-u',str(Path(__file__)),'snapshots','--root',str(root),
            '--window',window,'--candidate',str(destination/'candidate.json'),'--output',str(output)],
            info['files'],destination/'builder.log')
        if rc:
            raise ValueError('Baseline reconstruction failed: '+str(destination/'builder.log'))
    receipt=read(receipt_path)
    if (receipt['snapshots_sha256']!=sha(output) or receipt['candidate_hash']!=baseline['mutations_hash']
            or receipt['source_hash']!=info['study']['source_generation_hash']):
        raise ValueError('Completed reconstruction receipt does not authenticate this baseline')
    original=read(Path(info['files'])/info['study']['olr_snapshots'])
    rebuilt=read(output)
    days=[date.fromisoformat(r['trade_date']) for r in original]
    required_context=[str(d) for d in reconstruction_days(Path(info['files'])/info['study']['database'],days) if d not in set(days)]
    if receipt.get('context_only_dates',[])!=required_context:
        raise ValueError('Baseline reconstruction omitted available prior-session context')
    def economic(rows):
        # Artifact/source identifiers differ because the dataset identity is different.
        metadata_keys=('afternoon_features','afternoon_score_band_rule','prior_day_date')
        return {r['trade_date']:[{**{k:v for k,v in c.items() if k not in ('metadata','source_fingerprint','source_artifact_hash')},
            'execution_metadata':{k:(c.get('metadata') or {}).get(k) for k in metadata_keys}}
            for c in r['candidates']] for r in rows}
    old,new=economic(original),economic(rebuilt)
    errors=[d for d in sorted(set(old)|set(new)) if old.get(d)!=new.get(d)]
    write(destination/'bridge.json',{'pass':not errors,'changed_dates':errors,'sessions':len(old),
        'source':str(Path(info['files'])/info['study']['olr_snapshots']),'rebuilt':str(output),
        'context_only_dates':required_context,'receipt_sha256':sha(receipt_path),
        'comparison':'All candidate economics plus afternoon features, score-band identity and prior-day date; source/provenance identifiers intentionally differ'})
    if errors:
        raise ValueError('Baseline candidate reconstruction differs: '+str(destination/'bridge.json'))


def run(root, workers):
    if (root/'superseded_run.json').exists():
        raise ValueError('This screen was stopped by the user. Use the registered mechanism-study replacement, not this run command.')
    if not (root/'candidate_plan.json').exists():
        register_plan(root)
    protocol=read(root/'protocol.json')
    if sha(__file__) != protocol['harness_sha256']:
        raise ValueError('Harness differs from registered bytes; create a revision')
    for window in ('oos','train'):
        prepare_window(root,window)
    plan=read(root/'candidate_plan.json')
    baseline=plan['candidates'][0]
    event(root,'baseline_reproduction_started')
    batch(root,[baseline],workers,'baseline')
    pairs={row['window']:row for row in results(root) if row['candidate_id']==baseline['id']}
    if len(pairs)!=2 or not all(r['qualified'] for r in pairs.values()):
        raise ValueError('Baseline native integrity failed; fix the data or harness before searching')
    original=55.0465907665475
    if abs(pairs['train']['return_pct']-original)>1e-7 or pairs['train']['trades']!=176:
        raise ValueError('Training reproduction differs from selected round; bridge required before searching')
    write(root/'baseline_diagnostics.json',{w:concentration(r['output']) for w,r in pairs.items()})
    report(root)
    for window in ('oos','train'):
        reconstruction_bridge(root,window,baseline)
    event(root,'screen_started',candidates=len(plan['candidates']))
    batch(root,plan['candidates'],workers,'screen')
    report(root)
    targeted=(read(root/'targeted_plan.json')['candidates'] if (root/'targeted_plan.json').exists() else targeted_plan(root))
    event(root,'targeted_started',candidates=len(targeted))
    batch(root,targeted,workers,'targeted')
    report(root)
    leaders,_=rank_pairs(root)
    all_candidates={c['id']:c for c in [*plan['candidates'],*targeted]}
    finalists=[baseline,*[all_candidates[r['id']] for r in leaders[:5]]]
    event(root,'robustness_started',finalists=len(finalists))
    for candidate in finalists:
        for window in ('oos','train'):
            for capital in (10_000_000,25_000_000,50_000_000,100_000_000):
                for scenario in ('baseline','combined'):
                    evaluate(root,candidate,window,capital,scenario)
            for scenario in ('cost','participation'):
                evaluate(root,candidate,window,100_000_000,scenario)
            evaluate(root,candidate,window,100_000_000,'baseline',True)
    report(root)
    finish_analytics(root,finalists)
    write(root/'completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'primary_candidates':len(plan['candidates']),'targeted_candidates':len(targeted),
        'finalists':[x['id'] for x in finalists],'research_only':True,'independent_performance_established':False,
        'requires_final_interpretation':True})
    event(root,'completed',finalists=len(finalists))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','run','snapshots','report','acceptance'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--window',choices=['train','oos'])
    p.add_argument('--candidate',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--input',type=Path)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if a.command=='register':register_plan(a.root)
    elif a.command=='snapshots':snapshot_worker(a.root,a.window,a.candidate,a.output)
    elif a.command=='acceptance':acceptance_worker(a.input,a.output)
    elif a.command=='report':report(a.root)
    else:
        write(a.root/'process.json',{'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
            'script':str(Path(__file__).resolve()),'timeout_seconds_per_process':172800})
        try:run(a.root,a.workers)
        except Exception as exc:
            event(a.root,'needs_repair',error=str(exc),traceback=traceback.format_exc())
            raise


if __name__=='__main__':
    main()
