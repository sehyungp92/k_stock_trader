"""Bounded, training-only fixed-request diagnostic. Never a portfolio policy."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/structural_v6')
PARENT = BASE / 'native_controlled'
ROOT = BASE / 'fixed_signal_v1'
OUT = REPO / 'data/backtests/analysis/olr_structural_20261012/fixed_signal_v1'
INVENTORY = OUT.parent / 'opportunity_v1'
BLOCKS = [('2025-08-07', '2025-10-31'), ('2025-11-01', '2025-12-31'), ('2026-01-01', '2026-03-31')]
BUDGETS = (1_000_000, 20_000_000)
SCENARIOS = {'baseline': (.01, 5), 'combined': (.005, 15)}
NAMES = ['scripts/olr_fixed_signal_v8.py', 'scripts/olr_fixed_signal_worker_v8.py',
         'tests/strategy_olr/test_fixed_signal_v8.py', 'scripts/olr_opportunity_clock_v7.py']
PATCHES = {
    'strategy_olr/core/logic.py': [
        ('    target_notional = equity * min(float(config.max_position_pct), weight)',
         "    target_notional = float(state.meta.get('fixed_signal_budget_v8', equity * min(float(config.max_position_pct), weight)))"),
        ('    if capacity_enabled(config):\n        capacity_qty, _, evidence = capacity_forecast(state, bar, config)',
         "    if capacity_enabled(config) and 'fixed_signal_budget_v8' not in state.meta:\n        capacity_qty, _, evidence = capacity_forecast(state, bar, config)"),
    ],
    'strategy_olr/core/allocation.py': [
        ('    if config.capacity_redistribute:\n        used = sum(d.reserved_notional for d in decisions)',
         "    if config.capacity_redistribute and 'fixed_signal_budget_v8' not in state.meta:\n        used = sum(d.reserved_notional for d in decisions)"),
        ('        if config.capacity_redistribute:\n            meta.update(allocation_before_redistribution=native_approved[decision.intent_id],',
         "        if config.capacity_redistribute and 'fixed_signal_budget_v8' not in state.meta:\n            meta.update(allocation_before_redistribution=native_approved[decision.intent_id],"),
    ],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def temp():
    target = ROOT / 'runtime_tmp'
    target.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(target)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    tempfile.tempdir = str(target)


def transform(name, source):
    for old, new in PATCHES[name]:
        if source.count(old) != 1:
            raise ValueError('Exact native diagnostic substitution mismatch: ' + name)
        source = source.replace(old, new)
    return source


def select(rows, per_block=12):
    # No fill status, PNL, rank score, sector or symbol is used for date sampling.
    eligible = [dict(day=r['day'], symbol=r['symbol'], rank=r['rank'],
                     native_signal_time=r['native_signal_time']) for r in rows
                if r['native_signal_time'] and r['rank'] <= 6]
    dates = []
    for lo, hi in BLOCKS:
        options = sorted({r['day'] for r in eligible if lo <= r['day'] <= hi})
        if len(options) < per_block:
            raise ValueError('Insufficient pre-payoff date support')
        dates += [options[(2*i+1)*len(options)//(2*per_block)] for i in range(per_block)]
    assert len(dates) == len(set(dates)) == 3*per_block
    return sorted([r for r in eligible if r['day'] in dates], key=lambda r: (r['day'], r['symbol']))


class NativeOverlay(importlib.abc.MetaPathFinder):
    """Check original or explicitly overlaid bytes BEFORE module execution."""
    def __init__(self, files, manifest, overlays):
        self.files = Path(files).resolve()
        self.manifest, self.overlays = manifest, overlays
        self.roots = {p.split('/')[0] for p in manifest if p.count('/') == 1 and p.endswith('/__init__.py')} - {'tests'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in self.roots:
            return None
        key = fullname.replace('.', '/') + '.py'
        if key in self.overlays:
            item = self.overlays[key]
            source = Path(item['path']).resolve()
            if sha(source) != item['sha256']:
                raise ImportError('Changed explicit native overlay: ' + fullname)
            return importlib.util.spec_from_file_location(fullname, source)
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(self.files)] if path is None else path)
        if spec is None or not spec.origin:
            raise ImportError('Missing frozen native module: ' + fullname)
        source = Path(spec.origin).resolve()
        if not source.is_relative_to(self.files):
            raise ImportError('Foreign native import: ' + fullname)
        key = source.relative_to(self.files).as_posix()
        if self.manifest.get(key) != sha(source):
            raise ImportError('Unbound native import: ' + fullname)
        return spec


def imports(guard):
    rows = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in guard.roots or not getattr(module, '__file__', None):
            continue
        path = Path(module.__file__).resolve()
        key = name.replace('.', '/') + '.py'
        if key in guard.overlays:
            expected = guard.overlays[key]
            assert path == Path(expected['path']).resolve() and sha(path) == expected['sha256']
        else:
            assert path.is_relative_to(guard.files), name
            key = path.relative_to(guard.files).as_posix()
            assert guard.manifest.get(key) == sha(path), name
        rows[name] = dict(path=str(path), sha256=sha(path))
    return rows


def prepare():
    """Metadata, code and outcome-blind budget only; no label price stream."""
    temp()
    if (ROOT / 'design.json').exists():
        raise FileExistsError('Preserve existing design; no silent rebinding')
    checks = read(INVENTORY / 'inventory_checks.json')
    assert checks['passed'] and sha(INVENTORY / 'inventory_checks.json') == '0cf23202a01d2858ce6cf94eae1c84167cc4abee4adb8fc121ec7bb53ba28902'
    funnel = INVENTORY / 'incumbent_funnel'
    assert sha(funnel / 'checks.json') == 'c52513412a0bb5b6d2ffa2fb62bdb0d45c3ce3e26c74666c369cd235768fbc3f'
    ready_path = PARENT / 'native_engines/train/ready.json'
    ready = read(ready_path)
    files = Path(ready['files'])
    cohort = select(read(funnel / 'funnel.json')['rows'])
    assert len(cohort) <= 216
    overlays = {}
    for name in PATCHES:
        assert sha(files/name) == ready['manifest']['files'][name]
        changed = transform(name, (files/name).read_text(encoding='utf-8'))
        compile(changed, name, 'exec')
        path = ROOT / 'overlay' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(changed)
        overlays[name] = dict(path=str(path), sha256=sha(path), parent_sha256=sha(files/name))
    paths = [ready_path, Path(ready['study']['manifest']), files/ready['study']['olr_request'],
             files/ready['study']['olr_snapshots'], PARENT/'candidate_plan.json',
             funnel/'checks.json', funnel/'funnel.json', funnel/'receipt.json',
             INVENTORY/'inventory_checks.json', BASE/'opportunity_v1/clock_repair.json',
             BASE/'opportunity_v1/ledger_complete.json', BASE/'opportunity_v1/opportunities_train.json']
    design = dict(version='fixed-incumbent-native-diagnostic-v8', utc=utc(),
        hypothesis='Separate net payoff and quantity-dependent execution of existing confirmed signals from account funding and forecast-based request-size feedback.',
        source_ready=str(ready_path), source_manifest_sha256=sha(Path(ready['study']['manifest'])),
        inputs={str(p): sha(p) for p in paths}, overlays=overlays,
        cohort=cohort, dates=sorted({r['day'] for r in cohort}), blocks=BLOCKS,
        sampling='12 evenly spaced chronological confirmed-incumbent dates per block; include every original first-six confirmed candidate on each date, irrespective of original fill or profit. No score-only rejected names.',
        target_budgets=list(BUDGETS), budget_reason='KRW1m is the existing meaningful-notional floor; KRW20m is a round diagnostic scale near the already observed KRW23.17m mean original nominal acquisition. Neither is optimized.',
        scenarios=SCENARIOS, max_lanes=len(cohort)*4, max_sessions_per_lane=5,
        right_censor='Stop observation after five sourced trading sessions including entry or March31, whichever is earlier. Never force a sale. Unfilled, unsupported and unresolved labels are null, with coverage and actual residuals.',
        interventions=['Original entry artifacts and first native confirmation, original six slots; no new signals.',
          'Fixed target replaces rank allocation; disable speculative request forecast clipping and recycling ONLY for diagnostic state. Native physical fill capacity, cash checks, loss limits, all entry/exit thresholds, fees, partials, cancels and acknowledgements unchanged.',
          'Fresh KRW100m cash account for each lane. Other positions and signals do not compete. These shadow lanes are not jointly fundable portfolio results.',
          'Actual request may be reduced by unchanged native loss limits or integer rounding; report requested and filled amounts separately.',
          'Use verified close-anchored feature completeness on the unchanged, independently reconstructed original incumbent snapshots; no new selected cohort.'],
        comparisons=['Paired budgets under identical scenarios; acquired/requested notional, fill coverage, matured net bps and fixed-budget PNL, residual risk and both clock grids.',
          'Date-weighted chronological block means, winner concentration and descriptive rank association. No fitted model, cutoff or allocation is selected in this diagnostic.'],
        advancement='No automatic portfolio advancement. Require at least10 paired mature dates per block and80% paired mature opportunity coverage; positive date-weighted stressed net payoff in every block at the small budget before a separately preregistered allocation calibration is considered. Otherwise report rejection or insufficient support; no rescue.',
        no_promotion=True, no_oos=True, no_live_activation=True, retained='c4f49be01ed09b8dc',
        independent_validation=False, known_limitations=['Fixed universe/taxonomy', 'Sparse auctions', 'Inferred archive clocks', 'Conservative capacity and execution proxies', 'April-July repeatedly inspected development; not read here'],
        before_new_payoff_labels=True)
    write(ROOT/'design.json', design)
    write(OUT/'design.json', design)
    print(json.dumps(dict(opportunities=len(cohort), dates=len(design['dates']), lanes=design['max_lanes'], design_sha256=sha(ROOT/'design.json'))))


def authenticate(frozen=True, full=False):
    design = read(ROOT/'design.json')
    for path, digest in design['inputs'].items():
        assert sha(path) == digest, path
    ready = read(design['source_ready'])
    files = Path(ready['files'])
    revision_path = ROOT/'engineering_revision.json'
    if revision_path.exists():
        revision = read(revision_path)
        assert revision['design_sha256'] == sha(ROOT/'design.json')
        for name, item in design['overlays'].items():
            assert sha(item['path']) == item['sha256'], name
        for path, digest in revision['preserved'].items():
            assert sha(path) == digest, path
        design = {**design, 'overlays': revision['overlays']}
    for name, item in design['overlays'].items():
        assert sha(files/name) == item['parent_sha256'] == ready['manifest']['files'][name]
        assert sha(item['path']) == item['sha256']
        assert Path(item['path']).read_text(encoding='utf-8') == transform(name, (files/name).read_text(encoding='utf-8'))
    if full:
        for i, (name, digest) in enumerate(ready['manifest']['files'].items()):
            assert sha(files/name) == digest, name
            if i % 250 == 0:
                print(json.dumps(dict(phase='authenticate_unchanged_native_inputs', files=i)), flush=True)
    if frozen:
        freeze = read(ROOT/'frozen_harness.json')
        assert freeze['design_sha256'] == sha(ROOT/'design.json')
        for path, digest in freeze['files'].items():
            assert sha(path) == digest, path
    return design, ready


def freeze():
    import xml.etree.ElementTree as ET
    temp()
    design, ready = authenticate(False, True)
    suites = list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
    counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts['tests'] >= 12 and not any(counts[k] for k in ('failures', 'errors', 'skipped'))
    tests = read(ROOT/'test_imports.json')
    assert tests['exit_code'] == 0 and tests['overlay_imports_verified']
    assert tests['test_file_sha256'] == sha(REPO/'tests/strategy_olr/test_fixed_signal_v8.py')
    assert tests['worker_sha256'] == sha(REPO/'scripts/olr_fixed_signal_worker_v8.py')
    assert tests['contract_sha256'] == sha(__file__)
    paths = [REPO/n for n in NAMES] + [ROOT/n for n in ('tests.xml', 'tests.log', 'test_imports.json')]
    if (ROOT/'engineering_revision.json').exists():
        paths.append(ROOT/'engineering_revision.json')
    write(ROOT/'frozen_harness.json', dict(utc=utc(), design_sha256=sha(ROOT/'design.json'),
        files={str(p):sha(p) for p in paths}, tests=counts, before_new_payoff_labels=True,
        exact_source_overlays=design['overlays'], native_imports=tests['imports']))
    print(json.dumps(dict(frozen=str(ROOT/'frozen_harness.json'), sha256=sha(ROOT/'frozen_harness.json'))))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('phase', choices=['prepare', 'freeze'])
    globals()[args.parse_args().phase]()
