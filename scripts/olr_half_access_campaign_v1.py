"""0040: two-account funded factorial interaction, with immutable preparation."""
from __future__ import annotations
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import psutil
import olr_selective_half_continuation_v1 as shared
from olr_relative_context_campaign_v1 import private_text

read, write, sha, sig, now, status = shared.read, shared.write, shared.sha, shared.sig, shared.now, shared.status
REPO = shared.REPO
BASE = Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016')
PARENT = BASE/'expired_selection_v4'
HALF = Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013/generation_v3/cases/train/structural_partial_baseline')
HELPERS = ['olr_selective_half_continuation_v1.py', 'olr_relative_context_campaign_v1.py',
    'olr_half_access_tests_v1.py', 'olr_half_access_review_v1.py',
    'olr_expired_selection_review_v1.py', 'olr_opportunity_continuity_v1.py',
    'olr_relative_context_assessment_v1.py']
TESTS = ['test_expired_selection.py', 'test_capacity_allocation.py', 'test_precision_v3.py',
         'test_opening_risk_v2.py', 'test_structural_service.py', 'test_economic_evidence.py',
         'test_half_access.py']


def verify(root):
    protocol = read(root/'protocol.json')
    assert sha(__file__) == protocol['runner_sha256']
    assert sha(root/'design_at_freeze.md') == protocol['design_sha256']
    for name, digest in protocol['helpers'].items():
        assert sha(REPO/'scripts'/name) == digest, name
    for name, digest in protocol['parent_bindings'].items():
        assert sha(PARENT/name) == digest, name
    assert sha(HALF/'completion.json') == protocol['half_reference_sha256']
    assert sha(BASE/'opportunity_continuity_v2/train_rows.json') == protocol['census_sha256']
    assert sha(BASE/'selective_half_v1/protocol.json') == protocol['deferred_protocol_sha256']
    return protocol


def completed(case, completed_case):
    result = completed_case(case)
    assert result['valid'], result
    receipt = read(case/'completion.json')
    for name, digest in receipt['artifacts'].items():
        assert sha(case/name) == digest, (str(case), name)
    return len(receipt['artifacts'])


def prepare(root):
    verify(root)
    identities = []
    for path in [PARENT/'controller_process.json', PARENT/'review_dispatch_process.json',
                 *PARENT.glob('runtime/*/*/process.json')]:
        receipt = read(path)
        try:
            process = psutil.Process(receipt['pid'])
            live = abs(process.create_time()-receipt['created_at']) < .001 and process.cmdline() == receipt['command']
        except psutil.NoSuchProcess:
            live = False
        assert not live, ('Parent still active', str(path))
        identities.append(dict(receipt=str(path), authenticated_live=False))
    old = read(PARENT/'native_engines/train/ready.json')
    origin = Path(old['files']); files = root/'native_engines/train/files'
    files.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(old['manifest']); manifest.pop('manifest_hash', None)
    status(root, 'preparing_private_source', queued=['half', 'joint'])
    for name, digest in manifest['files'].items():
        dest = files/name; dest.parent.mkdir(parents=True, exist_ok=True); os.link(origin/name, dest)
        if name.endswith('.py'):
            assert sha(dest) == digest, name
    source_hash, completed_case, _ = shared.frozen_imports(root)
    parent_source = source_hash(origin)
    assert source_hash(files) == parent_source == old['study']['source_generation_hash']
    bindings = {key: sha(files/old['study'][key]) for key in
                ('database', 'olr_snapshots', 'olr_request', 'minute_source_definition')}
    for key, digest in bindings.items():
        assert manifest['files'][old['study'][key]] == digest, key
    test_name = 'tests/strategy_olr/test_half_access.py'
    private_text(files, test_name, (REPO/'scripts/olr_half_access_tests_v1.py').read_text(encoding='utf8'), manifest)
    source = source_hash(files)
    study = copy.deepcopy(old['study']); study.update(code=str(files), source_generation_hash=source)
    manifest['source_generation_hash'] = source
    write(root/'native_engines/train/ready.json', dict(files=str(files), study=study,
          manifest=manifest, interval_qualification=old['interval_qualification']))
    reused = {}; parent_summaries = read(PARENT/'assessment.json')['summaries']; artifact_counts = {}
    for label, key in [('control', 'train_control'), ('access', 'train_access')]:
        value = copy.deepcopy(parent_summaries[key]); case = Path(value['case'])
        assert sha(case/'completion.json') == value['completion_sha256']
        artifact_counts[label] = completed(case, completed_case)
        write(root/'accepted'/('train_'+label+'.json'), value)
        assert read(root/'accepted'/('train_'+label+'.json')) == value
        reused[label] = value
    artifact_counts['half_reference'] = completed(HALF, completed_case)
    report = root/'focused_tests.xml'
    command = [sys.executable, '-B', '-m', 'pytest', *['tests/strategy_olr/'+n for n in TESTS],
        '-q', '-p', 'no:cacheprovider', '--basetemp='+str(root/'tmp/pytest'), '--junitxml='+str(report)]
    with (root/'focused_tests.log').open('x', encoding='utf8') as stream:
        worker = subprocess.Popen(command, cwd=files, env=shared.env_for(root), stdout=stream,
            stderr=subprocess.STDOUT, creationflags=0x08000000)
        write(root/'focused_tests_process.json', dict(at=now(), pid=worker.pid,
            created_at=psutil.Process(worker.pid).create_time(), command=command))
        assert worker.wait() == 0, 'Preserve failed focused readiness tests'
    suite = ET.parse(report).getroot().find('testsuite')
    assert all(suite.get(k) == '0' for k in ('failures', 'errors', 'skipped'))
    assert int(suite.get('tests')) == 85
    assert source_hash(files) == source and source_hash(origin) == parent_source
    write(root/'preflight.json', dict(at=now(), passed=True, tests=85,
        source=source, parent_source=parent_source, input_hashes=bindings,
        source_changes={test_name: sha(files/test_name)}, trading_python_unchanged=True,
        test_report_sha256=sha(report), protocol_sha256=sha(root/'protocol.json'),
        parent_process_checks=identities, authenticated_artifact_counts=artifact_counts,
        actual_acceptance_roundtrip_passed=True))
    return reused


def launch(root, policy):
    ready = read(root/'native_engines/train/ready.json'); files = Path(ready['files'])
    study, manifest = copy.deepcopy(ready['study']), copy.deepcopy(ready['manifest'])
    prefix = f'research/half_access_interaction/train/{policy}/'
    values = [('structural_settings', 'structural.json', dict(mode='control', interval_service=True)),
        ('funded_probe_settings', 'probe.json', dict(version='first-native-fade-funded-probe-v1', action='hold', control_reference=None)),
        ('funded_population_settings', 'population.json', dict(version='monthly-native-funded-population-v1', cell=None, catalog_reference=None)),
        ('relative_context_settings', 'context.json', dict(fraction=0., version='relative-context-v1')),
        ('expired_selection_settings', 'selection.json', dict(enabled=policy == 'joint', version='expired-selection-v1'))]
    for key, name, value in values:
        study[key] = prefix+name; write(files/study[key], value); manifest['files'][study[key]] = sha(files/study[key])
    study['candidates']['candidate']['olr_capacity'].update(capacity_structural_partial=True, capacity_weak_partial_fraction=.5)
    manifest['evidence']['qualification_path'] = ready['interval_qualification']
    study['manifest'] = str(files.parent/(policy+'.manifest.json'))
    study['configuration']['input_manifest_path'] = study['manifest']
    study['spec_path'] = prefix+'spec.json'
    write(files/study['spec_path'], {k:v for k,v in study.items() if k not in ('code', 'manifest', 'spec_path', 'source_generation_hash')})
    manifest['files'][study['spec_path']] = sha(files/study['spec_path'])
    write(study['manifest'], dict(**manifest, manifest_hash=sig(manifest)))
    spec = root/'runtime/train'/policy/'study.json'; write(spec, study)
    case = root/'cases/train'/policy; log = root/'logs/train'/(policy+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-u', str(files/'scripts/portfolio_synergy_study.py'), 'replay',
        '--study', str(spec), '--recipe', 'candidate', '--scenario', 'baseline', '--standalone', 'OLR', '--output', str(case), '--full-audit']
    with log.open('x', encoding='utf8') as stream:
        worker = subprocess.Popen(command, cwd=files, env=shared.env_for(root), stdout=stream,
            stderr=subprocess.STDOUT, creationflags=0x08000000)
    write(spec.parent/'process.json', dict(at=now(), pid=worker.pid, created_at=psutil.Process(worker.pid).create_time(),
        command=command, log=str(log), case=str(case)))
    return worker, case


def accept(root, policy, case, completed_case):
    count = completed(case, completed_case)
    assert read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
    assert read(case/'funded_population_receipt.json')['audit']['pass_']
    if policy == 'half':
        checks = {n: read(case/n) == read(HALF/n) for n in
                  ('equity.json', 'daily.json', 'trades.json', 'minute_nav_evidence.json')}
        write(root/'half_disabled_parity.json', dict(at=now(), passed=all(checks.values()), checks=checks,
            reference=str(HALF), reference_completion_sha256=sha(HALF/'completion.json')))
        assert all(checks.values()), checks
    value = shared.economic_summary(case)
    assert value['completion_sha256'] == sha(case/'completion.json')
    extras = []; halves = []
    with gzip.open(case/'native_decisions.jsonl.gz', 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('metadata', {}).get('expired_selection_v1'): extras.append(row)
            if row.get('decision_code') == 'EXIT_SUBMITTED' and row.get('reason') == 'weak_followthrough_partial': halves.append(row)
    assert halves, 'Half mechanism inactive'
    assert bool(extras) == (policy == 'joint'), 'Access activation differs from design'
    value.update(selection_decisions=extras, half_decisions=halves, authenticated_artifacts=count)
    target = root/'accepted'/('train_'+policy+'.json'); write(target, value)
    assert read(target) == value
    return value


def run(root):
    summaries = prepare(root)
    source_hash, completed_case, contrast = shared.frozen_imports(root)
    for index, policy in enumerate(('half', 'joint')):
        verify(root)
        ready = read(root/'native_engines/train/ready.json')
        assert source_hash(Path(ready['files'])) == ready['study']['source_generation_hash']
        status(root, 'running', case=policy, accepted=list(summaries), queued=list(('half', 'joint')[index+1:]))
        worker, case = launch(root, policy)
        assert worker.wait() == 0, 'Preserve failed account: '+str(case)
        summaries[policy] = accept(root, policy, case, completed_case)
        status(root, 'case_accepted', case=policy, accepted=list(summaries))
    metrics = ('return_pct', 'full_dd_pct', 'common_dd_pct', 'maximum_currency_decline', 'meaningful', 'meaningful_days',
        'buy_notional', 'winning_wealth', 'losing_wealth', 'longest_underwater_calendar_days', 'mean_economic_event_exposure')
    pairs = {}
    for left, right in [('control', 'access'), ('control', 'half'), ('control', 'joint'), ('half', 'joint'), ('access', 'joint')]:
        pairs[left+'_to_'+right] = dict(control=left, treatment=right,
            delta={k:summaries[right][k]-summaries[left][k] for k in metrics},
            accounting=contrast(read(Path(summaries[left]['case'])/'trades.json'), read(Path(summaries[right]['case'])/'trades.json')))
    interaction = {k:summaries['joint'][k]-summaries['half'][k]-summaries['access'][k]+summaries['control'][k] for k in metrics}
    write(root/'assessment.json', dict(at=now(), summaries=summaries, comparisons=pairs, endpoint_interaction=interaction,
        economic_objective_achieved=False, scope='Bounded IS development diagnostic. Await predefined attribution and substantive readiness.'))
    status(root, 'causal_review', accepted=list(summaries), queued=[])
    command = [sys.executable, '-X', 'utf8', '-B', '-u', str(REPO/'scripts/olr_half_access_review_v1.py'), str(root)]
    with (root/'review.log').open('x', encoding='utf8') as stream:
        worker = subprocess.Popen(command, cwd=REPO, env=shared.env_for(root), stdout=stream,
            stderr=subprocess.STDOUT, creationflags=0x08000000)
        write(root/'review_process.json', dict(at=now(), pid=worker.pid, created_at=psutil.Process(worker.pid).create_time(), command=command))
        assert worker.wait() == 0, 'Preserve failed analysis only; do not rerun accepted accounts'
    status(root, 'awaiting_economic_interpretation', accepted=list(summaries), queued=[], review='causal_review_v1/assessment.json')


def start(root):
    root.mkdir(parents=True, exist_ok=False)
    design = REPO/'docs/optimisation/olr/experiments/olr-exp-0040.md'
    (root/'design_at_freeze.md').write_bytes(design.read_bytes())
    bindings = ['assessment.json', 'preflight.json', 'native_engines/train/ready.json']
    write(root/'protocol.json', dict(at=now(), experiment='0040', runner_sha256=sha(__file__), design_sha256=sha(design),
        helpers={n:sha(REPO/'scripts'/n) for n in HELPERS}, parent_bindings={n:sha(PARENT/n) for n in bindings},
        half_reference_sha256=sha(HALF/'completion.json'), census_sha256=sha(BASE/'opportunity_continuity_v2/train_rows.json'),
        deferred_protocol_sha256=sha(BASE/'selective_half_v1/protocol.json'),
        fresh_accounts=['half', 'joint'], reused_accounts=['control', 'access'], concurrency=1, planned_minutes=90,
        new_data=False, canonical_promotion=False, automatic_financial_successor=False))
    for name in [Path(__file__).name, *HELPERS]:
        target = root/'launch_sources'/name; target.parent.mkdir(exist_ok=True)
        target.write_bytes((REPO/'scripts'/name).read_bytes())
    command = [sys.executable, '-X', 'utf8', '-B', '-u', str(Path(__file__).resolve()), 'run', str(root)]
    with (root/'controller.log').open('x', encoding='utf8') as stream:
        worker = subprocess.Popen(command, cwd=REPO, env=shared.env_for(root), stdout=stream,
            stderr=subprocess.STDOUT, creationflags=0x08000000)
    receipt = dict(at=now(), pid=worker.pid, created_at=psutil.Process(worker.pid).create_time(), command=command)
    write(root/'controller_process.json', receipt); print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['start', 'run']); parser.add_argument('root', type=Path)
    args = parser.parse_args()
    try: globals()[args.mode](args.root)
    except Exception:
        status(args.root, 'needs_repair', error=traceback.format_exc()); raise
