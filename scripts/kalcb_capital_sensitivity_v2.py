"""Exp0030 R1: finish unattempted cells; retain rejected accounts as rejected."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, env
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as diag
from scripts.kalcb_capital_sensitivity_v1 import status, WORKER, SCENARIOS

PREVIOUS = Path('E:/CodexResearch/k_stock_trader/kalcb_capital_sensitivity_20261016/v1')
DESIGN = REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0030-continuation-r1.md'
ATTEMPTED = ('bridge_compound', 'bridge_fixed', 'compound_baseline', 'fixed_baseline', 'compound_participation')
REMAINING = ('fixed_participation', 'compound_cost', 'fixed_cost', 'compound_combined', 'fixed_combined')
PROGRAMS = ('scripts/kalcb_capital_sensitivity_v2.py',
            'tests/backtests/strategies/test_kalcb_capital_sensitivity_v2.py')


def no_other_producer():
    import psutil
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid == os.getpid() or 'python' not in (process.info['name'] or '').lower():
            continue
        args = process.info['cmdline'] or []
        if any('kalcb' in a.lower() and a.lower().endswith('.py') and
               ('worker' in a.lower() or 'capital_sensitivity_v' in a.lower() or
                'frontier_completion' in a.lower() or 'semantic_feature_ablation' in a.lower()) for a in args):
            raise ValueError('Another KALCB producer is active: '+str(process.pid))


def classify_case(folder, item, *, save=False):
    """Only classify the original verifier's explicit nonflat rejection; never pass it."""
    folder = Path(folder)
    try:
        validation = diag.validate_capital_native(folder, item, save=save)
    except ValueError as error:
        if str(error) != 'Unresolved native account':
            raise
        result = read(folder/'result.json')
        flags = {k: result[k] for k in ('residual_sessions', 'terminal_inventory', 'terminal_pending')}
        if not any(flags.values()):
            raise ValueError('Residual rejection lacks explicit account evidence') from error
        if (folder/'independent_account_verification.json').exists():
            raise ValueError('Rejected account has an independent pass sidecar') from error
        return dict(qualified=False, reason=str(error), flags=flags,
                    remaining_independent_checks_not_reached=True,
                    reported_return_excluded_from_qualified_comparisons=True)
    return dict(qualified=True, validation=validation)


def register(root, tests):
    no_other_producer()
    previous = read(PREVIOUS/'registration.json'); protocol = read(PREVIOUS/'protocol.json')
    inherited = previous['programs'] + protocol['inputs'] + [previous[k] for k in
        ('protocol', 'source_preflight', 'source_inventory')]
    verify(inherited)
    receipt = read(tests)
    if receipt['exit_code'] != 0:
        raise ValueError('Continuation tests failed')
    verify(receipt['bindings'])
    if read(PREVIOUS/'failure_all.json')['error'] != "ValueError('Unresolved native account')":
        raise ValueError('Unexpected previous failure')
    if [x['id'] for x in protocol['cases']] != list(ATTEMPTED+REMAINING):
        raise ValueError('Original matrix changed')
    if sorted(p.name for p in (PREVIOUS/'cases').iterdir()) != sorted(ATTEMPTED):
        raise ValueError('Unexpected prior case population')
    if [x['id'] for x in read(PREVIOUS/'completed_cases.json')] != list(ATTEMPTED[:4]):
        raise ValueError('Changed accepted predecessor set')
    ledger = {}; inputs = inherited + [bound(tests)] + receipt['bindings']
    for name in ('registration.json', 'failure_all.json', 'completed_cases.json', 'residual_diagnosis_v1.json'):
        inputs.append(bound(PREVIOUS/name))
    for case in protocol['cases'][:5]:
        folder = PREVIOUS/'cases'/case['id']
        verdict = classify_case(folder, previous['profiles'][case['id']])
        if verdict['qualified'] != (case['id'] != 'compound_participation'):
            raise ValueError('Changed predecessor qualification')
        if verdict['qualified']:
            diag.funding_review(folder, case['initial_equity'])
        if case.get('bridge') and not read(folder/'identity_bridge.json')['passed']:
            raise ValueError('Missing native identity bridge')
        ledger[case['id']] = dict(folder=str(folder), reused=True, **verdict)
        inputs.extend(bound(path) for path in sorted(folder.rglob('*')) if path.is_file())
    root.mkdir(parents=True, exist_ok=False); (root/'temp').mkdir()
    shutil.copyfile(DESIGN, root/'registered_continuation.md')
    programs = previous['programs'] + [bound(REPO/p) for p in PROGRAMS]
    inputs.append(bound(root/'registered_continuation.md'))
    verify(inputs+programs)
    write(root/'registration.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        experiment='kalcb-exp-0030', continuation='R1', criteria='unchanged C6',
        programs=programs, inputs=inputs, previous_registration=bound(PREVIOUS/'registration.json'),
        previous_protocol=bound(PREVIOUS/'protocol.json'), original_attempts=ledger,
        remaining_case_ids=list(REMAINING), source_and_policy_unchanged=True,
        qualification_rule='Original validator unchanged. Explicit residual failures remain unqualified; other errors stop.',
        economic_objective_achieved=False))
    write(root/'case_ledger.json', ledger)
    status(root, 'registered', reused_attempts=5, qualified_predecessors=4, rejected_predecessors=1, remaining_cases=5)


def run(root):
    import psutil, msvcrt
    no_other_producer()
    registration = read(root/'registration.json')
    verify(registration['inputs']+registration['programs'])
    previous = read(registration['previous_registration']['path'])
    protocol = read(registration['previous_protocol']['path'])
    if (root/'cases').exists() or (root/'assessment.json').exists():
        raise ValueError('Preserve existing generation')
    lock = (root/'run.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    code = PREVIOUS/'inputs/files'
    environment = env(code); environment.update(TEMP=str(root/'temp'), TMP=str(root/'temp'))
    ledger = read(root/'case_ledger.json')
    try:
        for case in protocol['cases']:
            if case['id'] not in registration['remaining_case_ids']:
                continue
            verify(registration['programs'])
            item = previous['profiles'][case['id']]
            target = root/'cases'/case['id']; target.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-B', '-u', str(code/WORKER), '--code', str(code),
                       '--plan', item['plan']['path'], '--policies', str(Path(item['policy']['path']).parent),
                       '--audit-manifest', item['manifest']['path'], '--output', str(target)]
            started = time.monotonic()
            with (root/(case['id']+'.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=code, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                identity = dict(pid=child.pid, created_at=psutil.Process(child.pid).create_time(), command=command)
                status(root, 'native', case=case['id'], child=identity, attempted_cases=len(ledger))
                child.wait()
            if child.returncode:
                raise RuntimeError('Native case failed: '+case['id'])
            verdict = classify_case(target, item, save=True)
            if verdict['qualified']:
                write(target/'funding_review.json', diag.funding_review(target, case['initial_equity']))
            else:
                write(target/'qualification_rejection.json', verdict)
            ledger[case['id']] = dict(folder=str(target), reused=False, seconds=time.monotonic()-started, **verdict)
            write(root/'case_ledger.json', ledger)
            status(root, 'case_assessed', case=case['id'], qualified=verdict['qualified'], attempted_cases=len(ledger))
        assess(root, registration, previous, protocol, ledger)
        status(root, 'assessment_complete', attempted_cases=len(ledger),
               qualified_cases=sum(v['qualified'] for v in ledger.values()))
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


def assess(root, registration, previous, protocol, ledger):
    if (root/'assessment.json').exists():
        raise ValueError('Preserve prior assessment')
    if set(ledger) != {c['id'] for c in protocol['cases']}:
        raise ValueError('Incomplete attempt ledger')
    rows = {}; diagnostics = {}; activity = {}; funding = {}; comparisons = {}; inputs = []
    for case in protocol['cases']:
        entry = ledger[case['id']]; folder = Path(entry['folder'])
        verdict = classify_case(folder, previous['profiles'][case['id']])
        if verdict['qualified'] != entry['qualified']:
            raise ValueError('Changed qualification')
        inputs.extend(bound(path) for path in sorted(folder.rglob('*')) if path.is_file())
        if case.get('bridge') or not verdict['qualified']:
            continue
        initial = case['initial_equity']; key = case['id']
        rows[key] = diag.metrics(folder, initial); rows[key]['initial_equity_krw'] = initial
        diagnostics[key] = diag.account(folder, initial); activity[key] = diag.opportunities(folder, initial)
        funding[key] = diag.funding_review(folder, initial)
    for scenario in SCENARIOS:
        a = 'fixed_'+scenario; b = 'compound_'+scenario
        rejected = [k for k in (a, b) if not ledger[k]['qualified']]
        comparison = dict(qualified_matched_comparison=not rejected, unqualified_case_ids=rejected, capital_changes={})
        if not rejected:
            comparison.update(path=diag.paired_attribution(diagnostics[a], diagnostics[b]),
                activity=diag.compare(activity[a], activity[b]),
                delta_money_dd_krw=rows[a]['max_money_dd_krw']-rows[b]['max_money_dd_krw'])
        for mode in ('compound', 'fixed'):
            key = mode+'_'+scenario
            if key not in rows:
                continue
            old = Path(protocol['historical_accounts'][mode+'_prior100/oos/'+scenario])
            comparison['capital_changes'][mode] = dict(activity=diag.compare(activity[key], diag.opportunities(old)),
                path=diag.paired_attribution(diagnostics[key], diag.account(old)))
        comparisons[scenario] = comparison
    qualified = {s: c for s, c in comparisons.items() if c['qualified_matched_comparison']}
    deltas = {s: c['activity']['net_pnl_delta_krw'] for s, c in qualified.items()}
    predictions = dict(total_registered_stresses=4, qualified_matched_stresses=len(qualified),
        fixed_positive_qualified_stresses=sum(x>0 for x in deltas.values()),
        combined_positive=deltas['combined']>0 if 'combined' in deltas else None,
        any_qualified_gain_ge_1m=any(x>=1_000_000 for x in deltas.values()),
        lower_money_dd_qualified_stresses=sum(c['delta_money_dd_krw']<0 for c in qualified.values()),
        full_path_qualification=all(v['qualified'] for v in ledger.values()),
        full_C6_support_claim=False,
        scope='Missing matched cells remain unavailable, never imputed or counted as qualified successes.')
    verify(registration['programs']+registration['inputs'])
    write(root/'assessment.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        registration=bound(root/'registration.json'), case_ledger=ledger, accounts=rows,
        diagnostics=diagnostics, activity=activity, funding=funding, comparisons=comparisons,
        predictions=predictions, bindings=inputs, scope=protocol['scope'],
        economic_objective_achieved=False))
    write(root/'completion.json', dict(complete=True, assessment_complete=True, planned_native_cases=10,
        fresh_native_cases=5, reused_attempts=5, qualified_cases=sum(v['qualified'] for v in ledger.values()),
        unqualified_case_ids=[k for k, v in ledger.items() if not v['qualified']],
        all_accounts_qualified=predictions['full_path_qualification'], assessment=bound(root/'assessment.json'),
        economic_decision_required=True, economic_objective_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('register', 'run', 'all'), required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--tests', type=Path)
    args = parser.parse_args()
    try:
        if args.phase in ('register', 'all'): register(args.output, args.tests)
        if args.phase in ('run', 'all'): run(args.output)
    except Exception as error:
        if args.output.exists():
            write(args.output/('failure_'+args.phase+'.json'), dict(at_utc=datetime.now(timezone.utc).isoformat(),
                error=repr(error), traceback=traceback.format_exc()))
            status(args.output, 'failed', error=repr(error))
        raise
