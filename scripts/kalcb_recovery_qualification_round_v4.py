"""Version4: resume qualified native evidence with the full independent-check context."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT, POLICY, FULL, read, write, bound, digest, link, env
from scripts import kalcb_alignment_walk_forward_v3 as aligned
from scripts import kalcb_recovery_mechanism_round as mechanism
from scripts import kalcb_recovery_blend_checkpoint_round as blend

PRIOR = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v3')
DEST = PRIOR.with_name('v4')
SCENARIOS = ('baseline', 'participation', 'cost', 'combined')
PROFILES = {
    'expanding': {'policy': str(aligned.DEST/'fit/policies/original_target_full_history'),
                  'reference': str(ROOT/'aligned_target_round_v3/cases/original_target_full_history')},
    'expanding_atr5': {'policy': str(aligned.DEST/'fit/policies/original_target_full_history'),
                       'transform': 'atr_quantity_5pct',
                       'reference': str(ROOT/'targeted_round/cases/continuous_atr_quantity_5pct')},
    'decay60': {'policy': str(mechanism.DEST/'fit/policies/original_decay60'),
                'reference': str(ROOT/'mechanism_round_v1/cases/original_decay60')},
    'decay60_atr5': {'policy': str(mechanism.DEST/'fit/policies/original_decay60'),
                    'transform': 'atr_quantity_5pct',
                    'reference': str(ROOT/'blend_checkpoint_round_v1/cases/decay60_atr5')},
}
PERIODS = {'is': ('2025-08-07', '2026-03-31', 156), 'oos': ('2026-04-01', '2026-07-10', 69)}
IS_PARENT = Path('E:/CodexResearch/k_stock_trader/kalcb_intraday_advantage_20261007/audit_cost_margin_final/manifest.json')


def check(out):
    p = read(out/'protocol.json')
    if p['profiles'] != PROFILES or p['scenarios'] != list(SCENARIOS):
        raise ValueError('Changed qualification contract')
    aligned.verify_bindings(p['bindings'])
    return p


def register(out):
    from copy import deepcopy
    import psutil
    from scripts.kalcb_recovery_qualification_round_v3 import check as prior_check
    if out.exists(): raise ValueError('Preserve prior successor')
    prior_check(PRIOR)
    state=read(PRIOR/'status.json')
    if state['stage']!='attention_required' or state.get('error')!="KeyError('audit_manifest')":
        raise ValueError('Only the known independent-verifier metadata omission is resumable')
    for identity in (state,state.get('child') or {}):
        if identity.get('pid') and psutil.pid_exists(identity['pid']):
            process=psutil.Process(identity['pid'])
            if abs(process.create_time()-identity['process_created'])<.1:
                raise ValueError('Prior process is still alive; never duplicate it')
    prep=read(PRIOR/'preparation.json')
    if not prep['complete']:raise ValueError('Missing complete immutable inputs')
    first=Path(prep['profiles'][0]['folder'])/'baseline'
    # Smoke the actual completed native receipt through the unchanged public
    # validator with the missing manifest context supplied, before registering.
    validate(first,prep['profiles'][0],'baseline',Path(PROFILES[prep['profiles'][0]['name']]['reference'])/prep['profiles'][0]['period']/'baseline',save=False)
    from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
    protocol=deepcopy(read(PRIOR/'protocol.json'))
    protocol['registered_at_utc']=datetime.now(timezone.utc).isoformat()
    protocol['repair']='Supply the audit_manifest required by the existing independent account validator. Reuse already completed source-audited native evidence only after full validation; run remaining cases on unchanged v3 frozen inputs. No execution/model/source/audit criteria changes.'
    protocol['bindings'] += [bound(__file__),bound(PRIOR/'protocol.json'),bound(PRIOR/'preparation.json'),bound(PRIOR/'status.json'),bound(PRIOR/'background.err.log')]
    protocol['bindings'] += [bound(first/f) for f in REPLAY_FILES]
    out.mkdir(parents=True)
    write(out/'protocol.json',protocol)
    print({'registered':str(out),'native_cases':32,'first_source_audited_case_reused':str(first)},flush=True)


def collect_lineage(out):
    """Archive authenticated actual model/target/feature/program bytes, not paths alone."""
    _, _, sessions, evidence = blend.original_targets()
    calendar = read(aligned.DEST/'fit/calendar.json')['sessions']
    if sessions != calendar:
        raise ValueError('Changed full history clock')
    refs = dict(evidence)
    def add(path, expected=None):
        path = Path(path).resolve(); value = digest(path)
        if expected and value != expected:
            raise ValueError('Changed lineage input '+str(path))
        refs[str(path)] = value
    for root in (aligned.DEST, mechanism.DEST, blend.DEST):
        p = read(root/'protocol.json')
        for item in p.get('bound_inputs', [])+p.get('inputs', [])+p.get('source_programs', [])+p.get('programs', []):
            add(item['path'], item['sha256'])
        add(root/'protocol.json')
    for root, recipe in ((aligned.DEST, 'original_target_full_history'), (mechanism.DEST, 'original_decay60')):
        add(root/'fit/completion.json')
        for kind in ('opening', 'timed'):
            add(root/'fit/forecasts'/recipe/(kind+'.jsonl'))
        for period in PERIODS:
            policy = root/'fit/policies'/recipe/period
            for path in policy.iterdir():
                if path.is_file(): add(path)
            c = read(policy/'chronology.json')
            for fold in c['folds']:
                if fold['train_through'] >= fold['valid_from'] or fold['label_available_through'] >= fold['fit_at']:
                    raise ValueError('Noncausal model history')
                if sessions.index(fold['valid_from'])-sessions.index(fold['train_through']) < 2:
                    raise ValueError('Missing one-session embargo')
                add(fold['model_path'], fold['model_sha256'])
            for cal in c['calibration']:
                if cal.get('latest_label_available_at') and cal['latest_label_available_at'] >= cal['valid_from']:
                    raise ValueError('Noncausal calibration label')
                if cal.get('observed_through') and cal['observed_through'] >= cal['valid_from']:
                    raise ValueError('Noncausal calibration window')
    # Complete native evidence for the four original boundary outcomes; preserve
    # the selected-family outcomes too as existing causal repair evidence.
    for path in aligned.BOUNDARY.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.gz'): add(path)
    for prepared in (Path('E:/CodexResearch/k_stock_trader/kalcb_continuation_20261006/prepared_v1'),
                     Path('E:/CodexResearch/k_stock_trader/kalcb_intraday_advantage_20261007/checkpoint/prepared'),
                     FULL/'forward_features/continuation', FULL/'forward_features/checkpoint'):
        for path in prepared.iterdir():
            if path.is_file(): add(path)
    for item in read(out/'protocol.json')['bindings']: add(item['path'], item['sha256'])
    target = out/'lineage'; target.mkdir(exist_ok=False)
    mapping = []
    for path, sha in sorted(refs.items()):
        source = Path(path); relative = Path('blobs')/(sha+'_'+source.name)
        if digest(source) != sha: raise ValueError('Changed target evidence')
        if not (target/relative).exists(): link(source, target/relative)
        mapping.append({'path': path, 'sha256': sha, 'archived': relative.as_posix()})
    write(target/'lineage.json', {'files': mapping, 'full_history_sessions': len(sessions),
          'source_and_model_clocks_passed': True, 'scope': 'Original-target expanding/decay policies and fixed ATR sizing; all input bytes archived.'})
    return target


def prepare(out):
    from scripts.kalcb_recovery_qualification_round_v3 import check as prior_check
    check(out);prior_check(PRIOR)
    prior=read(PRIOR/'preparation.json')
    if not prior['complete']:raise ValueError('Incomplete prior preparation')
    items=[]
    for old in prior['profiles']:
        folder=out/'profiles'/old['name']/old['period']
        folder.mkdir(parents=True,exist_ok=False)
        items.append({**old,'prior_folder':old['folder'],'input_folder':old['folder'],'folder':str(folder)})
    write(out/'preparation.json',{**prior,'profiles':items,'protocol':bound(out/'protocol.json'),'prior_preparation':bound(PRIOR/'preparation.json'),'repair':'Independent validation context only; all native inputs remain immutable v3 artifacts.'})


def validate(folder, item, scenario, reference, save=True):
    from backtests.strategies.kalcb.qualified_artifacts import validate_case
    r = read(folder/'result.json'); a = read(folder/'execution_audit.json')
    request = read(folder/'request.json'); completion = read(folder/'completion.json')
    if not completion['complete'] or {r['source_generation_hash'], request['source_generation_hash'], completion['source_generation_hash']} != {item['source_generation_hash']}:
        raise ValueError('Wrong source identity')
    if r['audit_status'] != 'passed' or a['audit_status'] != 'passed' or not a['audit_pass'] or not r['execution_validity_pass']:
        raise ValueError('Failed full source qualification')
    if any(r[k] for k in ('residual_sessions', 'terminal_inventory', 'terminal_pending')):
        raise ValueError('Unresolved inventory')
    daily = read(folder/'daily.json'); trades = read(folder/'trades.json')
    start, end, count = PERIODS[item['period']]
    if request['plan'] != read(item['plan']['path']) or Path(request['arguments']['audit_manifest']).resolve() != Path(item['manifest']['path']).resolve():
        raise ValueError('Wrong saved qualification request')
    native_validation = validate_case(folder, {
        'execution_source': {'source_generation_hash': item['source_generation_hash']},
        'audit_manifest': item['manifest']['path'],
        'policy': {'name': POLICY, 'manifest_sha256': item['policy']['sha256']},
        'initial_equity_krw': 100_000_000,
        'date_range': {'start': start, 'end': end, 'sessions': count},
        'metrics': {scenario: r}}, scenario)
    if save: write(folder/'independent_account_verification.json', native_validation['validation'])
    if len(daily) != count or daily[0]['day'] != start or daily[-1]['day'] != end:
        raise ValueError('Incomplete source calendar')
    if abs(sum(t['net_pnl'] for t in trades)-(daily[-1]['equity']-100_000_000)) > .01:
        raise ValueError('Cash mismatch')
    if scenario in ('baseline', 'combined'):
        previous = read(reference/'result.json')
        for key in ('return_pct', 'drawdown_pct', 'trades', 'active_days'):
            if abs(r[key]-previous[key]) > 1e-8: raise ValueError('Screening bridge drift '+key)
        prior_days = read(reference/'daily.json')
        if len(prior_days) != count or any(x['day'] != y['day'] or abs(x['equity']-y['equity']) > .01 for x, y in zip(daily, prior_days)):
            raise ValueError('Continuous equity bridge drift')
    return {k: r[k] for k in ('return_pct', 'drawdown_pct', 'trades', 'active_days', 'audit_status')}


def run(out):
    import msvcrt
    import psutil
    check(out)
    lock = (out/'run.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {'pid': os.getpid(), 'process_created': psutil.Process().create_time(), 'child': None}
    def status(stage, **kwargs):
        state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
        write(out/'status.tmp', state); os.replace(out/'status.tmp', out/'status.json')
    def child(command, cwd, log, stage):
        while psutil.virtual_memory().available < 2*1024**3:
            status('waiting_for_memory'); time.sleep(30)
        if (out/'hold.json').exists(): raise ValueError('Explicit hold')
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=cwd, env=env(cwd), stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            status(stage, child={'pid': process.pid, 'process_created': psutil.Process(process.pid).create_time(), 'command': command})
            while process.poll() is None: time.sleep(15)
        if process.returncode: raise ValueError('Failed native child; preserve '+str(log))
        status(stage+'_completed', child=None)
    try:
        if (out/'preparation.json').exists(): raise ValueError('Preserve existing qualification; no implicit restart')
        child([sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'prepare', '--root', str(out)], REPO, out/'prepare.log', 'prepare')
        check(out); preparation = read(out/'preparation.json')
        results = []
        for item in preparation['profiles']:
            aligned.verify_bindings([item['manifest'], item['plan'], item['policy']])
            folder = Path(item['folder']); inputs = Path(item['input_folder']); code = inputs/'inputs/files'
            for scenario in SCENARIOS:
                target = folder/scenario
                if target.exists(): raise ValueError('Preserve existing native evidence')
                command = [sys.executable, '-B', '-u', '-m', 'scripts.kalcb_executed_value_replay', '--plan', item['plan']['path'],
                           '--policies', str(code/'research/kalcb_executed_value/policies'), '--policy', POLICY,
                           '--scenario', scenario, '--audit-manifest', item['manifest']['path'], '--output', str(target)]
                previous=Path(item['prior_folder'])/scenario
                if (previous/'completion.json').exists():
                    from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
                    validate(previous,item,scenario,Path(PROFILES[item['name']]['reference'])/item['period']/scenario,save=False)
                    target.mkdir()
                    for filename in REPLAY_FILES:link(previous/filename,target/filename)
                    write(target/'reused_native_evidence.json',{'source':str(previous),'files':[bound(previous/f) for f in REPLAY_FILES],'reason':'Already source-audited native replay; independent verifier metadata fixed only.'})
                else:
                    child(command, code, folder/(scenario+'.log'), item['name']+'/'+item['period']+'/'+scenario)
                metrics = validate(target, item, scenario, Path(PROFILES[item['name']]['reference'])/item['period']/scenario)
                result = {'name': item['name'], 'period': item['period'], 'scenario': scenario, 'metrics': metrics, 'result': bound(target/'result.json')}
                results.append(result); write(out/'progress.json', results)
                print(result, flush=True)
        check(out)
        write(out/'completion.json', {'complete': True, 'results': results, 'preparation': bound(out/'preparation.json'),
              'scope': '32 source-audited native cases; no promotion. Direct intervention/stability/concentration/uncertainty and final diagnostics still require review.'})
        status('ready_for_qualification_review', child=None)
    except Exception as error:
        status('attention_required', error=repr(error)); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'prepare', 'run'])
    parser.add_argument('--root', type=Path, default=DEST)
    args = parser.parse_args()
    {'register': register, 'prepare': prepare, 'run': run}[args.stage](args.root)
