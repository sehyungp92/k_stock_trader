"""Version3: preserve period-qualified source contracts and prove calendar parity."""
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

PRIOR = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v2')
DEST = PRIOR.with_name('v3')
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
    if out.exists():
        raise ValueError('Preserve previous qualification')
    blend.check(blend.DEST)
    if not read(blend.DEST/'replay_completion.json')['complete']:
        raise ValueError('Finite experiments incomplete')
    tests = read(ROOT/'qualification_gate_tests_20261010.json')
    if not tests['complete'] or tests['exit_code']:
        raise ValueError('Qualification gate tests failed')
    paths = [Path(__file__), ROOT/'stage5_review.json', ROOT/'stage4_review.json', ROOT/'stage3_review.json',
             ROOT/'stage2_review.json', ROOT/'qualification_gate_tests_20261010.json', Path(tests['log']['path']),
             ROOT/'inputs/manifest.json', IS_PARENT, ROOT/'boundary_continuity.json',
             ROOT/'history_boundary_audit.json', REPO/'docs/strategy-implementation-lessons.md',
             REPO/'config/research/kalcb_intraday_advantage_20261007/selected_config.json',
             REPO/'scripts/kalcb_executed_value_freeze.py', REPO/'scripts/kalcb_recovery_ablation.py',
             REPO/'backtests/strategies/kalcb/oos_ablation.py',
             REPO/'backtests/strategies/kalcb/qualified_artifacts.py']
    for root in (aligned.DEST, mechanism.DEST, blend.DEST):
        paths += [root/'protocol.json', root/'replay_completion.json']
    for spec in PROFILES.values():
        for period in PERIODS:
            paths += [Path(spec['policy'])/period/'policies_manifest.json']
            for scenario in ('baseline', 'combined'):
                folder = Path(spec['reference'])/period/scenario
                paths += [folder/f for f in ('request.json', 'completion.json', 'result.json', 'daily.json', 'trades.json')]
    out.mkdir(parents=True)
    write(out/'protocol.json', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'profiles': PROFILES, 'scenarios': SCENARIOS, 'periods': PERIODS,
        'bindings': [bound(p) for p in sorted(set(paths))],
        'purpose': 'Qualify four representative recovery tradeoffs after all61 trial entries; not a new parameter search or automatic canonical replacement.',
        'history': 'Continuous original60-session minimum/one-session embargo/global20-session folds; own earlier OOF calibration; boundary outcomes mature before use.',
        'scope': 'Reused development; independent100m IS and OOS accounts. ExpandingATR5 now receives a fresh full IS replay; historical references are bridge targets only.',
        'acceptance': 'Every case must pass native source-bar/fill/activation/cash/position audit with exact bound configuration and source; baseline/combined must reproduce prior economic paths. Negative performance is retained, never a physical acceptance failure.',
        'research_evidence': 'Before canonical promotion consume component/backout, ATR4/5 and blend neighborhoods, issuer concentration, paired uncertainty and direct feasible intervention support (prefer10 issuer-days/8 sessions). Candidate changes and descendants are not independent interventions.',
        'promotion': 'Disabled in this program. Source qualification alone is insufficient. No live changes.'})
    repair = read(out/'protocol.json')
    repair['repair'] = 'IS bundle mixed newer calendar/parser runtime with older retained-source contract. Preserve the fully qualified IS generation and its source contract, replace only the explicitly bound audit-context replay helper, and prove economic code plus all in-scope calendar facts equal the full-window runtime. OOS keeps its already qualified generation. No price/model/config/policy change.'
    repair['bindings'] += [bound(PRIOR/p) for p in ('protocol.json','preparation.json','status.json','background.err.log','profiles/expanding/is/baseline.log')]
    repair['bindings'] += [bound(REPO/'scripts/kalcb_recovery_qualification_round.py'),bound(REPO/'scripts/kalcb_recovery_qualification_round_v2.py'),bound(IS_PARENT)]
    for item in read(PRIOR/'preparation.json')['profiles']:
        repair['bindings'] += [item['manifest'],item['plan'],item['policy']]
    write(out/'protocol.json',repair)
    print({'registered': str(out), 'native_cases': 32}, flush=True)


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
    from copy import deepcopy
    from backtests.strategies.kalcb.executed_value import signature
    from scripts.kalcb_recovery_qualification_round_v2 import check as prior_check
    check(out); prior_check(PRIOR)
    prior = read(PRIOR/'preparation.json')
    if not prior['complete']: raise ValueError('Incomplete prior preparation')
    historical = read(IS_PARENT); hid = historical.pop('manifest_hash')
    if signature(historical) != hid: raise ValueError('Changed qualified IS source generation')
    runtime = ROOT/'inputs/files'; current = read(ROOT/'inputs/manifest.json')
    economic_prefixes = ('strategy_kalcb/','strategy_common/','backtests/engine/','backtests/core/',
                         'backtests/strategies/kalcb/runner.py','scripts/kalcb_executed_value_study.py','kis_core/trading_calendar.py')
    comparisons = [n for n in historical['files'] if n.endswith('.py') and n.startswith(economic_prefixes)]
    if any(historical['files'][n] != current['files'].get(n) for n in comparisons):
        raise ValueError('Qualified IS economic/audit code differs from screening')
    # The sole replay helper delta is bound audit context, not decision/fill code.
    import ast
    old_ast = ast.parse((IS_PARENT.parent/'files/scripts/kalcb_executed_value_replay.py').read_text())
    new_ast = ast.parse((runtime/'scripts/kalcb_executed_value_replay.py').read_text())
    # All functions except run and the newly added audit_configuration are exact.
    functions = lambda module: {n.name:ast.dump(n) for n in module.body if isinstance(n,ast.FunctionDef) and n.name not in ('run','audit_configuration')}
    if functions(old_ast) != functions(new_ast): raise ValueError('Unexpected replay helper change')
    old_text = (IS_PARENT.parent/'files/scripts/kalcb_executed_value_replay.py').read_text()
    new_text = (runtime/'scripts/kalcb_executed_value_replay.py').read_text()
    start = lambda text: text[text.index('def run('):text.index('    identity = ',text.index('def run('))]
    if start(old_text) != start(new_text): raise ValueError('Economic replay loop changed')
    verified = set(); items = []; bridges = []
    def authenticate(path, sha):
        stat=path.stat(); key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,sha)
        if key not in verified:
            if digest(path)!=sha: raise ValueError('Changed frozen dependency '+str(path))
            verified.add(key)
    calendar_probe = """import sys,json,hashlib,yaml
from pathlib import Path
from datetime import date,timedelta
from kis_core.trading_calendar import get_trading_calendar
c=get_trading_calendar();symbols=sorted(yaml.safe_load(Path(sys.argv[1]).read_text())['symbols']);facts=[]
day=date(2025,5,12);end=date(2026,3,31)
while day<=end:
 facts.append([str(day),c.is_trading_day(day),str(c.session_hours(day))])
 for s in symbols:facts.append([str(day),s,str(c.closing_match_time(s,day)),str(c.intraday_halts_for(s,day)),str(c.periodic_auction_sources(s,day))])
 day+=timedelta(days=1)
print(json.dumps({'facts':len(facts),'sha256':hashlib.sha256(json.dumps(facts,sort_keys=True).encode()).hexdigest()}))"""
    source_probe = """import sys,json,yaml
from pathlib import Path
from backtests.core.input_manifest import InputManifest
from backtests.engine.retained_source import verify_acceptance
from backtests.auto.shared.cache_keys import source_generation_hash
m=InputManifest(Path(sys.argv[1]));receipt=m.receipt()
q=json.loads(m.read(m.root/m.payload['evidence']['qualification_path']))
a=verify_acceptance(lambda n:json.loads(m.read(m.root/n)),q,generation=m)
plan=json.loads(Path(sys.argv[2]).read_text());symbols=set(yaml.safe_load(Path(plan['universe']).read_text())['symbols'])
failures=[x for x in a['basis_failures'] if x.get('symbol') in symbols and plan['start']<=x.get('date','')<=plan['end']]
if failures:raise ValueError('In-scope retained-source basis failure')
print(json.dumps({'receipt_passed':True,'source_generation_hash':source_generation_hash(),'market_input_hash':receipt['market_input_hash'],'in_scope_basis_failures':len(failures),'out_of_scope_failures_retained':len(a['basis_failures'])}))"""
    for old in prior['profiles']:
        source_folder=Path(old['folder']);folder=out/'profiles'/old['name']/old['period'];folder.mkdir(parents=True)
        original=read(source_folder/'inputs/manifest.json');identity=original.pop('manifest_hash')
        if signature(original)!=identity:raise ValueError('Changed prior manifest')
        if old['period']=='is':
            selected={n:(IS_PARENT.parent/'files'/n,h) for n,h in historical['files'].items() if not n.startswith('research/kalcb_executed_value/')}
            selected.update({n:(source_folder/'inputs/files'/n,h) for n,h in original['files'].items() if n.startswith('research/kalcb_executed_value/')})
            selected['scripts/kalcb_executed_value_replay.py']=(runtime/'scripts/kalcb_executed_value_replay.py',current['files']['scripts/kalcb_executed_value_replay.py'])
            payload=deepcopy(historical);payload['evidence']['executed_value_policy']=original['evidence']['executed_value_policy']
        else:
            selected={n:(source_folder/'inputs/files'/n,h) for n,h in original['files'].items()};payload=deepcopy(original)
        files={}
        for name,(source,sha) in selected.items():
            authenticate(source,sha);link(source,folder/'inputs/files'/name);files[name]=sha
        for name in ('plan.json','config.json','audit_configuration.json'):link(source_folder/name,folder/name)
        payload['files']=files
        payload['evidence']['qualification_period_source_repair']={'prior_manifest':old['manifest'],'qualified_is_parent':bound(IS_PARENT),'program':bound(__file__),'economic_code_comparisons':len(comparisons),'in_scope_calendar_bridge_required':True}
        code=folder/'inputs/files'
        source=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'],cwd=code,env=env(code),text=True).strip()
        payload['source_generation_hash']=source
        write(folder/'inputs/manifest.json',{**payload,'manifest_hash':signature(payload)})
        universe=read(folder/'plan.json')['universe']
        before=json.loads(subprocess.check_output([sys.executable,'-B','-c',calendar_probe,universe],cwd=runtime,env=env(runtime),text=True))
        after=json.loads(subprocess.check_output([sys.executable,'-B','-c',calendar_probe,universe],cwd=code,env=env(code),text=True))
        if before!=after:raise ValueError('IS calendar fact drift')
        preflight=json.loads(subprocess.check_output([sys.executable,'-B','-c',source_probe,str(folder/'inputs/manifest.json'),str(folder/'plan.json')],cwd=code,env=env(code),text=True))
        if preflight['source_generation_hash']!=source:raise ValueError('Preflight source mismatch')
        if digest(folder/'plan.json')!=old['plan']['sha256']:raise ValueError('Plan changed')
        bridge={'name':old['name'],'period':old['period'],'calendar':after,'source_preflight':preflight,'passed':True}
        bridges.append(bridge);print(bridge,flush=True)
        items.append({**old,'folder':str(folder),'source_generation_hash':source,'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json')})
    write(out/'source_contract_bridge.json',{'passed':True,'profiles':bridges,'economic_code_comparisons':len(comparisons),'distinct_verified_file_identities':len(verified)})
    write(out/'preparation.json',{'complete':True,'profiles':items,'lineage':prior['lineage'],'protocol':bound(out/'protocol.json'),'prior_preparation':bound(PRIOR/'preparation.json'),'source_contract_bridge':bound(out/'source_contract_bridge.json')})


def validate(folder, item, scenario, reference):
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
        'policy': {'name': POLICY, 'manifest_sha256': item['policy']['sha256']},
        'initial_equity_krw': 100_000_000,
        'date_range': {'start': start, 'end': end, 'sessions': count},
        'metrics': {scenario: r}}, scenario)
    write(folder/'independent_account_verification.json', native_validation['validation'])
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
            folder = Path(item['folder']); code = folder/'inputs/files'
            for scenario in SCENARIOS:
                target = folder/scenario
                if target.exists(): raise ValueError('Preserve existing native evidence')
                command = [sys.executable, '-B', '-u', '-m', 'scripts.kalcb_executed_value_replay', '--plan', str(folder/'plan.json'),
                           '--policies', str(code/'research/kalcb_executed_value/policies'), '--policy', POLICY,
                           '--scenario', scenario, '--audit-manifest', str(folder/'inputs/manifest.json'), '--output', str(target)]
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
