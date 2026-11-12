"""0036: frozen full-context lifecycle, ten new accounts and two reused controls."""
from collections import defaultdict
import argparse
import copy
from datetime import datetime, timedelta
import gzip
import json
import math
import os
from pathlib import Path
import sqlite3
from statistics import median
import subprocess
import sys
import traceback

import psutil
import yaml
import olr_selective_half_continuation_v1 as shared

REPO = shared.REPO
BASE = Path('E:/CodexResearch/k_stock_trader')
PARENT = BASE / 'olr_continuing_research_20261016/relative_context_v2'
OOS = BASE / 'olr_round1_optimization_20261010/minute_exit_v5/clockfix_generation_1'
ROOT = PARENT.parent / 'relative_qualification_v1'
SOURCE = '925e4e61454bed93a8b93a86c3ee12a7f648095546c98b76c91892e97bb5f6de'
read, sha, sig, write, status, now = shared.read, shared.sha, shared.sig, shared.write, shared.status, shared.now
SCENARIOS = [('nominal', 'baseline', False), ('stress', 'combined', False), ('capped', 'baseline', True)]
CASES = [(w, scenario, policy) for w in ['train', 'oos'] for scenario, _, _ in SCENARIOS
         for policy in ['control', 'full'] if (w, scenario) != ('train', 'nominal')]
HELPERS = ['olr_selective_half_continuation_v1.py', 'olr_relative_context_assessment_v1.py']


def install(source, destination, expected, authenticate=True):
    if authenticate:
        assert sha(source) == expected, source
    if destination.exists():
        destination.unlink()  # Only this new generation's link; never truncate a parent.
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def verify_protocol(root):
    p = read(root / 'protocol.json')
    assert sha(__file__) == p['runner_sha256']
    assert sha(root / 'design_at_freeze.md') == p['design_sha256']
    for name, digest in p['helpers'].items():
        assert sha(REPO / 'scripts' / name) == digest, name
    return p


def oos_features(files, study):
    universe = [str(s).zfill(6) for s in yaml.safe_load((files / 'config/universe_103.yaml').read_text(encoding='utf8'))['symbols']]
    assert len(universe) == 103
    database = files / study['database']
    con = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    window = study['configuration']['date_range']
    days = [x[0] for x in con.execute('SELECT DISTINCT day FROM bars WHERE day>=? AND day<=? ORDER BY day', (window['start'], window['end']))]
    assert len(days) == 69 and days[0] == '2026-04-01' and days[-1] == '2026-07-10'
    source = defaultdict(dict)
    for symbol in universe:
        rows = con.execute("SELECT day,ts,o,c,v FROM bars WHERE symbol=? AND ts>=? AND ts<=? AND substr(ts,12,5) BETWEEN '09:00' AND '09:25' ORDER BY ts",
                           (symbol, window['start']+'T00:00:00+09:00', window['end']+'T23:59:00+09:00')).fetchall()
        for day, stamp, opening, close, volume in rows:
            source[day].setdefault(symbol, []).append((stamp, opening, close, volume))
    values, coverage = {}, defaultdict(int)
    for day in days:
        opening = datetime.fromisoformat(day + 'T09:00:00+09:00')
        labels = [(opening+timedelta(minutes=5*i)).isoformat() for i in range(6)]
        own = {}
        for symbol in universe:
            rows = source[day].get(symbol, [])
            if [r[0] for r in rows] != labels or sum(r[3] for r in rows) <= 0:
                coverage['unsupported_own_prefix'] += 1
                continue
            assert rows[0][1] > 0 and rows[-1][2] > 0
            own[symbol] = dict(opening=rows[0][1], close=rows[-1][2], own_return=rows[-1][2]/rows[0][1]-1,
                               source_last_label=labels[-1], available_at=(opening+timedelta(minutes=30)).isoformat(),
                               observed_volume=sum(r[3] for r in rows))
        at = (opening+timedelta(minutes=30)).isoformat()
        values[at] = {}
        for symbol, row in own.items():
            peers = [x['own_return'] for s, x in own.items() if s != symbol]
            if len(peers) < 68:
                coverage['unsupported_peer_coverage'] += 1
                continue
            values[at][symbol] = dict(**row, peer_count=len(peers), market_median=median(peers))
            coverage['supported'] += 1
    # Independent day-wise SQL projection checks every published prefix/value.
    checks = 0
    for day in days:
        rows = con.execute("SELECT symbol,ts,o,c,v FROM bars WHERE day=? AND substr(ts,12,5) BETWEEN '09:00' AND '09:25' ORDER BY symbol,ts", (day,)).fetchall()
        grouped = defaultdict(list)
        for symbol, stamp, op, cl, vol in rows:
            if symbol in universe:
                grouped[symbol].append((stamp, op, cl, vol))
        eligible = {s: x for s, x in grouped.items() if len(x) == 6
                    and [r[0][11:16] for r in x] == ['09:00','09:05','09:10','09:15','09:20','09:25']
                    and sum(r[3] for r in x) > 0}
        published = values[day+'T09:30:00+09:00']
        expected = eligible if len(eligible) >= 69 else {}
        assert set(published) == set(expected)
        for symbol, row in published.items():
            x = eligible[symbol]
            other = [z[-1][2]/z[0][1]-1 for s, z in eligible.items() if s != symbol]
            assert row['opening'] == x[0][1] and row['close'] == x[-1][2]
            assert row['observed_volume'] == sum(z[3] for z in x)
            assert math.isclose(row['market_median'], median(other), abs_tol=1e-12)
            assert row['peer_count'] == len(other)
            checks += 1
    con.close()
    return dict(version='relative-context-v1', universe=universe, rows=values, coverage=dict(coverage),
                evaluation_days=days, independent_context_checks=checks, source_database_sha256=sha(database),
                universe_sha256=sha(files/'config/universe_103.yaml'),
                label_scope='Completed 09:00-09:25 prefixes only; unchanged 0035 predicates, no outcome selection')


def prepare(root):
    protocol = verify_protocol(root)
    assert sha(PARENT/'native_engines/train/ready.json') == protocol['parent_ready_sha256']
    assert sha(PARENT/'assessment.json') == protocol['parent_assessment_sha256']
    assert sha(OOS/'native_engines/oos/ready.json') == protocol['oos_ready_sha256']
    status(root, 'preparing_frozen_sources', queued=CASES)
    parent = read(PARENT/'native_engines/train/ready.json')
    pre = read(PARENT/'preflight.json')
    assert pre['tests_passed'] and pre['tests'] == 69 and pre['source'] == SOURCE
    assert sha(pre['test_report']) == pre['test_report_sha256']
    assert read(PARENT/'disabled_parity.json')['pass_']
    origin = Path(parent['files'])
    for window in ['train', 'oos']:
        files = root/'native_engines'/window/'files'
        files.mkdir(parents=True, exist_ok=False)
        manifest = copy.deepcopy(parent['manifest'])
        manifest.pop('manifest_hash', None)
        for name, digest in manifest['files'].items():
            install(origin/name, files/name, digest, authenticate=False)
        if window == 'train':
            source_hash, completed_case, _ = shared.frozen_imports(root)
            assert source_hash(origin) == source_hash(files) == SOURCE
            study = copy.deepcopy(parent['study'])
            qualification = parent['interval_qualification']
        else:
            old = read(OOS/'native_engines/oos/ready.json')
            old_files = Path(old['files'])
            for i, (name, digest) in enumerate(old['manifest']['files'].items()):
                if Path(name).suffix == '.py' or name.startswith('kis_core/data/'):
                    continue  # The complete accepted N0 implementation is retained.
                install(old_files/name, files/name, digest)
                manifest['files'][name] = digest
                if i % 1500 == 0:
                    status(root, 'authenticating_existing_oos_inputs', files=i)
            manifest['evidence'] = copy.deepcopy(old['manifest']['evidence'])
            study = copy.deepcopy(old['study'])
            study['candidates'] = copy.deepcopy(parent['study']['candidates'])
            for name in ['config/universe_103.yaml','config/olr/sector_map.yaml','kis_core/data/krx_holidays.yaml']:
                assert old['manifest']['files'][name] == parent['manifest']['files'][name], name
            for key in ['economic_policy_settings','drawdown_reserve_contract','structural_entry_settings','structural_liquidity_contract']:
                study.pop(key, None)
            from backtests.engine.olr_interval_service import IntervalParticipationScenario
            q = read(files/manifest['evidence']['qualification_path'])
            q['capacity']['model_version'] = 'olr-completed-interval-service-v1'
            for row in q['capacity']['registered_execution_policies']:
                capacity = row['policy'].get('capacity')
                if capacity:
                    row['policy']['capacity'] = IntervalParticipationScenario(capacity['participation'], capacity['name']).contract()
            qualification = 'research/relative_qualification/qualification_interval.json'
            write(files/qualification, q)
            manifest['files'][qualification] = sha(files/qualification)
            feature_path = 'research/relative_qualification/oos_features.json'
            features = oos_features(files, study)
            write(files/feature_path, features)
            manifest['files'][feature_path] = sha(files/feature_path)
            study['relative_context_features'] = feature_path
            connection = sqlite3.connect((files/study['database']).as_uri()+'?mode=ro',uri=True)
            history_checks = {}
            for policy in ['control','full']:
                checkpoint = read(PARENT/'cases/train'/policy/'capacity_terminal_state.json')
                checked = 0
                for symbol, history in checkpoint['capacity_history_v1'].items():
                    for kind in ['auction','continuous']:
                        for stamp, volume in history[kind]:
                            rows = connection.execute('SELECT v FROM bars WHERE day=? AND symbol=? AND ts=?',
                                                      (stamp[:10],symbol,stamp)).fetchall()
                            assert len(rows)==1 and rows[0][0]==volume, (policy,symbol,stamp)
                            assert stamp[:10]<'2026-04-01'
                            checked += 1
                history_checks[policy] = checked
            connection.close()
            write(root/'oos_input_receipt.json', dict(at=now(), parent_ready_sha256=sha(OOS/'native_engines/oos/ready.json'),
                  source_database_sha256=features['source_database_sha256'], features_sha256=sha(files/feature_path),
                  coverage=features['coverage'], evaluation_sessions=len(features['evaluation_days']), independent_prefix_checks=features['independent_context_checks'],
                  reused_nominal_history_source_checks=history_checks))
        study.update(code=str(files), source_generation_hash=SOURCE)
        manifest['source_generation_hash'] = SOURCE
        assert source_hash(files) == SOURCE
        write(root/'native_engines'/window/'ready.json', dict(files=str(files), study=study, manifest=manifest, interval_qualification=qualification))
    reused = {}
    for name in ['control','full']:
        case = PARENT/'cases/train'/name
        assert completed_case(case)['valid']
        reused[name] = dict(case=str(case), completion_sha256=sha(case/'completion.json'))
    # Exact unchanged engine: reuse accepted boundary tests, explicitly scoped.
    write(root/'preflight.json', dict(at=now(), source=SOURCE, reused_69_tests_sha256=pre['test_report_sha256'],
          same_code_both_windows=True, oos_input_receipt_sha256=sha(root/'oos_input_receipt.json'),
          reused_nominal=reused, protocol_sha256=sha(root/'protocol.json'), runner_sha256=sha(__file__), passed=True))
    status(root, 'ready', queued=CASES)


def launch(root, window, scenario_tag, policy):
    ready = read(root/'native_engines'/window/'ready.json')
    files = Path(ready['files'])
    study, manifest = copy.deepcopy(ready['study']), copy.deepcopy(ready['manifest'])
    scenario, capped = next((s,c) for n,s,c in SCENARIOS if n == scenario_tag)
    tag = policy+'_'+scenario_tag
    out = root/'cases'/window/tag
    assert not out.exists(), 'Preserve existing native case'
    prefix = 'research/relative_qualification/'+window+'/'+tag+'/'
    for key, name, value in [
        ('structural_settings','structural.json',dict(mode='control',interval_service=True)),
        ('funded_probe_settings','probe.json',dict(version='first-native-fade-funded-probe-v1',action='hold',control_reference=None)),
        ('funded_population_settings','population.json',dict(version='monthly-native-funded-population-v1',cell=None,catalog_reference=None)),
        ('relative_context_settings','context.json',dict(fraction=1. if policy=='full' else 0.,version='relative-context-v1'))]:
        study[key] = prefix+name
        write(files/study[key],value)
        manifest['files'][study[key]] = sha(files/study[key])
    if window == 'oos':
        prior = PARENT/'cases/train'/policy if scenario_tag=='nominal' else root/'cases/train'/tag
        for key, name in [('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_completion','completion.json'),('capacity_checkpoint_result','result.json')]:
            relative = prefix+'parent_'+name
            install(prior/name, files/relative, sha(prior/name))
            study[key] = relative
            manifest['files'][relative] = sha(files/relative)
    manifest['evidence']['qualification_path'] = ready['interval_qualification']
    study['manifest'] = str(files.parent/(tag+'.manifest.json'))
    study['configuration']['input_manifest_path'] = study['manifest']
    study['spec_path'] = prefix+'spec.json'
    write(files/study['spec_path'], {k:v for k,v in study.items() if k not in ['code','manifest','source_generation_hash','spec_path']})
    manifest['files'][study['spec_path']] = sha(files/study['spec_path'])
    write(study['manifest'], dict(**manifest, manifest_hash=sig(manifest)))
    runtime = root/'runtime'/window/tag
    write(runtime/'study.json', study)
    command = [sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(runtime/'study.json'),
               '--recipe','candidate','--scenario',scenario,'--standalone','OLR','--output',str(out),'--full-audit']
    if capped:
        command.append('--no-upward-compounding')
    log = root/'logs'/window/(tag+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf8') as stream:
        process = subprocess.Popen(command,cwd=files,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
    identity = dict(at=now(),pid=process.pid,created_at=psutil.Process(process.pid).create_time(),command=command,case=str(out),log=str(log))
    write(runtime/'process.json',identity)
    return process, out


def summarize(case):
    summary = shared.economic_summary(case)
    summary['context_decisions'] = []
    with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('reason') in ['relative_context_reduce','relative_context_exit']:
                summary['context_decisions'].append(row)
    return summary


def assess_pair(root, window, scenario, summaries, contrast):
    directory = root/'pairs'/(window+'_'+scenario)
    control, full = [summaries[window+'_'+scenario+'_'+p] for p in ['control','full']]
    account = contrast(read(Path(control['case'])/'trades.json'), read(Path(full['case'])/'trades.json'))
    dates, months = defaultdict(float), defaultdict(float)
    for row in account['per_opportunity']:
        dates[row['day']] += row['delta_pnl']
        months[row['day'][:7]] += row['delta_pnl']
    pair = dict(delta={k:full[k]-control[k] for k in ['return_pct','full_dd_pct','common_dd_pct','maximum_currency_decline','meaningful','meaningful_days','buy_notional','winning_wealth','losing_wealth','longest_underwater_calendar_days']},
                accounting=account,entry_date_attribution=dict(dates),entry_month_attribution=dict(months),
                excluding_best_entry_date=sum(dates.values())-max(dates.values(),default=0))
    write(directory/'assessment.json',dict(at=now(),summaries=dict(control=control,full=full),comparisons=dict(control_to_full=pair),economic_objective_achieved=False))
    command = [sys.executable,'-B',str(REPO/'scripts/olr_relative_context_assessment_v1.py'),str(directory)]
    with (directory/'causal_review.log').open('x',encoding='utf8') as stream:
        result = subprocess.run(command,cwd=REPO,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
    assert result.returncode == 0, 'Preserve failed pair review: '+str(directory)
    return pair


def participation_value(directory):
    pair = read(directory/'assessment.json')
    review = read(directory/'causal_review_v1/assessment.json')['actions_and_funding']['full']
    trades = {name:{(t['symbol'],t['entry_decision_time'][:10]):t for t in read(Path(s['case'])/'trades.json')}
              for name,s in pair['summaries'].items()}
    keys = trades['control'].keys() | trades['full'].keys()
    meaningful = lambda t: bool(t and t['qty']*t['entry_price']>=1e6)
    changed = {k for k in keys if meaningful(trades['control'].get(k)) != meaningful(trades['full'].get(k))}
    for renewal in review['same_day_native_renewals']:
        if (renewal['symbol'],renewal['day']) in changed:
            changed.add((renewal['symbol'],renewal['prior_entry_day']))
    rows = pair['comparisons']['control_to_full']['accounting']['per_opportunity']
    return dict(keys=sorted(changed),net_pnl_attribution=sum(x['delta_pnl'] for x in rows if (x['symbol'],x['day']) in changed),
                scope='Changed meaningful participation plus prior lifecycle for counted renewals; additive account attribution, not isolated causal mediation.')


def execute(root):
    verify_protocol(root)
    prepare(root)
    source_hash, completed_case, contrast = shared.frozen_imports(root)
    summaries, comparisons = {}, {}
    previous = read(PARENT/'assessment.json')
    for policy in ['control','full']:
        summaries['train_nominal_'+policy] = previous['summaries'][policy]
    comparisons['train_nominal'] = assess_pair(root,'train','nominal',summaries,contrast)
    for index,(window,scenario,policy) in enumerate(CASES):
        verify_protocol(root)
        assert source_hash(root/'native_engines'/window/'files') == SOURCE
        key = window+'_'+scenario+'_'+policy
        status(root,'running',case=key,accepted=list(summaries),queued=CASES[index+1:])
        process, case = launch(root,window,scenario,policy)
        assert process.wait() == 0, 'Native replay failed: '+str(case)
        assert completed_case(case)['valid']
        assert read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
        assert read(case/'funded_population_receipt.json')['audit']['pass_']
        if window == 'oos':
            restart = read(case/'capacity_restart_receipt.json')
            assert restart['same_configuration_and_scenario'] and restart['all_samples_verified_against_oos_retained_ohlcv']
        summaries[key] = summarize(case)
        if policy == 'control':
            assert not summaries[key]['context_decisions']
        write(root/'accepted'/(key+'.json'),summaries[key])
        if policy == 'full':
            comparisons[window+'_'+scenario] = assess_pair(root,window,scenario,summaries,contrast)
        status(root,'case_accepted',case=key,accepted=list(summaries),queued=CASES[index+1:])
    activity = {w:participation_value(root/'pairs'/(w+'_nominal')) for w in ['train','oos']}
    checks = {w+'_nominal_material_return':comparisons[w+'_nominal']['delta']['return_pct']>=.5 for w in ['train','oos']}
    for scenario in ['stress','capped']:
        values = [comparisons[w+'_'+scenario]['delta']['return_pct'] for w in ['train','oos']]
        checks[scenario+'_return_support'] = sum(values)>0 and min(values)>=-.5
    changes = [comparisons[w+'_nominal']['delta']['meaningful'] for w in ['train','oos']]
    checks['meaningful_participation'] = sum(changes)>=2 and changes[1]>=-1
    checks['positive_changed_participation_value'] = sum(v['net_pnl_attribution'] for v in activity.values())>0
    write(root/'assessment.json',dict(at=now(),summaries=summaries,comparisons=comparisons,activity=activity,necessary_checks=checks,
          necessary_checks_pass=all(checks.values()),economic_objective_achieved=False,
          next='Complete registered joint risk/recovery, concentration, transfer and causal assessment. Necessary numerical checks alone do not establish delivery; no automatic redesign.'))
    status(root,'awaiting_qualification_disposition',accepted=list(summaries),queued=[],necessary_checks=checks,
           next='Economic objective remains unproven until substantive registered assessment; no new variants.')


def start(root):
    root.mkdir(parents=True,exist_ok=False)
    design = REPO/'docs/optimisation/olr/experiments/olr-exp-0036.md'
    (root/'design_at_freeze.md').write_bytes(design.read_bytes())
    write(root/'protocol.json',dict(at=now(),experiment='olr-exp-0036',design_sha256=sha(design),runner_sha256=sha(__file__),
          helpers={n:sha(REPO/'scripts'/n) for n in HELPERS},cases=CASES,reused_cases=['train_nominal_control','train_nominal_full'],
          source=SOURCE,parent_ready_sha256=sha(PARENT/'native_engines/train/ready.json'),parent_assessment_sha256=sha(PARENT/'assessment.json'),
          oos_ready_sha256=sha(OOS/'native_engines/oos/ready.json'),concurrency=1,planned_wall_hours=6,new_data=False))
    command = [sys.executable,'-B','-u',str(Path(__file__).resolve()),'execute','--root',str(root)]
    with (root/'controller.log').open('x',encoding='utf8') as stream:
        p = subprocess.Popen(command,cwd=REPO,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
    identity = dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=command,log=str(root/'controller.log'))
    write(root/'controller_process.json',identity)
    print(json.dumps(identity),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['start','execute'])
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    try:
        globals()[args.phase](args.root)
    except Exception as exc:
        if args.root.exists():
            status(args.root,'needs_repair',error=repr(exc),traceback=traceback.format_exc())
        raise
