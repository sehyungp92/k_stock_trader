"""Bounded portfolio mechanism experiment and sealed chronological validation.

No live orders. Later-window prices are used by the source audit only; policy
outcomes cannot be evaluated before development selection and the OOS seal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.executed_value import digest, write_json

ROOT = Path.home() / '.codex/research/portfolio_targeted_validation_20261009'
PARENT = Path.home() / '.codex/research/portfolio_overnight_frontier_20261009/coarse/study.json'
OVERLAY = ('backtests/strategies/portfolio_synergy/targeted_policy.py',
           'strategy_olr/config.py', 'strategy_olr/research.py', 'strategy_olr/core/logic.py', 'strategy_olr/core/stale_release.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    protocol = {
        'version': 'portfolio-targeted-validation-v1', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'capital_krw': 25_000_000, 'development': {'start': '2025-08-07', 'end': '2026-03-31'},
        'consumed_validation': {'start': '2026-04-01', 'end': '2026-05-12', 'use': 'regression_only'},
        'reserved_validation': {'start': '2026-06-01', 'end': '2026-07-10',
            'use': 'Single development-selected finalist versus fixed selected and higher-return incumbent controls. Not an OOS winner sweep.',
            'source_gate': 'Require compatible point-in-time daily/flow/intraday inputs, calendar/coverage and frozen policy inference. Never infer support from filenames or synthesize missing prices.',
            'tail': 'Last two available sessions prohibit new entries; retained inventory must liquidate through ordinary fills.',
            'independence': 'Audit retained run windows and human statement. Missing historical logs remain an explicit limitation.'},
        'budgets': {'kalcb': .75, 'olr': .625, 'higher_return_reference_olr': .75},
        'candidates': ['control', 'k_marginal', 'olr_stale_release', 'joint', 'native_rank', 'proportional', 'incumbent'],
        'candidate_contracts': {
            'k_marginal': 'Opening KALCB incumbents only: compare current quantity tier with next smaller .25/.5/1 tier using the matching saved causal calibrated model lineage. Reduce one tier only when smaller has positive baseline and combined forecasts, both quantity cohorts have >=20 mature stock-days and >=8 sessions, failure probability <.001, and the current tier has no positive increment in at least one scenario. Otherwise retain current action. No added signals or hindsight labels; timed additions unchanged.',
            'olr_stale_release': 'After the intended next-session horizon has elapsed (second exchange session after entry or later), after three complete contiguous opening bars, release half of remaining shares once per session only if the latest close is below completed-session VWAP. Native managed exits retain priority; no pending exit may be replaced. Actual next-event partial fills release funding. Research risk heuristic, not a learned continuation-value claim.',
            'joint': 'Combine the two exact interventions; no extra knobs.',
            'native_rank': 'Existing native-rank allocation at selected budgets; mechanical priority comparator.',
            'proportional': 'Existing integer proportional allocation at selected budgets; coverage comparator.',
            'incumbent': 'Preserved .75/.75 control; never eligible to bypass risk gates.'},
        'stages': {'screen': ['baseline', 'combined'],
            'battery': ['baseline', 'cost', 'participation', 'combined', 'auction_price_baseline',
                        'auction_price_combined', 'auction_nonfill_baseline', 'auction_nonfill_combined'],
            'battery_trigger': 'At least +1pp return in both screen scenarios, DD no worse than control by >.25pp, entries >=98% of matched control in each screen, >=10 directly changed issuer-days on >=8 sessions, and no account/execution failures. Full battery only for screen qualifiers and fixed controls.',
            'stop': 'No parameter refinement after results. A failed or unsupported candidate is recorded, not rescued by changing gates.'},
        'development_risk': {'baseline_dd_pct': 11., 'adverse_dd_pct': 12.5, 'overnight_max': .8,
                             'overnight_mean': .3, 'worst_session_loss_pct': 4.5, 'entries_min': 515},
        'selection': 'Among full-battery feasible screen qualifiers, maximize minimum adverse return; within 1pp prefer minimum scenario entry count, then lower worst DD. Retain control if none qualifies. Report return and activity Pareto tradeoffs, and comparison to the higher-return incumbent.',
        'oos_acceptance': {'require_source_and_execution_pass': True, 'require_positive_baseline_and_combined_net_return': True,
            'require_no_lower_baseline_or_combined_return_than_control': True, 'min_frequency_ratio_to_control': .98,
            'max_dd_increase_pp_vs_control': .25, 'risk': 'same DD/exposure/worst-session limits; activity relative to OOS control, never a 515-trade requirement on a shorter window',
            'failure': 'No tuning, runner-up swap, or independent-performance promotion after failure. Report short-window uncertainty and failed gates.'},
        'references': {'selected_pointer': bound(REPO/'config/research/portfolio_synergy_selected.json'),
                       'parent_study': bound(PARENT),
                       'diagnosis': bound(REPO/'docs/research/portfolio_synergy_risk_return_diagnosis_20261009.txt')},
        'research_only': True, 'live_deployment_enabled': False}
    write_json(root/'protocol.json', protocol)
    print(json.dumps({'registered': str(root/'protocol.json'), 'candidates': protocol['candidates']}), flush=True)


def source_audit(root):
    """Source consistency, date coverage and split metadata, no strategy outcomes."""
    import pandas as pd
    import yaml
    from kis_core.trading_calendar import get_trading_calendar
    from datetime import date
    protocol = read(root/'protocol.json')
    universe = [str(s).zfill(6) for s in yaml.safe_load((REPO/'config/universe_103.yaml').read_text(encoding='utf-8'))['symbols']]
    start, end = (date.fromisoformat(protocol['reserved_validation'][k]) for k in ('start', 'end'))
    calendar = get_trading_calendar()
    sessions = []
    day = start
    while day <= end:
        if calendar.is_trading_day(day): sessions.append(day.isoformat())
        from datetime import timedelta
        day += timedelta(days=1)
    records, problems = [], []
    for symbol in universe:
        paths = sorted((REPO/'data/kis_intraday_parquet'/symbol).glob(symbol+'_5m_*.parquet'))
        paths = [p for p in paths if p.stem.split('_')[-1] >= end.strftime('%Y%m%d')]
        daily_paths = sorted((REPO/'data/krx_daily_parquet/daily_ohlcv'/symbol).glob('*.parquet'))
        flow_paths = sorted((REPO/'data/krx_daily_parquet/daily_flow'/symbol).glob('*.parquet'))
        item = {'symbol': symbol, 'intraday_candidates': [str(p) for p in paths]}
        if len(paths) != 1 or len(daily_paths) != 1 or len(flow_paths) != 1:
            problems.append({'symbol': symbol, 'reason': 'ambiguous_or_missing_source'}); records.append(item); continue
        ip, dp, fp = paths[0], daily_paths[0], flow_paths[0]
        intraday = pd.read_parquet(ip, columns=['timestamp','close'])
        intraday['timestamp'] = pd.to_datetime(intraday.timestamp, utc=True).dt.tz_convert('Asia/Seoul')
        intraday['day'] = intraday.timestamp.dt.strftime('%Y-%m-%d')
        intraday = intraday[intraday.day.isin(sessions)].sort_values('timestamp')
        daily = pd.read_parquet(dp, columns=['date','close']); daily['day'] = pd.to_datetime(daily.date).dt.strftime('%Y-%m-%d')
        flow = pd.read_parquet(fp, columns=['date']); flow['day'] = pd.to_datetime(flow.date).dt.strftime('%Y-%m-%d')
        closes = daily.set_index('day')['close'].to_dict()
        missing = {'intraday_sessions': sorted(set(sessions)-set(intraday.day)),
                   'daily_sessions': sorted(set(sessions)-set(daily.day)),
                   'flow_sessions': sorted(set(sessions)-set(flow.day))}
        mismatch = []
        for day, bars in intraday.groupby('day'):
            last = bars.iloc[-1]
            # Only exact exchange closing marks can challenge the daily-close basis.
            closing = calendar.session_hours(date.fromisoformat(day))[1]
            if last.timestamp.time().replace(tzinfo=None) != closing or day not in closes:
                continue
            a, b = float(last['close']), float(closes[day])
            if abs(a-b) > max(1., .001*abs(b)):
                mismatch.append({'day': day, 'intraday_close': a, 'daily_close': b, 'relative_difference': a/b-1 if b else None})
        item.update(sources=[bound(p) for p in (ip,dp,fp)], missing=missing,
                    closing_basis_discrepancies=mismatch, timestamp_rows=len(intraday),
                    intraday_duplicate_timestamps=int(intraday.timestamp.duplicated().sum()))
        if any(missing.values()) or mismatch or item['intraday_duplicate_timestamps']:
            problems.append({'symbol': symbol, 'reason': 'coverage_or_basis_requires_qualification',
                             'missing': missing, 'closing_basis_discrepancies': len(mismatch)})
        records.append(item)
    split_paths = [REPO/'data/backtests/output/kalcb/round_1/selected_research_profile.json',
        Path('E:/CodexResearch/k_stock_trader/kalcb_campaign_20261007/boundary_verification.json'),
        Path('E:/CodexResearch/k_stock_trader/olr_forward_validation_20261007/lineage_audit.json'),
        Path('E:/CodexResearch/k_stock_trader/olr_forward_validation_20261007/delivery_result.json')]
    splits = []
    for p in split_paths:
        value = read(p)
        allowed = ('status','date_range','training_coverage','june_july_2026_used','last_permitted_development_date',
                   'reserved_2026_june_july_outcomes_opened','verdict','candidate_window','retained_run_windows',
                   'dynamic_loader','historical_pnl_opened','historical_tail_independence')
        splits.append({'source': bound(p), 'metadata': {k:value[k] for k in allowed if k in value}})
    result = {'protocol': bound(root/'protocol.json'), 'scope': 'Date/coverage/source-basis audit only. No strategy PNL or OOS model fit.',
              'candidate_sessions': sessions, 'universe_size': len(universe), 'symbols': records,
              'problems': problems, 'source_gate_pass': not problems, 'split_lineage': splits,
              'limitations': ['Coverage of calendar dates is not complete bar/auction/exception qualification.',
                              'Timestamp-only flow coverage is not proof of flow units or point-in-time availability.',
                              'Missing original logs limit a categorical no-prior-use claim.']}
    write_json(root/'source_audit.json', result)
    print(json.dumps({'sessions':len(sessions), 'symbols':len(universe), 'source_gate_pass':not problems,
                      'problem_symbols':len(problems), 'basis_discrepancy_rows':sum(len(r.get('closing_basis_discrepancies',[])) for r in records)}), flush=True)


def prepare(root):
    from scripts.portfolio_native_budget_study import freeze
    protocol=read(root/'protocol.json'); parent=read(PARENT)
    if digest(PARENT)!=protocol['references']['parent_study']['sha256']: raise ValueError('Changed parent')
    base=parent['candidates']['k75_o62.5']; recipes={}; jobs=[]
    for name in protocol['candidates']:
        r=copy.deepcopy(base)
        if name in ('k_marginal','joint'):r['kalcb_marginal_quantity']='supported_next_lower_tier_v1'
        if name in ('olr_stale_release','joint'):r['olr_capacity']['capacity_stale_release']=True
        if name in ('native_rank','proportional'):r['native_budget']['allocation_mode']=name
        if name=='incumbent':r['native_budget']['olr_fraction']=.75
        recipes[name]=r
        jobs.extend({'recipe':name,'scenario':s,'group':'screen','candidate':name,'battery':s} for s in protocol['stages']['screen'])
    freeze(PARENT,root/'development',recipes,jobs,overlay_files=OVERLAY,native_source_unchanged=False,
           registration={'protocol':bound(root/'protocol.json'),'purpose':'Bounded registered mechanism comparison; source/lifecycle changes explicitly declared.'})


def bridge(root):
    from scripts.portfolio_native_budget_study import checked
    evidence = {}
    for scenario in ('baseline','combined'):
        new = root/'development/cases'/('control_'+scenario)
        old = PARENT.parent/'cases'/('k75_o62.5_'+scenario)
        checked(new); checked(old)
        matches = {f:digest(new/f)==digest(old/f) for f in ('trades.json','daily.json','equity.json')}
        if not all(matches.values()): raise ValueError('Disabled-policy bridge differs: '+scenario)
        evidence[scenario] = {'new':bound(new/'completion.json'),'parent':bound(old/'completion.json'),'byte_identity':matches}
    write_json(root/'bridge.json',evidence)
    print(json.dumps({'bridge_pass':True}),flush=True)


def compressed_rows(path):
    import gzip
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream: yield json.loads(line)


def direct_support(case, candidate):
    identities, sessions = set(),set()
    details = {}
    if candidate in ('k_marginal','joint'):
        evidence = read(case/'marginal_quantity_evidence.json')
        changed = {r['identity'] for r in evidence['decisions'] if r['changed']}
        submitted = {r['timestamp'][:10]+'/'+r['symbol'] for r in compressed_rows(case/'allocation_events.jsonl.gz')
                     if r['strategy_id']=='KALCB' and r['native_qty']>0}
        direct = changed & submitted
        identities |= direct; sessions |= {key[:10] for key in direct}
        details['kalcb'] = {'changed_snapshot_issuer_days':len(changed),'changed_native_request_issuer_days':len(direct),
                           'identities':sorted(direct),'evidence':bound(case/'marginal_quantity_evidence.json')}
    if candidate in ('olr_stale_release','joint'):
        actual = [f for f in read(case/'fills.json') if f['side']=='SELL' and f['reason']=='stale_inventory_release']
        direct = {f['metadata']['stale_release']['entry_time'][:10]+'/'+f['symbol'] for f in actual
                  if 'stale_release' in f.get('metadata',{})}
        # Native final single-share exits still carry the reason, but no partial
        # metadata. Match these to the collapsed entry lifecycle.
        for t in read(case/'trades.json'):
            if t['strategy_id']=='OLR' and t['exit_reason']=='stale_inventory_release':
                direct.add(t['entry_fill_time'][:10]+'/'+t['symbol'])
        identities |= direct; sessions |= {f['timestamp'][:10] for f in actual}
        details['olr'] = {'changed_entry_issuer_days':len(direct),'release_sessions':len({f['timestamp'][:10] for f in actual}),
                          'partial_fill_events':len(actual),'identities':sorted(direct)}
    if candidate in ('native_rank','proportional'):
        # A conservative upper bound: all names in scarce simultaneous cohorts
        # where priority could matter. Subsequent NAV-path changes are excluded.
        submissions = {r['submitted_order_id']:r for r in compressed_rows(case/'allocation_events.jsonl.gz')}
        for row in compressed_rows(case/'native_budget_admissions.jsonl.gz'):
            members = [(submissions[oid],a) for oid,a in row['ledger']['allocations'].items() if oid in submissions]
            groups = {}
            for event,allocation in members:
                groups.setdefault((event['strategy_id'],event['available_at']),[]).append((event,allocation))
            for (_,at), group in groups.items():
                if len(group)<2 or all(a['qty']>=a['requested_qty'] for _,a in group): continue
                if candidate=='native_rank':
                    def rank(event):
                        m=event.get('native_metadata',{})
                        return float(m.get('frontier_rank') or m.get('candidate_rank') or 10**9),event['symbol']
                    symbol_order=sorted(e['symbol'] for e,_ in group)
                    ranked=[e['symbol'] for e,_ in sorted(group,key=lambda pair:rank(pair[0]))]
                    if symbol_order==ranked:continue
                identities |= {at[:10]+'/'+e['symbol'] for e,_ in group}; sessions.add(at[:10])
        details['scope']='Upper bound of directly contested cohorts, not a causal count of changed fills. Cannot establish support by itself.'
    exact = candidate not in ('native_rank','proportional')
    return {'issuer_days':len(identities),'sessions':len(sessions),'exact_mechanism_activation':exact,
            'pass':len(identities)>=10 and len(sessions)>=8 and exact,'details':details}


def screen(root):
    from scripts.portfolio_allocation_challenge import risk_metrics
    protocol=read(root/'protocol.json'); results={}
    for name in protocol['candidates']:
        results[name]={}
        for scenario in protocol['stages']['screen']:
            case=root/'development/cases'/(name+'_'+scenario)
            results[name][scenario]={**risk_metrics(case),'support':direct_support(case,name)}
    qualifiers=[]
    for name,values in results.items():
        for scenario,m in values.items():
            control=results['control'][scenario]
            m['screen_checks']={'return_improvement':m['return_pct']>=control['return_pct']+1,
                'drawdown':m['drawdown_pct']<=control['drawdown_pct']+.25,
                'frequency':m['trades']>=.98*control['trades'],'direct_support':m['support']['pass']}
        if name not in ('control','incumbent') and all(all(m['screen_checks'].values()) for m in values.values()):
            qualifiers.append(name)
    payload={'protocol':bound(root/'protocol.json'),'development':bound(root/'development/study.json'),
             'results':results,'qualifiers':qualifiers,'independent_performance_established':False}
    write_json(root/'screen.json',payload)
    print(json.dumps({'screen_qualifiers':qualifiers,'results':{n:{s:{k:m[k] for k in ('return_pct','drawdown_pct','trades','screen_checks')}
          for s,m in v.items()} for n,v in results.items()}}),flush=True)


def prepare_battery(root):
    from scripts.portfolio_native_budget_study import freeze
    from scripts.portfolio_allocation_challenge import BATTERY
    stage=read(root/'screen.json'); study=read(root/'development/study.json')
    if digest(root/'protocol.json')!=stage['protocol']['sha256']:raise ValueError('Changed protocol')
    if not stage['qualifiers']:
        print(json.dumps({'battery_candidates':[],'reason':'No registered screen qualifier; reuse fixed-control batteries after exact bridge.'}));return
    recipes={};jobs=[]
    for name in stage['qualifiers']:
        for label,(scenario,stress) in BATTERY.items():
            suffix='_price' if 'auction_price' in label else '_nonfill' if 'auction_nonfill' in label else ''
            key=name+suffix;r=copy.deepcopy(study['candidates'][name])
            if stress:r['olr_execution_stress']=stress
            recipes[key]=r;jobs.append({'recipe':key,'scenario':scenario,'candidate':name,'battery':label,'group':'battery'})
    freeze(root/'development/study.json',root/'battery',recipes,jobs,overlay=False,native_source_unchanged=False,
           registration={'protocol':bound(root/'protocol.json'),'screen':bound(root/'screen.json'),
                         'purpose':'Full eight-scenario qualification of registered screen qualifiers only.'})


def development_checks(metrics,label,risk):
    return {'drawdown':metrics['drawdown_pct']<=risk['baseline_dd_pct' if label=='baseline' else 'adverse_dd_pct'],
            'frequency':metrics['trades']>=risk['entries_min'],
            'overnight_max':metrics['overnight_max']<=risk['overnight_max'],
            'overnight_mean':metrics['overnight_mean']<=risk['overnight_mean'],
            'worst_session':metrics['worst_session_pct']>=-risk['worst_session_loss_pct']}


def choose_finalist(qualified):
    if not qualified:return 'control'
    minimum={n:min(m['return_pct'] for s,m in b.items() if s!='baseline') for n,b in qualified.items()}
    best=max(minimum.values());near=[n for n in qualified if minimum[n]>=best-1]
    return max(near,key=lambda n:(min(m['trades'] for m in qualified[n].values()),
                               -max(m['drawdown_pct'] for m in qualified[n].values()),minimum[n],n))


def oos_checks(finalist,control,risk):
    """A short holdout uses relative activity, never the 156-session count floor."""
    checks={}
    for scenario in ('baseline','combined'):
        a,b=finalist[scenario],control[scenario]
        checks[scenario]={
            'positive_return':a['return_pct']>0,'not_lower_return':a['return_pct']>=b['return_pct'],
            'frequency':a['trades']>=.98*b['trades'],'drawdown_increase':a['drawdown_pct']<=b['drawdown_pct']+.25,
            **{k:v for k,v in development_checks(a,scenario,risk).items() if k!='frequency'}}
    return {'pass':all(all(v.values()) for v in checks.values()),'tests':checks}


def select(root):
    """Seal development choice. OOS never selects a runner-up or changes gates."""
    from scripts.portfolio_allocation_challenge import risk_metrics,BATTERY
    from scripts.portfolio_native_budget_study import checked
    protocol=read(root/'protocol.json'); stage=read(root/'screen.json')
    if digest(root/'protocol.json')!=stage['protocol']['sha256']:raise ValueError('Changed protocol')
    # Fixed controls already have a full, source-qualified identical development
    # path. Reuse all eight cases only after the exact disabled-policy bridge.
    bridge(root)
    controls={}; reuse={}
    for name in ('control','incumbent'):
        generation=PARENT.parent if name=='control' else Path.home()/'.codex/research/portfolio_allocation_challenge_20261009/mechanisms'
        prefix='k75_o62.5' if name=='control' else 'k075_o075'
        controls[name]={}
        for label,(scenario,stress) in BATTERY.items():
            suffix='_price' if 'auction_adverse_bps' in stress else '_nonfill' if 'auction_nonfill_rate' in stress else ''
            case=generation/'cases'/(prefix+suffix+'_'+scenario)
            if name=='control':
                audited=generation/'qualified'/case.name
                checked(audited,True)
                identity={f:digest(case/f)==digest(audited/f) for f in ('trades.json','daily.json','equity.json')}
                if not all(identity.values()):raise ValueError('Control full-audit path differs: '+label)
                reuse[label]={'case':bound(case/'completion.json'),'qualified':bound(audited/'completion.json'),'byte_identity':identity}
                controls[name][label]=risk_metrics(audited)
            else:
                controls[name][label]=risk_metrics(case)
                if label in ('baseline','combined'):
                    current=root/'development/cases'/('incumbent_'+label)
                    if any(digest(case/f)!=digest(current/f) for f in ('trades.json','daily.json','equity.json')):
                        raise ValueError('Higher-return reference bridge differs: '+label)
    candidates={};qualified={}
    if stage['qualifiers']:
        registration=read(root/'battery/registration.json')
        if digest(root/'screen.json')!=registration['screen']['sha256']:raise ValueError('Changed screen decision')
        for job in registration['jobs']:
            case=root/'battery/qualified'/(job['recipe']+'_'+job['scenario']);checked(case,True)
            if job['battery'] in ('baseline','combined'):
                fast=root/'development/cases'/(job['candidate']+'_'+job['scenario'])
                if any(digest(case/f)!=digest(fast/f) for f in ('trades.json','daily.json','equity.json')):
                    raise ValueError('Candidate full-audit path differs from screen: '+str(case))
            m=risk_metrics(case);m['risk_checks']=development_checks(m,job['battery'],protocol['development_risk'])
            candidates.setdefault(job['candidate'],{})[job['battery']]=m
        for name,battery in candidates.items():
            if set(battery)!=set(BATTERY):raise ValueError('Incomplete candidate battery')
            if all(all(m['risk_checks'].values()) for m in battery.values()):qualified[name]=battery
    finalist=choose_finalist(qualified)
    payload={'sealed_utc':datetime.now(timezone.utc).isoformat(),'protocol':bound(root/'protocol.json'),
        'screen':bound(root/'screen.json'),'bridge':bound(root/'bridge.json'),'finalist':finalist,
        'reason':'Registered development selection only; no OOS winner search. Retain control if no challenger meets all gates.',
        'recipe':read(root/'development/study.json')['candidates'][finalist],'control_batteries':controls,
        'candidate_batteries':candidates,'qualified_control_reuse':reuse,
        'incumbent_reference_status':'Execution/account checked; historical reference battery is not promoted as fully source-qualified.',
        'oos_performance_opened':False,'independent_performance_established':False,'live_enabled':False}
    path=root/'development_selection.json'
    if path.exists(): raise FileExistsError('Selection already sealed')
    write_json(path,payload)
    print(json.dumps({'sealed_finalist':finalist}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register','source-audit','prepare','bridge','screen','prepare-battery','select'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    {'register':register,'source-audit':source_audit,'prepare':prepare,'bridge':bridge,'screen':screen,
     'prepare-battery':prepare_battery,'select':select}[args.command](args.root)
