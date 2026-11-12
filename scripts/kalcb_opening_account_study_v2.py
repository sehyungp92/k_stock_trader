"""Register and run a finite complete-account study using the shared PhaseRunner."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.opening_account_recovery_v2 import authenticate,publish
from backtests.strategies.kalcb.opening_account_io_v1 import OVERLAYS,freeze

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SCENARIOS=('baseline','participation','cost','combined')


def register(root,tests,preflight,predecessor):
    proof=authenticate(predecessor)
    if root.exists():
        raise ValueError('Preserve existing experiment; choose a fresh version')
    test=read(tests)
    if test.get('exit_code')!=0 or test.get('tests',0)<13:
        raise ValueError('Passed complete-account tests required')
    verify(test['programs']+[test['log']])
    pre=read(preflight)
    if any('error' in r for r in pre['rows']) or not any(r.get('solution',{}).get('changed') for r in pre['rows']):
        raise ValueError('Changed feasible real opening batches required before economic dispatch')
    profiles={}
    bindings=[bound(tests),bound(preflight)]
    references={}
    for period in ('is','oos'):
        path=BASE/'v25_pending_funding_screen/cases/control'/period/'input_profile.json'
        profiles[period]=read(path)
        verify([profiles[period][k] for k in ('plan','manifest','policy')])
        bindings.append(bound(path))
        references[period]={}
        for scenario in SCENARIOS:
            folder=(BASE/('v25_pending_funding_screen' if scenario in ('baseline','combined') else 'v28_pending_funding_qualification')
                    /'cases/control'/period/scenario)
            references[period][scenario]=str(folder)
            for name in ('completion.json','result.json','fills.json','equity.json','daily.json','execution_audit.json'):
                bindings.append(bound(folder/name))
    databases=read(BASE/'v51_pending_entry_account_pilot/protocol.json')['databases']
    root.mkdir(parents=True,exist_ok=False)
    overlays={}
    for rel in OVERLAYS:
        path=root/'overlays'/rel;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((REPO/rel).read_bytes());overlays[rel]=bound(path)
    code_files=[*OVERLAYS,'backtests/strategies/kalcb/opening_account_io_v1.py',
        'backtests/strategies/kalcb/opening_account_plugin_v2.py','scripts/kalcb_opening_account_study_v2.py',
        'backtests/strategies/kalcb/opening_account_recovery_v2.py','backtests/strategies/kalcb/late_artifact_validation_v2.py',
        'tests/test_kalcb_opening_account_v1.py']
    code_files+=['backtests/strategies/kalcb/'+p+'.py' for p in ('oos_ablation','retention_research_io_v1',
        'late_entry_io_v5','research_process_v1','qualified_artifacts')]
    code_files += [x.relative_to(REPO).as_posix() for x in (REPO/'backtests/auto/shared').glob('*.py')]
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),'version':'opening-complete-account-v2-validator-recovery','recovery':proof,
        'overlays':overlays,'programs':[bound(REPO/p) for p in code_files], 'profiles':profiles,
        'references':references,'bindings':bindings,'databases':databases,'database_bindings':[bound(p) for p in databases],
        'recipes':['control','joint','uniform'],'scenarios':SCENARIOS,'native_cases':24,
        'mechanism':'At the completed09:25 decision, preview all opening intents from the complete account. '
          'Use synchronized actual half-hour opens in exactly60 completed prior sessions with one full session embargo. '
          'The archive mixes5m/30m rows, so no coarse close is assigned a false5m clock. '
          'Exchange-shortened sessions use their actual calendar. Interrupted days remain included; '
          'unobserved common clocks are disclosed, not interpolated. This coarse stress proxy does not bound intrabar DD or gaps. '
          'Budget the mean worst6 daily adverse basket excursions plus full roundtrip costs to the unchanged '
          'risk_per_trade_pct*heat_cap_r (4% of equity); reserve pending buys and retain filled holdings. '
          'Maximize sum of normalized unchanged parent value forecasts times log(1+retained fraction), '
          'subject to original integer quantity caps and shared funding. Uniform control matches estimated tail risk. '
          'No new entry signal, external data, fitted model, thresholds grid, stop, hindsight symbol filter or forced exit.',
        'readiness':'Targets simultaneous financial exposure missed by isolated q labels and nominal unexecuted stop risk. '
          'Different from broad ATR cuts and V28 order-dependent funding cap. Forecast utility is a prior; '
          'native account returns decide whether it transfers. Unsupported history preserves the parent and is reported.',
        'objective':{'min_is_return_retention_each_stress':.98,'min_oos_return_gain_pp_each_stress':1.,
            'max_dd_deterioration_pp_each_period_stress':0.,'min_material_lifetime_retention':.95,
            'material_entry_notional_fraction_of_equity':.01,
            'scope':'Prespecified development frontier reflecting aggressive controlled risk; not a unique mathematical optimum. '
                    'Report every raw tradeoff, activity change and sacrificed parent winner.'},
        'forks':'First4 chronological supported changed IS control states per baseline/combined scenario. '
                'Exact disabled whole-state bridge; joint/uniform EOD full-account suffix labels cannot be summed as strategy return.',
        'learning':'Unchanged parent100 causal forecast/quantity policies and original history, splits, clocks, refits. '
                   'prior125 retained as historical sensitivity comparator. No new alpha model fit.',
        'scope':'IS156 and reused OOS69 sessions, all4stresses; one finite intervention plus matched-risk control. '
                'Process completion is separate from objective achievement; no live/canonical change.',
        'study_complete':False}
    if {k:v['sha256'] for k,v in overlays.items()}!=proof['overlays']:
        raise ValueError('Economic overlay changed during validator repair')
    protocol['bindings']+=proof['bindings']
    write(root/'protocol.json',protocol)
    publish(root,proof)
    write(root/'progress.json',[])
    write(root/'study.json',{'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
        'bound_inputs':[bound(root/'protocol.json')], 'source_programs':protocol['programs'],
        'recipes':{x:{'allocation':x} for x in protocol['recipes']},'periods':{'is':{'sessions':156},'oos':{'sessions':69}}})
    print(json.dumps({'registered':str(root),'native_cases':24}),flush=True)


def activity(folder):
    fills=read(folder/'fills.json');equity=read(folder/'equity.json')
    marks={str(t):v for t,v in equity};groups={}
    for f in fills:
        if f['side']!='BUY':
            continue
        item=groups.setdefault((f['symbol'],f['order_id']),{'notional':0.,'equity':float(f['metadata'].get('portfolio_equity') or 100000000.)})
        item['notional']+=f['qty']*f['price']
    return {'funded_entry_orders':len(groups),'material_funded_entries':sum(v['notional']>=.01*v['equity'] for v in groups.values()),
            'buy_notional':sum(v['notional'] for v in groups.values())}


def run(root):
    import psutil
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    from backtests.strategies.kalcb.opening_account_plugin_v2 import OpeningAccountPlugin
    from backtests.strategies.kalcb.research_process_v1 import execute
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+p['database_bindings'])
    me=psutil.Process()
    write(root/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    for process in psutil.process_iter(['pid','name','cmdline']):
        cmd=process.info['cmdline'] or []
        if process.pid!=me.pid and 'python' in (process.info['name'] or '').lower() and any('kalcb_' in x for x in cmd) and any(x in ('run','worker','pipeline','produce') for x in cmd):
            raise ValueError('Another KALCB research producer is active')
    (root/'temp').mkdir(exist_ok=True)
    os.environ.update(TEMP=str(root/'temp'),TMP=str(root/'temp'))
    def native_case(name,period,scenario):
        verify(p['programs'])
        item=freeze(root,name,period,p['profiles'][period])
        out=root/'cases'/name/period/scenario
        if not (out/'completion.json').exists():
            if out.exists():
                raise ValueError('Preserve incomplete native case: '+str(out))
            code=Path(item['manifest']['path']).parent/'files'
            command=[sys.executable,'-B','-u','-m','scripts.kalcb_opening_account_worker_v1',
                '--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),
                '--audit-manifest',item['manifest']['path'],'--output',str(out),'--scenario',scenario,'--mode',name]
            if name=='control':
                command+=['--bridge',p['references'][period][scenario]]
                if period=='is' and scenario in ('baseline','combined'):
                    command+=['--pairs','4']
            print(json.dumps({'start':name,'period':period,'scenario':scenario}),flush=True)
            execute(root,command,f'{name}_{period}_{scenario}',code,{'recipe':name,'period':period,'scenario':scenario,'study_complete':False})
        validate_native(out,item,scenario)
        if name=='control' and not read(out/'identity_bridge.json')['passed']:
            raise ValueError('Parent account bridge failed')
        result=read(out/'result.json')
        details=activity(out);write(out/'activity.json',details)
        result.update(details,result_binding=bound(out/'result.json'))
        rows=read(root/'progress.json')
        key=name,period,scenario
        if key not in {(r['name'],r['period'],r['scenario']) for r in rows}:
            rows.append({'name':name,'period':period,'scenario':scenario,'result':bound(out/'result.json'),
                'return_pct':result['return_pct'],'drawdown_pct':result['drawdown_pct'],'activity':details})
            write(root/'progress.json',rows)
        print(json.dumps({'complete':name,'period':period,'scenario':scenario,'return':result['return_pct'],'dd':result['drawdown_pct']}),flush=True)
        return result
    plugin=OpeningAccountPlugin(root/'study.json',native_case)
    manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
    state=PhaseRunner(plugin,manager.get_round_dir(1),round_name='Complete-account opening allocation',
        round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    verify(p['programs']+p['bindings']+p['database_bindings'])
    write(root/'completion.json',{'complete':True,'native_cases':len(read(root/'progress.json')),
        'objective':plugin.diagnostic()['objective_achieved_on_reused_development'],
        'independent_validation':False,'canonical_changed':False})
    write(root/'status.json',{'stage':'complete','objective':read(root/'completion.json')['objective']})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    ap.add_argument('--preflight',type=Path);ap.add_argument('--predecessor',type=Path)
    args=ap.parse_args()
    if args.mode=='register':register(args.root,args.tests,args.preflight,args.predecessor)
    else:run(args.root)
