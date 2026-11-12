"""Registered complete funded C9-C diagnostic, bounded ten-account matrix."""
from collections import Counter
from datetime import datetime,timezone
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import joblib
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify
from backtests.strategies.kalcb.holding_account_io_v1 import freeze
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
from backtests.strategies.kalcb.research_process_v1 import execute
from scripts.kalcb_population_assessment_v1 import account,paired_attribution
from scripts.kalcb_frontier_activity_attribution_v1 import opportunities,compare
from scripts.kalcb_entry_lifecycle_panel_v2 import lines

BASE=Path('E:/CodexResearch/k_stock_trader')
PREP=BASE/'kalcb_structural_preparation_20261016'
INFORMATION=PREP/'information_v1'
PARENT=BASE/'holding_account_20261013/study_v2/cases/control'
ALIGN=BASE/'kalcb_selected_target_alignment_20261010/v2'
AUDIT=BASE/'kalcb_late_entry_20261011/v26_opening_decision_audit_v2'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0033-funded-design.md'
OVERLAYS=('backtests/strategies/kalcb/funded_horizon_v1.py','scripts/kalcb_funded_horizon_worker_v1.py')


def status(root,phase,**kw):
    import psutil
    data={'at_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'pid':os.getpid(),
          'created_epoch':psutil.Process().create_time(),'argv':sys.argv,**kw}
    write(root/'status.json',data);print(json.dumps(data),flush=True)


def opening_features(source,q,a):
    bars=source['opening_bars'];atr=source['candidate']['daily_atr']
    if len(bars)!=6 or source['feature_available_at']!=source['trade_date']+'T09:30:00+09:00':
        raise ValueError('Not a causal completed opening')
    for i,b in enumerate(bars):
        if datetime.fromisoformat(b['timestamp']).strftime('%H:%M')!=f'09:{i*5:02}':raise ValueError('Opening gap')
    high=max(b['high'] for b in bars);low=min(b['low'] for b in bars);close=bars[-1]['close']
    volume=sum(b['volume'] for b in bars);vwap=sum((b['high']+b['low']+b['close'])/3*b['volume'] for b in bars)/volume
    return [(close-bars[0]['open'])/atr,(high-low)/atr,(close-vwap)/atr,(close-low)/max(high-low,1e-9),
            math.log1p(source['features']['first30_rel_volume']),atr/close,q,
            a['calibrated_core']['baseline']/a['notional_equity'],a['calibrated_core']['combined']/a['notional_equity']]


def register(root,tests):
    import psutil
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid!=os.getpid() and 'python' in (p.info['name'] or '').lower() and any('kalcb' in x.lower() and ('worker' in x.lower() or 'funded_horizon_v' in x.lower()) for x in p.info['cmdline'] or []):
            raise ValueError('Another KALCB producer active: '+str(p.pid))
    receipt=read(tests)
    if receipt['exit_code']!=0:raise ValueError('Failed controller/account tests')
    verify(receipt['bindings'])
    done=read(INFORMATION/'completion.json');verify([done['registration']]+done['outputs'])
    root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    status(root,'registering_causal_map')
    inputs=[bound(tests),bound(DESIGN),bound(INFORMATION/'completion.json'),done['registration']]+done['outputs']+receipt['bindings']
    sources={}
    for cell in read(ALIGN/'protocol.json')['cells']:
        if cell['kind']!='opening':continue
        p=Path(cell['prepared']);meta=read(p/'prepared.json');binding={'path':str(p/'opportunities.jsonl'),'sha256':meta['opportunities_sha256']};verify([binding]);inputs+=[binding,bound(p/'prepared.json')]
        for r in lines(p/'opportunities.jsonl'):
            if cell['period']=='oos' and r['trade_date']<'2026-04-01':continue
            sources[r['opportunity_id']]=r
    ab=next(b for b in read(AUDIT/'completion.json')['artifacts'] if Path(b['path']).name=='decisions.jsonl.gz');verify([ab]);inputs.append(ab)
    proofs=read(INFORMATION/'model_support.json');origins=[p['origin'] for p in proofs];models={i:joblib.load(INFORMATION/f'models_{i:02}.joblib') for i in range(len(origins))}
    signals={};counts=Counter()
    for a in lines(AUDIT/'decisions.jsonl.gz'):
        if 'control' not in a['selected']:continue
        eligible=[i for i,day in enumerate(origins) if day<=a['day']]
        if not eligible:counts['warmup_parent_unchanged']+=1;continue
        i=eligible[-1];source=sources[a['base_id']];x=opening_features(source,a['quantity_scale'],a)
        predictions={s:float(models[i][s]['tree_B']['tree'].predict(np.array([x]))[0]) for s in ('baseline','combined')}
        row={'day':a['day'],'symbol':a['base_id'].split('/')[1],'action_id':a['action_id'],'quantity_scale':a['quantity_scale'],
             'origin':origins[i],'observed_through':proofs[i]['observed_through'],'available_at':source['feature_available_at'],
             'features':x,'predictions':predictions,'source_original_admission':a['selected']['control']}
        if row['observed_through']>=a['day']:raise ValueError('Future model evidence')
        if a['base_id'] in signals:raise ValueError('Duplicate selected opening')
        signals[a['base_id']]=row;counts[a['period']+'_forecasts']+=1
        if predictions['baseline'] < -23 and predictions['combined'] < -33:counts[a['period']+'_signals']+=1
    write(root/'horizon_map.json',signals);write(root/'map_coverage.json',dict(counts))
    matches=0
    for r in read(INFORMATION/'renewal_predictions.json'):
        if not r['selected'] or r['base_id'] not in signals:continue
        row=signals[r['base_id']]
        if row['action_id']!=r['action_id'] or row['origin']!=r['origin']:
            raise ValueError('Causal map changed original quantity/model lineage')
        if any(abs(row['predictions'][s]-r['predictions']['tree_B'][s])>1e-8 for s in ('baseline','combined')):
            raise ValueError('Opening-only reconstruction does not reproduce B prediction')
        matches+=1
    if not matches:raise ValueError('No real source prediction bridge')
    write(root/'prediction_bridge.json',{'passed':True,'matched_scenario_rows':matches,
          'scope':'Common selected opening predictions exactly reproduced from opening-only inputs; no future bar or fill determines C membership.'})
    profiles={};references={};databases=set()
    for period in ('is','oos'):
        old=read(PARENT/period/'baseline/input_profile.json');verify([old[k] for k in ('manifest','plan','policy')]);profiles[period]=old;references[period]={}
        plan=read(old['plan']['path']);databases.add(plan['database'])
        for s in ('baseline','combined'):
            folder=PARENT/period/s;item=read(folder/'input_profile.json');validate_native(folder,item,s,save=False)
            references[period][s]=str(folder)
            inputs.extend(bound(folder/name) for name in ('input_profile.json','completion.json','result.json','execution_audit.json','fills.json','trades.json','daily.json','equity.json'))
    overlays={}
    for rel in OVERLAYS:
        p=root/'overlays'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/rel).read_bytes());overlays[rel]=bound(p)
    overlays['research/horizon_policy/map.json']=bound(root/'horizon_map.json')
    programs=[bound(REPO/p) for p in [*OVERLAYS,'scripts/kalcb_funded_horizon_v1.py','tests/test_kalcb_funded_horizon_v1.py',
        'backtests/strategies/kalcb/holding_account_io_v1.py','backtests/strategies/kalcb/late_artifact_validation_v2.py',
        'backtests/strategies/kalcb/fixed_allocation_funding_v1.py','scripts/kalcb_population_assessment_v1.py',
        'scripts/kalcb_frontier_activity_attribution_v1.py',
        'docs/optimisation/kalcb/evidence/reconstruct_frontier.py']]
    protocol={'at_utc':datetime.now(timezone.utc).isoformat(),'design':bound(DESIGN),'inputs':inputs,'programs':programs,
        'overlays':overlays,'profiles':profiles,'references':references,'databases':sorted(databases),
        'database_bindings':[bound(p) for p in sorted(databases)],'signal_map':bound(root/'horizon_map.json'),
        'native_accounts':10,'diagnostic':True,'qualification':False,'initial_equity':100000000,
        'matrix':[{'mode':'disabled','period':'is','scenario':'baseline'},{'mode':'disabled','period':'oos','scenario':'combined'}]+
                 [{'mode':mode,'period':period,'scenario':s} for mode in ('veto','renew') for period in ('is','oos') for s in ('baseline','combined')]}
    write(root/'protocol.json',protocol);write(root/'completed_cases.json',[])
    verify(inputs+programs+protocol['database_bindings'])
    status(root,'freezing_native_sources',coverage=dict(counts))
    for period in ('is','oos'):freeze(root,period,profiles[period])
    write(root/'registration.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'protocol':bound(root/'protocol.json'),
        'profiles':{p:bound(root/'runtime'/p/'profile.json') for p in ('is','oos')},'source_authentication':'cached_link verified every consumed parent file against immutable manifest'})


def assess(root):
    protocol=read(root/'protocol.json');spec=importlib.util.spec_from_file_location('horizon_metrics',REPO/'docs/optimisation/kalcb/evidence/reconstruct_frontier.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    outputs={};controls={};control_activity={}
    for period in ('is','oos'):
        for s in ('baseline','combined'):
            ref=Path(protocol['references'][period][s]);controls[period+'/'+s]=account(ref)
            controls[period+'/'+s]['mtm']=m.metrics(ref);control_activity[period+'/'+s]=opportunities(ref)
            for mode in ('veto','renew'):
                folder=root/'cases'/mode/period/s;candidate=account(folder);activity=opportunities(folder)
                outputs[mode+'/'+period+'/'+s]={'account':candidate,'mtm':m.metrics(folder),
                    'paired_attribution':paired_attribution(candidate,controls[period+'/'+s]),
                    'activity':activity,'activity_attribution':compare(activity,control_activity[period+'/'+s]),
                    'activity_sensitivities':{str(fraction):sum(r['entry_notional_krw']>=fraction*r['decision_equity_krw'] and
                        r['net_pnl_krw']>=.0001*r['decision_equity_krw'] for r in activity['opportunities'].values()) for fraction in (.005,.01,.02)},
                    'actions':read(folder/'horizon_actions.json')}
    write(root/'assessment.json',{'controls':controls,'control_activity':control_activity,'treatments':outputs,'qualification':False,'economic_objective_achieved':False,
          'scope':'Full funded bounded diagnostic, all reused development; principal-solution selection and qualification still require explicit assessment.'})


def run(root):
    p=read(root/'protocol.json');verify(p['programs']+p['inputs']+p['database_bindings'])
    for i,case in enumerate(p['matrix']):
        mode,period,s=case['mode'],case['period'],case['scenario'];item=read(root/'runtime'/period/'profile.json');code=Path(item['manifest']['path']).parent/'files';out=root/'cases'/mode/period/s
        if out.exists():raise ValueError('Preserve existing account generation')
        if i==2:
            for b in p['matrix'][:2]:
                if not read(root/'cases'/b['mode']/b['period']/b['scenario']/'identity_bridge.json')['passed']:raise ValueError('Missing bridge')
        command=[sys.executable,'-B','-u','-m','scripts.kalcb_funded_horizon_worker_v1','--plan',item['plan']['path'],
             '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],'--output',str(out),'--scenario',s,'--mode',mode]
        if mode=='disabled':command+=['--bridge',p['references'][period][s]]
        status(root,'native_case_start',case=case,complete=i,total=10)
        execute(root,command,'_'.join((mode,period,s)),code,{'case':case,'cases_complete':i,'cases_total':10,'qualification':False})
        write(out/'input_profile.json',item);validate_native(out,item,s);write(out/'funding_review.json',funding_review(out))
        actions=read(out/'horizon_actions.json')
        kinds=Counter(a['kind'] for a in actions)
        if mode=='disabled' and actions:raise ValueError('Disabled controller activated')
        if mode=='veto' and not kinds['veto']:raise ValueError('Expected causal veto did not activate')
        if mode=='renew' and not (kinds['renew_exit']+kinds['pending_only_cancel']):raise ValueError('Expected renewal did not activate')
        write(out/'activation_review.json',{'passed':True,'mode':mode,'event_counts':dict(kinds),
            'distinct_issuer_days':len({a['key'] for a in actions}),
            'scope':'Native activation check; repeated route veto events are not independent opportunities or economic progress.'})
        ledger=read(root/'completed_cases.json');ledger.append({'case':case,'completion':bound(out/'completion.json'),'verification':bound(out/'independent_account_verification.json')});write(root/'completed_cases.json',ledger)
    status(root,'independent_assessment');assess(root)
    verify(p['programs']+p['inputs']+p['database_bindings'])
    write(root/'completion.json',{'complete':True,'at_utc':datetime.now(timezone.utc).isoformat(),'accounts':10,'assessment':bound(root/'assessment.json'),
          'registration':bound(root/'registration.json'),'economic_objective_achieved':False,'qualification':False})
    status(root,'diagnostic_assessed_disposition_pending')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',type=Path,required=True);a=p.parse_args()
    try:register(a.output,a.tests);run(a.output)
    except Exception as exc:
        if a.output.exists():write(a.output/'failure.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'error':repr(exc),'traceback':traceback.format_exc()})
        raise
