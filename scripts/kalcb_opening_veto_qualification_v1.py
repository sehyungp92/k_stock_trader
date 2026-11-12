"""C10 fixed-policy qualification: missing stresses and exact capital sensitivity."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,link,source_hash,env
from backtests.strategies.kalcb.fixed_allocation_reference_v1 import verify_identity_bridge
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as diag
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.research_process_v1 import execute
from scripts.kalcb_capital_sensitivity_v2 import classify_case
from scripts.kalcb_funded_horizon_v1 import status

BASE=Path('E:/CodexResearch/k_stock_trader')
PREP=BASE/'kalcb_structural_preparation_20261016';C=PREP/'funded_horizon_v1';D=PREP/'restoration_v3'
FRONT=BASE/'kalcb_frontier_completion_20261016/v3';CAP=BASE/'kalcb_capital_sensitivity_20261016'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0034.md'
WORKER='scripts/kalcb_opening_veto_capital_worker_v1.py'
SCENARIOS=('baseline','participation','cost','combined')


def control_profiles():
    controls={};items={}
    for p in ('is','oos'):
        for s in SCENARIOS:
            folder=BASE/'holding_account_20261013/study_v2/cases/control'/p/s
            controls[p+'/'+s]=str(folder);items[p+'/'+s]=read(folder/'input_profile.json')
    return controls,items


def classify_standard(folder,item,scenario):
    """Keep only the original validator's explicit residual rejection unqualified."""
    folder=Path(folder)
    try:validation=validate_native(folder,item,scenario)
    except ValueError as error:
        if str(error)!='Unresolved native account':raise
        result=read(folder/'result.json')
        flags={k:result[k] for k in ('residual_sessions','terminal_inventory','terminal_pending')}
        if not any(flags.values()) or (folder/'independent_account_verification.json').exists():
            raise ValueError('Residual rejection contradicts account evidence') from error
        return dict(qualified=False,reason=str(error),flags=flags,remaining_independent_checks_not_reached=True,
            reported_return_excluded_from_qualified_comparisons=True)
    return dict(qualified=True,validation=validation)


def no_producer():
    import psutil
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid==os.getpid() or 'python' not in (p.info['name'] or '').lower():continue
        if any('kalcb' in a.lower() and ('worker' in a.lower() or 'qualification_v1.py' in a or 'funded_horizon_v1.py' in a) for a in p.info['cmdline'] or []):
            raise ValueError('Another KALCB producer '+str(p.pid))


def capital_freeze(root,protocol):
    old=protocol['profiles']['oos'];verify([old[k] for k in ('manifest','plan','policy')])
    mp=Path(old['manifest']['path']);payload=read(mp);sig=payload.pop('manifest_hash')
    if stable_signature(payload)!=sig:raise ValueError('Parent manifest mismatch')
    folder=root/'capital';code=folder/'inputs/files';files={};seen=set()
    for i,(rel,sha) in enumerate(payload['files'].items()):
        src=mp.parent/'files'/rel;stat=src.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,sha)
        if key not in seen:
            if digest(src)!=sha:raise ValueError('Changed inherited source '+rel)
            seen.add(key)
        link(src,code/rel);files[rel]=sha
        if i%500==0:status(root,'capital_source_authentication',files_checked=i+1,total=len(payload['files']))
    if WORKER in files:raise ValueError('Worker already present')
    shutil.copyfile(REPO/WORKER,code/WORKER);files[WORKER]=digest(code/WORKER)
    qualification=read(code/payload['evidence']['qualification_path']);declaration=qualification['capacity']
    if declaration['kind']!='declared_bar_participation_model' or declaration['phase_liquidity_verified'] is not False:
        raise ValueError('Unexpected capacity evidence')
    for s in SCENARIOS:
        audit=read(Path(protocol['controls']['oos/'+s])/'execution_audit.json')['audit_receipt']
        policy=audit['qualification']['capacity']['policy']
        rows=[r for r in declaration['registered_execution_policies'] if r['initial_equity']==100000000 and r['buying_power_leverage']==2 and r['policy']==policy]
        if len(rows)!=1:raise ValueError('Ambiguous original capacity declaration')
        row=deepcopy(rows[0]);row['initial_equity']=protocol['capitals'][s]
        if row in declaration['registered_execution_policies']:raise ValueError('Capital declaration already present')
        declaration['registered_execution_policies'].append(row)
    qrel='research/opening_veto_qualification/capacity.json';write(code/qrel,qualification);files[qrel]=digest(code/qrel)
    payload['evidence']['qualification_path']=qrel;oldplan=read(old['plan']['path']);oldcontext=read(oldplan['audit_configuration']);profiles={}
    for case in protocol['cases']:
        if case['kind']!='capital':continue
        dest=folder/'plans'/case['id'];contract={**case,'mode':'compound','reference':100000000.,'protocol':bound(root/'protocol.json')}
        context={**oldcontext,'initial_equity':case['initial_equity'],'output_root':str(root/'cases'/case['id']),'promotion_status':'offline_component_qualification_only'}
        write(dest/'audit_configuration.json',context);plan=deepcopy(oldplan)
        plan['input_sha256'].pop(plan['audit_configuration'],None);plan['audit_configuration']=str(dest/'audit_configuration.json')
        plan.pop('fixed_allocation_reference',None);plan['capital_sensitivity']=contract
        for path in (dest/'audit_configuration.json',root/'protocol.json'):plan['input_sha256'][str(path)]=digest(path)
        write(dest/'plan.json',plan)
        for name in ('plan.json','audit_configuration.json'):
            rel='research/opening_veto_qualification/'+case['id']+'/'+name;link(dest/name,code/rel);files[rel]=digest(dest/name)
        profiles[case['id']]=dict(name=case['id'],period='oos',folder=str(folder),plan=bound(dest/'plan.json'),
            policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'))
    rel='research/opening_veto_qualification/protocol.json';link(root/'protocol.json',code/rel);files[rel]=digest(root/'protocol.json')
    generation=source_hash(code);payload['files']=files;payload['source_generation_hash']=generation
    payload['evidence']['capital_sensitivity']=dict(protocol=bound(root/'protocol.json'),parent_manifest=old['manifest'],full_continuous_state_claim=False,core_engine_policy_changes=False)
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    for item in profiles.values():
        item.update(manifest=bound(folder/'inputs/manifest.json'),source_generation_hash=generation)
        if item['policy']['sha256']!=old['policy']['sha256']:raise ValueError('Changed original entry policy')
    preflight=subprocess.run([sys.executable,'-B','-m',WORKER[:-3].replace('/','.'),'--help'],
        cwd=code,env=env(code),capture_output=True,text=True,check=True)
    write(root/'capital_source_preflight.json',dict(files_authenticated=len(payload['files']),
        generation=generation,worker_help=preflight.stdout,capital_declaration=bound(code/qrel)))
    return profiles


def register(root,tests):
    no_producer();receipt=read(tests)
    if receipt['exit_code']!=0:raise ValueError('Tests failed')
    verify(receipt['bindings']);root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    status(root,'authenticating_predecessors')
    done=read(C/'completion.json');verify([done['assessment'],done['registration']])
    dd=read(D/'completion.json');verify([dd[k] for k in ('registration','events','rows','assessment')])
    if done['accounts']!=10 or not dd['complete']:raise ValueError('Preparation incomplete')
    cp=read(C/'protocol.json');verify(cp['programs']+cp['inputs']+cp['database_bindings'])
    profiles={p:read(C/'runtime'/p/'profile.json') for p in ('is','oos')}
    index=read(FRONT/'account_index.json');capreg=read(CAP/'v1/registration.json');capproto=read(CAP/'v1/protocol.json');ledger=read(CAP/'v2/case_ledger.json')
    controls,control_items=control_profiles()
    inputs=[bound(DESIGN),bound(tests),bound(C/'completion.json'),bound(C/'target_attribution_v1.json'),bound(D/'completion.json'),
        bound(FRONT/'account_index.json'),bound(CAP/'v1/registration.json'),bound(CAP/'v1/protocol.json'),bound(CAP/'v2/case_ledger.json'),
        bound(C/'protocol.json')]+receipt['bindings']+cp['programs']+cp['inputs']
    reused={}
    for p in ('is','oos'):
        for s in SCENARIOS:
            ref=Path(controls[p+'/'+s]);item=control_items[p+'/'+s]
            validate_native(ref,item,s,save=False)
            verify_identity_bridge(ref,index['compound_prior100/'+p+'/'+s])
            if item['policy']['sha256']!=profiles[p]['policy']['sha256']:raise ValueError('Different parent policy')
            inputs.extend(item[k] for k in ('manifest','plan','policy'))
            inputs.extend(bound(ref/f) for f in ('request.json','completion.json','execution_audit.json','result.json','daily.json','equity.json','trades.json','fills.json','orders.jsonl.gz','independent_account_verification.json'))
            if s in ('baseline','combined'):
                folder=C/'cases/veto'/p/s;validate_native(folder,profiles[p],s,save=False);reused[p+'/'+s]=str(folder)
                inputs.extend(bound(folder/f) for f in ('request.json','completion.json','execution_audit.json','result.json','daily.json','equity.json','trades.json','fills.json','orders.jsonl.gz','independent_account_verification.json'))
        inputs.extend(profiles[p][k] for k in ('manifest','plan','policy'))
    capcontrols={}
    for s in SCENARIOS:
        key='compound_'+s;old=ledger[key];verdict=classify_case(old['folder'],capreg['profiles'][key])
        if verdict['qualified']!=old['qualified']:raise ValueError('Changed capital control verdict')
        capcontrols[s]=dict(folder=old['folder'],qualified=verdict['qualified'],profile=capreg['profiles'][key],verdict=verdict)
        inputs.extend(bound(path) for path in Path(old['folder']).iterdir() if path.is_file())
    cases=[dict(id='stress_'+p+'_'+s,kind='standard',period=p,scenario=s,initial_equity=100000000.,horizon_mode='veto') for p in ('is','oos') for s in ('participation','cost')]
    cases += [dict(id='bridge_'+mode,kind='capital',period='oos',scenario='baseline',initial_equity=100000000.,horizon_mode=mode,
        bridge=str(C/'cases'/('veto' if mode=='veto' else 'disabled')/'oos'/'baseline') if mode=='veto' else controls['oos/baseline']) for mode in ('disabled','veto')]
    cases += [dict(id='capital_'+s,kind='capital',period='oos',scenario=s,initial_equity=capproto['capitals'][s],horizon_mode='veto') for s in SCENARIOS]
    programs=[bound(REPO/f) for f in (WORKER,'scripts/kalcb_opening_veto_qualification_v1.py',
        'backtests/strategies/kalcb/capital_sensitivity_diagnostics_v1.py','backtests/strategies/kalcb/late_artifact_validation_v2.py',
        'scripts/kalcb_capital_sensitivity_v2.py','scripts/kalcb_funded_horizon_v1.py','scripts/kalcb_frontier_activity_attribution_v1.py',
        'scripts/kalcb_population_assessment_v1.py','backtests/strategies/kalcb/fixed_allocation_reference_v1.py',
        'backtests/strategies/kalcb/retention_research_io_v1.py','backtests/strategies/kalcb/research_process_v1.py',
        'backtests/auto/shared/cache_keys.py','tests/test_kalcb_opening_veto_qualification_v1.py')]
    write(root/'completed_cases.json',[])
    protocol=dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0034',criteria='C10',inputs=inputs,programs=programs,
        profiles=profiles,controls=controls,reused=reused,capital_controls=capcontrols,capitals=capproto['capitals'],
        databases=cp['database_bindings'],signal_map=cp['signal_map'],cases=cases,scope='Fixed policy; no restored admissions; reused development')
    write(root/'protocol.json',protocol);verify(inputs+programs+protocol['databases'])
    capprofiles=capital_freeze(root,protocol)
    write(root/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),protocol=bound(root/'protocol.json'),capital_profiles=capprofiles,ten_native_accounts=True))


def useful_days(activity):
    grouped=defaultdict(list)
    for key,r in sorted(activity['opportunities'].items()):grouped[key[:17]].append(r)
    # Group all lifetimes, including losses, and use the earliest decision equity.
    # Retries/partial fills never manufacture independent profitable issuer-days.
    return sum(any(r['entry_notional_krw']>=.01*r['decision_equity_krw'] for r in rr) and
        sum(r['net_pnl_krw'] for r in rr)>=.0001*rr[0]['decision_equity_krw'] for rr in grouped.values())


def assess(root):
    p=read(root/'protocol.json');ledger=read(root/'completed_cases.json');candidate=dict(p['reused']);cells={};capital={};rejected={}
    for row in ledger:
        c=row['case']
        if c['kind']=='standard':
            key=c['period']+'/'+c['scenario']
            if row['verdict']['qualified']:candidate[key]=str(root/'cases'/c['id'])
            else:rejected[key]=row['verdict']
    for key,folder in candidate.items():
        a=diag.account(folder);b=diag.account(p['controls'][key]);aa=diag.opportunities(folder);bb=diag.opportunities(p['controls'][key])
        cells[key]=dict(account=a,control=b,metrics=diag.metrics(folder),control_metrics=diag.metrics(p['controls'][key]),
            activity=aa,control_activity=bb,path=diag.paired_attribution(a,b),activity_delta=diag.compare(aa,bb),
            profitable_material_issuer_days=useful_days(aa),control_profitable_material_issuer_days=useful_days(bb))
    for row in ledger:
        c=row['case']
        if not c['id'].startswith('capital_'):continue
        s=c['scenario'];initial=c['initial_equity'];old=p['capital_controls'][s];item=dict(verdict=row['verdict'],control_verdict=old['verdict'],initial_equity=initial,matched_qualified=row['verdict']['qualified'] and old['qualified'])
        if row['verdict']['qualified']:
            folder=root/'cases'/c['id'];item.update(metrics=diag.metrics(folder,initial),account=diag.account(folder,initial),activity=diag.opportunities(folder,initial))
        if item['matched_qualified']:
            b=diag.account(old['folder'],initial);bb=diag.opportunities(old['folder'],initial)
            item.update(control_metrics=diag.metrics(old['folder'],initial),net_delta_krw=item['account']['net_pnl_krw']-b['net_pnl_krw'],
                activity_delta=diag.compare(item['activity'],bb),path=diag.paired_attribution(item['account'],b))
        capital[s]=item
    od=[v['account']['net_pnl_krw']-v['control']['net_pnl_krw'] for k,v in cells.items() if k.startswith('oos/')]
    iid=[v['account']['net_pnl_krw']-v['control']['net_pnl_krw'] for k,v in cells.items() if k.startswith('is/')]
    fd={pfx:float(np.median([v['profitable_material_issuer_days']-v['control_profitable_material_issuer_days'] for k,v in cells.items() if k.startswith(pfx+'/')])) for pfx in ('is','oos')}
    criteria=dict(all_standard_valid=len(cells)==8 and not rejected,
        component_return_reference=len(cells)==8 and sum(x>=1000000 for x in od)>=3 and float(np.median(iid))>=0,
        capital_portability=len(capital)==4 and all(v['verdict']['qualified'] for v in capital.values()) and sum(v.get('net_delta_krw',0)>0 for v in capital.values() if v['matched_qualified'])>=2,
        useful_frequency_reference=len(cells)==8 and fd['oos']>=3 and fd['is']>=0,median_frequency_change=fd,
        human_economic_risk_chronology_assessment_required=True,economic_objective_achieved=False)
    write(root/'assessment.json',dict(cells=cells,rejected_standard=rejected,capital=capital,criteria=criteria,scope='C10 references; no automatic economic success or promotion. Reused development.'))


def run(root):
    p=read(root/'protocol.json');reg=read(root/'registration.json');verify([reg['protocol']]+p['programs']+p['inputs']+p['databases'])
    for i,c in enumerate(p['cases']):
        out=root/'cases'/c['id']
        if out.exists():raise ValueError('Preserve existing case')
        if c['id'].startswith('capital_'):
            for mode in ('disabled','veto'):
                if not read(root/'cases'/('bridge_'+mode)/'identity_bridge.json')['passed']:raise ValueError('Capital bridge missing')
        item=p['profiles'][c['period']] if c['kind']=='standard' else reg['capital_profiles'][c['id']]
        code=Path(item['manifest']['path']).parent/'files'
        module='scripts.kalcb_funded_horizon_worker_v1' if c['kind']=='standard' else WORKER[:-3].replace('/','.')
        cmd=[sys.executable,'-B','-u','-m',module,'--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],'--output',str(out)]
        if c['kind']=='standard':cmd+=['--scenario',c['scenario'],'--mode','veto']
        execute(root,cmd,c['id'],code,dict(case=c,cases_complete=i,cases_total=10))
        write(out/'input_profile.json',item)
        if c['kind']=='standard':verdict=classify_standard(out,item,c['scenario'])
        else:verdict=classify_case(out,item,save=True)
        if verdict['qualified']:write(out/'funding_review.json',diag.funding_review(out,c['initial_equity']))
        else:write(out/'qualification_rejection.json',verdict)
        actions=read(out/'horizon_actions.json')
        if c['horizon_mode']=='veto' and not any(a['kind']=='veto' for a in actions):raise ValueError('Veto failed activation')
        if c['horizon_mode']=='disabled' and actions:raise ValueError('Disabled activation')
        row=dict(case=c,verdict=verdict,completion=bound(out/'completion.json'))
        ledger=read(root/'completed_cases.json');ledger.append(row);write(root/'completed_cases.json',ledger)
    status(root,'independent_assessment');assess(root);verify(p['programs']+p['inputs']+p['databases'])
    write(root/'completion.json',dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),attempts=10,registration=bound(root/'registration.json'),assessment=bound(root/'assessment.json'),economic_objective_achieved=False))
    status(root,'qualified_matrix_assessed_disposition_pending')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',type=Path,required=True);a=p.parse_args()
    try:register(a.output,a.tests);run(a.output)
    except Exception as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
