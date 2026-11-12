"""C10 R1/R2: frozen dependency and assessor-interface repairs only."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,sys,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import kalcb_opening_veto_qualification_v1 as old
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,link,source_hash,env
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.research_process_v1 import execute
from scripts.kalcb_capital_sensitivity_v2 import classify_case
from scripts.kalcb_opening_veto_assessment_v2 import assess

PREVIOUS=old.BASE/'kalcb_opening_veto_qualification_20261016/v1'
REPAIR=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0034-repair-r1.md'
REPAIR2=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0034-repair-r2.md'
HELPER='backtests/strategies/kalcb/fixed_allocation_reference_v1.py'
ACCEPTED=('stress_is_participation','stress_is_cost','stress_oos_participation','stress_oos_cost')


def no_producer():
    import psutil
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid==os.getpid() or 'python' not in (p.info['name'] or '').lower():continue
        if any('kalcb' in a.lower() and ('worker' in a.lower() or 'opening_veto_qualification_v' in a or 'funded_horizon_v1.py' in a) for a in p.info['cmdline'] or []):
            raise ValueError('Another KALCB producer '+str(p.pid))


def rebind_plan(plan,new_context,new_protocol,new_protocol_binding):
    p=deepcopy(plan);contract=p['capital_sensitivity']
    for path in (p['audit_configuration'],contract['protocol']['path']):p['input_sha256'].pop(path,None)
    p['audit_configuration']=str(new_context);contract['protocol']=new_protocol_binding
    for path in (new_context,new_protocol):p['input_sha256'][str(path)]=digest(path)
    return p


def freeze(root,protocol,prior_registration):
    old_item=prior_registration['capital_profiles']['bridge_disabled'];mp=Path(old_item['manifest']['path'])
    verify([old_item['manifest']]);payload=read(mp);sig=payload.pop('manifest_hash')
    if stable_signature(payload)!=sig:raise ValueError('Changed v1 manifest')
    if HELPER in payload['files']:raise ValueError('Original missing-dependency diagnosis contradicted')
    folder=root/'capital';code=folder/'inputs/files';files={};replacements={};profiles={}
    for c in protocol['cases']:
        source=prior_registration['capital_profiles'][c['id']];verify([source[k] for k in ('manifest','plan','policy')])
        plan=read(source['plan']['path']);context=read(plan['audit_configuration']);dest=folder/'plans'/c['id']
        if {k:plan['capital_sensitivity'][k] for k in c}!=c:raise ValueError('Changed original capital cell')
        context['output_root']=str(root/'cases'/c['id']);write(dest/'audit_configuration.json',context)
        plan=rebind_plan(plan,dest/'audit_configuration.json',root/'protocol.json',bound(root/'protocol.json'))
        write(dest/'plan.json',plan)
        for name in ('plan.json','audit_configuration.json'):
            replacements['research/opening_veto_qualification/'+c['id']+'/'+name]=dest/name
        profiles[c['id']]=dict(name=c['id'],period='oos',folder=str(folder),plan=bound(dest/'plan.json'))
    replacements['research/opening_veto_qualification/protocol.json']=root/'protocol.json'
    for i,(rel,sha) in enumerate(payload['files'].items()):
        src=mp.parent/'files'/rel
        if digest(src)!=sha:raise ValueError('Changed v1 frozen source '+rel)
        if rel not in replacements:link(src,code/rel);files[rel]=sha
        if i%500==0:old.status(root,'r1_source_authentication',files_checked=i+1,total=len(payload['files']))
    for rel,src in replacements.items():link(src,code/rel);files[rel]=digest(src)
    donor=old.CAP/'v1/inputs';donor_manifest=read(donor/'manifest.json');helper=donor/'files'/HELPER
    if digest(helper)!=donor_manifest['files'][HELPER]:raise ValueError('Changed historical allocation helper')
    link(helper,code/HELPER);files[HELPER]=digest(helper)
    generation=source_hash(code);payload['files']=files;payload['source_generation_hash']=generation
    payload['evidence']['capital_sensitivity'].update(protocol=bound(root/'protocol.json'),repair='R1 omitted helper',parent_manifest=old_item['manifest'])
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    for item in profiles.values():
        item.update(manifest=bound(folder/'inputs/manifest.json'),policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'),source_generation_hash=generation)
        if item['policy']['sha256']!=old_item['policy']['sha256']:raise ValueError('Policy changed in repair')
    return profiles


PROBE='''import json,sys
from pathlib import Path
from scripts import kalcb_executed_value_replay as replay
from scripts import kalcb_opening_veto_capital_worker_v1 as worker
from backtests.strategies.kalcb.runner import KALCBReplayAdapter
class ReachedReplay(Exception):pass
def sentinel(*args,**kwargs):raise ReachedReplay()
original=KALCBReplayAdapter._portfolio_view
replay.run=sentinel
try:worker.main()
except ReachedReplay:pass
else:raise AssertionError('Probe did not reach the real replay boundary')
assert KALCBReplayAdapter._portfolio_view is original
print(json.dumps(dict(reached_replay_boundary=True,financial_replay_executed=False,hooks_restored=True)))
'''


def preflight(root,profiles):
    results={}
    for name,item in profiles.items():
        code=Path(item['manifest']['path']).parent/'files'
        cmd=[sys.executable,'-B','-c',PROBE,'--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),
            '--audit-manifest',item['manifest']['path'],'--output',str(root/'preflight'/name)]
        result=subprocess.run(cmd,cwd=code,env=env(code),capture_output=True,text=True)
        write(root/'preflight_logs'/(name+'.json'),dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError('Frozen worker preflight failed '+name)
        receipt=json.loads(result.stdout.strip().splitlines()[-1])
        if not receipt['reached_replay_boundary'] or receipt['financial_replay_executed']:raise ValueError('Invalid dry probe')
        if (root/'preflight'/name/'completion.json').exists():raise ValueError('Dry probe cannot be a completed account')
        results[name]=receipt
    write(root/'runtime_preflight.json',results)


def register(root,tests):
    no_producer();test=read(tests)
    if test['exit_code']:raise ValueError('Failed repair tests')
    verify(test['bindings']);root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir();old.status(root,'r1_predecessor_authentication')
    parent=read(PREVIOUS/'protocol.json');reg=read(PREVIOUS/'registration.json');verify([reg['protocol']]+parent['inputs']+parent['programs']+parent['databases'])
    failure=read(PREVIOUS/'failed_stage.json')
    if failure['stage']!='bridge_disabled' or 'No module named' not in (PREVIOUS/'bridge_disabled.log').read_text():raise ValueError('Unexpected predecessor failure')
    if (PREVIOUS/'cases/bridge_disabled').exists():raise ValueError('Unexpected bridge outcome')
    ledger=read(PREVIOUS/'completed_cases.json')
    if [r['case']['id'] for r in ledger]!=list(ACCEPTED) or not all(r['verdict']['qualified'] for r in ledger):raise ValueError('Unexpected accepted prefix')
    stop=read(PREVIOUS.parent/'v2/operational_preflight_stop.json')
    if not stop['verified_exited'] or stop['native_cases']!=0:raise ValueError('Unexpected v2 native work')
    protocol=deepcopy(parent);protocol['at_utc']=datetime.now(timezone.utc).isoformat();protocol['repair']='R1/R2';protocol['cases']=[c for c in parent['cases'] if c['kind']=='capital']
    protocol['inputs'] += [bound(PREVIOUS/f) for f in ('registration.json','failure.json','failed_stage.json','bridge_disabled.log','completed_cases.json')]+[bound(REPAIR),bound(tests)]+test['bindings']
    protocol['inputs'] += [bound(REPAIR2),bound(PREVIOUS.parent/'v2/operational_preflight_stop.json')]
    for row in ledger:
        c=row['case'];out=PREVIOUS/'cases'/c['id'];old.validate_native(out,parent['profiles'][c['period']],c['scenario'],save=False)
        old.diag.funding_review(out,c['initial_equity']);protocol['reused'][c['period']+'/'+c['scenario']]=str(out)
        protocol['inputs'].extend(bound(p) for p in out.iterdir() if p.is_file())
    helper=old.CAP/'v1/inputs/files'/HELPER;protocol['inputs'] += [bound(old.CAP/'v1/inputs/manifest.json'),bound(helper)]
    protocol['programs'] += [bound(__file__),bound(REPO/'tests/test_kalcb_opening_veto_qualification_v2.py')]
    protocol['programs'] += [bound(REPO/'scripts/kalcb_opening_veto_assessment_v2.py'),bound(REPO/'tests/test_kalcb_opening_veto_assessment_v2.py')]
    write(root/'protocol.json',protocol);write(root/'completed_cases.json',[]);write(root/'reused_cases.json',ledger)
    profiles=freeze(root,protocol,reg);preflight(root,profiles)
    verify(protocol['inputs']+protocol['programs']+protocol['databases'])
    write(root/'registration.json',dict(protocol=bound(root/'protocol.json'),capital_profiles=profiles,preflight=bound(root/'runtime_preflight.json'),native_accounts=6,reused_accounts=4,criteria='unchanged C10'))


def run(root):
    p=read(root/'protocol.json');reg=read(root/'registration.json');verify([reg['protocol'],reg['preflight']]+p['programs']+p['inputs']+p['databases'])
    for i,c in enumerate(p['cases']):
        out=root/'cases'/c['id']
        if out.exists():raise ValueError('Preserve existing case')
        if c['id'].startswith('capital_'):
            for mode in ('disabled','veto'):
                if not read(root/'cases'/('bridge_'+mode)/'identity_bridge.json')['passed']:raise ValueError('Missing capital identity bridge')
        item=reg['capital_profiles'][c['id']];code=Path(item['manifest']['path']).parent/'files'
        cmd=[sys.executable,'-B','-u','-m','scripts.kalcb_opening_veto_capital_worker_v1','--plan',item['plan']['path'],
            '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],'--output',str(out)]
        execute(root,cmd,c['id'],code,dict(case=c,cases_complete=4+i,cases_total=10,reused=4))
        write(out/'input_profile.json',item);verdict=classify_case(out,item,save=True)
        if verdict['qualified']:write(out/'funding_review.json',old.diag.funding_review(out,c['initial_equity']))
        else:write(out/'qualification_rejection.json',verdict)
        actions=read(out/'horizon_actions.json')
        if c['horizon_mode']=='veto' and not any(a['kind']=='veto' for a in actions):raise ValueError('Missing veto activation')
        if c['horizon_mode']=='disabled' and actions:raise ValueError('Disabled activation')
        ledger=read(root/'completed_cases.json');ledger.append(dict(case=c,verdict=verdict,completion=bound(out/'completion.json')));write(root/'completed_cases.json',ledger)
    old.status(root,'independent_assessment');assess(root);verify(p['programs']+p['inputs']+p['databases'])
    write(root/'completion.json',dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),registered_accounts=10,reused=4,new=6,registration=bound(root/'registration.json'),assessment=bound(root/'assessment.json'),economic_objective_achieved=False))
    old.status(root,'qualified_matrix_assessed_disposition_pending')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',type=Path,required=True);a=p.parse_args()
    try:register(a.output,a.tests);run(a.output)
    except Exception as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
