"""Complete-window, paired native ablations with explicit cached IS evidence."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT,DIAG,POLICY,read,write,bound,digest,jsonl,rows,link
from backtests.strategies.kalcb.oos_ablation import KALCBOOSAblationPlugin,transform_snapshot
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager
from scripts.kalcb_executed_value_replay import verify_policy,timed_policy

def transform(parent,dest,mode):
    parent=Path(parent);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    pm=read(parent/'policies_manifest.json')
    opening=read(parent/(POLICY+'.snapshots.json'))
    admissions=rows(parent/(POLICY+'.admission.jsonl'))
    timed=rows(parent/'timed.snapshots.jsonl');tads=rows(parent/'timed.admission.jsonl')
    changed={POLICY+'.snapshots.json',POLICY+'.admission.jsonl','timed.snapshots.jsonl','timed.admission.jsonl','policies_manifest.json','completion.json'}
    for p in parent.iterdir():
        if p.is_file() and p.name not in changed:link(p,dest/p.name)
    excluded={}
    if mode=='no_support_additions':
        for r in admissions:
            if r.get('forecast_origin')=='new_forecast':excluded.setdefault(r['trade_date'],set()).add(r['opportunity_id'].split('/')[1])
    osn=[transform_snapshot(r,mode,excluded.get(r['trade_date'],())) for r in opening]
    tsn=[] if mode=='no_checkpoint' else [transform_snapshot(r,mode,excluded.get(r['trade_date'],())) for r in timed]
    def adjust(admissions,snaps,timed=False):
        lookup={(r['trade_date'],c['symbol']):c for r in snaps for c in r['candidates']};result=[]
        for r in admissions:
            key=(r['trade_date'],r['symbol'] if timed else r['opportunity_id'].split('/')[1])
            if key not in lookup:continue
            q=float(lookup[key]['metadata'].get('action_quantity_scale',1));v=dict(r)
            if q!=float(r.get('quantity_scale',q)):
                v.update(source_action_id=r.get('action_id'),quantity_scale=q,action_id=f'{key[0]}/{key[1]}/diagnostic_q{q:.12g}')
            result.append(v)
        return result
    write(dest/(POLICY+'.snapshots.json'),osn)
    jsonl(dest/(POLICY+'.admission.jsonl'),adjust(admissions,osn))
    jsonl(dest/'timed.snapshots.jsonl',tsn);jsonl(dest/'timed.admission.jsonl',adjust(tads,tsn,True))
    manifest={**pm,'diagnostic_transform':{'mode':mode,'parent_manifest':bound(parent/'policies_manifest.json'),
        'no_new_fit_or_outcome_condition':True,'quantity_estimates_not_rescored':mode.startswith(('atr_quantity','q1_'))}}
    manifest['files']={p.name:digest(p) for p in dest.iterdir() if p.is_file()}
    write(dest/'policies_manifest.json',manifest)
    write(dest/'completion.json',{'complete':True})
    m,s=verify_policy(dest,POLICY);timed_policy(dest,m,s)
    write(dest/'completion.json',{'complete':True,'transformation_verified':True})

def prepare(root,round_name='ablation',extra=None,workers=2):
    old=read(DIAG/'ablation_v2/study.json');out=root/round_name;out.mkdir(parents=True,exist_ok=False)
    recipes=extra or {k:v for k,v in old['recipes'].items() if k!='hard_stop_restored'}
    if 'control' not in recipes:recipes={'control':old['recipes']['control'],**recipes}
    periods={'is':old['periods']['is'],'oos':{**old['periods']['oos'],'plan':str(root/'plan.json'),
        'policies':str(root/'inputs/files/research/kalcb_executed_value/policies'),'sessions':69,
        'references':{s:str(root/'control'/s/'result.json') for s in ('baseline','combined')}}}
    bindings=[];cached={}
    for name,r in recipes.items():
        if name in old['recipes'] and r==old['recipes'][name]:
            cached[name]={s:str(DIAG/'ablation_v2/cases'/name/'is'/s) for s in ('baseline','combined')}
            for p in cached[name].values():
                for f in ('request.json','result.json','completion.json','daily.json','trades.json'):bindings.append(bound(Path(p)/f))
    for period,source in periods.items():
        for p in (source['plan'],source['config'],str(Path(source['policies'])/'policies_manifest.json')):bindings.append(bound(p))
        for mode in sorted({v['policy'] for v in recipes.values()}-{'control'}):
            dest=out/'policies'/period/mode
            prior=DIAG/'ablation_v2/policies/is'/mode
            if period=='is' and prior.exists():
                shutil.copytree(prior,dest,copy_function=link)
            else:transform(source['policies'],dest,mode)
            bindings.append(bound(dest/'policies_manifest.json'))
    spec={**old,'registered_at_utc':datetime.now(timezone.utc).isoformat(),'recipes':recipes,'periods':periods,
        'code':str(root/'inputs/files'),'source_generation_hash':read(root/'preparation.json')['source_generation_hash'],
        'bound_inputs':bindings,'source_programs':[bound(__file__),bound(REPO/'backtests/strategies/kalcb/oos_ablation.py')],
        'workers':workers,'cached_is':cached,'cached_is_source':old['source_generation_hash'],
        'oos_independence':'April 1-July 10 is reused development history; 69 sessions, all entry eligible.',
        'purpose':'Full-period controlled ablation; carry forward exact qualified-core IS experiments, never June-only OOS.',
        'physical_failure_policy':'Retain rejected evidence, exclude from ranking; a variant cannot terminate or validate other variants.'}
    write(out/'study.json',spec);print(json.dumps({'study':str(out/'study.json'),'recipes':len(recipes),'cached_is':len(cached)}),flush=True)

class RecoveryPlugin(KALCBOOSAblationPlugin):
    name='kalcb_full_oos_recovery_ablation'
    def run_native(self,name,period,scenario):
        if name=='control' and period=='oos':
            parent=Path(self.spec['periods']['oos']['references'][scenario]).parent
            r=read(parent/'result.json')
            if r['audit_status']!='passed' or not r['execution_validity_pass'] or r['source_generation_hash']!=self.spec['source_generation_hash']:raise ValueError('Unqualified cached full OOS control')
            folder,_,_=self.paths(name,period);out=folder/scenario
            if not out.exists():
                out.mkdir()
                for p in parent.iterdir():
                    if p.is_file():link(p,out/p.name)
                write(folder/(scenario+'.reuse.json'),{'source':str(parent),'result':bound(parent/'result.json'),'reason':'Exact fully audited 69-session control'})
            return r
        if period=='is' and name in self.spec['cached_is']:
            parent=Path(self.spec['cached_is'][name][scenario]);r=read(parent/'result.json')
            request=read(parent/'request.json');done=read(parent/'completion.json')
            expected=self.spec['cached_is_source']
            if {r['source_generation_hash'],request['source_generation_hash'],done['source_generation_hash']}!={expected}:raise ValueError('Changed cached IS source')
            old=read(DIAG/'ablation_v2/study.json')
            if self.spec['recipes'][name]!=old['recipes'][name]:raise ValueError('Cannot cache changed IS recipe')
            if not r['execution_validity_pass'] or r['terminal_inventory'] or r['terminal_pending'] or r['residual_sessions']:raise ValueError('Invalid cached IS')
            folder,plan,policies=self.paths(name,period);out=folder/scenario
            if not out.exists():
                out.mkdir()
                for p in parent.iterdir():
                    if p.is_file():link(p,out/p.name)
                write(folder/(scenario+'.reuse.json'),{'source':str(parent),'result':bound(parent/'result.json'),
                    'reason':'Identical IS configuration, policies, database and economic core; qualified economic parity held.',
                    'historical_source_generation_hash':expected,'new_market_source_generation_hash':self.spec['source_generation_hash']})
            return r
        return super().run_native(name,period,scenario)

    def evaluate_names(self,names):
        for name in names:
            for period in ('is','oos'):self.paths(name,period)
        jobs=[(n,p,s) for n in names for p in ('is','oos') for s in ('baseline','combined')]
        def one(job):
            try:return job,self.run_native(*job),None
            except Exception as e:return job,None,str(e)
        with ThreadPoolExecutor(max_workers=self.spec['workers']) as pool:results=list(pool.map(one,jobs))
        errors={}
        for (name,period,scenario),_,error in results:
            if error:errors.setdefault(name,[]).append({'period':period,'scenario':scenario,'error':error})
        if 'control' in errors:raise ValueError('Control failure: '+json.dumps(errors['control']))
        for name,failures in errors.items():
            record={'recipe':name,'changes':self.spec['recipes'][name],'rejected':True,'failures':failures,
                'metrics':{'physical_valid':False,'diagnostic_score':0,'research_only':True},'cases':{}}
            write(self.root/'cases'/name/'rejected_record.json',record);self.records[name]=record
        return [self.records[n] if n in errors else self.record(n) for n in names]

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        result=super().write_full_diagnostics(state,output_dir,**kwargs)
        p=Path(output_dir)/'round_final_diagnostics.txt'
        p.write_text(p.read_text().replace('All June-July variants','All April 1-July 10 variants'),encoding='utf-8')
        rejected=[read(p) for p in (self.root/'cases').glob('*/rejected_record.json')]
        write(Path(output_dir)/'rejected_cases.json',rejected)
        return {**result,'rejected_recipe_count':len(rejected)}

def run(study):
    plugin=RecoveryPlugin(study);manager=RoundManager('stock',plugin.name,base_dir=plugin.root/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Complete April-July paired recovery experiments',
        round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--round-name',default='ablation');p.add_argument('--recipes',type=Path);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.stage=='prepare':prepare(a.root,a.round_name,read(a.recipes) if a.recipes else None,a.workers)
    else:run(a.root/a.round_name/'study.json')

if __name__=='__main__':main()
