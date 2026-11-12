"""Register targeted policy/configuration interactions for native IS/OOS replay."""
from datetime import datetime,timezone
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,DIAG,read,write,bound,link
from scripts.kalcb_recovery_ablation import RecoveryPlugin,transform
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager

def prepare(root,name,definitions,workers):
    parent=read(root/'ablation_v3/study.json');dest=root/name;dest.mkdir(exist_ok=False)
    specs={'control':{'config':{},'policy':'control'}};overrides={'is':{},'oos':{}};cached={};bindings=list(parent['bound_inputs'])
    old=read(DIAG/'ablation_v2/study.json')
    for case,recipe in definitions.items():
        specs[case]={'config':recipe.get('config',{}),'policy':'control'}
        for period in ('is','oos'):
            source=Path(recipe.get(period+'_policy',parent['periods'][period]['policies']))
            mode=recipe.get('quantity_rule')
            if mode:
                target=dest/'policies'/period/case;transform(source,target,mode);source=target
            if not read(source/'completion.json')['complete']:raise ValueError('Incomplete policy '+str(source))
            overrides[period][case]=str(source);bindings.append(bound(source/'policies_manifest.json'))
        # Reuse only the exact historical policy/config combination. OOS
        # policies may differ, but their forward performance is always replayed.
        if not recipe.get('is_policy'):
            match=next((k for k,v in old['recipes'].items() if v['config']==recipe.get('config',{}) and v['policy']==(recipe.get('quantity_rule') or 'control')),None)
            if match and match!='hard_stop_restored':
                cached[case]={s:str(DIAG/'ablation_v2/cases'/match/'is'/s) for s in ('baseline','combined')}
                for p in cached[case].values():
                    for f in ('result.json','request.json','completion.json','daily.json','trades.json'):bindings.append(bound(Path(p)/f))
    spec={**parent,'registered_at_utc':datetime.now(timezone.utc).isoformat(),'recipes':specs,'experiment_definitions':definitions,
        'policy_overrides_by_period':overrides,'cached_interaction_is':cached,'bound_inputs':bindings,'workers':workers,
        'source_programs':[bound(__file__),bound(Path(__file__).with_name('kalcb_recovery_ablation.py'))],
        'purpose':'Precisely registered interactions; actual changed IS rules replayed, exact unchanged IS accounts explicitly reused.',
        'is_interpretation':'Forward learner recipes inherit selected IS as a reference. Configuration and quantity changes are compared on the same historical IS policy; this is not a new IS test of an OOS-only training schedule.'}
    write(dest/'study.json',spec)

class ExperimentPlugin(RecoveryPlugin):
    name='kalcb_full_oos_targeted_experiments'
    def paths(self,name,period):
        folder,plan,policies=super().paths(name,period)
        if name!='control':policies=Path(self.spec['policy_overrides_by_period'][period][name])
        return folder,plan,policies
    def run_native(self,name,period,scenario):
        if period=='is' and name in self.spec['cached_interaction_is']:
            parent=Path(self.spec['cached_interaction_is'][name][scenario]);r=read(parent/'result.json');request=read(parent/'request.json')
            if {r['source_generation_hash'],request['source_generation_hash']}!={self.spec['cached_is_source']} or not r['execution_validity_pass'] or r['residual_sessions'] or r['terminal_inventory'] or r['terminal_pending']:raise ValueError('Invalid historical interaction reference')
            folder,_,_=self.paths(name,period);out=folder/scenario
            if not out.exists():
                out.mkdir()
                for p in parent.iterdir():
                    if p.is_file():link(p,out/p.name)
                write(folder/(scenario+'.reuse.json'),{'source':str(parent),'result':bound(parent/'result.json'),
                    'scope':self.spec['is_interpretation'],'configuration_and_policy_identity':'Matched registered recipe against original IS study; source code generation identical.'})
            return r
        return super().run_native(name,period,scenario)

def run(root,name):
    plugin=ExperimentPlugin(root/name/'study.json');manager=RoundManager('stock',plugin.name,base_dir=plugin.root/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name=name,round_manager=manager,round_num=1,
        max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--round-name',required=True);p.add_argument('--recipes',type=Path);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    prepare(a.root,a.round_name,read(a.recipes),a.workers) if a.stage=='prepare' else run(a.root,a.round_name)
