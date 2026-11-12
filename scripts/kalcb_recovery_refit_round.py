"""Native registered replay of completed causal recovery policy experiments."""
from datetime import datetime,timezone
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,DIAG,read,write,bound,link
from scripts.kalcb_recovery_ablation import RecoveryPlugin
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager

def prepare(root,names,round_name):
    parent=read(root/'ablation_v3/study.json');dest=root/round_name;dest.mkdir(exist_ok=False)
    policies={n:str(root/'refits/policies'/n) for n in names}
    bindings=list(parent['bound_inputs'])
    for p in policies.values():
        if not read(Path(p)/'completion.json')['complete']:raise ValueError('Incomplete fitted policy')
        bindings.append(bound(Path(p)/'policies_manifest.json'))
    spec={**parent,'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'recipes':{'control':{'config':{},'policy':'control'},**{n:{'config':{},'policy':'control'} for n in names}},
        'policy_overrides':policies,'bound_inputs':bindings,
        'source_programs':[bound(__file__),bound(Path(__file__).with_name('kalcb_recovery_ablation.py'))],
        'workers':1,'purpose':'Causal deployment continuation from the selected March 31 IS state; values/calibration controls.',
        'is_interpretation':'Unchanged selected IS reference, not a new IS test of the alternative forward fitting schedule.'}
    write(dest/'study.json',spec)

class RefitPlugin(RecoveryPlugin):
    name='kalcb_full_oos_causal_refits'
    def paths(self,name,period):
        folder,plan,policies=super().paths(name,period)
        if period=='oos' and name!='control':policies=Path(self.spec['policy_overrides'][name])
        return folder,plan,policies
    def run_native(self,name,period,scenario):
        if period=='is' and name!='control':
            parent=DIAG/'ablation_v2/cases/control/is'/scenario;r=read(parent/'result.json')
            if not r['execution_validity_pass'] or r['source_generation_hash']!=self.spec['cached_is_source']:raise ValueError('Changed IS reference')
            folder,_,_=self.paths(name,period);out=folder/scenario
            if not out.exists():
                out.mkdir()
                for p in parent.iterdir():
                    if p.is_file():link(p,out/p.name)
                write(folder/(scenario+'.reuse.json'),{'source':str(parent),'result':bound(parent/'result.json'),
                    'scope':self.spec['is_interpretation']})
            return r
        return super().run_native(name,period,scenario)

def run(root,round_name):
    plugin=RefitPlugin(root/round_name/'study.json');manager=RoundManager('stock',plugin.name,base_dir=plugin.root/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Causal economic refits on April-July development history',
        round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--round-name',default='refit_round');p.add_argument('--recipes',nargs='+');a=p.parse_args()
    if a.stage=='prepare':prepare(a.root,a.recipes,a.round_name)
    else:run(a.root,a.round_name)
