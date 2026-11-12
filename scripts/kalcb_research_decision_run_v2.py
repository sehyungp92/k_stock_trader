"""C18 finite diagnostic plugin in the existing RoundManager/PhaseRunner."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,time,threading,traceback
import psutil

OLD=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2/round_run_v2')
sys.path.insert(0,str(OLD/'controller/files'))
from backtests.auto.shared.round_manager import RoundManager
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.plugin import PhaseSpec,PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment,GateCriterion,ScoredCandidate,EndOfRoundArtifacts,PhaseDecision
from backtests.strategies.kalcb.round3_native_io_v1 import read,write,bound,profile_for
from backtests.strategies.kalcb.retention_research_io_v1 import env,digest
from backtests.strategies.kalcb.round3_contract_v1 import verify_acceptance
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_case

def execute(item,scenario,folder,bridge=None,hydrate=False):
    folder=Path(folder);accept=folder.parent/(folder.name+'_acceptance.json')
    if accept.exists():
        r=read(accept);verify_acceptance(r);return r
    if folder.exists():raise ValueError('Preserve interrupted generation '+str(folder))
    folder.parent.mkdir(parents=True,exist_ok=True);code=Path(item['manifest']['path']).parent/'files'
    for k in ['manifest','plan','policy']:
        assert digest(item[k]['path'])==item[k]['sha256']
    cmd=[sys.executable,'-B','-u','-m','scripts.kalcb_research_decision_worker_v1','--plan',item['plan']['path'],
        '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
        '--output',str(folder),'--scenario',scenario,'--observe']
    if bridge:cmd+=['--bridge',str(bridge)]
    if hydrate:cmd+=['--hydrate']
    if item['arm']=='off' and item['period']=='is':
        cmd=cmd[:3]+['-m','cProfile','-o',str(folder.parent/(folder.name+'_profile.pstats'))]+cmd[3:]
    began=time.monotonic();peak=0;guard=None
    with folder.with_suffix('.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,cwd=code,env={**env(code),'ARROW_IO_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        proc=psutil.Process(child.pid)
        write(folder.parent/(folder.name+'_launch.json'),dict(pid=child.pid,created=proc.create_time(),argv=cmd,
            at_utc=datetime.now(timezone.utc).isoformat(),nested_native_workers=0))
        while child.poll() is None:
            try:peak=max(peak,proc.memory_info().rss)
            except psutil.NoSuchProcess:break
            if peak>2*1024**3 or time.monotonic()-began>2400:
                guard='2GiB' if peak>2*1024**3 else '40min';child.terminate();break
            time.sleep(2)
        rc=child.wait()
    process=dict(seconds=time.monotonic()-began,peak_rss=peak,exit_code=rc,guard=guard,arm=item['arm'],period=item['period'],scenario=scenario,capital=item['initial'])
    write(folder.parent/(folder.name+'_process.json'),process)
    if rc or guard:
        write(folder.parent/(folder.name+'_invalid.json'),dict(passed=False,eligible_for_scoring=False,process=process,log=bound(folder.with_suffix('.log'))))
        raise RuntimeError('Invalid native generation '+str(folder))
    try:
        profile=profile_for(item,folder,scenario);case=validate_case(folder,profile,scenario)
        independent=read(folder/'independent_journal.json')
        assert independent['passed'] and not independent['terminal_inventory'] and not independent['pending']
        if bridge:assert read(folder/'full_path_bridge.json')['passed']
        if hydrate:assert read(folder/'hydration.json')
        completion=read(folder/'completion.json');assert completion['c18_target']==item['arm']
        if item['arm']!='off':assert completion['delayed_decisions']>0
        write(folder/'independent_account_verification.json',case['validation'])
    except Exception as e:
        write(folder/'qualification_failure.json',dict(passed=False,eligible_for_scoring=False,error=repr(e)));raise
    names=['request.json','completion.json','result.json','execution_audit.json','fills.json','trades.json','daily.json','equity.json',
        'decisions.jsonl.gz','orders.jsonl.gz','broker_journal.jsonl.gz','account_states.jsonl.gz','horizon_actions.json',
        'independent_journal.json','independent_account_verification.json']
    if bridge:names+=['full_path_bridge.json']
    if hydrate:names+=['hydration.json']
    receipt=dict(passed=True,eligible_for_scoring=False,diagnostic_only=True,process=process,item=item,profile=profile,
        folder=str(folder),files={n:bound(folder/n) for n in names})
    write(accept,receipt);return receipt

class DiagnosticPlugin:
    name='kalcb_research_decision';num_phases=2;ultimate_targets={};max_workers=2;capability_level='real_replay';requires_full_diagnostics=False
    def __init__(self,registration,phase_limit):
        self.registration=Path(registration);self.root=self.registration.parent;self.reg=read(registration)
        self.num_phases=phase_limit;self.records={};self.prep=read(self.reg['native_preparation']['path'])
        source=read(OLD/'round_artifacts/phase_state.json');self.baseline_metrics=source['phase_results']['4']['final_metrics']
        self.initial_mutations=source['cumulative_mutations'];self.initial_mutations_override=deepcopy(self.initial_mutations)
        self.config=dict(initial_mutations=self.initial_mutations,previous_promotion_guard_required=False,production_enabled=False,
            initial_equity=1e8,max_workers=2,capability_level='real_replay',promotion_status='diagnostic_only_no_selection')
        self.execution_context=deepcopy(self.baseline_metrics.get('execution_contract',{}))
        self.execution_context.update(production_enabled=False,max_workers=2,diagnostic_registration=bound(registration))
        self.controls={r['item']['period']+'/'+r['process']['scenario']:r for r in read(OLD/'evaluations/v6_h0.seal.json')['receipts']}
        assert read(self.reg['saved_evidence']['path'])['passed']
        assert read(self.reg['unit_receipt']['path'])['passed']
        for b in self.reg['bindings']:
            assert digest(b['path'])==b['sha256'],'C18 controller dependency drift'
    def canonicalize_mutations(self,mutations):
        if mutations!=self.initial_mutations:raise ValueError('Diagnostic cannot select or mutate a trading policy')
        return deepcopy(mutations)
    def get_phase_spec(self,phase,state):
        names=['off'] if phase==1 else ['entry','exit']
        return PhaseSpec('C18 '+('disabled identity and hydration' if phase==1 else 'entry-only versus exit-only execution delay'),
            [Experiment(n,{'__diagnostic__':n}) for n in names],lambda m:[GateCriterion('retained_control_valid',1,1,True)],None,{},
            PhaseAnalysisPolicy(focus_metrics=['c16_S'],decide_action_fn=lambda *a:PhaseDecision('advance','Finite diagnostic only; no policy selection')),
            max_rounds=1,phase_metric_basis='authenticated_retained_control',primary_promotion_metric='c16_S',
            official_metric_keys=('c16_S',),promotion_requires_audit_pass=True)
    def evaluate(self,name,callback):
        if time.time()>self.reg['financial_deadline_epoch']:raise RuntimeError('Registered financial-work deadline reached; reporting reserve preserved')
        path=self.root/'evaluations'/(name+'.json')
        if path.exists():
            data=read(path)
            for r in data['receipts']:verify_acceptance(r)
            self.records[name]=data;return
        if name!='off':
            assert read(self.root/'evaluations/off.json')['passed']
            off=read(self.root/'evaluations/off.json')
            worst=max(r['process']['seconds'] for r in off['receipts'])
            estimated_remaining=16*max(worst,60)/2*1.5
            if time.time()+estimated_remaining>self.reg['financial_deadline_epoch']:raise RuntimeError('Measured timing runtime exceeds registered remaining envelope')
        pairs=[('is','baseline')] if name=='off' else [(p,s) for p in ['is','oos'] for s in ['baseline','participation','cost','combined']]
        receipts=[]
        if name=='off':
            reused=read(self.reg['reused_off_receipt']['path']);verify_acceptance(reused)
            assert reused['item']['arm']=='off' and reused['item']['period']=='oos' and reused['process']['scenario']=='combined'
            assert read(Path(reused['folder'])/'full_path_bridge.json')['passed']
            receipts.append(reused)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs={pool.submit(execute,self.prep['profiles'][p][name],s,self.root/'accounts'/name/p/s,
                self.controls[p+'/'+s]['folder'] if name=='off' else None,name=='off'):(p,s) for p,s in pairs}
            for f in as_completed(jobs):
                receipts.append(f.result())
                if callback:callback(dict(event='accepted_diagnostic_account',arm=name,cell='/'.join(jobs[f]),accepted=len(receipts),planned=len(pairs)))
        data=dict(passed=True,diagnostic_only=True,eligible_for_scoring=False,receipts=receipts)
        write(path,data);self.records[name]=data
    def create_evaluate_batch(self,phase,cumulative_mutations,**kwargs):
        owner=self
        class Eval:
            callback=None
            def set_progress_callback(self,callback):self.callback=callback
            def __call__(self,candidates,current):
                result=[]
                for c in candidates:
                    if c.name=='__baseline__':result.append(ScoredCandidate(c.name,owner.baseline_metrics['c16_S'],metrics={'accepted_control':True}));continue
                    owner.evaluate(c.name,self.callback)
                    # No diagnostic result can enter scoring or adoption, irrespective of its return.
                return result
        return Eval()
    def compute_final_metrics(self,mutations):
        self.canonicalize_mutations(mutations);return deepcopy(self.baseline_metrics)
    def validate_archived_execution(self,metrics):
        assert metrics['c16_S']==self.baseline_metrics['c16_S']
        return read(self.reg['saved_evidence']['path'])['passed']
    def run_phase_diagnostics(self,phase,state,metrics,greedy_result):return json.dumps({'phase':phase,'evaluations':list(self.records),'no_policy_adopted':True})
    run_enhanced_diagnostics=run_phase_diagnostics
    def build_end_of_round_artifacts(self,state):return EndOfRoundArtifacts(json.dumps({'diagnostics':list(self.records),'comparison_only':True}),{},'C18 timing subphase; research decision/report remains pending')

def monitor(root):
    root=Path(root);running=[];invalid=[]
    for f in (root/'accounts').rglob('*_launch.json'):
        d=read(f)
        try:
            p=psutil.Process(d['pid']);valid=abs(p.create_time()-d['created'])<.01 and p.cmdline()==d['argv']
            if valid:running.append(dict(pid=p.pid,created=p.create_time(),argv=p.cmdline(),cpu=p.cpu_times().user,rss=p.memory_info().rss,launch=str(f)))
        except psutil.NoSuchProcess:pass
    invalid=[str(p) for p in (root/'accounts').rglob('*_invalid.json')]+[str(p) for p in (root/'accounts').rglob('qualification_failure.json')]
    if len(running)>2:raise ValueError('C18 native concurrency exceeded')
    progress={str(p.relative_to(root)):read(p) for p in (root/'framework').rglob('progress.json')}
    at=datetime.now(timezone.utc);directory=root/'monitor_checks';directory.mkdir(exist_ok=True)
    write(directory/(at.strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(at_utc=at.isoformat(),workers=running,accepted=len(list((root/'accounts').rglob('*_acceptance.json'))),
        invalid=invalid,progress=progress,prediction_review='Timing outcomes require paired complete scenarios; no mid-run retuning',
        phase_complete=(root/'timing_complete.json').exists()))
    return dict(workers=len(running),accepted=len(list((root/'accounts').rglob('*_acceptance.json'))),invalid=invalid)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);ap.add_argument('--phase-limit',type=int,choices=[1,2],default=2);ap.add_argument('--monitor',action='store_true');a=ap.parse_args()
    reg=Path(a.registration);root=reg.parent
    if a.monitor:print(json.dumps(monitor(root)));return
    plugin=DiagnosticPlugin(reg,a.phase_limit);proc=psutil.Process();write(root/'launch.json',dict(pid=proc.pid,created=proc.create_time(),argv=proc.cmdline(),registration=bound(reg),max_workers=2))
    stop=threading.Event()
    def loop():
        while not stop.wait(1800):
            try:monitor(root)
            except Exception as e:
                with (root/'monitor_errors.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(error=repr(e),time=time.time()))+'\n')
    threading.Thread(target=loop,daemon=True).start();began=time.monotonic()
    try:
        manager=RoundManager('stock','kalcb_research_decision',base_dir=root/'framework')
        runner=PhaseRunner(plugin,root/'framework/diagnostic',round_name='C18 timing diagnostic; no optimisation',max_rounds=1,min_delta=0,max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=1)
        runner.run_all_phases()
        write(root/'timing_complete.json',dict(passed=True,seconds=time.monotonic()-began,accounts=sum(len(x['receipts']) for x in plugin.records.values()),research_decision_complete=False))
    except Exception as e:
        write(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:stop.set();monitor(root)

if __name__=='__main__':main()
