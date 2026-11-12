"""Finite C18 oracle diagnostics through the existing shared phase framework."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,importlib.util,json,os,subprocess,sys,time,threading,traceback
import psutil

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    import hashlib
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(p),sha256=sha(p))
def verify(b):assert sha(b['path'])==b['sha256']
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def load(name,b):
    verify(b);s=importlib.util.spec_from_file_location(name,b['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def execute(registration,arm,period,scenario,smoke=False):
    reg=read(registration);root=Path(registration).parent;tag='smoke_'+arm if smoke else arm
    folder=root/'accounts'/tag/period/scenario;receipt=folder.parent/(folder.name+'_acceptance.json')
    if receipt.exists():
        result=read(receipt)
        for b in result['files'].values():verify(b)
        assert result['passed'] and not result['eligible_for_scoring'];return result
    if folder.exists():raise ValueError('Preserve incomplete oracle generation '+str(folder))
    folder.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-B','-u',reg['worker_program']['path'],'--registration',str(registration),'--arm',arm,
        '--period',period,'--scenario',scenario,'--output',str(folder)]
    if smoke:cmd+=['--smoke']
    environment={**os.environ,**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_IO_THREADS']},'PYTHONDONTWRITEBYTECODE':'1'}
    start=time.monotonic();peak=0;guard=None
    with folder.with_suffix('.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        process=psutil.Process(child.pid)
        write(folder.parent/(folder.name+'_launch.json'),dict(pid=child.pid,created=process.create_time(),argv=cmd,at_utc=datetime.now(timezone.utc).isoformat(),nested_workers=0))
        while child.poll() is None:
            try:peak=max(peak,process.memory_info().rss)
            except psutil.NoSuchProcess:break
            if peak>2*1024**3 or time.monotonic()-start>2400 or time.time()>reg['financial_deadline_epoch']:
                guard='registered memory/time/deadline';child.terminate();break
            time.sleep(2)
        rc=child.wait()
    process_record=dict(seconds=time.monotonic()-start,peak_rss=peak,exit_code=rc,guard=guard,arm=arm,period=period,scenario=scenario,smoke=smoke)
    write(folder.parent/(folder.name+'_process.json'),process_record)
    if rc or guard:
        write(folder.parent/(folder.name+'_invalid.json'),dict(passed=False,eligible_for_scoring=False,process=process_record))
        raise RuntimeError('Invalid oracle generation '+str(folder))
    if smoke:
        native=read(folder/'smoke.json');source=read(folder/'smoke_source.json')
        assert native['native_lifecycle_completed'] and source['passed']
        choices=[r for r in source['changed_selection_choices'] if r['day']<=reg['smoke_through'][period]]
        fills=read(folder/'smoke_fills.json');base=read(Path(reg['controls'][period][scenario]['folder'])/'fills.json')
        base=[f for f in base if f['timestamp'][:10]<=reg['smoke_through'][period]]
        economic=lambda f:(f['symbol'],str(f['timestamp']).replace(' ','T'),f['side'],f['qty'],f['price'])
        from collections import Counter
        left=Counter(map(economic,base));right=Counter(map(economic,fills))
        reach=dict(eligible_menu_choices=sum(x['positive_menu_candidates'] for x in choices),selected_choices=sum(len(x['selected']) for x in choices),
            actual_exit_choices=len(source['exit_choices']),scheduled_exit_decisions=source['scheduled_exits'],
            changed_economic_fills=sum((left-right).values())+sum((right-left).values()),prefix_only=True,
            inactive_prefix_is_not_economic_failure=True)
        write(folder/'reach.json',reach)
    else:
        complete=read(folder/'completion.json');assert complete['passed'] and complete['conditional_native_execution_qualified']
        audit=read(folder/'execution_audit.json');assert audit['audit_status']=='passed' and audit['execution_validity_pass']
        separate=load('c18_separate_oracle_accounting',reg['independent_accounting_program'])
        record,_=separate.account(folder);assert record['passed']
        write(folder/'independent_funding_verification.json',record)
        if arm=='off':assert read(folder/'full_path_bridge.json')['passed'] and read(folder/'hydration.json')
    files={p.name:bound(p) for p in folder.iterdir() if p.is_file()}
    result=dict(passed=True,eligible_for_scoring=False,oracle_only=arm!='off',smoke=smoke,process=process_record,
        folder=str(folder),files=files,item={**reg['profiles'][period],'arm':arm},causal_policy_qualified=False,
        native_reach=bool(reach['changed_economic_fills']) if smoke else True)
    write(receipt,result);return result

def plugin_class(base):
    class OraclePlugin(base.DiagnosticPlugin):
        def __init__(self,registration):
            reg=read(registration);super().__init__(reg['timing_registration']['path'],2)
            self.num_phases=3;self.reg=reg;self.root=Path(registration).parent;self.registration=Path(registration);self.records={}
            for b in reg['programs']+reg['bindings']:verify(b)
            for b in reg['required_completions']:verify(b);assert read(b['path'])['passed']
        def get_phase_spec(self,phase,state):
            names={1:['off'],2:['smoke_S','smoke_E','smoke_SE'],3:['S','E','SE']}[phase]
            return base.PhaseSpec('C18 '+{1:'oracle worker disabled identity',2:'native oracle reach',3:'funded hindsight diagnostics'}[phase],
                [base.Experiment(n,{'__diagnostic__':n}) for n in names],lambda m:[base.GateCriterion('retained_control_valid',1,1,True)],None,{},
                base.PhaseAnalysisPolicy(focus_metrics=['c16_S'],decide_action_fn=lambda *a:base.PhaseDecision('advance','Fixed hindsight diagnostic; no trading-policy adoption')),
                max_rounds=1,phase_metric_basis='authenticated_retained_control',primary_promotion_metric='c16_S',official_metric_keys=('c16_S',),promotion_requires_audit_pass=True)
        def evaluate(self,name,callback):
            if time.time()>self.reg['financial_deadline_epoch']:raise RuntimeError('Reporting reserve reached')
            if name=='off':items=[('off','oos','combined',False)]
            elif name.startswith('smoke_'):
                assert read(self.root/'evaluations/off.json')['passed'];items=[(name[6:],'is','baseline',True)]
            else:
                for arm in ['S','E','SE']:assert read(self.root/('evaluations/smoke_'+arm+'.json'))['passed']
                own=read(self.root/('evaluations/smoke_'+name+'.json'))
                if not all(r['native_reach'] for r in own['receipts']):
                    record=dict(passed=True,eligible_for_scoring=False,receipts=[],status='withheld_fixed_prefix_inactive',
                        economic_verdict='untested; no full account dispatched',coverage_gap=True)
                    write(self.root/'evaluations'/(name+'.json'),record);self.records[name]=record
                    if callback:callback(dict(event='oracle_coverage_gap',arm=name,reason='fixed prefix inactive, not economic failure'))
                    return
                items=[(name,p,s,False) for p in ['is','oos'] for s in ['baseline','combined']]
            if name in ('S','E','SE'):
                worst=max(self.reg['planning_native_seconds'],max(r['process']['seconds'] for r in read(self.root/'evaluations/off.json')['receipts']))
                remaining={'S':12,'E':8,'SE':4}[name]*worst/2*1.3
                if time.time()+remaining>self.reg['financial_deadline_epoch']:raise RuntimeError('Measured remaining oracle cost exceeds chosen budget')
            receipts=[]
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs={pool.submit(execute,self.registration,*item):item for item in items}
                for f in as_completed(jobs):
                    receipts.append(f.result())
                    if callback:callback(dict(event='accepted_oracle_diagnostic',arm=name,cell=jobs[f],accepted=len(receipts),planned=len(items)))
            record=dict(passed=True,eligible_for_scoring=False,receipts=receipts);write(self.root/'evaluations'/(name+'.json'),record);self.records[name]=record
        def build_end_of_round_artifacts(self,state):
            return base.EndOfRoundArtifacts(json.dumps({'diagnostics':list(self.records),'oracle_not_causal_policy':True}),{},'C18 funded oracle execution complete; independent research decision/report remains pending')
    return OraclePlugin

def monitor(root):
    root=Path(root);active=[]
    for f in root.glob('accounts/*/*/*_launch.json'):
        d=read(f)
        try:
            p=psutil.Process(d['pid'])
            if abs(p.create_time()-d['created'])<.01 and p.cmdline()==d['argv']:active.append(dict(pid=p.pid,created=p.create_time(),argv=p.cmdline(),rss=p.memory_info().rss,cpu=p.cpu_times().user))
        except psutil.NoSuchProcess:pass
    assert len(active)<=2
    value=dict(at_utc=datetime.now(timezone.utc).isoformat(),workers=active,accepted=len(list(root.glob('accounts/*/*/*_acceptance.json'))),
        invalid=[str(p) for p in root.glob('accounts/*/*/*_invalid.json')],
        progress={str(p.relative_to(root)):read(p) for p in (root/'framework').rglob('progress.json')},
        prediction_review='Fixed feasible hindsight heuristics; no retuning, no optimum/capturable-alpha claim',complete=(root/'oracle_complete.json').exists())
    write(root/'monitor_checks'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),value)
    return dict(workers=len(active),accepted=value['accepted'],invalid=value['invalid'],complete=value['complete'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);ap.add_argument('--monitor',action='store_true');a=ap.parse_args()
    root=Path(a.registration).parent;reg=read(a.registration)
    if a.monitor:print(json.dumps(monitor(root)));return
    for prerequisite in reg['prerequisite_launches']:
        d=read(prerequisite['path'])
        if psutil.pid_exists(d['pid']):
            p=psutil.Process(d['pid'])
            if abs(p.create_time()-d['created'])<.01:raise RuntimeError('Prerequisite controller still active; no overlapping financial dispatch')
    base=load('c18_shared_base',reg['base_controller']);plugin=plugin_class(base)(a.registration)
    p=psutil.Process();write(root/'launch.json',dict(pid=p.pid,created=p.create_time(),argv=p.cmdline(),registration=bound(a.registration),max_workers=2))
    stop=threading.Event()
    def loop():
        while not stop.wait(1800):monitor(root)
    threading.Thread(target=loop,daemon=True).start();start=time.monotonic()
    try:
        manager=base.RoundManager('stock','kalcb_research_decision_oracles',base_dir=root/'framework')
        runner=base.PhaseRunner(plugin,root/'framework/oracles',round_name='C18 fixed funded oracles; no optimisation',max_rounds=1,min_delta=0,
            max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=1)
        runner.run_all_phases();write(root/'oracle_complete.json',dict(passed=True,seconds=time.monotonic()-start,eligible_for_scoring=False,research_decision_complete=False))
    except Exception as e:
        write(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:stop.set();monitor(root)
if __name__=='__main__':main()
