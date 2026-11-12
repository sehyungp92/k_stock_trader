"""C18 diagnostic labels in the existing shared framework; never selectable."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,importlib.util,json,os,subprocess,sys,time,traceback,threading
import psutil

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    import hashlib
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def verify(b):assert sha(b['path'])==b['sha256'],str(b)

def aggregate_seconds(root):
    total=0.
    for f in Path(root).rglob('*_process.json'):total+=read(f)['seconds']
    for f in Path(root).rglob('*_launch.json'):
        if f.with_name(f.name.replace('_launch.json','_process.json')).exists():continue
        d=read(f)
        try:
            p=psutil.Process(d['pid'])
            if abs(p.create_time()-d['created'])<.01 and p.cmdline()==d['argv']:total+=max(0,time.time()-p.create_time())
        except psutil.NoSuchProcess:pass
    return total

def launch(registration,item):
    reg=read(registration);root=Path(registration).parent;period,scenario,pilot=item
    folder=root/('pilot' if pilot else 'full')/period/scenario
    receipt=folder.parent/(folder.name+'_acceptance.json')
    if receipt.exists():
        result=read(receipt)
        for b in result['files'].values():verify(b)
        assert result['passed'] and not result['eligible_for_scoring'];return result
    folder.parent.mkdir(parents=True,exist_ok=True)
    if folder.exists():raise RuntimeError('Preserve incomplete label generation '+str(folder))
    cmd=[sys.executable,'-B','-u',reg['worker_program']['path'],'--registration',str(registration),
        '--period',period,'--scenario',scenario,'--output',str(folder)]
    if pilot:cmd+=['--pilot']
    began=time.monotonic();peak=0;guard=None;budget_check_at=0.;spent=0.
    env={**os.environ,**{n:'1' for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_IO_THREADS']},'PYTHONDONTWRITEBYTECODE':'1'}
    with folder.with_suffix('.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        proc=psutil.Process(child.pid)
        write(folder.parent/(folder.name+'_launch.json'),dict(pid=proc.pid,created=proc.create_time(),argv=cmd,at_utc=datetime.now(timezone.utc).isoformat(),nested_workers=0))
        while child.poll() is None:
            try:peak=max(peak,proc.memory_info().rss)
            except psutil.NoSuchProcess:break
            if time.monotonic()-budget_check_at>20:
                spent=aggregate_seconds(root);budget_check_at=time.monotonic()
            if peak>2*1024**3 or time.monotonic()-began>reg['worker_seconds_guard'] or time.time()>reg['financial_deadline_epoch'] or spent>=reg['aggregate_label_seconds']-30:
                guard='registered memory/time/deadline';child.terminate();break
            time.sleep(2)
        rc=child.wait()
    process=dict(seconds=time.monotonic()-began,peak_rss=peak,exit_code=rc,guard=guard,period=period,scenario=scenario,pilot=pilot)
    write(folder.parent/(folder.name+'_process.json'),process)
    if rc or guard:
        write(folder.parent/(folder.name+'_invalid.json'),dict(passed=False,process=process,eligible_for_scoring=False))
        raise RuntimeError('Invalid label batch '+str(folder))
    complete=read(folder/'completion.json');assert complete['passed'] and not complete['eligible_for_scoring']
    assert complete['outcomes']==4*complete['distinct_episodes'] and complete['max_independent_cash_error']<.02
    assert complete['statuses'].get('physical_invalid',0)==0
    for b in complete['files'].values():verify(b)
    result=dict(passed=True,eligible_for_scoring=False,process=process,completion=complete,
        files={**complete['files'],'completion.json':bound(folder/'completion.json')})
    write(receipt,result);return result

def install_plugin(base):
    class LabelPlugin(base.DiagnosticPlugin):
        def __init__(self,registration):
            reg=read(registration);super().__init__(reg['timing_registration']['path'],2)
            self.registration=Path(registration);self.root=self.registration.parent;self.reg=reg;self.records={}
            for b in reg['programs']+reg['bindings']:verify(b)
            assert read(reg['horizon_unit']['path'])['passed']
        def get_phase_spec(self,phase,state):
            name='pilot' if phase==1 else 'full'
            return base.PhaseSpec('C18 standalone labels '+name,[base.Experiment(name,{'__diagnostic__':name})],
                lambda m:[base.GateCriterion('retained_control_valid',1,1,True)],None,{},
                base.PhaseAnalysisPolicy(focus_metrics=['c16_S'],decide_action_fn=lambda *a:base.PhaseDecision('advance','Bounded diagnostic labels; no policy adoption')),
                max_rounds=1,phase_metric_basis='authenticated_retained_control',primary_promotion_metric='c16_S',
                official_metric_keys=('c16_S',),promotion_requires_audit_pass=True)
        def evaluate(self,name,callback):
            if time.time()>self.reg['financial_deadline_epoch']:raise RuntimeError('Reporting reserve reached')
            pilot=name=='pilot'
            if not pilot:
                previous=read(self.root/'evaluations/pilot.json');assert previous['passed']
                measured={r['process']['period']:r for r in previous['receipts']}
                estimates={p:max(60,r['process']['seconds'])*self.reg['expected_episode_counts'][p]/max(1,r['completion']['distinct_episodes'])*2*1.5 for p,r in measured.items()}
                spent=sum(r['process']['seconds'] for r in previous['receipts'])
                estimate=dict(at_utc=datetime.now(timezone.utc).isoformat(),period_process_seconds=estimates,
                    aggregate_process_seconds=sum(estimates.values())+spent,pilot_spent_seconds=spent,
                    factor=1.5,source='measured first2sessions; conservative includes repeated source preparation',
                    enough_budget=(sum(estimates.values())+spent<=self.reg['aggregate_label_seconds'] and time.time()+sum(estimates.values())/2<self.reg['financial_deadline_epoch']))
                write(self.root/'measured_full_cost.json',estimate)
                if not estimate['enough_budget']:raise RuntimeError('Measured enumeration does not fit registered label budget; no full launch')
            items=[(p,s,pilot) for p in ['is','oos'] for s in (['baseline'] if pilot else ['baseline','combined'])]
            receipts=[]
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs={pool.submit(launch,self.registration,item):item for item in items}
                for f in as_completed(jobs):
                    r=f.result();receipts.append(r)
                    if callback:callback(dict(event='accepted_standalone_label_batch',phase=name,cell=jobs[f],accepted=len(receipts),planned=len(items)))
            record=dict(passed=True,eligible_for_scoring=False,receipts=receipts)
            write(self.root/'evaluations'/(name+'.json'),record);self.records[name]=record
        def build_end_of_round_artifacts(self,state):
            return base.EndOfRoundArtifacts(json.dumps({'label_phases':list(self.records),'comparison_only':True}),{},
                'C18 standalone labels; funded/predictive diagnostics and research decision remain pending')
    return LabelPlugin

def monitor(root):
    root=Path(root);workers=[]
    for f in root.rglob('*_launch.json'):
        d=read(f)
        try:
            p=psutil.Process(d['pid'])
            if abs(p.create_time()-d['created'])<.01 and p.cmdline()==d['argv']:
                workers.append(dict(pid=p.pid,created=p.create_time(),argv=p.cmdline(),cpu=p.cpu_times().user,rss=p.memory_info().rss))
        except psutil.NoSuchProcess:pass
    assert len(workers)<=2
    result=dict(at_utc=datetime.now(timezone.utc).isoformat(),workers=workers,
        accepted=[str(p) for p in root.rglob('*_acceptance.json')],invalid=[str(p) for p in root.rglob('*_invalid.json')],
        progress={str(p.relative_to(root)):read(p) for p in (root/'framework').rglob('progress.json')},
        predictions='Labels do not establish executable funded value; no mid-run changes',complete=(root/'labels_complete.json').exists())
    write(root/'monitor_checks'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),result)
    return dict(workers=len(workers),accepted=len(result['accepted']),invalid=result['invalid'],complete=result['complete'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);ap.add_argument('--monitor',action='store_true');a=ap.parse_args()
    root=Path(a.registration).parent;reg=read(a.registration)
    if a.monitor:print(json.dumps(monitor(root)));return
    for b in reg['programs']+reg['bindings']:verify(b)
    proc=psutil.Process();write(root/'launch.json',dict(pid=proc.pid,created=proc.create_time(),argv=proc.cmdline(),registration=bound(a.registration),max_workers=2))
    timing=Path(reg['timing_registration']['path']).parent
    # This serial dependency prevents auto-dispatched timing workers and label
    # workers from ever exceeding the global native-worker bound.
    while not (timing/'timing_complete.json').exists():
        if (timing/'failure.json').exists():raise RuntimeError('Timing prerequisite failed; labels withheld')
        if time.time()>reg['financial_deadline_epoch']:raise RuntimeError('Reporting reserve reached while waiting')
        time.sleep(15)
    assert read(timing/'timing_complete.json')['passed']
    ident=read(timing/'launch.json')
    while psutil.pid_exists(ident['pid']):
        p=psutil.Process(ident['pid'])
        if abs(p.create_time()-ident['created'])>.01:break
        time.sleep(2)
    spec=importlib.util.spec_from_file_location('c18_base',reg['base_controller']['path']);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    plugin=install_plugin(base)(a.registration);stop=threading.Event()
    def loop():
        while not stop.wait(1800):monitor(root)
    threading.Thread(target=loop,daemon=True).start();began=time.monotonic()
    try:
        manager=base.RoundManager('stock','kalcb_research_decision_labels',base_dir=root/'framework')
        runner=base.PhaseRunner(plugin,root/'framework/labels',round_name='C18 labels; no optimisation',max_rounds=1,min_delta=0,
            max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=1)
        runner.run_all_phases()
        write(root/'labels_complete.json',dict(passed=True,seconds=time.monotonic()-began,research_decision_complete=False,eligible_for_scoring=False))
    except Exception as e:
        write(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:stop.set();monitor(root)

if __name__=='__main__':main()
