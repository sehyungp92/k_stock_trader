"""One-shot dependency launcher for frozen C18 diagnostics, not an optimiser.

All financial dispatch remains in the existing registered shared PhaseRunners.
This launcher never creates candidates, chooses policies or changes a budget.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,sys,os,time,traceback
import psutil
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path('C:/Users/sehyu/Documents/Other/Projects/k_stock_trader')
OUT=ROOT/'stage_chain_v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def verify(b):assert bound(b['path'])['sha256']==b['sha256']
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def alive(launch):
    if not launch.exists():return False
    d=read(launch)
    try:p=psutil.Process(d['pid']);return abs(p.create_time()-d['created'])<.01 and p.cmdline()==d['argv']
    except psutil.NoSuchProcess:return False
def wait_complete(directory,name,deadline):
    while not (directory/name).exists() or alive(directory/'launch.json'):
        if (directory/'failure.json').exists():raise RuntimeError('Prerequisite failed: '+str(directory))
        if time.time()>deadline:raise RuntimeError('Chosen dependency deadline reached')
        time.sleep(15)
    assert read(directory/name)['passed']
def command(stage,args,deadline,*,check=True):
    env={**os.environ,**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_IO_THREADS']},'PYTHONDONTWRITEBYTECODE':'1'}
    argv=[sys.executable,'-B','-u',*map(str,args)];start=time.monotonic();peak=0;guard=None
    with (OUT/(stage+'.log')).open('x',encoding='utf-8') as log:
        child=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));p=psutil.Process(child.pid)
        save(OUT/(stage+'_launch.json'),dict(pid=p.pid,created=p.create_time(),argv=argv))
        while child.poll() is None:
            try:peak=max(peak,p.memory_info().rss)
            except psutil.NoSuchProcess:break
            if time.time()>deadline or peak>2*1024**3:
                guard='chosen deadline or2GiB';child.terminate();break
            time.sleep(2)
        rc=child.wait()
    save(OUT/(stage+'_receipt.json'),dict(exit_code=rc,guard=guard,seconds=time.monotonic()-start,peak_rss=peak,passed=not rc and not guard))
    if check and (rc or guard):raise RuntimeError('Stage failed: '+stage)
    return not rc and not guard
def main():
    assert OUT.exists();reg=read(OUT/'registration.json')
    for b in reg['programs']:verify(b)
    p=psutil.Process();save(OUT/'launch.json',dict(pid=p.pid,created=p.create_time(),argv=p.cmdline(),registration=bound(OUT/'registration.json')))
    deadline=reg['financial_deadline_epoch'];report_deadline=reg['report_deadline_epoch']
    try:
        wait_complete(ROOT/'timing_run_v2','timing_complete.json',deadline)
        completed=read(ROOT/'timing_analysis_v1/completion.json');assert completed['passed']
        for b in completed['files'].values():verify(b)
        save(OUT/'timing_analysis_reuse.json',dict(passed=True,completion=bound(ROOT/'timing_analysis_v1/completion.json'),native_evaluations=0))
        wait_complete(ROOT/'labels_run_v6','labels_complete.json',deadline)
        command('bind_counterfactuals',[REPO/'scripts/kalcb_research_decision_register_counterfactual_v1.py','--labels',ROOT/'labels_run_v6'],deadline)
        command('oracle_framework_unit',[REPO/'scripts/kalcb_research_decision_oracle_framework_unit_v1.py'],deadline)
        oracle=ROOT/'oracle_run_v1';oreg=read(oracle/'registration.json')
        checks=[oracle/'framework_preflight/receipt.json',ROOT/'oracle_unit_v1/receipt.json',ROOT/'horizon_unit_v3/receipt.json',ROOT/'information_unit_v3/receipt.json']
        for path in checks:assert read(path)['passed']
        for b in oreg['programs']+oreg['bindings']:verify(b)
        assert not (oracle/'launch.json').exists()
        assert not alive(ROOT/'labels_run_v6/launch.json') and not alive(ROOT/'timing_run_v2/launch.json')
        save(oracle/'launch_qualification.json',dict(passed=True,registration=bound(oracle/'registration.json'),checks=[bound(x) for x in checks],
            native_workers_max=2,off_bridge_and_per_arm_native_reach_required_before_full=True,financial_deadline_epoch=deadline))
        # Run the shared oracle controller in a separate child. Its own receipts,
        # worker guards, phase gates and persistent monitor own financial work.
        env={**os.environ,**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_IO_THREADS']},'PYTHONDONTWRITEBYTECODE':'1'}
        args=[sys.executable,'-B','-u',oreg['controller_program']['path'],'--registration',str(oracle/'registration.json')]
        with (OUT/'oracle_controller.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));proc=psutil.Process(child.pid)
            save(OUT/'oracle_controller_launch.json',dict(pid=proc.pid,created=proc.create_time(),argv=args))
            try:
                info=ROOT/'information_v1';ir=read(info/'registration.json');verify(ir['program'])
                command('information',[ir['program']['path'],'--registration',info/'registration.json'],min(report_deadline,time.time()+3600),check=False)
                command('finer_paths',[REPO/'scripts/kalcb_research_decision_finer_paths_v2.py','--labels',ROOT/'labels_run_v6'],min(report_deadline,time.time()+3600),check=False)
            finally:
                # Do not interrupt an independently qualified healthy oracle if
                # a reporting diagnostic fails. Preserve both outcomes.
                while child.poll() is None:time.sleep(15)
            save(OUT/'oracle_controller_receipt.json',dict(exit_code=child.returncode,passed=child.returncode==0))
        if child.returncode==0:
            command('oracle_analysis',[REPO/'scripts/kalcb_research_decision_oracle_analysis_v1.py','--registration',oracle/'registration.json'],report_deadline)
        save(OUT/'pipeline_complete.json',dict(passed=True,oracle_passed=child.returncode==0,independent_review_and_documentation_pending=True))
    except Exception as e:
        save(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
