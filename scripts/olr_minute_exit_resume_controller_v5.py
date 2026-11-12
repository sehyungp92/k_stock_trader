"""Resume preserved preflight failure with exact successful source/engine evidence."""
from datetime import datetime,timezone
from pathlib import Path
import os,subprocess,sys,tempfile,traceback
from scripts import olr_minute_exit_study_v5 as t
n=t.n


def run():
    temp_root=t.ROOT/'runtime_tmp';temp_root.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(temp_root);tempfile.tempdir=str(temp_root)
    if (t.ROOT/'cases').exists():raise RuntimeError('Cannot repair engineering after outcomes')
    expected=n.read(t.ROOT/'engineering_controller_harness.json')
    for name,digest in expected['files'].items():assert n.sha(t.REPO/name)==digest,name
    repair=n.read(t.ROOT/'entry_identity_repair.json')
    assert repair['no_performance_consumed'] and repair['native_engine_and_sources_unchanged']
    resume=t.ROOT/'temporary_storage_repair.json'
    if resume.exists():
        recovery=n.read(resume)
        assert recovery['entry_identity_repair_sha256']==n.sha(t.ROOT/'entry_identity_repair.json')
        reuse=recovery['reuse_files']
    else:reuse=repair['reuse_files']
    for name,digest in reuse.items():assert n.sha(name)==digest,name
    print('Authenticating completed native generations and IS source audit for reuse.',flush=True)
    for w in ('train','oos'):
        ready=n.read(t.ENGINES/w/'ready.json')
        assert n.source_hash(Path(ready['files']))==ready['study']['source_generation_hash']
        for name,digest in ready['manifest']['files'].items():assert n.sha(Path(ready['files'])/name)==digest,name
    tests=Path(repair['engine_test_receipt']);receipt=n.read(tests)
    assert receipt['passed']>=65 and receipt['exit_code']==receipt['failures']==receipt['errors']==receipt['skipped']==0
    for name,digest in receipt['source_hashes'].items():assert n.sha(t.REPO/name)==digest,name
    assert n.sha(tests.parent/'tests.log')==receipt['log_sha256'] and n.sha(tests.parent/'tests.xml')==receipt['junit_sha256']
    print('Completed source audit, native inputs and 65 engine tests authenticated; resuming entry parity.',flush=True)
    for phase,command in [
        ('entry_identity_tests',[sys.executable,'-B','-m','pytest','tests/strategy_olr/test_minute_entry_identity_v5.py','--noconftest','-q']),
        ('preflight_native',[sys.executable,'-B','-u','-m','scripts.olr_minute_exit_preflight_v5'])]:
        for name,digest in expected['files'].items():assert n.sha(t.REPO/name)==digest,name
        with (t.ROOT/(phase+'.controller.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=t.REPO,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Resumed engineering failed: '+phase)
        print(dict(phase=phase,status='complete'),flush=True)
    n.write(t.ROOT/'engineering_complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        preflight_sha256=n.sha(t.ROOT/'preflight_complete.json'),harness_sha256=n.sha(t.ROOT/'engineering_controller_harness.json'),
        no_performance_executed=True,authenticated_reuse_sha256=n.sha(t.ROOT/'entry_identity_repair.json')))


if __name__=='__main__':
    try:run()
    except Exception:
        n.write(t.ROOT/'engineering_failure.json',dict(utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
