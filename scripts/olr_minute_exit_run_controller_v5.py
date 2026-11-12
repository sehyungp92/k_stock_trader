"""Wait for authenticated engineering; freeze and run the bounded experiment."""
from datetime import datetime,timezone
import os,subprocess,sys,tempfile,time,traceback
from scripts import olr_minute_exit_study_v5 as t
from scripts.olr_minute_exit_prepare_controller_v5 import verify_process
n=t.n


def run():
    temp_root=t.ROOT/'runtime_tmp';temp_root.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(temp_root);tempfile.tempdir=str(temp_root)
    bound=n.read(t.ROOT/'qualification_controller_harness.json')
    identity=n.read(t.ROOT/'engineering_process.json')
    for name,digest in bound['files'].items():assert n.sha(t.REPO/name)==digest,name
    assert n.sha(t.ROOT/'engineering_process.json')==bound['engineering_identity_sha256']
    while not (t.ROOT/'engineering_complete.json').exists():
        if (t.ROOT/'engineering_failure.json').exists():raise RuntimeError('Engineering failed; inspect and preserve before any restart')
        if not verify_process(identity):raise RuntimeError('Exact engineering process exited without completion')
        time.sleep(20)
    for phase,module in [('freeze','scripts.olr_minute_exit_freeze_v5'),('qualification','scripts.olr_minute_exit_qualify_v5')]:
        for name,digest in bound['files'].items():assert n.sha(t.REPO/name)==digest,name
        command=[sys.executable,'-B','-u','-m',module]
        with (t.ROOT/(phase+'.controller.log')).open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=t.REPO,stdout=stream,stderr=subprocess.STDOUT)
            pid=process.pid
            query="Get-CimInstance Win32_Process -Filter 'ProcessId="+str(pid)+"' | Select-Object ProcessId,CommandLine,@{n='creation_time_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
            import json
            actual=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],text=True))
            n.write(t.ROOT/(phase+'_process.json'),dict(pid=pid,command_line=actual['CommandLine'],creation_time_utc=actual['creation_time_utc'],
                module=module,launch_utc=datetime.now(timezone.utc).isoformat()))
            print(dict(phase=phase,pid=pid,status='running'),flush=True)
            rc=process.wait()
        if rc:raise RuntimeError('Bounded '+phase+' failed; preserve current attempt and inspect its log')
        print(dict(phase=phase,status='complete'),flush=True)
    n.write(t.ROOT/'run_controller_complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        qualification_complete_sha256=n.sha(t.ROOT/'qualification_complete.json'),next='Reviewed combined report and independent delivery verification remain required.'))


if __name__=='__main__':
    try:run()
    except Exception:
        n.write(t.ROOT/'run_controller_failure.json',dict(utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
