"""Continue source preparation into native engineering checks, never performance."""
from datetime import datetime,timezone
from pathlib import Path
import json,subprocess,sys,time,traceback
from scripts import olr_minute_exit_study_v5 as t
n=t.n


def verify_process(identity):
    pid=int(identity['pid'])
    command="Get-CimInstance Win32_Process -Filter 'ProcessId="+str(pid)+"' | Select-Object CommandLine,@{n='utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
    value=subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],text=True).strip()
    if not value:return False
    process=json.loads(value)
    if (process['CommandLine']!=identity['command_line'] or
            datetime.fromisoformat(process['utc'].replace('Z','+00:00'))!=datetime.fromisoformat(identity['creation_time_utc'].replace('Z','+00:00'))):
        raise ValueError('Source preparation process identity changed')
    return True


def run():
    identity=n.read(t.ROOT/'source_preparation_process.json')
    expected=n.read(t.ROOT/'engineering_controller_harness.json')
    for name,digest in expected['files'].items():assert n.sha(t.REPO/name)==digest,name
    if (t.ROOT/'cases').exists():raise RuntimeError('Engineering controller forbidden after performance')
    while not (t.ROOT/'source_preparation_complete.json').exists():
        if (t.ROOT/'source_preparation_failure.json').exists():raise RuntimeError('Source preparation failed; preserve and inspect exact cause')
        if not verify_process(identity):raise RuntimeError('Source process exited without authenticated completion')
        time.sleep(20)
    print('Authenticated source completion observed; checking current engineering.',flush=True)
    commands=[('focused_tests',[sys.executable,'-B','-u','-m','scripts.olr_minute_exit_test_v5']),
        ('prepare_native',[sys.executable,'-B','-u','-m','scripts.olr_minute_exit_prepare_v5']),
        ('preflight_native',[sys.executable,'-B','-u','-m','scripts.olr_minute_exit_preflight_v5'])]
    for phase,command in commands:
        for name,digest in expected['files'].items():assert n.sha(t.REPO/name)==digest,name
        print(json.dumps(dict(phase=phase,status='running')),flush=True)
        log=t.ROOT/(phase+'.controller.log')
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=t.REPO,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Engineering phase failed: '+str(log))
        print(json.dumps(dict(phase=phase,status='complete')),flush=True)
    n.write(t.ROOT/'engineering_complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        preflight_sha256=n.sha(t.ROOT/'preflight_complete.json'),harness_sha256=n.sha(t.ROOT/'engineering_controller_harness.json'),
        no_performance_executed=True,next='Freeze final qualification harness then only the registered12 native comparisons.'))


if __name__=='__main__':
    try:run()
    except Exception:
        n.write(t.ROOT/'engineering_failure.json',dict(utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
