"""C15 R1 single-producer execution with process and system resource evidence.

No optimization phases, search, scoring, retries, or promotion ownership.
"""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import json
import subprocess
import time
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,env


def exact_process(identity):
    import psutil
    try:
        p=psutil.Process(identity['pid'])
        return abs(p.create_time()-identity['process_created'])<.01 and p.cmdline()==identity['command']
    except psutil.NoSuchProcess:return False


def completed_dependency(root,identity,count):
    root=Path(root)
    status=read(root/'status.json')
    if status.get('stage')=='attention_required':raise RuntimeError('Dependency requires preserved versioned repair')
    if exact_process(identity):return False
    if not (root/'stage_completion.json').exists():raise RuntimeError('Dependency exited without completion')
    done=read(root/'stage_completion.json');rows=read(root/'progress.json');recipes=read(root/'protocol.json')['recipes']
    expected={(n,p,s) for n in recipes for p in ('is','oos') for s in ('baseline','combined')}
    actual=[(r['name'],r['period'],r['scenario']) for r in rows]
    if not done.get('complete') or done.get('native_cases')!=count or len(actual)!=count or set(actual)!=expected:
        raise ValueError('Incomplete or duplicate dependency cases')
    if sum(r['is_identity'] for r in rows)!=8 or any(r['metrics']['audit_status']!='passed' for r in rows):
        raise ValueError('Missing audited identity controls')
    for n,p,s in expected:
        recipe=recipes[n]
        if recipe.get('quantity',1)==0:
            if not read(root/'cases'/n/p/s/'identity_bridge.json')['passed']:raise ValueError('Failed parent bridge')
    return True


def commit_available():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('Missing Windows commit capacity')
    return m.avail_page


def execute(root,command,label,cwd,state):
    import psutil
    root=Path(root)
    while psutil.virtual_memory().available<384*1024**2 or commit_available()<1280*1024**2:
        write(root/'status.json',{**state,'stage':'waiting_measured_headroom','next':label,'child':None});time.sleep(15)
    log=root/(label+'.log')
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=cwd,env=env(cwd),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        identity=dict(pid=child.pid,process_created=psutil.Process(child.pid).create_time(),command=command)
        if not exact_process(identity):raise RuntimeError('Child identity mismatch')
        write(root/'status.json',{**state,'stage':label,'child':identity,'log':str(log),'updated_at_utc':datetime.now(timezone.utc).isoformat()})
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid)
                with (root/'resource_observations.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),stage=label,identity=identity,
                        memory=p.memory_info()._asdict(),cpu=p.cpu_times()._asdict(),io=p.io_counters()._asdict(),
                        system_physical_available=psutil.virtual_memory().available,system_commit_available=commit_available(),
                        disk_free=psutil.disk_usage(str(root)).free,
                        descendants=[dict(pid=c.pid,created=c.create_time(),memory=c.memory_info()._asdict()) for c in p.children(recursive=True)]))+'\n')
            except psutil.NoSuchProcess:pass
            time.sleep(5)
    if child.returncode:
        write(root/'failed_stage.json',dict(stage=label,returncode=child.returncode,log=bound(log),preserve_evidence=True))
        raise RuntimeError('Registered native case failed: '+label)
    return bound(log)
