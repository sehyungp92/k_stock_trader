"""Low-overhead validation telemetry; observes only, changes no process."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

v=Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10/revisions/revision_19_native_overheads')
parents={}
with (v/'validation_resource_samples.jsonl').open('a',encoding='utf-8') as f:
    deadline=time.monotonic()+4*3600
    while time.monotonic()<deadline:
        active=[]
        for p in psutil.process_iter(['pid','name','cmdline','create_time','ppid']):
            args=p.info['cmdline'] or []
            if p.info['name']=='python.exe' and any('portfolio_synergy_study.py' in x for x in args) and any('revision_19_native_overheads' in x for x in args):
                try:
                    cpu=p.cpu_times();mem=p.memory_info()
                    active.append({'pid':p.pid,'created':p.create_time(),'cpu_seconds':cpu.user+cpu.system,
                        'rss':mem.rss,'private':mem.private,'peak_wset':mem.peak_wset,'args':args})
                    parent=p.parent()
                    if parent:parents[parent.pid]=parent.create_time()
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        row={'utc':datetime.now(timezone.utc).isoformat(),'host_cpu_percent':psutil.cpu_percent(),
             'available_mb':psutil.virtual_memory().available/2**20,'native':active}
        f.write(json.dumps(row)+'\n');f.flush()
        if parents and not active:
            live=False
            for pid,created in parents.items():
                try:live |= psutil.Process(pid).create_time()==created
                except psutil.NoSuchProcess:pass
            if not live:break
        time.sleep(5)
