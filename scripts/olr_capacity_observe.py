"""Capture native execution evidence without changing any decisions or fills."""
import argparse
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
import gzip
import json
import os
from pathlib import Path
import runpy
import sys

p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
p.add_argument('--recipe',required=True);p.add_argument('--scenario',required=True);p.add_argument('--full-audit',action='store_true')
p.add_argument('--no-upward-compounding',action='store_true')
a=p.parse_args();study=json.loads(Path(a.study).read_text());code=Path(study['code']);output=Path(a.output)
sys.path.insert(0,str(code));os.chdir(code)
import backtests.engine.integrity as integrity
import backtests.engine.replay as replay
native_drive=replay.drive_timestamp_batch;native_audit=integrity.audit_broker
events=[]
def serial(value):
    if is_dataclass(value):return {f.name:serial(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value
def observe_drive(*args,**kwargs):
    for event in native_drive(*args,**kwargs):
        events.append(serial(event));yield event
def observe_audit(broker,*args,**kwargs):
    for name,rows in [('journal',broker.journal),('fills',broker.fills),('decisions',events)]:
        with gzip.open(output/(name+'.jsonl.gz'),'xt',encoding='utf-8',compresslevel=1) as stream:
            for row in rows:stream.write(json.dumps(serial(row),default=str)+'\n')
    return native_audit(broker,*args,**kwargs)
replay.drive_timestamp_batch=observe_drive;integrity.audit_broker=observe_audit
sys.argv=[str(code/'scripts/portfolio_synergy_study.py'),'replay','--study',a.study,
    '--recipe',a.recipe,'--scenario',a.scenario,'--standalone','OLR','--output',a.output]
if a.full_audit:sys.argv+=['--full-audit']
if a.no_upward_compounding:sys.argv+=['--no-upward-compounding']
runpy.run_path(str(code/'scripts/portfolio_synergy_study.py'),run_name='__main__')
