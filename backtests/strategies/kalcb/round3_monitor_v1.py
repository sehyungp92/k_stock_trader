"""Read-only 30-minute progress/prediction check, scoped to this C17 attempt."""
from datetime import datetime,timezone
from pathlib import Path
import json,os
import psutil
from .round3_native_io_v1 import read,write,bound

def check(registration):
    registration=Path(registration);data=read(registration);out=registration.parent;canonical=Path(data['repository'])/'data/backtests/output/kalcb/round_3'
    active=[]
    for p in psutil.process_iter(['pid','ppid','cmdline','create_time']):
        try:
            args=p.info['cmdline'] or [];cmd=' '.join(args)
            if 'kalcb_round3_native_worker_v3' in cmd and str(out).replace('\\','/').lower() in cmd.replace('\\','/').lower():active.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    errors=[]
    if len(active)>2:errors.append('More than two own native workers')
    for b in data['programs']:
        if bound(b['path'])['sha256']!=b['sha256']:errors.append('Frozen program changed '+b['path'])
    evaluations={};contradictions=[]
    for path in (out/'evaluations').glob('*.json'):
        if path.name.endswith('.seal.json'):continue
        value=read(path);evaluations[path.stem]=dict(native_valid=value.get('native_valid'),qualification=value.get('qualification'))
        q=value.get('qualification',{})
        if q.get('gain_pp',0)<0:contradictions.append(dict(arm=path.stem,kind='negative_funded_increment',gain_pp=q['gain_pp']))
        if 'qualification:mechanism' in q.get('failures',[]):contradictions.append(dict(arm=path.stem,kind='predicted_mechanism_not_verified'))
    failed=list((out/'accounts').rglob('*_rejected.json')) if (out/'accounts').exists() else []
    if (out/'failure.json').exists():errors.append(read(out/'failure.json')['error'])
    launch=read(out/'launch.json') if (out/'launch.json').exists() else {};alive=False
    if launch:
        try:alive=abs(psutil.Process(launch['pid']).create_time()-launch['created'])<.01
        except psutil.NoSuchProcess:pass
    result=dict(at_utc=datetime.now(timezone.utc).isoformat(),controller_alive=alive,native_workers=active,
        shared_progress=read(canonical/'progress.json') if (canonical/'progress.json').exists() else None,
        phase_state=bound(canonical/'phase_state.json') if (canonical/'phase_state.json').exists() else None,
        accepted_accounts=len(list((out/'accounts').rglob('*_acceptance.json'))) if (out/'accounts').exists() else 0,
        invalid_receipts=[str(p) for p in failed],evaluations=evaluations,prediction_contradictions=contradictions,errors=errors,
        process_completed=(out/'closeout.json').exists(),documentation_completed=(out/'record_closeout.json').exists(),
        policy='Observe only: no retuning, retries, extra arms, holdout relabelling or stopped-study restart.')
    target=out/'monitor_checks'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');write(target,result)
    return result
