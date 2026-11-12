"""Preserve terminal invalid trial receipts without scoring or retrying them."""
from pathlib import Path
from .round3_native_io_v1 import execute as native_execute,read,write,bound
from .round3_contract_v1 import verify_acceptance

def execute(item,scenario,folder,**options):
    folder=Path(folder);rejected=folder.parent/(folder.name+'_rejected.json')
    if rejected.exists():
        value=read(rejected)
        if value['item']!=item or value['scenario']!=scenario or value['options']!=options:raise ValueError('Invalid-trial cache key changed')
        for b in value['failure_bindings']:
            if bound(b['path'])['sha256']!=b['sha256']:raise ValueError('Invalid-trial evidence changed')
        return value
    try:value=native_execute(item,scenario,folder,**options)
    except RuntimeError as exc:
        process=folder.parent/(folder.name+'_process.json')
        if not process.exists():raise
        value=dict(passed=False,eligible_for_scoring=False,folder=str(folder),process=read(process),error=repr(exc),
            disposition='invalid_or_interrupted_execution_not_economic_rejection')
    if value.get('passed'):return value
    paths=[folder.with_suffix('.log'),folder.parent/(folder.name+'_process.json'),folder/'qualification_failure.json']
    value.update(item=item,scenario=scenario,options=options,failure_bindings=[bound(p) for p in paths if p.exists()])
    write(rejected,value);return value
