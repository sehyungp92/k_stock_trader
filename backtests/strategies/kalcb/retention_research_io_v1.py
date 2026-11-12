"""Immutable packaging and account validation for retention research cases."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from backtests.auto.shared.cache_keys import stable_signature

PERIODS={'is':('2025-08-07','2026-03-31',156),'oos':('2026-04-01','2026-07-10',69)}
POLICY='interaction_cost_margin'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bound(path):return {'path':str(Path(path).resolve()),'sha256':digest(path)}
def verify(bindings):
    for b in bindings:
        if digest(b['path'])!=b['sha256']:raise ValueError('Changed binding '+b['path'])
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
def link(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)
def env(code):
    return {**os.environ,'PYTHONPATH':str(code),'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8',
            **{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS')}}
def source_hash(code):
    return subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'],cwd=code,env=env(code),text=True).strip()


def freeze_case(out,name,period,recipe,old,authenticated):
    folder=out/'cases'/name/period
    receipt=folder/'input_profile.json'
    if receipt.exists():
        item=read(receipt);verify([item[k] for k in ('manifest','plan','policy')]);return item
    verify([old[k] for k in ('manifest','plan','policy')])
    pp=Path(old['manifest']['path']);parent=read(pp);identity=parent.pop('manifest_hash')
    if stable_signature(parent)!=identity:raise ValueError('Changed parent manifest')
    folder.mkdir(parents=True,exist_ok=False)
    plan=read(old['plan']['path']);config=read(plan['config']);context=read(plan['audit_configuration'])
    changes=recipe['config']
    if not set(changes)<={'exit_profit_retention_start_atr','exit_profit_retention_confirmation'}:
        raise ValueError('Unregistered retention dimension')
    config.update(changes)
    for key,value in changes.items():
        if config[key]!=value:raise ValueError('Ineffective config application')
    old_paths=[plan['config'],plan['audit_configuration']]
    write(folder/'config.json',config)
    context.update(output_root=str(folder),promotion_status='retention_ablation_research_only')
    write(folder/'audit_configuration.json',context)
    for path in old_paths:plan['input_sha256'].pop(path,None)
    for key in ('config','audit_configuration'):
        plan[key]=str(folder/(key+'.json'));plan['input_sha256'][plan[key]]=digest(plan[key])
    plan['retention_research']={'recipe':name,'period':period,'protocol':bound(out/'protocol.json'),
                              'frozen_parent':recipe['parent'],'management_only':True}
    write(folder/'plan.json',plan)
    replacements={'research/kalcb_executed_value/plan.json':folder/'plan.json',
                  'research/retention_research/protocol.json':out/'protocol.json',
                  'research/retention_research/config.json':folder/'config.json',
                  'research/retention_research/audit_configuration.json':folder/'audit_configuration.json'}
    files={}
    for relative,sha in parent['files'].items():
        if relative in replacements:continue
        source=pp.parent/'files'/relative;stat=source.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,sha)
        if key not in authenticated:
            if digest(source)!=sha:raise ValueError('Changed frozen file '+relative)
            authenticated.add(key)
        link(source,folder/'inputs/files'/relative);files[relative]=sha
    for relative,source in replacements.items():
        link(source,folder/'inputs/files'/relative);files[relative]=digest(source)
    payload=deepcopy(parent);payload['files']=files
    payload['evidence']['retention_research']={'parent_manifest':bound(pp),'registration':bound(out/'protocol.json'),
                                             'config_changes':changes,'original_policy':old['policy']}
    code=folder/'inputs/files';actual_source=source_hash(code)
    if actual_source!=old['source_generation_hash']:raise ValueError('Unexpected economic source change')
    payload['source_generation_hash']=actual_source
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy)!=old['policy']['sha256']:raise ValueError('Changed parent selection')
    item={'name':name,'period':period,'folder':str(folder),'source_generation_hash':actual_source,
          'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(receipt,item)
    return item


def validate_native(folder,item,scenario,save=True):
    from backtests.strategies.kalcb.qualified_artifacts import validate_case
    folder=Path(folder);result=read(folder/'result.json');request=read(folder/'request.json')
    if request['plan']!=read(item['plan']['path']):raise ValueError('Replay plan mismatch')
    if Path(request['arguments']['audit_manifest']).resolve()!=Path(item['manifest']['path']).resolve():
        raise ValueError('Source manifest mismatch')
    # Match exact frozen policy bytes, whether the historical request used an
    # equivalent external export or the manifest-local directory.
    if digest(Path(request['arguments']['policies'])/'policies_manifest.json')!=item['policy']['sha256']:
        raise ValueError('Policy identity mismatch')
    start,end,sessions=PERIODS[item['period']]
    checked=validate_case(folder,{'execution_source':{'source_generation_hash':item['source_generation_hash']},
        'audit_manifest':item['manifest']['path'],'policy':{'name':POLICY,'manifest_sha256':item['policy']['sha256']},
        'initial_equity_krw':100000000,'date_range':{'start':start,'end':end,'sessions':sessions},'metrics':{scenario:result}},scenario)
    if checked['daily'][0]['day']!=start or checked['daily'][-1]['day']!=end:raise ValueError('Calendar truncated')
    if save:write(folder/'independent_account_verification.json',checked['validation'])
    return {k:result[k] for k in ('return_pct','drawdown_pct','trades','active_days','audit_status')}
