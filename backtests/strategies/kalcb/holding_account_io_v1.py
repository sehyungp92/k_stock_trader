"""Publish explicit research services over the authenticated first-hour parent."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from .retention_research_io_v1 import read,write,bound,digest,verify,source_hash
from .late_entry_io_v5 import cached_link

OVERLAYS=(
    'backtests/strategies/kalcb/account_fork_v2.py',
    'backtests/strategies/kalcb/holding_action_v1.py',
    'backtests/strategies/kalcb/holding_account_replay_v1.py',
    'backtests/strategies/kalcb/holding_account_model_v1.py',
    'scripts/kalcb_holding_account_worker_v1.py',
)


def freeze(root,period,old):
    root=Path(root);p=read(root/'protocol.json');folder=root/'runtime'/period
    if (folder/'profile.json').exists():
        item=read(folder/'profile.json');verify([item[k] for k in ('manifest','plan','policy')]);return item
    verify([old[k] for k in ('manifest','plan','policy')])
    mp=Path(old['manifest']['path']);parent=read(mp);sig=parent.pop('manifest_hash')
    if stable_signature(parent)!=sig:raise ValueError('Changed authenticated parent generation')
    folder.mkdir(parents=True,exist_ok=False)
    plan=read(old['plan']['path']);cfg=read(plan['config']);context=read(plan['audit_configuration'])
    write(folder/'config.json',cfg)
    context.update(output_root=str(folder),promotion_status='holding_account_research_only')
    write(folder/'audit_configuration.json',context)
    for k in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[k],None);plan[k]=str((folder/(k+'.json')).resolve())
        plan['input_sha256'][plan[k]]=digest(plan[k])
    plan['holding_account_research']={'protocol':bound(root/'protocol.json'),'parent':old,'period':period,
        'unchanged_entry_policy':True,'unchanged_management_reference':True}
    write(folder/'plan.json',plan)
    code=folder/'inputs/files'
    replacements={rel:Path(b['path']) for rel,b in p['overlays'].items()}
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/holding_account/protocol.json':root/'protocol.json',
        'research/holding_account/config.json':folder/'config.json',
        'research/holding_account/audit_configuration.json':folder/'audit_configuration.json'})
    files={}
    for rel,sha in parent['files'].items():
        if rel not in replacements:
            cached_link(mp.parent/'files'/rel,code/rel,sha,root/'content_cache');files[rel]=sha
    for rel,path in replacements.items():
        sha=digest(path);cached_link(path,code/rel,sha,root/'content_cache');files[rel]=sha
    payload=deepcopy(parent);payload['files']=files;payload['source_generation_hash']=source_hash(code)
    payload['evidence']['holding_account_research']={'parent_manifest':old['manifest'],
        'protocol':bound(root/'protocol.json'),'services':p['overlays']}
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy)!=old['policy']['sha256']:raise ValueError('Changed causal parent policy')
    item={'name':'control','period':period,'folder':str(folder),'source_generation_hash':payload['source_generation_hash'],
        'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(folder/'profile.json',item);return item
