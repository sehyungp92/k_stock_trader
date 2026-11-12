"""Immutable observer-only publication; preserve all parent economic bytes."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,source_hash
from backtests.strategies.kalcb.late_entry_io_v5 import cached_link


def freeze(root,name,period,old):
    p=read(root/'protocol.json');folder=Path(p['runtime_root'])/'cases'/name/period
    receipt=root/'cases'/name/period/'input_profile.json'
    if receipt.exists():
        item=read(receipt);verify([item[k] for k in ('manifest','plan','policy')]);return item
    verify([old[k] for k in ('manifest','plan','policy')]+list(p['overrides'].values()))
    pp=Path(old['manifest']['path']);parent=read(pp);identity=parent.pop('manifest_hash')
    if stable_signature(parent)!=identity:raise ValueError('Changed parent manifest')
    folder.mkdir(parents=True,exist_ok=False)
    plan=read(old['plan']['path']);old_config=Path(plan['config'])
    (folder/'config.json').write_bytes(old_config.read_bytes())
    context=read(plan['audit_configuration']);context.update(output_root=str(folder),promotion_status='unchanged_parent_observation_only')
    write(folder/'audit_configuration.json',context)
    for key in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[key],None);plan[key]=str((folder/(key+'.json')).resolve());plan['input_sha256'][plan[key]]=digest(plan[key])
    plan['reclaim_observation']={'protocol':bound(root/'protocol.json'),'parent':old,'economics_unchanged':True}
    write(folder/'plan.json',plan)
    replacements={rel:Path(b['path']) for rel,b in p['overrides'].items()}
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/reclaim_observer/protocol.json':root/'protocol.json',
        'research/reclaim_observer/config.json':folder/'config.json',
        'research/reclaim_observer/audit_configuration.json':folder/'audit_configuration.json',
        'research/reclaim_observer/opportunities.json':root/'opportunities.json'})
    files={};cache=Path(p['runtime_root'])/'content_cache'
    for rel,sha in parent['files'].items():
        if rel in replacements:continue
        cached_link(pp.parent/'files'/rel,folder/'inputs/files'/rel,sha,cache);files[rel]=sha
    for rel,src in replacements.items():
        sha=digest(src);cached_link(src,folder/'inputs/files'/rel,sha,cache);files[rel]=sha
    payload=deepcopy(parent);payload['files']=files
    payload['evidence']['reclaim_observation']={'protocol':bound(root/'protocol.json'),
        'parent_manifest':bound(pp),'config_unchanged':bound(old_config),'policy_unchanged':old['policy'],
        'scope':'Original parent source obligations plus additive raw minute90 observer evidence; no new portfolio action.'}
    code=folder/'inputs/files';actual=source_hash(code);payload['source_generation_hash']=actual
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy)!=old['policy']['sha256'] or digest(folder/'config.json')!=digest(old_config):raise ValueError('Changed parent economics')
    item={'name':name,'period':period,'folder':str(folder),'source_generation_hash':actual,
        'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(receipt,item);return item
