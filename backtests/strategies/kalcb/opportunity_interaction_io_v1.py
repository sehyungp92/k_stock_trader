"""Explicit C12 cap configuration over the authenticated native research source."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from .retention_research_io_v1 import read,write,bound,digest,verify,source_hash
from .late_entry_io_v5 import cached_link


def configuration(parent,cap):
    if cap not in (0,2) or parent.get('risk_max_gross_equity_pct',0)!=0:
        raise ValueError('Unexpected parent or unregistered cap')
    result=deepcopy(parent);result['risk_max_gross_equity_pct']=float(cap)
    return result


def freeze(root,period,cap,old):
    root=Path(root);p=read(root/'protocol.json');folder=root/'runtime'/f'{period}_cap{cap}'
    if folder.exists():raise ValueError('Preserve frozen generation')
    verify([old[k] for k in ('manifest','plan','policy')])
    mp=Path(old['manifest']['path']);parent=read(mp);sig=parent.pop('manifest_hash')
    if stable_signature(parent)!=sig:raise ValueError('Changed parent manifest')
    folder.mkdir(parents=True,exist_ok=False)
    plan=read(old['plan']['path']);cfg=configuration(read(plan['config']),cap)
    write(folder/'config.json',cfg);context=read(plan['audit_configuration'])
    context.update(output_root=str(folder),promotion_status='c12_diagnostic_only_no_promotion')
    write(folder/'audit_configuration.json',context)
    for k in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[k],None);plan[k]=str((folder/(k+'.json')).resolve())
        plan['input_sha256'][plan[k]]=digest(plan[k])
    plan['holding_account_research']={'protocol':bound(root/'protocol.json'),'parent':old,'period':period,
        'unchanged_entry_policy':False,'unchanged_management_reference':True,
        'declared_config_change':{'risk_max_gross_equity_pct':cap},'scope':'Opening pending cap only; unchanged frozen VETO map and native lifecycle'}
    write(folder/'plan.json',plan)
    code=folder/'inputs/files';replacements={rel:Path(b['path']) for rel,b in p['overlays'].items()}
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/holding_account/protocol.json':root/'protocol.json','research/holding_account/config.json':folder/'config.json',
        'research/holding_account/audit_configuration.json':folder/'audit_configuration.json'})
    files={}
    for rel,sha in parent['files'].items():
        if rel not in replacements:
            cached_link(mp.parent/'files'/rel,code/rel,sha,root/'content_cache');files[rel]=sha
    for rel,path in replacements.items():
        sha=digest(path);cached_link(path,code/rel,sha,root/'content_cache');files[rel]=sha
    payload=deepcopy(parent);payload['files']=files;payload['source_generation_hash']=source_hash(code)
    payload['evidence']['c12_interaction']={'parent_manifest':old['manifest'],'protocol':bound(root/'protocol.json'),
        'config_change':{'risk_max_gross_equity_pct':cap},'services':p['overlays']}
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy)!=old['policy']['sha256']:raise ValueError('Changed causal policy')
    item={'name':f'cap{cap}','period':period,'cap':cap,'folder':str(folder),'source_generation_hash':payload['source_generation_hash'],
        'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(folder/'profile.json',item);return item
