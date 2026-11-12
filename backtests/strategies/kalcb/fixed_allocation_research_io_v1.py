"""Publish a new, authenticated economic source for funded allocation comparisons."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.retention_research_io_v1 import (
    read, write, bound, digest, verify, link, source_hash)

OVERLAY='backtests/strategies/kalcb/fixed_allocation_reference_v1.py'
WORKER='scripts/kalcb_fixed_allocation_worker_v1.py'


def freeze_allocation(out,name,period,old,programs,authenticated,*,mode,reference=100_000_000.,bridge=None):
    out=Path(out);folder=out/('identity' if mode=='identity' else 'cases')/name/period
    receipt=folder/'input_profile.json'
    if mode not in ('identity','fixed') or reference!=100_000_000.:
        raise ValueError('Undeclared allocation benchmark')
    if set(programs)!={OVERLAY,WORKER}:raise ValueError('Unexpected economic source overlay')
    verify(list(programs.values())+[old[k] for k in ('manifest','plan','policy')])
    if receipt.exists():
        item=read(receipt);verify([item[k] for k in ('manifest','plan','policy')]);return item
    pp=Path(old['manifest']['path']);parent=read(pp);identity=parent.pop('manifest_hash')
    if stable_signature(parent)!=identity:raise ValueError('Changed parent manifest')
    contract={'mode':mode,'reference':reference,'protocol':bound(out/'protocol.json'),
        'scope':'Fixed nominal allocation reference minus invested cost; actual funded cash/equity/MTM retained. No external flows or cash reset.',
        'inherited_profile':old,'bridge_files':{}}
    if mode=='identity':
        if bridge is None or not read(Path(bridge)/'independent_account_verification.json')['passed']:
            raise ValueError('Identity needs an audited reference')
        contract['bridge_files']={str((Path(bridge)/f).resolve()):digest(Path(bridge)/f)
            for f in ('result.json','daily.json','equity.json','fills.json','independent_account_verification.json')}
    elif bridge is not None:raise ValueError('Fixed benchmark has no identity target')
    if folder.exists():raise ValueError('Preserve partial allocation publication')
    folder.mkdir(parents=True)
    plan=read(old['plan']['path']);config=read(plan['config']);context=read(plan['audit_configuration'])
    write(folder/'config.json',config)
    context.update(output_root=str(folder),promotion_status='funded_fixed_allocation_research_only',
                   fixed_allocation_reference={'mode':mode,'reference':reference,'actual_funding_retained':True})
    write(folder/'audit_configuration.json',context)
    for key in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[key],None)
        plan[key]=str((folder/(key+'.json')).resolve());plan['input_sha256'][plan[key]]=digest(plan[key])
    plan['input_sha256'][str((out/'protocol.json').resolve())]=digest(out/'protocol.json')
    plan['fixed_allocation_reference']=contract
    write(folder/'plan.json',plan)
    replacements={OVERLAY:Path(programs[OVERLAY]['path']),WORKER:Path(programs[WORKER]['path']),
        'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/fixed_allocation/protocol.json':out/'protocol.json',
        'research/fixed_allocation/config.json':folder/'config.json',
        'research/fixed_allocation/audit_configuration.json':folder/'audit_configuration.json'}
    files={}
    for relative,sha in parent['files'].items():
        if relative in replacements:continue
        source=pp.parent/'files'/relative;st=source.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,sha)
        if key not in authenticated:
            if digest(source)!=sha:raise ValueError('Changed inherited file '+relative)
            authenticated.add(key)
        link(source,folder/'inputs/files'/relative);files[relative]=sha
    for relative,source in replacements.items():
        link(source,folder/'inputs/files'/relative);files[relative]=digest(source)
    payload=deepcopy(parent);payload['files']=files
    payload['evidence']['fixed_allocation_reference']={'parent_manifest':bound(pp),
        'registration':bound(out/'protocol.json'),'overlay_programs':programs,'contract':contract,
        'original_policy':old['policy'],'new_economic_source':True}
    code=folder/'inputs/files';actual=source_hash(code)
    if actual==old['source_generation_hash']:raise ValueError('New benchmark source was not identified')
    payload['source_generation_hash']=actual
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy)!=old['policy']['sha256']:raise ValueError('Changed inherited policy')
    item={'name':name,'period':period,'folder':str(folder),'source_generation_hash':actual,
        'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(receipt,item);return item
