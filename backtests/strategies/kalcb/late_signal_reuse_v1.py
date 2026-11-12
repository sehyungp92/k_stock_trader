"""Authenticate signal producer outputs and a completed native control, read-only."""
from pathlib import Path
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,link
from backtests.strategies.kalcb.late_completed_inputs_v1 import file_manifest
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native,validate_first_hour_contract
from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES


def require_closed(identity):
    import psutil
    try:
        process=psutil.Process(identity['pid'])
        if abs(process.create_time()-identity['process_created'])<.01:
            raise ValueError('Predecessor process still alive: '+str(process.cmdline()))
    except psutil.NoSuchProcess:pass


def authenticate(root):
    root=Path(root);p=read(root/'protocol.json')
    require_closed(read(root/'supervisor_start.json'))
    if read(root/'status.json').get('error')!="ValueError('Wrong first-hour policy contract')":
        raise ValueError('Not the registered signal validator failure')
    verify(p['programs']+p['input_bindings']+p['data_bindings'])
    meta=read(root/'prepared/prepared.json')
    if (not meta['complete'] or meta['prepared_sessions']!=286 or meta['first_oos_prior_sessions']!=217
            or meta['sessions'][-1]!='2026-07-10'):
        raise ValueError('Incomplete continuous preparation')
    verify([meta['protocol'],meta['previews'],{'path':str(root/'prepared/opportunities.jsonl'),'sha256':meta['opportunities_sha256']}])
    for scenario in ('baseline','combined'):
        folder=Path(p['label_root'])/'labels'/scenario;done=read(folder/'completion.json')
        if done['smoke'] or done['scenario']!=scenario or done['source_generation_hash']!=p['producer_runtime']['source_generation_hash']:
            raise ValueError('Changed original full-cost labels')
        verify([done['prepared'],done['config']]);file_manifest(folder,done)
    file_manifest(root/'models',read(root/'models/completion.json'))
    if not read(root/'policies/completion.json')['complete']:raise ValueError('Incomplete policies')
    for name in p['recipes']:
        for period in ('is','oos'):
            folder=root/'policies'/name/period;pm=read(folder/'policies_manifest.json')
            if not read(folder/'completion.json')['complete']:raise ValueError('Incomplete policy')
            file_manifest(folder,{'complete':True,'files':pm['files']})
            validate_first_hour_contract(folder/'policies_manifest.json',pm)
            if read(folder/'later_lineage.json')['recipe']!=p['recipes'][name]:raise ValueError('Changed recipe identity')
    for phase,count in (('before',78),('after',85)):
        proof=read(root/f'producer_imports_{phase}.json')
        verify([proof['runtime_manifest'],*proof['origins'].values()])
        if proof['checked_modules']!=count or len(proof['origins'])!=count or proof['source_generation_hash']!=p['producer_runtime']['source_generation_hash']:
            raise ValueError('Incomplete isolated runtime proof')
        manifest=read(proof['runtime_manifest']['path']);code=Path(proof['runtime_manifest']['path']).parent/'files'
        for name,origin in proof['origins'].items():
            if origin.get('registered_new_signal'):
                if name!='backtests.strategies.kalcb.late_signal_v1' or origin not in [dict(x,registered_new_signal=True) for x in p['programs']]:
                    raise ValueError('Unregistered external model source')
            else:
                rel=Path(origin['path']).relative_to(code).as_posix()
                if manifest['files'].get(rel)!=origin['sha256']:raise ValueError('Foreign application import')
    derived=read(root/'derived_feature_audit_v1.json');verify([derived[k] for k in ('source','program','protocol')])
    if not derived['passed'] or derived['counts']!={'feature_comparisons':53732,'issuer_days':7676,'inherited_comparisons':30704,'poison_rejections':2}:
        raise ValueError('Missing independent derived-feature proof')
    bindings=[bound(root/name) for name in ('protocol.json','study.json','status.json','supervisor_start.json',
        'background.err.log','assessment_inheritance.json','producer_imports_before.json','producer_imports_after.json',
        'derived_feature_audit_v1.json','derived_feature_audit_v1.py')]
    for directory in ('prepared','models','policies'):
        bindings.extend(bound(x) for x in sorted((root/directory).rglob('*')) if x.is_file())
    out=root/'cases/control/is/baseline';item=read(root/'cases/control/is/input_profile.json')
    metrics=validate_native(out,item,'baseline',save=False)
    bridge=bridge_accounts(out,Path(p['profiles']['100']['is']['references']['baseline']).parent)
    bindings.extend(bound(out/name) for name in REPLAY_FILES)
    bindings.append(bound(root/'cases/control/is/input_profile.json'))
    return {'root':str(root),'bindings':bindings,'producer_reuse':True,'strategy_unchanged':True,
        'control':{'metrics':metrics,'bridge':bridge,'folder':str(out)},
        'scope':'Completed V9 producer outputs and source-passed control authenticated under strict versioned signal contract. Failed evidence remains unchanged.'}


def publish(root,proof):
    root=Path(root);old=Path(proof['root']);verify(proof['bindings'])
    for directory in ('prepared','models','policies'):
        for source in (old/directory).rglob('*'):
            if source.is_file():link(source,root/source.relative_to(old))
    for name in REPLAY_FILES:link(old/'cases/control/is/baseline'/name,root/'cases/control/is/baseline'/name)
    link(old/'cases/control/is/input_profile.json',root/'cases/control/is/input_profile.json')


def seed_verified_cache(source,target):
    """Hash every inherited content blob before creating immutable local hardlinks."""
    source=Path(source);target=Path(target);count=0
    if target.exists():raise ValueError('Preserve earlier cache publication')
    for blob in source.glob('*/*'):
        if not blob.is_file() or len(blob.name)!=64 or blob.parent.name!=blob.name[:2]:
            raise ValueError('Unexpected cache member')
        if digest(blob)!=blob.name:raise ValueError('Corrupt inherited cache blob')
        link(blob,target/blob.relative_to(source));count+=1
    return {'verified_blobs':count,'source':str(source),'target':str(target),'method':'verified content-addressed hardlinks, no economic change'}
