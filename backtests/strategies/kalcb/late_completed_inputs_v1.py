"""Authenticate completed producer artifacts for a validator-only successor."""
from pathlib import Path
from backtests.strategies.kalcb.retention_research_io_v1 import read, bound, verify, digest, link
from backtests.strategies.kalcb.late_artifact_validation_v1 import validate_native
from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES


def file_manifest(folder, receipt):
    folder=Path(folder)
    if receipt.get('complete') is not True:
        raise ValueError('Incomplete producer')
    bindings=[{'path':str(folder/name),'sha256':sha} for name,sha in receipt['files'].items()]
    if any(Path(name).name!=name for name in receipt['files']):
        raise ValueError('Unexpected producer file path')
    verify(bindings)
    return bindings


def authenticate(root):
    root=Path(root); protocol=read(root/'protocol.json')
    verify(protocol['programs']+protocol['bindings']+protocol['data_bindings'])
    if read(root/'status.json').get('error')!="ValueError('Shortened source audit scope')":
        raise ValueError('Not the registered validator compatibility failure')
    bindings=[bound(root/name) for name in ('protocol.json','study.json','status.json','background.err.log',
        'progress.json','preflight/completion.json','assessment_inheritance.json')]
    preflight=read(root/'preflight/completion.json')
    if not preflight['passed'] or preflight['native_identity_bridges']!=4:
        raise ValueError('Preflight gates incomplete')
    from backtests.strategies.kalcb.late_runtime_recovery_v1 import authenticate_predecessor
    authenticate_predecessor(protocol['preflight_reuse']['root'])
    meta=read(root/'prepared/prepared.json')
    if (not meta['complete'] or meta['prepared_sessions']!=286 or
        meta['first_oos_prior_sessions']!=217 or meta['sessions'][-1]!='2026-07-10'):
        raise ValueError('Incomplete continuous preparation')
    verify([meta['protocol'],meta['previews'],{'path':str(root/'prepared/opportunities.jsonl'),'sha256':meta['opportunities_sha256']}])
    for scenario in ('baseline','combined'):
        folder=root/'labels'/scenario; done=read(folder/'completion.json')
        if done['smoke'] or done['scenario']!=scenario or done['source_generation_hash']!=protocol['producer_runtime']['source_generation_hash']:
            raise ValueError('Wrong fresh label lineage')
        verify([done['prepared'],done['config']]); file_manifest(folder,done)
    file_manifest(root/'models',read(root/'models/completion.json'))
    if not read(root/'policies/completion.json')['complete']:
        raise ValueError('Unfinished policies')
    for name in protocol['recipes']:
        for period in ('is','oos'):
            folder=root/'policies'/name/period
            if not read(folder/'completion.json')['complete']:
                raise ValueError('Incomplete policy')
            pm=read(folder/'policies_manifest.json')
            file_manifest(folder,{'complete':True,'files':pm['files']})
            lineage=read(folder/'later_lineage.json')
            verify([lineage[k] for k in ('registration','preparation','models','parent_policy')]+lineage['label_manifests'])
            if lineage['recipe']!=protocol['recipes'][name] or not lineage['parent_snapshots_preserved']:
                raise ValueError('Changed economic recipe')
    # Capture every completed file, including unsuccessful action observations.
    for directory in ('prepared','labels','models','policies'):
        bindings.extend(bound(p) for p in sorted((root/directory).rglob('*')) if p.is_file())
    out=root/'cases/control/is/baseline'; item=read(root/'cases/control/is/input_profile.json')
    metrics=validate_native(out,item,'baseline',save=False)
    bridge=bridge_accounts(out,Path(protocol['profiles']['100']['is']['references']['baseline']).parent)
    bindings.extend(bound(out/name) for name in REPLAY_FILES)
    bindings.append(bound(root/'cases/control/is/input_profile.json'))
    return {'root':str(root),'bindings':bindings,'completed_producers':list(('prepared','labels','models','policies')),
        'continuous_sessions':286,'first_oos_prior_sessions':217,'recipes_unchanged':True,
        'control':{'folder':str(out),'input_profile':bound(root/'cases/control/is/input_profile.json'),
                   'metrics':metrics,'bridge':bridge,'result':bound(out/'result.json')},
        'scope':'Completed V6 artifacts authenticated without rewriting failed evidence. V6 first control now passes strict scope-aware independent validation and exact parent bridge. No strategy or economic result changed.'}


def publish(root,proof):
    """Link completed immutable inputs and the revalidated control to successor."""
    root=Path(root); old=Path(proof['root']); verify(proof['bindings'])
    for directory in ('prepared','policies'):
        for source in (old/directory).rglob('*'):
            if source.is_file():link(source,root/source.relative_to(old))
    source=Path(proof['control']['folder'])
    for name in REPLAY_FILES:link(source/name,root/'cases/control/is/baseline'/name)
    link(old/'cases/control/is/input_profile.json',root/'cases/control/is/input_profile.json')
