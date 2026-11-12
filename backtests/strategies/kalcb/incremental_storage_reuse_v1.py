"""Authenticate the closed disk-full attempt; never reuse its partial publication."""
from pathlib import Path
import os
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify,link
from backtests.strategies.kalcb.research_process_v1 import exact_process
from backtests.strategies.kalcb.late_incremental_validation_v1 import validate_native,validate_first_hour_contract
from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES


def require_closed(root):
    identity=read(Path(root)/'supervisor_start.json')
    if exact_process(identity):raise ValueError('Healthy predecessor must not be replaced')


def authenticate(root):
    root=Path(root);require_closed(root);p=read(root/'protocol.json')
    if read(root/'status.json').get('error')!="OSError(28, 'No space left on device')":raise ValueError('Unexpected predecessor failure')
    verify(p['programs']+p['input_bindings']+p['data_bindings'])
    memo={}
    def binding(path,expected=None):
        path=Path(path);st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in memo:memo[key]=digest(path)
        if expected is not None and memo[key]!=expected:raise ValueError('Changed completed input '+str(path))
        return dict(path=str(path.absolute()),sha256=memo[key])
    bindings=[]
    prepared=read(root/'prepared/prepared.json')
    if not prepared['complete'] or prepared['prepared_sessions']!=286 or prepared['first_oos_prior_sessions']!=217 or prepared['sessions'][-1]!='2026-07-10':
        raise ValueError('Incomplete continuous history')
    bindings.append(binding(root/'prepared/opportunities.jsonl',prepared['opportunities_sha256']))
    for phase,count in (('before',79),('after',83)):
        proof=read(root/f'producer_imports_{phase}.json');verify([proof['runtime_manifest'],*proof['origins'].values()])
        if len(proof['origins'])!=count:raise ValueError('Incomplete isolated runtime proof')
        bindings.append(binding(root/f'producer_imports_{phase}.json'))
    model=read(root/'models/completion.json')
    if not model['complete'] or not read(root/'policies/completion.json')['complete']:raise ValueError('Incomplete producer')
    bindings += [binding(root/'models'/f,h) for f,h in model['files'].items()]
    for name,recipe in p['recipes'].items():
        for period in ('is','oos'):
            folder=root/'policies'/name/period;pm=read(folder/'policies_manifest.json')
            bindings += [binding(folder/f,h) for f,h in pm['files'].items()]
            validate_first_hour_contract(folder/'policies_manifest.json',pm)
            if read(folder/'later_lineage.json')['recipe']!=recipe:raise ValueError('Changed registered recipe')
    for folder in ('prepared','models','policies'):
        bindings += [binding(f) for f in (root/folder).rglob('*') if f.is_file()]
    for scenario in ('baseline','combined'):
        label=Path(p['label_root'])/'labels'/scenario;done=read(label/'completion.json')
        if not done['complete'] or done.get('smoke'):raise ValueError('Incomplete full-cost labels')
        bindings.append(binding(label/'completion.json'))
        bindings += [binding(label/f,h) for f,h in done['files'].items()]
    oldrows=read(root/'progress.json');actual={(r['name'],r['period'],r['scenario']) for r in oldrows}
    if len(oldrows)!=2 or actual!={('control','is',s) for s in ('baseline','combined')}:raise ValueError('Unexpected completed matrix')
    item=read(root/'cases/control/is/input_profile.json');controls=[]
    for row in oldrows:
        scenario=row['scenario'];folder=root/'cases/control/is'/scenario
        metrics=validate_native(folder,item,scenario,save=False)
        bridge=bridge_accounts(folder,Path(p['profiles']['100']['is']['references'][scenario]).parent)
        if not bridge['passed']:raise ValueError('Completed control differs')
        controls.append(dict(row=row,metrics=metrics,bridge=bridge))
        bindings += [binding(folder/f) for f in (*REPLAY_FILES,'identity_bridge.json')]
    bindings += [binding(root/f) for f in ('protocol.json','study.json','status.json','supervisor_start.json',
        'background.err.log','progress.json','assessment_inheritance.json','preparation_review_v1.json','cases/control/is/input_profile.json')]
    return dict(root=str(root),bindings=bindings,controls=controls,producer_reuse=True,
        partial_publication_reused=False,scope='Disk/storage transport repair only; V13 model/policy lineage and two audited IS controls retained unchanged.')


def publish(root,proof):
    root=Path(root);old=Path(proof['root']);verify(proof['bindings'])
    for directory in ('prepared','models','policies'):
        for src in (old/directory).rglob('*'):
            if src.is_file():link(src,root/src.relative_to(old))
    for f in ('assessment_inheritance.json','preparation_review_v1.json','producer_imports_before.json','producer_imports_after.json'):
        link(old/f,root/f)
    item=read(old/'cases/control/is/input_profile.json');write(root/'cases/control/is/input_profile.json',item)
    for scenario in ('baseline','combined'):
        for f in (*REPLAY_FILES,'identity_bridge.json'):
            link(old/'cases/control/is'/scenario/f,root/'cases/control/is'/scenario/f)


def seed_cache(source,target):
    source=Path(source);target=Path(target)
    if target.exists():raise ValueError('Earlier cache attempt must be preserved')
    count=0
    for blob in source.glob('*/*'):
        if not blob.is_file() or len(blob.name)!=64 or blob.parent.name!=blob.name[:2]:
            raise ValueError('Partial or unexpected cache member; never reuse failed bytes')
        if digest(blob)!=blob.name:raise ValueError('Corrupt cache blob')
        link(blob,target/blob.relative_to(source));count+=1
    return dict(source=str(source),target=str(target),verified_blobs=count)
