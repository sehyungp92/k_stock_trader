"""Explicit successor provenance; only complete, authenticated IS accounts reuse."""
from pathlib import Path
import hashlib
import shutil
from .retention_research_io_v1 import read,write,bound,verify,digest
from .late_artifact_validation_v2 import validate_native
from .population_research_io_v1 import bridge_accounts
from .pending_entry_account_pilot_v1 import canonical,neutral

SCENARIOS=('baseline','participation','cost','combined')


def verify_terminal_transition(folder):
    folder=Path(folder);done=read(folder/'completion.json')
    if not done.get('complete'):raise ValueError('Incomplete repaired account')
    verify([done['terminal_audit_transition'],done['holding']])
    t=read(folder/'terminal_audit_transition.json')
    if digest(folder/'terminal_audit_transition.json')!=done['terminal_audit_transition']['sha256']:
        raise ValueError('Terminal transition receipt substitution')
    identity=neutral(read(folder/'execution_audit.json')['audit_receipt']['identity'])
    if (t.get('passed') is not True or t['after_sha256']!=t['expected_after_sha256']
        or t['published_evaluation_identity']!=identity):
        raise ValueError('Unverified complete terminal audit transition')
    return {'passed':True,'receipt':bound(folder/'terminal_audit_transition.json')}


def authenticate_predecessor(root):
    root=Path(root);p=read(root/'protocol.json')
    verify(p['programs']+p['bindings']+p['database_bindings'])
    if (root/'completion.json').exists() or not (root/'supervisor_failure.json').exists():
        raise ValueError('Expected preserved failed predecessor')
    progress=read(root/'progress.json')
    expected={('control','is',s) for s in SCENARIOS}
    if len(progress)!=4 or {(r['name'],r['period'],r['scenario']) for r in progress}!=expected:
        raise ValueError('Only the four complete IS predecessor accounts may be reused')
    cases={};verified=set()
    for row in progress:
        folder=root/'cases/control/is'/row['scenario'];profile=read(folder/'input_profile.json')
        validate_native(folder,profile,row['scenario'],save=False)
        done=read(folder/'completion.json');holding=read(folder/'holding_complete.json')
        verify([done['holding'],row['result']])
        if not holding.get('complete') or holding['action_forks']!=2*holding['rows']:
            raise ValueError('Partial holding evidence cannot be reused')
        identities={x['day']:x['exact_hold_eod_state'] for x in read(folder/'hold_identity.json')}
        count=0
        import json
        for line in (folder/'labels.jsonl').open(encoding='utf-8'):
            label=json.loads(line);count+=1
            if identities.get(label['observation']['day'])!=label['hold_eod_state']:
                raise ValueError('Unauthenticated predecessor holding state')
            bindings=[label['prefix'],label['source'],label['hold']]+[c['ledger'] for c in label['contrasts'].values()]
            for binding in bindings:
                key=(binding['path'],binding['sha256'])
                if key not in verified:verify([binding]);verified.add(key)
        if count!=holding['rows'] or len(identities)!=holding['exact_hold_days']:
            raise ValueError('Truncated predecessor evidence')
        bridge=bridge_accounts(folder,Path(p['references']['is'][row['scenario']]))
        if not bridge['passed']:raise ValueError('Predecessor does not match original parent')
        files=[{'relative':f.relative_to(folder).as_posix(),**bound(f)}
            for f in sorted(folder.rglob('*')) if f.is_file()]
        cases['control/is/'+row['scenario']]={'source':str(folder),'files':files,
            'original_profile':profile,'complete_holding':holding,'exact_parent_bridge':bridge}
    return {'predecessor_protocol':bound(root/'protocol.json'),
        'preserved_failure':bound(root/'supervisor_failure.json'),'completed_cases':cases,
        'excluded_partial_case':str(root/'cases/control/oos/baseline'),
        'partial_output_reused':False}


def reuse_complete(root,spec,out,scenario):
    out=Path(out)
    if out.exists():raise ValueError('Successor account already exists')
    verify(spec['files']);source=Path(spec['source'])
    # Private copies preserve every original absolute source/receipt binding.
    # No validation or future write can mutate predecessor files via hardlinks.
    shutil.copytree(source,out)
    for f in spec['files']:
        if digest(out/f['relative'])!=f['sha256']:raise ValueError('Copied account differs')
    if read(out/'input_profile.json')!=spec['original_profile']:
        raise ValueError('Reused account source identity was rewritten')
    validate_native(out,spec['original_profile'],scenario,save=False)
    write(out/'successor_reuse.json',{'authenticated_complete_account':True,'source':spec['source'],
        'registration':bound(Path(root)/'protocol.json'),'original_profile_preserved':True,
        'partial_output_reused':False})


def check_successor_sources(root):
    root=Path(root);p=read(root/'protocol.json')
    old=read(p['recovery']['predecessor_protocol']['path'])
    for key in ('recipes','scenarios','native_accounts_min','native_accounts_max','sampling','actions',
        'continuation','value_contract','features','model_spec','learning','readiness','objective','coverage_limit'):
        if p[key]!=old[key]:raise ValueError('Changed economic design during terminal-audit recovery: '+key)
    # Every original native service is byte-identical; new entry points wrap the
    # old replay only at its audit boundary. Preserve the previous failure.
    for rel,b in old['overlays'].items():
        if p['overlays'][rel]['sha256']!=b['sha256']:raise ValueError('Changed original economic service')
    verify([p['recovery']['predecessor_protocol'],p['recovery']['preserved_failure']])
    return {'passed':True,'original_economic_overlays_unchanged':len(old['overlays']),
        'complete_accounts_reused':len(p['recovery']['completed_cases']),'partial_output_reused':False}
