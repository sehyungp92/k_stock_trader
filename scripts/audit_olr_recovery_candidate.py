"""Audit a frozen research candidate without overriding its qualification status.

The native source/execution auditor reruns both anchor scenarios and the median
reference. Saved trades, marked equity and daily accounting must match exactly.
This is an integrity diagnostic, not an acceptance or promotion workflow.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register','run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--workers', type=int, default=3)
    args=parser.parse_args(); root=args.root.resolve()
    study=read(root/'study_100000000.json'); sys.path.insert(0,study['code'])
    from backtests.strategies.olr.allocation_recovery import run_one, location, support
    from backtests.strategies.olr.capacity_qualification import checked_case, native_audit_valid
    registration=root/(args.candidate+'_audit_registration.json')
    if args.command=='register':
        if registration.exists(): raise FileExistsError(registration)
        if args.candidate not in read(root/'registration.json')['profiles']: raise ValueError('Unregistered profile')
        target=root/'audit_workflow'; target.mkdir(exist_ok=True)
        wrapper=target/Path(__file__).name
        if wrapper.exists() and digest(wrapper)!=digest(Path(__file__)):
            raise ValueError('Do not replace a frozen audit workflow')
        if not wrapper.exists(): shutil.copy2(__file__,wrapper)
        write(registration,{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
            'candidate':args.candidate,'capitals':[100000000], 'scenarios':['baseline','combined'],
            'references':['median',args.candidate], 'wrapper':str(wrapper),'wrapper_hash':digest(wrapper),
            'study_hash':digest(root/'study_100000000.json'),
            'original_registration_hash':digest(root/'registration.json'),
            'source_generation_hash':study['source_generation_hash'],
            'promotion_permitted':False,'independent_evidence':False,
            'purpose':'Verify native source, execution and exact economic repeatability of the promising but sparsely supported mechanism; do not waive any economic or support gate.'})
        print(json.dumps({'registered':str(registration)}),flush=True); return
    plan=read(registration)
    if (Path(__file__).resolve()!=Path(plan['wrapper']).resolve()
            or digest(Path(__file__))!=plan['wrapper_hash']
            or digest(root/'study_100000000.json')!=plan['study_hash']
            or digest(root/'registration.json')!=plan['original_registration_hash']
            or args.candidate!=plan['candidate']):
        raise ValueError('Frozen audit identity changed')
    jobs=[(name,capital,scenario) for name in plan['references']
          for capital in plan['capitals'] for scenario in plan['scenarios']]
    # Check existing complete economic evidence before creating any audit runs.
    for name,capital,scenario in jobs: checked_case(location(root,name,capital,scenario))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,root,name,capital,scenario,True) for name,capital,scenario in jobs]
        for future in futures: future.result()
    records=[]
    for name,capital,scenario in jobs:
        path=location(root,name,capital,scenario,True); result,audit=checked_case(path)
        reference=location(root,name,capital,scenario)
        parity={file:read(path/file)==read(reference/file) for file in ('trades.json','equity.json','daily.json')}
        valid=native_audit_valid(audit,path)
        records.append({'recipe':name,'capital':capital,'scenario':scenario,
            'native_audit_valid':valid,'exact_economic_parity':parity,
            'pass':valid and all(parity.values()),'support':support(path),
            'audit_sha256':digest(path/'execution_audit.json'),
            'completion_sha256':digest(path/'completion.json')})
    report={'complete':True,'pass':all(row['pass'] for row in records),
        'registration_hash':digest(registration),'source_generation_hash':study['source_generation_hash'],
        'records':records,'promotion_permitted':False,'independent_evidence':False,
        'scope':'Full source/execution audits for the two anchor scenarios only; capital-grid qualification and direct-support requirements remain separate and unchanged.'}
    write(root/(args.candidate+'_integrity_diagnostic.json'),report)
    print(json.dumps(report,indent=2),flush=True)
    if not report['pass']: raise SystemExit(1)


if __name__=='__main__': main()
