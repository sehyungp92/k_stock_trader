"""Authenticate every derived selection receipt, including conditional checks."""
from pathlib import Path
from .round3_native_io_v1 import read,write,bound
from .round3_contract_v1 import verify_acceptance

def seal(path,value,receipts,extra_bindings=()):
    path=Path(path);write(path,value)
    bindings={}
    for r in receipts:
        if r.get('passed'):
            verify_acceptance(r)
            for b in r['files'].values():bindings[b['path']]=b
        elif r.get('folder'):
            failed=Path(r['folder'])/'qualification_failure.json'
            if failed.exists():bindings[str(failed)]=bound(failed)
    for b in extra_bindings:bindings[b['path']]=b
    write(path.with_suffix('.seal.json'),dict(evaluation=bound(path),evidence=list(bindings.values()),
        receipts=receipts,complete=True,scope='All native accounts and mandatory conditional qualification evidence'))

def authenticate(path):
    path=Path(path);proof=read(path.with_suffix('.seal.json'))
    if proof.get('complete') is not True:raise ValueError('Incomplete evaluation seal')
    for binding in [proof['evaluation'],*proof['evidence']]:
        if bound(binding['path'])['sha256']!=binding['sha256']:raise ValueError('Stale evaluation or qualification '+binding['path'])
    for receipt in proof['receipts']:
        if receipt.get('passed'):verify_acceptance(receipt)
    return read(path)
