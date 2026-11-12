"""Relocate immutable native research to an SSD without changing its identity.

Original studies and receipts remain intact. Complete case copies retain their
original request/receipt provenance; the ordinary runner validates them on reuse.
No incomplete case is reused and no executing source file is modified.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def prepare(source, target, cache):
    target.mkdir(parents=True,exist_ok=False)
    manifest=read(source/'inputs/manifest.json')
    cached=read(cache/'inputs/manifest.json')['files']
    code=target/'inputs/files';code.mkdir(parents=True)
    def copy(item):
        name,expected=item;out=code/name;out.parent.mkdir(parents=True,exist_ok=True)
        original=source/'inputs/files'/name
        local=cache/'inputs/files'/name
        if cached.get(name)==expected and local.drive.lower()==out.drive.lower():
            os.link(local,out)
        else:
            shutil.copy2(original,out)
        # Inputs retain their original per-file digests, verified on actual
        # consumption. Executing source is verified in full below.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy,manifest['files'].items()))
    shutil.copy2(source/'inputs/manifest.json',target/'inputs/manifest.json')
    shutil.copy2(source/'registration.json',target/'registration.json')
    registration=read(target/'registration.json')
    study_bindings=[]
    for path in source.glob('study_*.json'):
        study=read(path)
        study.update(code=str(code),manifest=str(target/'inputs/manifest.json'))
        write(target/path.name,study)
        study_bindings.append({'name':path.name,'original_hash':digest(path),
                               'relocated_hash':digest(target/path.name)})
    program='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    actual=subprocess.check_output([sys.executable,'-B','-c',program],cwd=code,text=True).strip()
    if actual!=registration['source_generation_hash']:
        raise ValueError('Relocated execution source differs')
    write(target/'relocation.json',{'original':str(source),'local':str(target),
        'source_generation_hash':actual,'input_manifest_hash':manifest['manifest_hash'],
        'registration_hash':digest(source/'registration.json'),'study_bindings':study_bindings,
        'changes':'Only execution-location fields code/manifest in external study locators; frozen specs, registration, inputs and code unchanged',
        'original_files_retained':True,'completed_case_provenance':'Original request and completion receipts preserved verbatim',
        'complete_cases':[]})


def adopt(source,target):
    relocation=read(target/'relocation.json')
    assert relocation['original']==str(source)
    cases=target/'cases';cases.mkdir(exist_ok=True)
    for old in (source/'cases').iterdir():
        if not (old/'completion.json').exists() or (cases/old.name).exists():
            continue
        shutil.copytree(old,cases/old.name)
        relocation['complete_cases'].append({'name':old.name,
            'completion_hash':digest(old/'completion.json'),'original_path':str(old)})
    write(target/'relocation.json',relocation)
    print(json.dumps({'relocated':str(target),'reused_complete_cases':len(relocation['complete_cases'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','adopt']);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--cache',type=Path)
    a=p.parse_args();source=a.source.resolve();target=a.target.resolve()
    if a.command=='prepare':prepare(source,target,a.cache.resolve())
    else:adopt(source,target)
