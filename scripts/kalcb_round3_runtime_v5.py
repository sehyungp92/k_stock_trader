"""Immutable recorder-only successor; reuse authenticated immutable input inodes."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read,bound,digest,source_hash
from backtests.auto.shared.cache_keys import stable_signature
from scripts.kalcb_round3_freeze_v1 import write
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
OLD=ROOT/'native_v4';OUT=ROOT/'native_v5';REPO=Path(__file__).resolve().parents[1]
REL='backtests/strategies/kalcb/round3_integrity_v1.py'


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(program=bound(__file__),parent=bound(OLD/'preparation.json'),
        helper=bound(REPO/REL),repair='Complete-state commitment uses canonical non-string mapping-key serialization',
        unchanged_economics=True,unchanged_score=True,unchanged_arms=bound(OLD/'arms.json'),max_workers=2))
    old=read(OLD/'preparation.json');profiles={}
    for period,items in old['profiles'].items():
        item=next(iter(items.values()));mp=Path(item['manifest']['path']);payload=read(mp);sig=payload.pop('manifest_hash')
        if stable_signature(payload)!=sig:raise ValueError('Parent identity drift')
        out=OUT/'runtime'/period;code=out/'inputs/files';files={}
        for relative,sha in payload['files'].items():
            src=mp.parent/'files'/relative;dst=code/relative;dst.parent.mkdir(parents=True,exist_ok=True)
            if relative==REL:
                shutil.copyfile(REPO/REL,dst);files[relative]=digest(dst)
            else:
                try:
                    os.link(src,dst)
                    if not os.path.samefile(src,dst):raise ValueError('Input inode mismatch')
                except OSError:
                    shutil.copyfile(src,dst)
                    if digest(dst)!=sha:raise ValueError('Copied immutable input changed')
                files[relative]=sha
        payload['files']=files;payload['source_generation_hash']=source_hash(code)
        payload['evidence']['C17_recorder_successor']=dict(parent=bound(mp),registration=bound(OUT/'registration.json'),
            source_diff={REL:dict(before=read(mp)['files'][REL],after=files[REL])},
            data_authentication='Same immutable inodes as authenticated v4; external database/policy hashes rechecked by every native worker')
        write(out/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
        common=dict(folder=str(out),manifest=bound(out/'inputs/manifest.json'),source_generation_hash=payload['source_generation_hash'],
            policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'))
        profiles[period]={k:{**v,**common} for k,v in items.items()}
    shutil.copyfile(OLD/'arms.json',OUT/'arms.json')
    write(OUT/'preparation.json',dict(complete=True,profiles=profiles,seconds=time.monotonic()-start))
    print('Recorder successor ready',flush=True)


if __name__=='__main__':main()
