"""Freeze a C18 diagnostic-only extension of the authenticated C17 native engine."""
from pathlib import Path
from copy import deepcopy
import os,shutil,json,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read,bound,digest,source_hash
from backtests.auto.shared.cache_keys import stable_signature

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OLD=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
REPO=Path(__file__).resolve().parents[1]
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)

def main():
    start=time.monotonic();out=ROOT/'native_v1';out.mkdir(exist_ok=False)
    design=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043.md'
    save(out/'registration.json',dict(design=bound(design),program=bound(__file__),
        parent=bound(OLD/'native_v5/preparation.json'),economic_comparator_unchanged=True,
        diagnostics=['off','entry','exit'],new_treatments=False,stopped_operations=False))
    profiles={};old=read(OLD/'native_v5/preparation.json')
    helper='backtests/strategies/kalcb/research_decision_probe_v1.py'
    worker='scripts/kalcb_research_decision_worker_v1.py'
    for period,items in old['profiles'].items():
        item=next(v for v in items.values() if v['arm']=='v6_h0' and v['initial']==100000000)
        mp=Path(item['manifest']['path']);payload=read(mp);sig=payload.pop('manifest_hash')
        assert stable_signature(payload)==sig
        runtime=out/period;code=runtime/'inputs/files';files={}
        for rel,h in payload['files'].items():
            src=mp.parent/'files'/rel;dst=code/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(src,dst)
            except OSError:shutil.copyfile(src,dst);assert digest(dst)==h
            files[rel]=h
        for rel in [helper,worker]:
            dst=code/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dst);files[rel]=digest(dst)
        payload['files']=files;payload['source_generation_hash']=source_hash(code)
        payload['evidence']['C18_timing_diagnostic']=dict(parent=bound(mp),registration=bound(out/'registration.json'),
            added_sources={k:files[k] for k in [helper,worker]},existing_core_broker_and_policy_unchanged=True)
        manifest=runtime/'inputs/manifest.json';save(manifest,{**payload,'manifest_hash':stable_signature(payload)})
        profiles[period]={}
        for target in ['off','entry','exit']:
            plan=deepcopy(read(item['plan']['path']));plan['round3_execution']['c18_target']=target
            plan['round3_execution']['c18_registration']=bound(out/'registration.json')
            pp=runtime/'plans'/(target+'.json');save(pp,plan)
            profiles[period][target]=dict(period=period,arm=target,initial=100000000,folder=str(runtime),plan=bound(pp),
                manifest=bound(manifest),source_generation_hash=payload['source_generation_hash'],
                policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'))
    save(out/'preparation.json',dict(complete=True,profiles=profiles,seconds=time.monotonic()-start,
        existing_source_identity='original immutable hard links, except two added diagnostic modules; full original files remain unchanged'))
    print('C18 frozen diagnostic sources prepared',flush=True)

if __name__=='__main__':main()
