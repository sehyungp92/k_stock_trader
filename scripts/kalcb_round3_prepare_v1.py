"""Create source-only C17 bridge capsules; never launches an economic arm."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read,bound,digest,source_hash
from backtests.auto.shared.cache_keys import stable_signature
from scripts.kalcb_round3_freeze_v1 import write

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
ADDED=('backtests/strategies/kalcb/round3_integrity_v1.py','scripts/kalcb_round3_native_worker_v1.py')


def prepare(period):
    freeze=read(ROOT/'freeze/manifest.json');old=freeze['profiles'][period+'_100000000']
    parent=read(old['manifest']['path']);sig=parent.pop('manifest_hash')
    if stable_signature(parent)!=sig:raise ValueError('Parent manifest signature')
    out=ROOT/'bridge_v1/runtime'/period;out.mkdir(parents=True,exist_ok=False)
    code=out/'inputs/files';source=Path(old['manifest']['path']).parent/'files'
    files={};cache=[]
    for relative,sha in parent['files'].items():
        if '__pycache__' in Path(relative).parts or relative.endswith('.pyc'):
            cache.append(relative);continue
        src=source/relative;dst=code/relative
        if digest(src)!=sha:raise ValueError('Changed non-cache source '+relative)
        dst.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,dst)
        except OSError:shutil.copyfile(src,dst)
        files[relative]=sha
    for relative in ADDED:
        if relative in files:raise ValueError('Overlay collision')
        dst=code/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/relative,dst)
        files[relative]=digest(dst)
    plan=read(old['plan']['path']);ctx=read(plan['audit_configuration']);config=read(plan['config'])
    write(out/'config.json',config);ctx['output_root']=str(out)
    write(out/'audit_configuration.json',ctx)
    for key in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[key]);plan[key]=str(out/(key+'.json'));plan['input_sha256'][plan[key]]=digest(plan[key])
    plan['round3_execution']=dict(mode='veto',initial_equity=100000000,
        registration=bound(ROOT/'bridge_v1/registration.json'),signal_map=bound(code/'research/horizon_policy/map.json'))
    write(out/'plan.json',plan)
    for name in ('config.json','audit_configuration.json','plan.json'):
        rel='research/c17/bridge/'+name;dst=code/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(out/name,dst);files[rel]=digest(dst)
    payload=deepcopy(parent);payload['files']=files;payload['source_generation_hash']=source_hash(code)
    payload['evidence']['C17_bridge']=dict(parent_manifest=old['manifest'],registration=bound(ROOT/'bridge_v1/registration.json'),
        added_python={rel:files[rel] for rel in ADDED},excluded_generated_caches=cache,
        economic_source_unchanged=True)
    write(out/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    item=dict(period=period,folder=str(out),manifest=bound(out/'inputs/manifest.json'),plan=bound(out/'plan.json'),
        policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'),source_generation_hash=payload['source_generation_hash'])
    write(out/'profile.json',item);return item


def main():
    out=ROOT/'bridge_v1';out.mkdir(exist_ok=False)
    write(out/'registration.json',dict(protocol=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0042.md'),
        freeze=bound(ROOT/'freeze/manifest.json'),program=bound(__file__),helpers=[bound(REPO/p) for p in ADDED],
        scope='Native observation OFF-economics bridge and persisted pending/inventory hydration probe; no treatments',
        cases=[dict(period=p,scenario=s,hydrate=h) for p,s,h in [('is','baseline',False),('oos','combined',False),('is','baseline',True),('oos','combined',True)]],
        complete_ordered_path_equality_required=True,full_source_audit=True,max_workers=2))
    profiles={p:prepare(p) for p in ('is','oos')}
    write(out/'preparation.json',dict(complete=True,profiles=profiles))
    print('C17 bridge capsules prepared',flush=True)


if __name__=='__main__':main()
