"""Freeze finite policies and common native sources before any arm outcome."""
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

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
OUT=ROOT/'native_v3'
ADDED=('backtests/strategies/kalcb/round3_integrity_v1.py',
       'backtests/strategies/kalcb/round3_execution_options_v1.py','scripts/kalcb_round3_native_worker_v3.py')
SCENARIOS=('baseline','participation','cost','combined')


def recipes(base):
    arms={}
    for votes in (5,6,7):
        for hold in (0,6,8):
            cfg=deepcopy(base);cfg['entry_plan_routes'][0]['min_quality_votes']=votes
            cfg['research_opening_failure_min_hold_bars']=hold
            name=f'v{votes}_h{hold}'
            arms[name]=dict(config=cfg,mode='veto',selectable=(votes,hold)!=(6,0),
                phase=2 if hold==0 and votes!=6 else 3 if votes==6 and hold else 4,
                hypothesis='H2+H3' if votes!=6 and hold else 'H2' if hold else 'H3')
    for name in ('veto','cap','votes','checkpoint','pullback','heat','min_bar','min_vwap','context_relvol','quick_exit','followthrough','retention'):
        cfg=deepcopy(base);mode='veto'
        if name=='veto':mode='disabled'
        elif name=='cap':cfg['risk_max_gross_equity_pct']=0
        elif name=='votes':cfg['entry_plan_routes'][0]['min_quality_votes']=0
        elif name=='checkpoint':cfg['entry_checkpoint_enabled']=False
        elif name=='pullback':cfg['entry_plan_routes']=cfg['entry_plan_routes'][1:]
        elif name=='heat':cfg['heat_cap_r']=1e100
        elif name in ('min_bar','min_vwap'):
            field='min_bar_ret' if name=='min_bar' else 'min_vwap_ret'
            cfg['entry_plan_'+field]=-9.
            for route in cfg['entry_plan_routes']:route[field]=-9.
        elif name=='context_relvol':cfg['entry_plan_routes'][0]['context_min'].pop('first30_rel_volume')
        elif name=='quick_exit':cfg['quick_exit_enabled']=False
        elif name=='followthrough':cfg['exit_failed_followthrough_bars']=0
        elif name=='retention':cfg['exit_profit_retention_enabled']=False
        arms['remove_'+name]=dict(config=cfg,mode=mode,selectable=False,phase=1,hypothesis='component dependency')
    identity={}
    for name,arm in arms.items():
        key=stable_signature(dict(config=arm['config'],mode=arm['mode']))
        arm['effective_policy_hash']=key;arm['alias_of']=identity.setdefault(key,name)
    return arms


def modified_source(relative,text):
    if relative=='strategy_kalcb/core/logic.py':
        anchor='"min_bar_ret": 0., "min_vwap_ret": 0., "min_or_position": .5,'
        replacement='"min_bar_ret": (-9. if config.entry_plan_min_bar_ret <= -9. else 0.), "min_vwap_ret": (-9. if config.entry_plan_min_vwap_ret <= -9. else 0.), "min_or_position": .5,'
        if text.count(anchor)!=1:raise ValueError('Checkpoint gate patch anchor drift')
        return text.replace(anchor,replacement)
    if relative=='scripts/kalcb_executed_value_replay.py':
        anchor="    if updates and not cfg.entry_checkpoint_enabled:raise ValueError('Timed candidates require checkpoint execution')"
        if text.count(anchor)!=1:raise ValueError('Checkpoint package patch anchor drift')
        text=text.replace(anchor,'    if not cfg.entry_checkpoint_enabled: updates = {} # Explicit checkpoint-population diagnostic removal.')
        anchor="        if updates:\n            if policy_manifest.get('observation_minutes')"
        if text.count(anchor)!=1:raise ValueError('Observation scope anchor drift')
        return text.replace(anchor,"        if policy_manifest.get('observation_minutes'):\n            if policy_manifest.get('observation_minutes')")
    return text


def prepare(period,arms):
    freeze=read(ROOT/'freeze/manifest.json');old=freeze['profiles'][period+'_100000000']
    parent=read(old['manifest']['path']);sig=parent.pop('manifest_hash')
    if stable_signature(parent)!=sig:raise ValueError('Parent manifest changed')
    src=Path(old['manifest']['path']).parent/'files';out=OUT/'runtime'/period;code=out/'inputs/files'
    out.mkdir(parents=True,exist_ok=False);files={};changed={}
    modified=('strategy_kalcb/core/logic.py','scripts/kalcb_executed_value_replay.py')
    for relative,sha in parent['files'].items():
        if '__pycache__' in Path(relative).parts or relative.endswith('.pyc'):continue
        source=src/relative;dest=code/relative
        if digest(source)!=sha:raise ValueError('Parent bytes changed '+relative)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if relative in modified:
            dest.write_text(modified_source(relative,source.read_text(encoding='utf-8')),encoding='utf-8')
            changed[relative]=dict(before=sha,after=digest(dest))
        else:
            try:os.link(source,dest)
            except OSError:shutil.copyfile(source,dest)
        files[relative]=digest(dest) if relative in modified else sha
    for relative in ADDED:
        dest=code/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/relative,dest);files[relative]=digest(dest)
    # Exact existing capital/participation tuples, no general capacity exemption.
    qualification=read(src/parent['evidence']['qualification_path'])
    declaration=qualification['capacity']['registered_execution_policies']
    originals=[deepcopy(r) for r in declaration if r['initial_equity']==100000000 and r['buying_power_leverage']==2]
    if len(originals)!=4:raise ValueError('Unexpected reference capacity tuples')
    for capital in (95000000,105000000):
        for r in originals:
            r=deepcopy(r);r['initial_equity']=capital
            if r not in declaration:declaration.append(r)
    qrel='research/c17/qualification.json';write(code/qrel,qualification);files[qrel]=digest(code/qrel)
    plan0=read(old['plan']['path']);profiles={}
    for name,arm in arms.items():
        for capital in (95000000,100000000,105000000):
            folder=out/'policies'/name/str(capital);folder.mkdir(parents=True)
            cfg=deepcopy(arm['config']);write(folder/'config.json',cfg)
            plan=deepcopy(plan0);ctx=read(plan['audit_configuration']);ctx.update(initial_equity=capital,output_root=str(folder))
            write(folder/'audit_configuration.json',ctx)
            for k in ('config','audit_configuration'):
                plan['input_sha256'].pop(plan[k]);plan[k]=str(folder/(k+'.json'));plan['input_sha256'][plan[k]]=digest(plan[k])
            plan['round3_execution']=dict(mode=arm['mode'],initial_equity=capital,arm=name,
                registration=bound(OUT/'registration.json'),signal_map=bound(code/'research/horizon_policy/map.json'),
                smoke_through='2025-08-11' if period=='is' else '2026-05-21')
            write(folder/'plan.json',plan)
            for n in ('config.json','audit_configuration.json','plan.json'):
                rel=f'research/c17/policies/{name}/{capital}/{n}';dest=code/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(folder/n,dest);files[rel]=digest(dest)
            profiles[name+'/'+str(capital)]=dict(period=period,arm=name,initial=capital,folder=str(out),plan=bound(folder/'plan.json'))
    payload=deepcopy(parent);payload['files']=files;payload['source_generation_hash']=source_hash(code)
    payload['evidence']['qualification_path']=qrel
    payload['evidence']['C17']=dict(parent=old['manifest'],registration=bound(OUT/'registration.json'),
        diagnostic_source_patches=changed,default_off_economic_bridge_required=True)
    write(out/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    common=dict(manifest=bound(out/'inputs/manifest.json'),source_generation_hash=payload['source_generation_hash'],
        policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'))
    return {k:{**v,**common} for k,v in profiles.items()}


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    cfg=read(ROOT/'freeze/optimized_config_round2.json')['mutations'];arms=recipes(cfg)
    write(OUT/'arms.json',arms)
    write(OUT/'registration.json',dict(protocol=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0042.md'),
        binding=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0042-executable-binding.md'),
        freeze=bound(ROOT/'freeze/manifest.json'),program=bound(__file__),helpers=[bound(REPO/p) for p in ADDED],
        arms=bound(OUT/'arms.json'),max_workers=2,selectable_family_size=8,no_untouched_holdout=True))
    profiles={p:prepare(p,arms) for p in ('is','oos')}
    write(OUT/'preparation.json',dict(complete=True,profiles=profiles,seconds=time.monotonic()-start))
    print('C17 finite native runtime prepared',flush=True)


if __name__=='__main__':main()
