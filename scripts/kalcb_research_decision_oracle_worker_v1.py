"""Thin registered C18 oracle adapter; existing replay owns every financial step."""
from pathlib import Path
from dataclasses import asdict
import argparse,importlib.util,json,hashlib,sys

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def load(name,binding):
    assert sha(binding['path'])==binding['sha256']
    spec=importlib.util.spec_from_file_location(name,binding['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);ap.add_argument('--period',choices=['is','oos'],required=True)
    ap.add_argument('--scenario',choices=['baseline','combined'],required=True);ap.add_argument('--arm',choices=['off','S','E','SE'],default='off')
    ap.add_argument('--output',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    reg=read(a.registration);profile=reg['profiles'][a.period];reg['registration_binding']=bound(a.registration)
    for b in reg['programs']+reg['bindings']+[profile[k] for k in ('manifest','plan','policy')]:assert sha(b['path'])==b['sha256']
    code=Path(profile['manifest']['path']).parent/'files';sys.path.insert(0,str(code))
    projection_module=load('c18_oracle_projection',reg['oracle_program']);horizon_module=load('c18_oracle_horizon',reg['horizon_program'])
    from scripts.kalcb_executed_value_study import load_plan,config_for
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_pending_funding_worker_v1 import origins
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
    from backtests.strategies.kalcb.round3_execution_options_v1 import ExecutionOptions,ProbeComplete
    from backtests.strategies.kalcb.round3_integrity_v1 import neutral,exact_paths,reconstruct_journal
    from backtests.strategies.kalcb import executed_value as ev
    from strategy_kalcb.config import KALCBConfig
    plan=load_plan(Path(profile['plan']['path']));cfg=config_for(plan,a.scenario);generation=source_generation_hash()
    assert generation==profile['source_generation_hash']
    expected={**read(plan['config']),'slippage_bps':ev.SCENARIOS[a.scenario][1],'fast_replay_suppress_rejections':False}
    assert neutral(asdict(cfg))==neutral(asdict(KALCBConfig.from_mapping(expected)))
    assert all(neutral(asdict(cfg))[k]==v for k,v in expected.items())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);before=origins(code,read(profile['manifest']['path']))
    signals=code/'research/horizon_policy/map.json';assert sha(signals)==plan['round3_execution']['signal_map']['sha256']
    write(out/'request.json',dict(arguments=vars(a),registration=reg['registration_binding'],native_generation=generation,
        additional_programs=reg['programs'],effective_config=neutral(asdict(cfg)),origins_before=before,
        oracle_only=a.arm!='off',eligible_for_scoring=False,causal_trading_policy=False))
    veto=HorizonReplay('veto' if a.arm in ('off','E') else 'disabled',read(signals))
    observer=ExecutionOptions(out,enabled=True,hydration=a.arm=='off',smoke_through=reg['smoke_through'][a.period] if a.smoke else None)
    projection=projection_module.OracleProjection(a.arm,a.period,a.scenario,reg,out)
    horizon=horizon_module.DiagnosticHorizon(chooser=projection.horizon if a.arm in ('E','SE') else None)
    veto.install();observer.install();projection.install();horizon.install();smoke=False
    try:
        try:run(plan,Path(profile['policy']['path']).parent,'interaction_cost_margin',a.scenario,out,Path(profile['manifest']['path']))
        except ProbeComplete:smoke=True
    finally:horizon.restore();projection.restore();observer.restore();veto.restore()
    write(out/'horizon_actions.json',veto.records);write(out/'oracle_exit_choices.json',projection.exit_records)
    write(out/'scheduled_exits.json',horizon.records)
    assert source_generation_hash()==generation
    load_plan(Path(profile['plan']['path']))
    after=origins(code,read(profile['manifest']['path']));assert after==before
    for b in reg['programs']:assert sha(b['path'])==b['sha256']
    if smoke:
        observer.save_smoke()
        write(out/'smoke_source.json',dict(passed=True,native_generation=generation,additional_sources=reg['programs'],origins_before=before,origins_after=after,
            changed_selection_choices=projection.choices,exit_choices=projection.exit_records,scheduled_exits=len(horizon.records),eligible_for_scoring=False))
        return
    audit=read(out/'execution_audit.json')
    assert audit['audit_status']=='passed' and audit['execution_validity_pass'],audit
    independent=reconstruct_journal(out,1e8);assert independent['passed'] and not independent['terminal_inventory'] and not independent['pending']
    write(out/'independent_journal.json',independent)
    if a.arm=='off':
        assert observer.hydrated=={'pending','inventory'}
        write(out/'hydration.json',observer.receipts)
        write(out/'full_path_bridge.json',exact_paths(out,reg['controls'][a.period][a.scenario]['folder']))
    write(out/'completion.json',dict(passed=True,complete=True,arm=a.arm,period=a.period,scenario=a.scenario,
        source_generation_hash=generation,oracle_programs=reg['programs'],eligible_for_scoring=False,
        causal_policy_qualified=False,conditional_native_execution_qualified=True,independent_funding_review_still_required=True,
        policy_binding=getattr(projection,'policy_binding',None),hindsight_heuristic_not_optimum=a.arm!='off',holdout_accesses=0))

if __name__=='__main__':main()
