"""C18 thin worker: default-off selective timing, unchanged C12-B native economics."""
import argparse
from dataclasses import asdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','policies','audit-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scenario',required=True);p.add_argument('--observe',action='store_true')
    p.add_argument('--hydrate',action='store_true');p.add_argument('--bridge',type=Path)
    p.add_argument('--smoke-through');p.add_argument('--delay-minutes',type=int,default=0,choices=(0,5))
    a=p.parse_args();code=Path(__file__).resolve().parents[1]
    if code!=a.audit_manifest.resolve().parent/'files':raise ValueError('Execute frozen worker')
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
    from scripts.kalcb_executed_value_study import load_plan,config_for
    from scripts.kalcb_executed_value_replay import run
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_pending_funding_worker_v1 import origins
    from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
    from backtests.strategies.kalcb.round3_integrity_v1 import exact_paths,reconstruct_journal,neutral
    from backtests.strategies.kalcb.round3_execution_options_v1 import ProbeComplete
    from backtests.strategies.kalcb.research_decision_probe_v1 import DecisionStudyOptions
    from backtests.strategies.kalcb import executed_value as ev
    from strategy_kalcb.config import KALCBConfig
    manifest=read(a.audit_manifest);plan=load_plan(a.plan);contract=plan['round3_execution']
    if contract['mode'] not in ('veto','disabled'):raise ValueError('Stopped/unsupported controller')
    cfg=config_for(plan,a.scenario);expected=read(plan['config'])
    expected={**expected,'slippage_bps':ev.SCENARIOS[a.scenario][1],'fast_replay_suppress_rejections':False}
    effective=neutral(asdict(cfg))
    if effective!=neutral(asdict(KALCBConfig.from_mapping(expected))) or any(effective[k]!=v for k,v in expected.items()):
        raise ValueError('Effective config override/alias failure')
    initial=contract['initial_equity']
    if initial not in (95000000,100000000,105000000) or read(plan['audit_configuration'])['initial_equity']!=initial:
        raise ValueError('Capital identity mismatch')
    signals=code/'research/horizon_policy/map.json';verify([contract['signal_map'],contract['registration']])
    if bound(signals)['sha256']!=contract['signal_map']['sha256']:raise ValueError('VETO map drift')
    if a.smoke_through and a.smoke_through!=contract['smoke_through']:raise ValueError('Unregistered smoke window')
    generation=source_generation_hash();before=origins(code,manifest)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'request.json',dict(arguments={**vars(a),'policy':'interaction_cost_margin'},plan=plan,
        source_generation_hash=generation,import_bindings=before,effective_config=effective))
    controller=HorizonReplay(contract['mode'],read(signals))
    verify([contract['c18_registration']])
    if a.delay_minutes:raise ValueError('C18 selective timing cannot stack ordinary delay')
    observer=DecisionStudyOptions(a.output,target=contract['c18_target'],enabled=a.observe,hydration=a.hydrate,
        delay_minutes=a.delay_minutes,smoke_through=a.smoke_through)
    ev.REFERENCE_EQUITY=initial;controller.install();observer.install();smoke=False
    try:
        try:run(plan,a.policies,'interaction_cost_margin',a.scenario,a.output,a.audit_manifest)
        except ProbeComplete:smoke=True
    finally:observer.restore();controller.restore();ev.REFERENCE_EQUITY=100000000.
    write(a.output/'horizon_actions.json',controller.records)
    if source_generation_hash()!=generation:raise ValueError('Source changed during account')
    load_plan(a.plan)
    if smoke:
        observer.save_smoke()
        write(a.output/'smoke_source.json',dict(checked=True,source_generation_hash=generation,
            origins_before=before,origins_after=origins(code,manifest),eligible_for_scoring=False))
        return
    if a.observe:write(a.output/'independent_journal.json',reconstruct_journal(a.output,initial))
    if a.hydrate:
        if observer.hydrated!={'pending','inventory'}:raise ValueError('Hydration lacked native reach')
        write(a.output/'hydration.json',observer.receipts)
    if a.bridge:write(a.output/'full_path_bridge.json',exact_paths(a.output,a.bridge))
    write(a.output/'completion.json',dict(complete=True,source_generation_hash=generation,mode=contract['mode'],
        initial_equity=initial,origins_before=before,origins_after=origins(code,manifest),
        full_path_bridge=bool(a.bridge),observation=a.observe,hydration=a.hydrate,
        delay_minutes=a.delay_minutes,delayed_decisions=observer.delayed,c18_target=contract['c18_target']))


if __name__=='__main__':main()
