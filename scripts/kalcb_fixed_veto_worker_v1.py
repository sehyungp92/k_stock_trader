"""C13 frozen fixed/compound allocation with unchanged opening VETO."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','policies','audit-manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();code=Path(__file__).resolve().parents[1]
    if code!=a.audit_manifest.resolve().parent/'files':raise ValueError('Unfrozen worker')
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_executed_value_study import load_plan,config_for
    from scripts.kalcb_executed_value_replay import run
    from backtests.strategies.kalcb import executed_value as ev
    from backtests.strategies.kalcb.fixed_allocation_reference_v1 import FixedAllocationReference,verify_identity_bridge
    from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
    plan=load_plan(a.plan);contract=plan['capital_sensitivity'];initial=contract['initial_equity']
    if contract['mode'] not in ('compound','fixed') or contract['reference']!=100000000 or contract['horizon_mode'] not in ('disabled','veto') or initial<=0:
        raise ValueError('Unregistered capital/controller mode')
    cfg=config_for(plan,contract['scenario'])
    for key in ('risk_dynamic_notional_enabled','risk_max_single_equity_pct','risk_max_gross_equity_pct',
                'risk_daily_vol_target','risk_drawdown_full_pct','frontier_shadow_enabled','frontier_rotation_enabled'):
        if getattr(cfg,key):raise ValueError('Unsupported persistent/capital dependency '+key)
    if read(plan['audit_configuration'])['initial_equity']!=initial or ev.REFERENCE_EQUITY!=100000000:
        raise ValueError('Capital contract mismatch')
    protocol=read(contract['protocol']['path']);verify([contract['protocol']]+protocol['databases'])
    signals=code/'research/horizon_policy/map.json'
    if bound(signals)['sha256']!=protocol['signal_map']['sha256']:raise ValueError('Changed signal map')
    generation=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'request.json',dict(arguments={**vars(a),'policy':'interaction_cost_margin','scenario':contract['scenario']},
        plan=plan,capital_sensitivity=contract,source_generation_hash=generation))
    allocation=FixedAllocationReference(100000000,enabled=contract['mode']=='fixed');controller=HorizonReplay(contract['horizon_mode'],read(signals))
    ev.REFERENCE_EQUITY=initial;allocation.install();controller.install()
    try:run(plan,a.policies,'interaction_cost_margin',contract['scenario'],a.output,a.audit_manifest)
    finally:controller.restore();allocation.restore();ev.REFERENCE_EQUITY=100000000.
    write(a.output/'allocation_reference.json',allocation.evidence());write(a.output/'horizon_actions.json',controller.records)
    if contract.get('bridge'):write(a.output/'identity_bridge.json',verify_identity_bridge(a.output,contract['bridge']))
    if source_generation_hash()!=generation:raise ValueError('Changed executing source')
    load_plan(a.plan);verify(protocol['databases'])
    write(a.output/'completion.json',dict(complete=True,source_generation_hash=generation,initial_equity=initial,
        mode=contract['mode'],horizon_mode=contract['horizon_mode'],full_continuous_state_claim=False))


if __name__=='__main__':main()
