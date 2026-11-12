"""One source-bound complete native C9-C account; no label forks."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','policies','audit-manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--scenario',required=True);p.add_argument('--mode',choices=('disabled','veto','renew'),required=True)
    p.add_argument('--bridge',type=Path);a=p.parse_args()
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
    from scripts.kalcb_executed_value_study import load_plan
    from scripts.kalcb_executed_value_replay import run
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_pending_funding_worker_v1 import origins
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
    code=Path(__file__).resolve().parents[1];manifest=read(a.audit_manifest);plan=load_plan(a.plan)
    if code!=a.audit_manifest.resolve().parent/'files':raise ValueError('Unfrozen worker')
    protocol=read(plan['holding_account_research']['protocol']['path']);verify([plan['holding_account_research']['protocol']]+protocol['database_bindings'])
    signal_path=code/'research/horizon_policy/map.json';signals=read(signal_path)
    if bound(signal_path)['sha256']!=protocol['signal_map']['sha256']:raise ValueError('Changed causal map')
    generation=source_generation_hash();before=origins(code,manifest)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'request.json',{'arguments':{**vars(a),'policy':'interaction_cost_margin'},'plan':plan,
        'source_generation_hash':generation,'import_bindings':before,'signal_map':bound(signal_path)})
    controller=HorizonReplay(a.mode,signals);controller.install()
    try:run(plan,a.policies,'interaction_cost_margin',a.scenario,a.output,a.audit_manifest)
    finally:controller.restore()
    write(a.output/'horizon_actions.json',controller.records)
    if a.bridge:
        bridge=bridge_accounts(a.output,a.bridge);write(a.output/'identity_bridge.json',bridge)
        if not bridge['passed']:raise ValueError('Disabled overlay changed parent')
    if source_generation_hash()!=generation:raise ValueError('Changed executing source')
    load_plan(a.plan);verify(protocol['database_bindings'])
    write(a.output/'completion.json',{'complete':True,'source_generation_hash':generation,'mode':a.mode,
        'origins_before':before,'origins_after':origins(code,manifest),'actions':bound(a.output/'horizon_actions.json'),
        'objective_achieved':False,'qualification':False})


if __name__=='__main__':main()
