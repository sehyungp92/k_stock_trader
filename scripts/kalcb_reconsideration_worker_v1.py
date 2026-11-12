"""C15 source-bound complete native account; no local labels or fits."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','policies','audit-manifest','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--mode',choices=('disabled','reconsider'),required=True)
    parser.add_argument('--bridge',type=Path)
    a=parser.parse_args()
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
    from scripts.kalcb_executed_value_study import load_plan
    from scripts.kalcb_executed_value_replay import run
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_pending_funding_worker_v1 import origins
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    from backtests.strategies.kalcb.reconsideration_v1 import ReconsiderationReplay
    code=Path(__file__).resolve().parents[1];manifest=read(a.audit_manifest);plan=load_plan(a.plan)
    if code!=a.audit_manifest.resolve().parent/'files':raise ValueError('Unfrozen worker')
    binding=plan['holding_account_research']['protocol'];protocol=read(binding['path'])
    verify([binding]+protocol['database_bindings'])
    signals=code/'research/horizon_policy/map.json'
    if bound(signals)['sha256']!=protocol['signal_map']['sha256']:raise ValueError('Changed opening map')
    generation=source_generation_hash();before=origins(code,manifest)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'request.json',dict(arguments={**vars(a),'policy':'interaction_cost_margin'},plan=plan,
          source_generation_hash=generation,import_bindings=before,signal_map=bound(signals)))
    controller=ReconsiderationReplay(a.mode,read(signals));controller.install()
    try:run(plan,a.policies,'interaction_cost_margin',a.scenario,a.output,a.audit_manifest)
    finally:controller.restore()
    write(a.output/'horizon_actions.json',controller.records)
    write(a.output/'reconsideration_actions.json',controller.renewals)
    if a.bridge:
        bridge=bridge_accounts(a.output,a.bridge);write(a.output/'identity_bridge.json',bridge)
        if not bridge['passed']:raise ValueError('Disabled reconsideration changed VETO')
    if source_generation_hash()!=generation:raise ValueError('Changed executing source')
    load_plan(a.plan);verify(protocol['database_bindings'])
    write(a.output/'completion.json',dict(complete=True,source_generation_hash=generation,mode=a.mode,
          origins_before=before,origins_after=origins(code,manifest),actions=bound(a.output/'reconsideration_actions.json'),
          economic_objective_achieved=False,principal_qualification=False))


if __name__=='__main__':main()
