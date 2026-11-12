"""One audited native account with exact remaining-action evidence or policy."""
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','policies','audit-manifest','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--scenario',required=True);ap.add_argument('--mode',choices=('labels','policy','disabled'),required=True)
    ap.add_argument('--bridge',type=Path);ap.add_argument('--limit',type=int);ap.add_argument('--models',type=Path)
    a=ap.parse_args()
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
    from scripts.kalcb_executed_value_study import load_plan
    from scripts.kalcb_executed_value_replay import run
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_pending_funding_worker_v1 import origins
    from backtests.strategies.kalcb.holding_account_replay_v1 import HoldingAccountReplay
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    code=Path(__file__).resolve().parents[1];manifest=read(a.audit_manifest);plan=load_plan(a.plan)
    protocol=read(plan['holding_account_research']['protocol']['path'])
    verify([plan['holding_account_research']['protocol']]+protocol['database_bindings'])
    if code!=a.audit_manifest.resolve().parent/'files':raise ValueError('Unfrozen worker execution')
    source=source_generation_hash();before=origins(code,manifest)
    selector=None
    if a.mode=='policy':
        from backtests.strategies.kalcb.holding_account_model_v1 import Policy
        selector=Policy(a.models,a.scenario,plan['holding_account_research']['period'])
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'request.json',{'arguments':{**vars(a),'policy':'interaction_cost_margin'},'plan':plan,
        'source_generation_hash':source,'import_bindings':before,'models':bound(a.models/'manifest.json') if a.models else None})
    controller=HoldingAccountReplay(a.output,mode=a.mode,limit=a.limit,selector=selector)
    controller.install()
    try:
        run(plan,a.policies,'interaction_cost_margin',a.scenario,a.output,a.audit_manifest)
        controller.finish()
    finally:controller.restore()
    if a.bridge:
        result=bridge_accounts(a.output,a.bridge);write(a.output/'identity_bridge.json',result)
        if not result['passed']:raise ValueError('Holding observation changed native control')
    if source_generation_hash()!=source:raise ValueError('Changed executing source')
    load_plan(a.plan);verify(protocol['database_bindings'])
    write(a.output/'completion.json',{'complete':True,'source_generation_hash':source,'mode':a.mode,
        'origins_before':before,'origins_after':origins(code,manifest),'holding':bound(a.output/'holding_complete.json'),
        'objective_achieved':False,'study_complete':False})


if __name__=='__main__':main()
