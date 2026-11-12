"""One frozen native account, called by the shared research framework."""
import argparse
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','policies','audit-manifest','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--scenario',required=True)
    ap.add_argument('--mode',choices=('control','joint','uniform'),required=True)
    ap.add_argument('--bridge',type=Path)
    ap.add_argument('--pairs',type=int,default=0)
    args=ap.parse_args()
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
    from scripts.kalcb_executed_value_study import load_plan
    from scripts.kalcb_executed_value_replay import run
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb.opening_account_replay_v2 import OpeningAccountReplay
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    from scripts.kalcb_pending_funding_worker_v1 import origins
    code=Path(__file__).resolve().parents[1]
    manifest=read(args.audit_manifest)
    plan=load_plan(args.plan)
    if plan['opening_account_research']['mode']!=args.mode:
        raise ValueError('Requested action differs from immutable plan')
    verify(plan['opening_account_research']['database_bindings'])
    if code != args.audit_manifest.resolve().parent/'files':
        raise ValueError('Native worker must execute inside its audited frozen runtime')
    source=source_generation_hash()
    before=origins(code,manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'request.json',{'arguments':{**vars(args),'policy':'interaction_cost_margin'},
        'plan':plan,'source_generation_hash':source,'import_bindings':before})
    controller=OpeningAccountReplay(args.mode,plan['opening_account_research']['databases'],
        pair_limit=args.pairs,pair_store=args.output/'prefixes')
    controller.install()
    try:
        run(plan,args.policies,'interaction_cost_margin',args.scenario,args.output,args.audit_manifest)
        controller.finish_pair()
    finally:
        controller.restore()
    write(args.output/'opening_decisions.json',controller.records)
    write(args.output/'account_contrasts.json',controller.pairs)
    if args.bridge:
        bridge=bridge_accounts(args.output,args.bridge)
        write(args.output/'identity_bridge.json',bridge)
        if not bridge['passed']:
            raise ValueError('Disabled joint decision layer changed parent account')
    if source_generation_hash()!=source:
        raise ValueError('Changed executing source')
    load_plan(args.plan)
    verify(plan['opening_account_research']['database_bindings'])
    write(args.output/'completion.json',{'complete':True,'source_generation_hash':source,
        'origins_before':before,'origins_after':origins(code,manifest),
        'decisions':bound(args.output/'opening_decisions.json'),
        'contrasts':bound(args.output/'account_contrasts.json'),
        'mode':args.mode,'study_complete':False})


if __name__=='__main__':
    main()
