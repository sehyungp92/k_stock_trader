"""Frozen native parent traversal and single-opening marginal targets."""
import argparse
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('code','plan','policies','audit-manifest','output','population','bridge'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--scenario',choices=('baseline','combined'),required=True);args=ap.parse_args()
    code=args.code.resolve();sys.path.insert(0,str(code))
    if Path(__file__).resolve()!=code/'scripts/kalcb_opening_marginal_worker_v1.py':raise ValueError('Execute frozen worker')
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_executed_value_study import load_plan
    from scripts.kalcb_executed_value_replay import run
    from backtests.strategies.kalcb.opening_marginal_replay_v1 import OpeningMarginalReplay
    from backtests.strategies.kalcb.fixed_allocation_reference_v1 import verify_identity_bridge
    plan=load_plan(args.plan);population=read(args.population);verify(population['bindings'])
    expected=population['cases'][args.scenario]['units'];source=source_generation_hash()
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'request.json',dict(arguments={k:str(v) for k,v in vars(args).items()},plan=plan,
        source_generation_hash=source,population=bound(args.population),diagnostic_only=True))
    controller=OpeningMarginalReplay(args.output,expected);controller.install()
    try:
        run(plan,args.policies,'interaction_cost_margin',args.scenario,args.output,args.audit_manifest)
        controller.finish()
    finally:controller.restore()
    bridge=verify_identity_bridge(args.output,args.bridge);write(args.output/'identity_bridge.json',bridge)
    if source_generation_hash()!=source:raise ValueError('Executing source changed')
    load_plan(args.plan)
    write(args.output/'completion.json',dict(complete=True,source_generation_hash=source,
        independent_account_review_required=True,diagnostic_only=True,targets=bound(args.output/'marginal_targets.json'),
        omission_targets=len(controller.seen),opening_sessions=len(controller.records),economic_objective_achieved=False))


if __name__=='__main__':main()
