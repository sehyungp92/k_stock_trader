"""One frozen native execution case; no fitting or optimization orchestration."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('code','plan','policies','audit-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scenario',required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.code.resolve()))
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb import executed_value as ev
    plan=load_plan(args.plan);source=source_generation_hash()
    args.output.mkdir(parents=True,exist_ok=False)
    arguments={k:str(getattr(args,k.replace('-','_'))) for k in ('plan','policies','output','scenario')}
    arguments.update(policy='interaction_cost_margin',audit_manifest=str(args.audit_manifest))
    ev.write_json(args.output/'request.json',{'arguments':arguments,'plan':plan,'source_generation_hash':source,
        'worker':{'path':str(Path(__file__).resolve()),'sha256':ev.digest(__file__)}})
    run(plan,args.policies,'interaction_cost_margin',args.scenario,args.output,args.audit_manifest)
    if source_generation_hash()!=source:raise ValueError('Runtime changed during replay')
    load_plan(args.plan)
    ev.write_json(args.output/'completion.json',{'complete':True,'source_generation_hash':source})
    print(json.dumps({'completed':str(args.output)}),flush=True)


if __name__=='__main__':main()
