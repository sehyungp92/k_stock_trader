"""Single frozen funded allocation-reference benchmark; no orchestration."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('code','plan','policies','audit-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scenario',required=True)
    p.add_argument('--mode',choices=('identity','fixed'),required=True)
    p.add_argument('--reference',type=float,required=True)
    p.add_argument('--bridge',type=Path)
    args=p.parse_args();sys.path.insert(0,str(args.code.resolve()))
    # Both this worker and the overlay must be included in the new source bundle,
    # not loaded as an unrecorded economic patch against an old source identity.
    if Path(__file__).resolve()!=args.code.resolve()/'scripts/kalcb_exit_allocation_worker_v1.py':
        raise ValueError('Execute the frozen benchmark worker inside its source bundle')
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb.fixed_allocation_reference_v1 import FixedAllocationReference,verify_identity_bridge
    from backtests.strategies.kalcb import executed_value as ev
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    plan=load_plan(args.plan);source=source_generation_hash()
    contract=plan['fixed_allocation_reference']
    if contract['mode']!=args.mode or contract['reference']!=args.reference:
        raise ValueError('Unbound fixed-allocation contract')
    if args.mode=='identity':
        if not args.bridge or not read(args.bridge/'independent_account_verification.json')['passed']:
            raise ValueError('Identity requires an audited native reference')
        for name in ('result.json','fills.json','equity.json','daily.json','independent_account_verification.json'):
            expected=contract['bridge_files'].get(str((args.bridge/name).resolve()))
            if not expected or ev.digest(args.bridge/name)!=expected:raise ValueError('Unbound identity reference')
    elif args.bridge:raise ValueError('Fixed benchmark cannot claim identity bridge')
    # Authenticate every loaded application module before executing the account.
    from backtests.strategies.kalcb.mechanism_observer_v2 import MechanismObserver
    manifest=read(args.audit_manifest);origins={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file or name.split('.')[0] not in ('scripts','backtests','strategy_kalcb','strategy_common','kis_core'):continue
        path=Path(file).resolve()
        if not path.is_relative_to(args.code.resolve()):raise ValueError('External qualification module '+name)
        relative=path.relative_to(args.code.resolve()).as_posix()
        if relative.endswith('.py') and manifest['files'].get(relative)!=ev.digest(path):raise ValueError('Unbound qualification module '+name)
        origins[name]={'path':str(path),'sha256':ev.digest(path)}
    args.output.mkdir(parents=True,exist_ok=False)
    arguments={k:str(getattr(args,k)) for k in ('plan','policies','output','scenario','audit_manifest')}
    arguments['policy']='interaction_cost_margin'
    ev.write_json(args.output/'request.json',{'arguments':arguments,'plan':plan,
        'source_generation_hash':source,'allocation_benchmark':contract,'application_origins':origins})
    from backtests.strategies.kalcb.mechanism_observer_v2 import MechanismObserver
    observer=MechanismObserver(())
    benchmark=FixedAllocationReference(args.reference,enabled=args.mode=='fixed');benchmark.install();observer.install()
    try:run(plan,args.policies,'interaction_cost_margin',args.scenario,args.output,args.audit_manifest)
    finally:
        observer.restore();benchmark.restore()
    summary=observer.summarize(read(args.output/'fills.json'))
    ev.write_json(args.output/'mechanism_evidence.json',dict(summary=summary,exit_probes=observer.exit_probes,entry_probes=observer.entry_probes,
        scope='Actual funded exits; complete parent cascade has no exit in identical pre-decision state.'))
    if read(args.output/'result.json')['audit_status']!='passed':raise ValueError('Benchmark source audit failed')
    if args.mode=='identity':ev.write_json(args.output/'identity_bridge.json',verify_identity_bridge(args.output,args.bridge))
    ev.write_json(args.output/'allocation_reference.json',benchmark.evidence())
    if source_generation_hash()!=source:raise ValueError('Benchmark runtime changed')
    load_plan(args.plan)
    ev.write_json(args.output/'completion.json',{'complete':True,'source_generation_hash':source,
        'mode':args.mode,'independent_account_and_funding_review_required':True})
    print(json.dumps({'complete':True,'output':str(args.output),'mode':args.mode}),flush=True)


if __name__=='__main__':main()
