"""One frozen native account plus exact-state read-only mechanism evidence."""
import argparse
import json
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','policies','audit-manifest','output','alternatives'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--scenario',required=True); ap.add_argument('--bridge',type=Path)
    args=ap.parse_args()
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound
    from backtests.strategies.kalcb.mechanism_observer_v2 import MechanismObserver
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    plan=load_plan(args.plan); source=source_generation_hash()
    # All application imports must resolve inside this authenticated bundle.
    code=Path(__file__).resolve().parents[1]; manifest=read(args.audit_manifest)
    checked=[]
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file: continue
        name=getattr(module,'__name__','')
        if name.split('.')[0] not in ('scripts','backtests','strategy_kalcb','strategy_common','kis_core'): continue
        path=Path(file).resolve()
        if not path.is_relative_to(code): raise ValueError('External application import '+str(path))
        rel=path.relative_to(code).as_posix()
        if rel.endswith('.py'):
            proof=bound(path)
            if manifest['files'].get(rel)!=proof['sha256']: raise ValueError('Unregistered import '+rel)
            checked.append(proof)
    args.output.mkdir(parents=True,exist_ok=False)
    arguments={'plan':str(args.plan),'policies':str(args.policies),'policy':'interaction_cost_margin',
               'scenario':args.scenario,'audit_manifest':str(args.audit_manifest),'output':str(args.output)}
    write(args.output/'request.json',{'arguments':arguments,'plan':plan,'source_generation_hash':source,
        'mechanism_observer':bound(code/'backtests/strategies/kalcb/mechanism_observer_v2.py'),
        'alternative_binding':bound(args.alternatives),'import_bindings':checked})
    observer=MechanismObserver(read(args.alternatives)['alternatives'] if args.bridge else ())
    observer.install()
    try: run(plan,args.policies,'interaction_cost_margin',args.scenario,args.output,args.audit_manifest)
    finally: observer.restore()
    summary=observer.summarize(read(args.output/'fills.json'))
    write(args.output/'mechanism_evidence.json',{'summary':summary,'exit_probes':observer.exit_probes,
        'entry_probes':observer.entry_probes,'scope':'Exact same-state observations; only matched native exit fills are executed interventions.'})
    if args.bridge:
        write(args.output/'identity_bridge.json',bridge_accounts(args.output,args.bridge))
        if summary['submitted_exit_orders']: raise ValueError('Disabled overlay submitted an exit')
    if source_generation_hash()!=source: raise ValueError('Source changed during case')
    load_plan(args.plan)
    write(args.output/'completion.json',{'complete':True,'source_generation_hash':source,
        'mechanism_evidence':bound(args.output/'mechanism_evidence.json')})
    print(json.dumps({'complete':True,'summary':{k:v for k,v in summary.items() if k!='executions'}}),flush=True)


if __name__=='__main__': main()
