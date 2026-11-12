"""One frozen native opening-reservation account; ordinary shared replay engine."""
import argparse
from pathlib import Path
import sys


def origins(code,manifest):
    from backtests.strategies.kalcb.retention_research_io_v1 import bound
    rows=[]
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file or name.split('.')[0] not in ('scripts','backtests','strategy_kalcb','strategy_common','kis_core'):continue
        path=Path(file).resolve()
        if not path.is_relative_to(code):raise ValueError('External application import '+name)
        proof=bound(path);rel=path.relative_to(code).as_posix()
        if manifest['files'].get(rel)!=proof['sha256']:raise ValueError('Changed import '+rel)
        rows.append(proof)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','policies','audit-manifest','output'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--scenario',required=True);ap.add_argument('--bridge',type=Path)
    a=ap.parse_args()
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    from backtests.strategies.kalcb.reclaim_observer_v1 import ReclaimObserver
    code=Path(__file__).resolve().parents[1];manifest=read(a.audit_manifest)
    plan=load_plan(a.plan);source=source_generation_hash();before=origins(code,manifest)
    a.output.mkdir(parents=True,exist_ok=False)
    args={k:str(getattr(a,k)) for k in ('plan','policies','scenario','output')}
    args.update(audit_manifest=str(a.audit_manifest),policy='interaction_cost_margin')
    write(a.output/'request.json',{'arguments':args,'plan':plan,'source_generation_hash':source,'import_bindings':before})
    # Empty opportunity list: observe actual entry exposures only, no copies,
    # feed extension, 90-minute action or alternative portfolio path.
    observer=ReclaimObserver([]);observer.install()
    try:run(plan,a.policies,'interaction_cost_margin',a.scenario,a.output,a.audit_manifest)
    finally:observer.restore()
    summary=observer.summarize(read(a.output/'fills.json'))
    write(a.output/'observer_evidence.json',{'entry_exposures':observer.entries,'previews':[],
        'summary':summary,'scope':'Actual entry submissions and funding state only; no new reclaim action.'})
    if a.bridge:
        bridge=bridge_accounts(a.output,a.bridge);write(a.output/'identity_bridge.json',bridge)
        if not bridge['passed']:raise ValueError('Disabled reservation changed parent path')
    if source_generation_hash()!=source:raise ValueError('Source changed')
    load_plan(a.plan)
    write(a.output/'completion.json',{'complete':True,'source_generation_hash':source,'study_complete':False,
        'evidence':bound(a.output/'observer_evidence.json'),'origins_before':before,'origins_after':origins(code,manifest),
        'promotion_status':'pending_funding_screening_only_no_promotion'})


if __name__=='__main__':main()
