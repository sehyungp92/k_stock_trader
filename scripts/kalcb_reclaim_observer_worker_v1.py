"""One exact unchanged parent account with separately copied minute90 previews."""
import argparse
from datetime import date
import json
from pathlib import Path
import sqlite3
import sys


def origins(code,manifest):
    from backtests.strategies.kalcb.retention_research_io_v1 import bound
    checked=[]
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None);name=getattr(module,'__name__','')
        if not file or name.split('.')[0] not in ('scripts','backtests','strategy_kalcb','strategy_common','kis_core'):continue
        path=Path(file).resolve()
        if not path.is_relative_to(code):raise ValueError('External application import '+str(path))
        rel=path.relative_to(code).as_posix()
        if rel.endswith('.py'):
            proof=bound(path)
            if manifest['files'].get(rel)!=proof['sha256']:raise ValueError('Unregistered import '+rel)
            checked.append(proof)
    return checked


def raw_proof(spec,start,end):
    from collections import defaultdict
    from backtests.strategies.kalcb.reclaim_source_audit_v1 import verify_rows
    from backtests.strategies.kalcb.retention_research_io_v1 import read,verify
    verify(spec['bindings'])
    features=defaultdict(list);population=defaultdict(list)
    for line in Path(spec['prepared']['path']).read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if start<=r['trade_date']<=end:features[r['trade_date']].append(r)
    for i,path in enumerate(spec['opening_prepared']):
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            r=json.loads(line);day=r['trade_date']
            if start<=day<=end and ((i==0 and day<='2026-03-31') or (i==1 and day>='2026-04-01')):population[day].append(r)
    proofs=[]
    for day,rows in sorted(features.items()):
        db=spec['databases'][int(day>='2026-04-01')]
        with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as conn:
            index={(s,t,'5m'):(o,h,l,c,v,True) for s,t,o,h,l,c,v in conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=?',(day,))}
        proofs.append({'day':day,**verify_rows(population[day],rows,index,{date.fromisoformat(day)})})
    return {'passed':True,'days':proofs,'scope':'Additive complete raw minute90 population/sector/prefix proof; parent native source audit remains mandatory.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','policies','audit-manifest','output','opportunities','bridge'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--scenario',required=True);ap.add_argument('--prior',type=int,required=True)
    a=ap.parse_args()
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    from backtests.strategies.kalcb.reclaim_observer_v1 import ReclaimObserver
    code=Path(__file__).resolve().parents[1];manifest=read(a.audit_manifest)
    plan=load_plan(a.plan);source=source_generation_hash();before=origins(code,manifest)
    spec=read(a.opportunities);policy=read(a.policies/'policies_manifest.json')
    start,end=policy['evaluation_from'],policy['evaluation_through']
    a.output.mkdir(parents=True,exist_ok=False)
    proof=raw_proof(spec,start,end);write(a.output/'reclaim_source_proof.json',proof)
    selected=[(r,[s for s in scores if s['prior']==a.prior]) for r,scores in spec['opportunities'] if start<=r['trade_date']<=end]
    arguments={'plan':str(a.plan),'policies':str(a.policies),'policy':'interaction_cost_margin',
        'scenario':a.scenario,'audit_manifest':str(a.audit_manifest),'output':str(a.output)}
    write(a.output/'request.json',{'arguments':arguments,'plan':plan,'source_generation_hash':source,
        'observer':bound(code/'backtests/strategies/kalcb/reclaim_observer_v1.py'),'opportunities':bound(a.opportunities),'import_bindings':before})
    observer=ReclaimObserver(selected);observer.install()
    try:run(plan,a.policies,'interaction_cost_margin',a.scenario,a.output,a.audit_manifest)
    finally:observer.restore()
    summary=observer.summarize(read(a.output/'fills.json'))
    write(a.output/'observer_evidence.json',{'summary':summary,'previews':observer.previews,'entry_exposures':observer.entries})
    bridge=bridge_accounts(a.output,a.bridge);write(a.output/'identity_bridge.json',bridge)
    if not bridge['passed']:raise ValueError('Observer changed parent path')
    if source_generation_hash()!=source:raise ValueError('Source changed')
    load_plan(a.plan);after=origins(code,manifest)
    write(a.output/'completion.json',{'complete':True,'source_generation_hash':source,'evidence':bound(a.output/'observer_evidence.json'),
        'raw_source':bound(a.output/'reclaim_source_proof.json'),'origins_before':before,'origins_after':after,
        'study_complete':False,'new_action_portfolio_qualified':False})
    print(json.dumps({'complete':True,'summary':summary}),flush=True)


if __name__=='__main__':main()
