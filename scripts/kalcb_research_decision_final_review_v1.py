"""Read-only C18 closeout audit. No financial replay or policy selection."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import time
import psutil

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OUT=ROOT/'final_review_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f: return {'path':str(p),'sha256':hashlib.file_digest(f,'sha256').hexdigest()}
def save(p,value):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)
def table(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def physical_path(folder,alter_ids=False,alter_quantity=False):
    fills=read(folder/'fills.json');equity=read(folder/'equity.json')
    if alter_ids:
        for f in fills:
            f['order_id']='different';f['reason']='different';f['metadata']['diagnostic_id']='different'
    if alter_quantity:fills[0]['qty']+=1
    values=[{k:f[k] for k in ['timestamp','symbol','side','qty','price']}|{'fee':f['metadata']['fee']} for f in fills]
    return hashlib.sha256(json.dumps({'fills':values,'equity':equity},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    started=time.monotonic();reg=read(OUT/'registration.json');assert bound(__file__)==reg['program']
    for b in reg['bindings']:assert bound(b['path'])==b
    analysis=read(ROOT/'timing_analysis_v1/completion.json');assert analysis['passed']
    for b in analysis['files'].values():assert bound(b['path'])==b
    accounting=list(csv.DictReader((ROOT/'timing_analysis_v1/accounting.csv').open()))
    paths=[]
    for i,r in enumerate(accounting,2):
        folder=Path(r['folder']);assert r['passed']=='True'
        h=physical_path(folder)
        paths.append({'arm':r['arm'],'cell':r['cell'],'accounting_row':i,'economic_path_hash':h})
    first=Path(accounting[0]['folder'])
    assert physical_path(first)==physical_path(first,alter_ids=True)
    assert physical_path(first)!=physical_path(first,alter_quantity=True)
    table('economic_paths.csv',paths)
    groups=defaultdict(list)
    for p in paths:groups[p['economic_path_hash']].append(p['arm']+'/'+p['cell'])
    save(OUT/'economic_path_groups.json',dict(groups))
    data=list(csv.DictReader((ROOT/'timing_analysis_v1/matched_lifetimes.csv').open()))
    cells=defaultdict(list)
    for number,r in enumerate(data,2):cells[r['arm'],r['cell']].append((number,r))
    summaries=[];tails=[]
    for (arm,cell),rows in sorted(cells.items()):
        changed=[(n,r) for n,r in rows if abs(float(r['delta']))>.02]
        sacrificed=[(n,r) for n,r in rows if float(r['baseline_net'])>0 and float(r['delta'])<-.02]
        avoided=[(n,r) for n,r in rows if float(r['baseline_net'])<0 and float(r['delta'])>.02]
        summaries.append({'arm':arm,'cell':cell,'matched_lifetimes':len(rows),
            'changed_pnl_lifetimes':len(changed),'changed_pnl_dates':len({r['decision'][:10] for _,r in changed}),
            'changed_pnl_issuer_days':len({(r['decision'][:10],r['symbol']) for _,r in changed}),
            'sacrificed_profitable_lifetimes':len(sacrificed),
            'sacrificed_profit_krw':sum(-float(r['delta']) for _,r in sacrificed),
            'improved_losing_lifetimes':len(avoided),
            'improved_loss_krw':sum(float(r['delta']) for _,r in avoided),
            'scope':'Matched lifetimes only; losses include sizing/entry/exit/cost interactions, not pure hold-policy regret.'})
        for direction in ['worst','best']:
            ordered=sorted(rows,key=lambda nr:float(nr[1]['delta']),reverse=direction=='best')[:5]
            for n,r in ordered:
                tails.append({'arm':arm,'cell':cell,'tail':direction,'source_row':n,
                    **{k:r[k] for k in ['symbol','decision','baseline_net','diagnostic_net','delta','baseline_reason','diagnostic_reason']}})
    table('timing_affected_population.csv',summaries);table('timing_tail_examples.csv',tails)
    processes=[];intervals=[]
    for name in ['timing_run_v1','timing_run_v2','labels_run_v5','labels_run_v6']:
        for path in sorted((ROOT/name).rglob('*_process.json')):
            d=read(path);launch=path.with_name(path.name.replace('_process.json','_launch.json'));l=read(launch)
            assert not any('ProcessPool' in a for a in l['argv'])
            try:
                p=psutil.Process(l['pid']);assert abs(p.create_time()-l['created'])>.01 or p.cmdline()!=l['argv']
            except psutil.NoSuchProcess:pass
            # Process receipt creation is conservative termination upper bound.
            intervals += [(l['created'],1),(path.stat().st_mtime,-1)]
            processes.append({'stage':name,'receipt':str(path),'pid':l['pid'],'created':l['created'],
                'ended_by':path.stat().st_mtime,'seconds':d['seconds'],'peak_rss':d['peak_rss'],
                'exit_code':d['exit_code'],'guard':d.get('guard'),
                'kind':'full_native_account' if name.startswith('timing') else 'standalone_label_batch'})
    concurrent=peak=0
    for _,delta in sorted(intervals,key=lambda x:(x[0],x[1])):concurrent+=delta;peak=max(peak,concurrent)
    assert peak<=2
    table('native_resources.csv',processes)
    statuses=[]
    for p in ['is','oos']:
        for s in ['baseline','combined']:
            f=ROOT/'labels_run_v6/full'/p/s/'completion.json';d=read(f)
            statuses.append({'period':p,'scenario':s,'episodes':d['distinct_episodes'],'outcomes':d['outcomes'],
                **{k:d['statuses'].get(k,0) for k in ['filled','censored','unfilled','physical_invalid']},
                'cash_error':d['max_independent_cash_error'],'accepted':False,'receipt':str(f)})
            assert d['statuses']['physical_invalid']>0
            assert not f.parent.parent.joinpath(s+'_acceptance.json').exists()
    table('label_qualification.csv',statuses)
    failures=[]
    for name in ['labels_run_v6','stage_chain_v3']:
        d=read(ROOT/name/'failure.json');l=read(ROOT/name/'launch.json')
        try:
            p=psutil.Process(l['pid']);assert abs(p.create_time()-l['created'])>.01 or p.cmdline()!=l['argv']
        except psutil.NoSuchProcess:pass
        failures.append({'stage':name,'error':d['error'],'failure':bound(ROOT/name/'failure.json')})
    for name in ['oracle_run_v1','information_v1','finer_paths_v2','oracle_analysis_v1']:
        assert not (ROOT/name).exists(),name
    assert not (ROOT/'labels_run_v6/labels_complete.json').exists()
    assert not (ROOT/'labels_run_v6/evaluations/full.json').exists()
    assert not list((ROOT/'labels_run_v6').rglob('*_acceptance.json'))
    ancillary=[]
    for folder in ['saved_evidence_v1','native_v1','cohorts_v1','scores_v1','rank_reach_v1',
                   'timing_analysis_v1','lifecycle_review_v1','label_integrity_review_v1',
                   'label_integrity_review_v2','capacity_source_review_v1','capacity_source_review_v2']:
        found=[]
        for name in ['completion.json','preparation.json','receipt.json']:
            f=ROOT/folder/name
            if f.exists():
                d=read(f);value=d.get('seconds',d.get('elapsed_seconds'))
                if value is not None:found.append({'stage':folder,'seconds':value,'receipt':str(f)})
        assert len(found)<=1
        ancillary+=found
    table('ancillary_recorded_resources.csv',ancillary)
    save(OUT/'receipt.json',{'passed':True,'review_pass_is_not_label_acceptance':True,
        'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,
        'timing_accounts':len(paths),'distinct_economic_paths':len(groups),'dedup_id_invariance_and_quantity_sensitivity':True,
        'native_full_account_attempts':sum(p['kind']=='full_native_account' for p in processes),
        'standalone_batch_attempts':sum(p['kind']=='standalone_label_batch' for p in processes),
        'native_elapsed_process_hours':sum(p['seconds'] for p in processes)/3600,
        'ancillary_recorded_hours':sum(r['seconds'] for r in ancillary)/3600,
        'resource_limitation':'Short probes, monitor commands and manual/model review lack complete process-time metering. Controller waiting time is not counted as native computation.',
        'peak_native_concurrency':peak,'full_label_outcomes':sum(r['outcomes'] for r in statuses),
        'full_label_physical_invalid':sum(r['physical_invalid'] for r in statuses),
        'failed_dependencies':failures,'downstream_dispatch':False,'policy_promotions':0,'holdout_accesses':0,
        'research_economic_question_complete':False,
        'files':{p.name:bound(p) for p in OUT.iterdir() if p.is_file() and p.name!='receipt.json'}})
    print(json.dumps({'timing_accounts':len(paths),'economic_paths':len(groups),'peak_workers':peak,
        'native_hours':sum(p['seconds'] for p in processes)/3600,'recorded_ancillary_hours':sum(r['seconds'] for r in ancillary)/3600}))


if __name__=='__main__':main()
