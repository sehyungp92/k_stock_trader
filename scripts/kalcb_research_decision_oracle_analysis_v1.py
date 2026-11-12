"""Independent fixed-menu and funded-account attribution; no policy search."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import argparse,importlib.util,json,gzip,hashlib,time,csv
import numpy as np
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def verify(b):assert bound(b['path'])['sha256']==b['sha256']
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for n,line in enumerate(f,1):yield n,json.loads(line)
def table(p,data):
    if not data:return
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def hash_path(folder):
    fills=read(folder/'fills.json');economics=[{k:f[k] for k in ['symbol','side','qty','price','timestamp']}|{'fee':f['metadata']['fee']} for f in fills]
    # IDs, reasons and diagnostic metadata do not establish a distinct economic path.
    return hashlib.sha256(json.dumps(dict(fills=economics,equity=read(folder/'equity.json')),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',default=str(ROOT/'oracle_run_v1/registration.json'));a=ap.parse_args()
    reg=read(a.registration);run=Path(a.registration).parent;assert read(run/'oracle_complete.json')['passed']
    out=ROOT/'oracle_analysis_v1';out.mkdir(exist_ok=False);start=time.monotonic()
    binding=reg['independent_accounting_program'];verify(binding);spec=importlib.util.spec_from_file_location('separate_accounting',binding['path']);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    save(out/'registration.json',dict(program=bound(__file__),oracle_registration=bound(a.registration),at_utc=datetime.now(timezone.utc).isoformat(),
        objective='Each own-funded feasible heuristic versus unchanged control; no global optimality claim',
        aggregation='IS/OOS one half; baseline/combined equal preference within this diagnostic subset, not the omitted four-scenario aggregate',
        uncertainty='paired5/10/20session blocks4000draws fixed43018; reused-history diagnostic, no search-adjusted policy claim'))
    accounts={};trades={};daily={};dates={};bindings=[];summary=[];decomposition=[];episodes=[];cis=[];gaps=[]
    for p in ['is','oos']:
        for s in ['baseline','combined']:accounts['control',p,s]=Path(reg['controls'][p][s]['folder'])
    for arm in ['S','E','SE']:
        record=read(run/f'evaluations/{arm}.json')
        if record.get('coverage_gap'):gaps.append(dict(arm=arm,reason=record['status'],economic_verdict=record['economic_verdict']));continue
        for r in record['receipts']:
            assert r['passed'] and not r['eligible_for_scoring']
            for b in r['files'].values():verify(b)
            bindings+=list(r['files'].values());accounts[arm,r['process']['period'],r['process']['scenario']]=Path(r['folder'])
    for key,folder in accounts.items():
        result,ts=audit.account(folder);trades[key]=ts;ds=read(folder/'daily.json');daily[key]=np.diff([1e8]+[r['equity'] for r in ds])/1e6;dates[key]=[r['day'] for r in ds]
        summary.append(dict(arm=key[0],period=key[1],scenario=key[2],**result,economic_path_hash=hash_path(folder),
            c16_historical_limit_pp=12 if key[2]=='baseline' else 15,
            exceeds_historical_limit=result['dd_pp']>(12 if key[2]=='baseline' else 15),eligible_for_scoring=False))
    def group(ts):
        g=defaultdict(list)
        for n,t in enumerate(ts,1):g[(t['entry_fill_time'][:10],t['symbol'])].append((n,t))
        return g
    for arm in ['S','E','SE']:
        if (arm,'is','baseline') not in accounts:continue
        for p in ['is','oos']:
            for s in ['baseline','combined']:
                bk=('control',p,s);ck=(arm,p,s);assert dates[bk]==dates[ck]
                bg,cg=group(trades[bk]),group(trades[ck]);added=removed=shared=0;changed_dates=set();changed_episodes=0;sacrificed=0
                for key in sorted(set(bg)|set(cg)):
                    b=sum(t['net_pnl'] for _,t in bg.get(key,[]));c=sum(t['net_pnl'] for _,t in cg.get(key,[]));delta=c-b
                    kind='shared' if key in bg and key in cg else 'added' if key in cg else 'removed'
                    if abs(delta)>.02:changed_dates.add(key[0]);changed_episodes+=1
                    if b>0 and c<b:sacrificed+=b-c
                    if kind=='shared':shared+=delta
                    elif kind=='added':added+=delta
                    else:removed+=delta
                    episodes.append(dict(arm=arm,period=p,scenario=s,day=key[0],symbol=key[1],kind=kind,baseline_cash=b,oracle_cash=c,delta_cash=delta,
                        sacrificed_profitable_episode=(b>0 and c<b),avoided_loss=(b<0 and c>b),
                        baseline_trade_rows=json.dumps([n for n,_ in bg.get(key,[])]),oracle_trade_rows=json.dumps([n for n,_ in cg.get(key,[])]),
                        baseline_folder=str(accounts[bk]),oracle_folder=str(accounts[ck])))
                delta=sum(daily[ck]-daily[bk])*1e6;assert abs(delta-added-removed-shared)<.1
                decomposition.append(dict(arm=arm,period=p,scenario=s,delta_krw=delta,added_krw=added,removed_contribution_krw=removed,
                    shared_lifecycle_krw=shared,residual_krw=delta-added-removed-shared,changed_pnl_dates=len(changed_dates),changed_issuer_days=changed_episodes,
                    added_issuer_days=len(set(cg)-set(bg)),removed_issuer_days=len(set(bg)-set(cg)),surrendered_profitable_issuer_day_cash=sacrificed,
                    scope='Exact own-account issuer-day identity; shared effects combine timing, quantity, exits, costs and later funding; not independent additive interventions'))
        for scenario in ['baseline','combined','two_scenario_aggregate']:
            vs={p:np.mean([daily[arm,p,s]-daily['control',p,s] for s in (['baseline','combined'] if scenario=='two_scenario_aggregate' else [scenario])],axis=0) for p in ['is','oos']}
            cis.extend(dict(arm=arm,scenario=scenario,**r) for r in audit.uncertainty(vs))
    if all((arm,'is','baseline') in accounts for arm in ['S','E','SE']):
        for scenario in ['baseline','combined','two_scenario_aggregate']:
            vs={p:np.mean([daily['SE',p,s]-daily['S',p,s]-daily['E',p,s]+daily['control',p,s] for s in (['baseline','combined'] if scenario=='two_scenario_aggregate' else [scenario])],axis=0) for p in ['is','oos']}
            cis.extend(dict(arm='SE_minus_S_minus_E_interaction',scenario=scenario,**r) for r in audit.uncertainty(vs))
    relaxed=[];status=Counter();choices=[]
    for p in ['is','oos']:
        for s in ['baseline','combined']:
            b=reg['labels'][p][s];verify(b);menus=defaultdict(list)
            for n,r in rows(b['path']):
                status[p,s,r['status']]+=1
                if r['label_valid']:menus[r['episode_id']].append((n,r))
            local=[];issuer=Counter()
            for episode,menu in menus.items():
                n,best=min(menu,key=lambda nr:(-nr[1]['net_cash_krw'],nr[1]['horizon_minutes']));value=max(0,best['net_cash_krw']);local.append(value);issuer[best['day'],best['symbol']]+=1
                choices.append(dict(period=p,scenario=s,episode_id=episode,finite_menu_bound_cash=value,horizon=best['horizon_minutes'] if value>0 else 'no_entry',
                    valid_menu_items=len(menu),source=b['path'],source_row=n,basis='standalone100m/rank1 actual native label quantities; not an account upper bound'))
            relaxed.append(dict(period=p,scenario=s,episodes_with_valid_menu=len(local),positive_finite_menu=sum(v>0 for v in local),
                distinct_issuer_days=len(issuer),overlapping_issuer_days=sum(v>1 for v in issuer.values()),
                cash_quantiles={str(q):float(np.quantile(local,q)) for q in [0,.1,.5,.9,.99,1]} if local else {},
                no_profit_sum_reported=True,global_optimum_certified=False,global_upper_bound_available=False))
    for name,data in [('accounts',summary),('funded_decomposition',decomposition),('issuer_day_contributions',episodes),('paired_uncertainty',cis),('relaxed_menu_choices',choices)]:table(out/(name+'.csv'),data)
    save(out/'relaxed_summary.json',relaxed);save(out/'coverage_gaps.json',gaps);save(out/'source_bindings.json',bindings)
    save(out/'completion.json',dict(passed=True,seconds=time.monotonic()-start,qualified_accounts=len(accounts),feasible_oracle_arms=sorted({k[0] for k in accounts if k[0]!='control'}),
        independent_accounting=True,independent_market_validation=False,holdout_accesses=0,eligible_for_scoring=False,
        files={p.name:bound(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':main()
