"""Auditable comparisons for the complete-history OLR research revision."""
from __future__ import annotations
from collections import Counter
import copy
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_continuous_state_study_v1 as x
s=x.s;n=x.n


def folder(cid,w,scenario='baseline',capped=False):
    return x.ROOT/'cases'/cid/w/('100000000_'+scenario+('_capped' if capped else ''))


def audit(folder,window,warm):
    proof=s.forensic.authenticate(folder)
    if warm:
        actual=n.read(folder/'capacity_history_inputs.json');expected=n.read(x.ROOT/f'boundary_validation_{window}.json')
        assert actual['seeded_state_hash']==expected['seeded_state_hash']
        assert actual['seeded_state']==expected['seeded_state']
        assert actual['seed_rows']==expected['prior_auction_rows']
        assert len(actual['seed_days'])==expected['prior_sessions'] and max(actual['seed_days'])<actual['asof']
        assert actual['database_hash']==expected['database_sha256']
        trades=n.read(folder/'trades.json')
        proof.update(actual_seed_matches_uninterrupted_history=True,prior_sessions=len(actual['seed_days']),
            entry_auction_sample_counts=dict(Counter(len(t['route_metadata']['capacity_forecast']['exit']['samples']) for t in trades)))
    return proof


def pairs():
    plan=n.read(x.ROOT/'candidate_plan.json');rows=[]
    original={w:n.read(s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline/assessment.json') for w in ('train','oos')}
    for c in plan['candidates']:
        row={'candidate_id':c['id'],'label':c['label'],'mutation':c['mutations']}
        for w in ('train','oos'):
            f=folder(c['id'],w)
            if not (f/'assessment.json').exists():break
            a=n.read(f/'assessment.json')
            if not a.get('qualified'):break
            proof=audit(f,w,c['mutations'].get('capacity_broad_auction_history',False))
            row[w]={k:a[k] for k in ('return_pct','drawdown_pct','trades')}
            row[w].update(return_gain_pp=a['return_pct']-original[w]['return_pct'],drawdown_change_pp=a['drawdown_pct']-original[w]['drawdown_pct'],folder=str(f),authentication=proof)
        if not all(w in row for w in ('train','oos')):continue
        row['both_returns_improved']=all(row[w]['return_gain_pp']>=0 for w in ('train','oos'))
        row['eligible']=row['train']['return_gain_pp']>=-3 and row['oos']['return_gain_pp']>=2 and row['oos']['return_pct']>0 and all(row[w]['drawdown_change_pp']<=2 for w in ('train','oos'))
        rows.append(row)
    n.write(s.OUT/'continuous_state_paired_results.json',rows)
    cold=next((r for r in rows if r['candidate_id']==plan['baseline_id']),None)
    if cold:
        parity={}
        fields=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
        for w in ('train','oos'):
            old=s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline';new=Path(cold[w]['folder'])
            economic=lambda p:[{k:t.get(k) for k in fields} for t in n.read(p/'trades.json')]
            assert economic(old)==economic(new)
            assert n.read(old/'daily.json')==n.read(new/'daily.json')
            assert n.read(old/'equity.json')==n.read(new/'equity.json')
            parity[w]={'trade_economics_exact':True,'daily_accounts_exact':True,'every_intraday_equity_mark_exact':True}
        n.write(s.OUT/'continuous_state_cold_parity.json',parity)
    return rows


def register_capacity_contingency():
    path=x.ROOT/'capacity_quantile_contingency_registration.json'
    if path.exists():return
    cp=n.read(x.ROOT/'candidate_plan.json');center=next(c for c in cp['candidates'] if c['label']=='both_histories_continuous_guard_1p0');candidates=[]
    for q in (.75,1.):
        m=copy.deepcopy(center['mutations']);m['capacity_exit_quantile']=q;key=n.sig(m)
        candidates.append({'id':'c'+key[:16],'label':f'continuous_both_guard_1p0_exit_q_{q}',
            'mutations':m,'mutations_hash':key,'kind':'conditional_forecast_calibration','aliases':[],
            'paths':[list(p) for p,_,_ in n.mutation_edits(cp['baseline'],m)]})
    assert not any(r.get('qualified') and r['candidate_id']!=cp['baseline_id'] for r in n.results(x.ROOT))
    n.write(path,{'registered_utc':s.utc(),'before_warm_history_native_outcomes':True,'candidates':candidates,
        'activation':'Only if the full-history median forecast with the 1.0R guard loses more than 3 percentage points of original IS return. Otherwise do not run.',
        'hypothesis':'Filling formerly missing median forecasts systematically tightens the original rule: it would bind 70 of 176 original IS fills versus 11 of 47 OOS fills. Calibrate the forecast separately from strict actual broker participation limits.',
        'forecast_quantiles':[.75,1.],'actual_capacity_limit_unchanged':.01,
        'risk':'A high forecast quantile is optimistic. Native next-bar/auction fills, terminal liquidation, combined half-participation/higher-slippage stress, and IS/OOS gates still apply.',
        'maximum_additional_configurations':2,'independent_validation':False,'promotion':False})


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='register':register_capacity_contingency()
    else:
        rows=pairs();print([{k:r[k] for k in ('label','train','oos','eligible')} for r in rows])
