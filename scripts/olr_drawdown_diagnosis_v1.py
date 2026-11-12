"""Audited drawdown attribution and descriptive exit/risk diagnostics."""
from __future__ import annotations
import argparse,bisect,gzip,json,sqlite3,sys
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_boundary_restart_results_v1 as r
b=r.b;s=r.s;n=r.n
ROOT=b.ROOT.parent/'drawdown_diagnosis_v1'
OUT=REPO/'data/backtests/analysis/olr_drawdown_diagnosis_20261010'


def dt(v):return datetime.fromisoformat(v)
def gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def episode(rows):
    values=np.array([float(v[1]) for v in rows]);high=np.maximum.accumulate(np.r_[1e8,values])[1:];dd=1-values/high
    ti=int(np.argmax(dd));pi=int(np.argmax(values[:ti+1]));return {'peak_index':pi,'trough_index':ti,'peak_time':rows[pi][0],'trough_time':rows[ti][0],
        'peak_nav':float(values[pi]),'trough_nav':float(values[ti]),'drawdown_pct':float(dd[ti]*100),'decline':float(values[pi]-values[ti]),
        'end_drawdown_pct':float((1-values[-1]/high[-1])*100),'peak_recovered':bool(any(v>=values[pi] for v in values[ti+1:]))}
def percentiles(v):return {str(q):float(np.percentile(v,q)) for q in (0,25,50,75,90,95,100)}


def static():
    OUT.mkdir(exist_ok=True,parents=True);plan=n.read(b.ROOT/'candidate_plan.json');catalog={c['label']:c for c in plan['candidates']}
    data={'created_utc':s.utc(),'interpretation':'Risk and MAE/MFE cohorts describe observed trades. Eventual extrema are not causal decision-time predictors. Exposure screens are not executable replays.',
        'policies':{}}
    for label in ('original_saved','literal_carry_original','literal_carry_guard_1p0'):
        windows={}
        for w in ('train','oos'):
            f=s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline' if label=='original_saved' else b.output(catalog[label],w)
            s.forensic.authenticate(f);trades=n.read(f/'trades.json');equity=n.read(f/'equity.json');daily=n.read(f/'daily.json');times=[dt(v[0]) for v in equity]
            rows=[]
            for t in trades:
                m=t['route_metadata'];risk=float(m.get('risk_per_share') or m.get('daily_atr') or 0);notional=t['qty']*t['entry_price']
                ix=bisect.bisect_left(times,dt(t['entry_fill_time']))-1;nav=equity[ix][1] if ix>=0 else 1e8
                rows.append({'symbol':t['symbol'],'entry_decision_time':t['entry_decision_time'],'entry_fill_time':t['entry_fill_time'],'exit_fill_time':t['exit_fill_time'],
                    'qty':t['qty'],'entry_price':t['entry_price'],'exit_price':t['exit_price'],'net_pnl':t['net_pnl'],'return_pct':100*t['net_pnl']/notional,
                    'notional':notional,'pre_first_fill_nav':nav,'notional_pct_pre_first_fill_nav':100*notional/nav,'risk_per_share':risk,'risk_pct_price':100*risk/t['entry_price'],
                    'one_r_pct_pre_first_fill_nav':100*t['qty']*risk/nav,'mfe_r':t['mfe']/risk,'mae_r':t['mae']/risk,'exit_reason':t['exit_reason'],
                    'rank':m.get('candidate_rank'),'sector':m.get('sector'),'capacity_exit_forecast':m.get('capacity_forecast',{}).get('exit',{}),
                    'first_fill_entry_notional_caveat':'Total aggregated entry quantity divided by NAV before first fill; exact timing-dependent holdings use the observational fill ledger.'})
            cohorts={}
            for name,selected in [('all',rows),('winners',[v for v in rows if v['net_pnl']>0]),('losers',[v for v in rows if v['net_pnl']<0])]:
                cohorts[name]={'count':len(selected),'pnl':sum(v['net_pnl'] for v in selected),'risk_price_pct':percentiles([v['risk_pct_price'] for v in selected]),
                    'one_r_account_pct':percentiles([v['one_r_pct_pre_first_fill_nav'] for v in selected]),
                    'mae_below_r':{str(k):sum(v['mae_r']<=-k for v in selected) for k in (.4,.8,1.,1.2,2.)},
                    'mfe_below_r':{str(k):sum(v['mfe_r']<k for v in selected) for k in (1.,4.)}}
            sizing=[]
            for riskbudget in (1.,2.,3.,4.):
                affected=[v for v in rows if v['one_r_pct_pre_first_fill_nav']>riskbudget]
                # Diagnostic accounting only: no native reallocation or liquidity feedback.
                pnl=sum(v['net_pnl']*min(1,riskbudget/v['one_r_pct_pre_first_fill_nav']) for v in rows)
                sizing.append({'risk_budget_pct':riskbudget,'affected':len(affected),'affected_winners':sum(v['net_pnl']>0 for v in affected),
                    'affected_winner_pnl':sum(v['net_pnl'] for v in affected if v['net_pnl']>0),'affected_loser_pnl':sum(v['net_pnl'] for v in affected if v['net_pnl']<0),
                    'scaled_observed_pnl_delta_non_executable':pnl-sum(v['net_pnl'] for v in rows)})
            windows[w]={'folder':str(f),'completion_sha256':n.sha(f/'completion.json'),'result':n.read(f/'result.json'),
                'intraday_episode':episode(equity),'daily_close_episode':episode([(v['day'],v['equity']) for v in daily]),'cohorts':cohorts,
                'risk_budget_exposure_screen':sizing,'trades':rows}
        data['policies'][label]=windows
    by={w:defaultdict(list) for w in ('train','oos')}
    for w in by:
        for t in data['policies']['literal_carry_original'][w]['trades']:by[w][t['symbol']].append(t['risk_pct_price'])
    common=set(by['train'])&set(by['oos']);diff=[float(np.median(by['oos'][sym])-np.median(by['train'][sym])) for sym in common]
    data['within_symbol_risk_shift']={'common_traded_symbols':len(common),'symbols_with_higher_oos_median_risk_pct':sum(v>0 for v in diff),'median_symbol_change_pp':float(np.median(diff)),
        'interpretation':'Descriptive medians at selected entry dates; reduces but does not eliminate selection/composition differences.'}
    n.write(OUT/'static_diagnostics.json',data);print({'static_diagnostics':'complete'},flush=True)


def trace(label):
    f=ROOT/label;proof=n.read(ROOT/(label+'_exact_parity.json'));assert proof['status']=='passed'
    completion=n.read(f/'completion.json')
    for name,h in completion['artifacts'].items():assert n.sha(f/name)==h,name
    native=n.read(f/'equity.json');collapsed=n.read(f/'trades.json');ep=episode(native)
    positions={};prices={};groups=[];cash=1e8;marks=[];maxerror=0.;matched=0
    for row in gz(f/'native_journal.jsonl.gz'):
        event=row['event']
        if event in ('opening_marks','closing_marks'):prices.update({k:float(v) for k,v in row['prices'].items()})
        if event=='fill':
            fill=row['fill'];symbol=fill['symbol'];qty=fill['qty'];price=fill['price'];fee=fill['metadata']['fee'];side=fill['side']
            assert abs(cash-row['cash_before'])<1e-5
            if side=='BUY' and symbol not in positions:
                g={'id':len(groups),'symbol':symbol,'entry_order_id':fill['order_id'],'entry_decision_time':row['order']['submitted_at'],
                    'first_fill_time':fill['timestamp'],'qty':0,'cashflow':0.,'buys':[],'sells':[],'risk_per_share':row['order']['metadata'].get('risk_per_share',0)}
                groups.append(g);positions[symbol]=g
            g=positions[symbol]
            sign=1 if side=='BUY' else -1;g['qty']+=sign*qty;flow=-sign*qty*price-fee;g['cashflow']+=flow;cash+=flow
            g['buys' if side=='BUY' else 'sells'].append({'timestamp':fill['timestamp'],'qty':qty,'price':price,'fee':fee,'order_id':fill['order_id'],'order_type':row['order']['order_type'],
                'reason':row['order'].get('reason'),'submitted_at':row['order']['submitted_at'],'bar':fill['metadata']['execution_bar'],'phase':fill['metadata']['execution_phase']})
            if not g['qty']:del positions[symbol]
            assert abs(cash-row['cash_after'])<1e-5
        if event=='closing_marks':
            values={str(g['id']):g['cashflow']+g['qty']*prices.get(g['symbol'],0) for g in groups}
            nav=cash+sum(g['qty']*prices[sym] for sym,g in positions.items())
            assert dt(row['timestamp'])==dt(native[matched][0]);error=abs(nav-native[matched][1]);maxerror=max(maxerror,error);assert error<1e-5,(row['timestamp'],error)
            marks.append({'timestamp':row['timestamp'],'nav':nav,'cash':cash,'pnl':values,
                'held':{str(g['id']):{'qty':g['qty'],'price':prices[sym]} for sym,g in positions.items()},
                'gross_exposure':sum(g['qty']*prices[sym] for sym,g in positions.items()),
                'sum_original_risk_dollars':sum(g['qty']*g['risk_per_share'] for g in positions.values())})
            matched+=1
    assert matched==len(native) and len(groups)==len(collapsed) and not positions
    bykey={(t['symbol'],dt(t['entry_fill_time'])):t for t in collapsed}
    for g in groups:
        t=bykey[(g['symbol'],dt(g['first_fill_time']))];assert abs(g['cashflow']-t['net_pnl'])<1e-5
        g['native_trade']=t;g['total_buy_qty']=sum(v['qty'] for v in g['buys']);assert g['total_buy_qty']==t['qty']
    pi,ti=ep['peak_index'],ep['trough_index'];peak,trough=marks[pi],marks[ti];contributions=[]
    for g in groups:
        key=str(g['id']);delta=trough['pnl'].get(key,0)-peak['pnl'].get(key,0)
        if abs(delta)>.001:contributions.append({'id':g['id'],'symbol':g['symbol'],'entry_decision_time':g['entry_decision_time'],
            'pnl_at_peak':peak['pnl'].get(key,0),'pnl_at_trough':trough['pnl'].get(key,0),'episode_pnl_delta':delta,'drawdown_points':-100*delta/peak['nav'],
            'held_at_peak':key in peak['held'],'held_at_trough':key in trough['held'],'final_trade_pnl':g['cashflow'],'exit_reason':g['native_trade']['exit_reason']})
    assert abs(sum(v['episode_pnl_delta'] for v in contributions)+(ep['decline']))<1e-5
    states=list(gz(f/'managed_state.jsonl.gz'));state_groups=defaultdict(list)
    for v in states:state_groups[v['entry_order_id']].append(v)
    pending=[];episodes=[]
    for g in groups:
        obs=state_groups[g['entry_order_id']];blocked=[v for v in obs if v['pending_exit_order_id'] and v['reason_without_pending_gate']]
        relevant=[v for v in obs if v['reason_without_pending_gate']]
        if blocked:
            byorder=defaultdict(list)
            for v in blocked:byorder[v['pending_exit_order_id']].append(v)
            for oid,vals in byorder.items():
                fills=[v for v in g['sells'] if v['order_id']==oid]
                pending.append({'symbol':g['symbol'],'entry_decision_time':g['entry_decision_time'],'order_id':oid,'blocked_observations':len(vals),'first':vals[0],'last':vals[-1],
                    'pending_order_fills':fills,'all_sells':g['sells'],'final_trade_pnl':g['cashflow']})
        overnight=[v for v in obs if dt(v['timestamp']).date()>dt(g['first_fill_time']).date()]
        firstovernight=overnight[0] if overnight else None
        firstbuy=g['buys'][0];prev=[v for v in marks if dt(v['timestamp'])<dt(firstbuy['timestamp'])];entrynav=prev[-1]['nav'] if prev else 1e8
        t=g['native_trade'];risk=g['risk_per_share'];q=g['total_buy_qty']
        episodes.append({'symbol':g['symbol'],'entry_decision_time':g['entry_decision_time'],'entry_order_id':g['entry_order_id'],'net_pnl':g['cashflow'],
            'native_exit_reason':t['exit_reason'],'risk_pct':100*risk/t['entry_price'],'one_r_pct_pre_first_fill_nav':100*q*risk/entrynav,
            'first_next_session_observation':firstovernight,'first_exit_rule_condition':relevant[0] if relevant else None,
            'buys':g['buys'],'sells':g['sells'],'max_position_drawdown_from_entry_pct':100*t['mae']/t['entry_price']})
    result={'label':label,'source':str(f),'source_completion_sha256':n.sha(f/'completion.json'),'exact_parity':proof,
        'all_bar_ledger_reconstruction_max_error':maxerror,'marks':matched,'episode':ep,'peak_state':peak,'trough_state':trough,
        'contributions':sorted(contributions,key=lambda v:v['episode_pnl_delta']),'pending_exit_rule_observations':pending,'trade_execution_paths':episodes,
        'peak_held_positions_delta':sum(v['episode_pnl_delta'] for v in contributions if v['held_at_peak']),
        'post_peak_entries_delta':sum(v['episode_pnl_delta'] for v in contributions if not v['held_at_peak']),
        'max_gross_exposure_pct':max(100*v['gross_exposure']/v['nav'] for v in marks),
        'max_sum_original_risk_pct':max(100*v['sum_original_risk_dollars']/v['nav'] for v in marks)}
    n.write(OUT/(label+'_drawdown.json'),result);n.write(OUT/(label+'_marks.json'),marks)
    print({'label':label,'mdd':ep['drawdown_pct'],'peak_held_delta':result['peak_held_positions_delta'],'post_peak_delta':result['post_peak_entries_delta'],'pending_orders_with_rule_conditions':len(pending),'reconstruction_error':maxerror},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['static','trace','all']);a=p.parse_args()
    if a.action in ('static','all'):static()
    if a.action in ('trace','all'):
        for label in ('literal_carry_original','literal_carry_guard_1p0'):trace(label)
