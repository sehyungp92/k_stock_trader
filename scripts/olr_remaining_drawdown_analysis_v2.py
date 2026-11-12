"""Descriptive causal-time screens, not counterfactual strategy performance."""
from __future__ import annotations
import argparse
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import sqlite3
import numpy as np
from scripts.olr_remaining_drawdown_v2 import ROOT, OUT, PRIOR, SELECTED, read, write, sha

def stamp(v): return datetime.fromisoformat(v)
def gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream: yield json.loads(line)
def pctiles(v):
    return {str(q): float(np.percentile(v,q)) for q in (0,25,50,75,90,95,100)} if v else {}
def summary(rows):
    winners=[r for r in rows if r['net_pnl']>0]; losers=[r for r in rows if r['net_pnl']<0]
    return {'trades':len(rows),'winners':len(winners),'losers':len(losers),
        'net_pnl':sum(r['net_pnl'] for r in rows),'winner_pnl':sum(r['net_pnl'] for r in winners),
        'loser_pnl':sum(r['net_pnl'] for r in losers)}

def static():
    from scripts.olr_drawdown_diagnosis_v1 import episode
    data={}
    for w in ('train','oos'):
        f=PRIOR/'cases'/SELECTED/w/'100000000_baseline'
        comp=read(f/'completion.json')
        for name,h in comp['artifacts'].items(): assert sha(f/name)==h,name
        ts=read(f/'trades.json'); eq=read(f/'equity.json'); times=[stamp(v[0]) for v in eq]
        rows=[]
        for t in ts:
            m=t['route_metadata']; risk=m['risk_per_share']; i=bisect.bisect_left(times,stamp(t['entry_fill_time']))-1
            nav=eq[i][1] if i>=0 else 1e8
            rows.append({'symbol':t['symbol'],'entry':t['entry_fill_time'],'exit':t['exit_fill_time'],
                'qty':t['qty'],'entry_price':t['entry_price'],'net_pnl':t['net_pnl'],'exit_reason':t['exit_reason'],
                'risk':risk,'risk_pct':100*risk/t['entry_price'],'risk_account_pct':100*risk*t['qty']/nav,
                'return_pct':100*t['net_pnl']/(t['qty']*t['entry_price']),
                'mfe_r':t['mfe']/risk,'mae_r':t['mae']/risk,
                'rank':m.get('candidate_rank'),'score':m.get('candidate_score'),
                'sector':m.get('sector'),'score_band':m.get('afternoon_score_band_rule'),
                'allocation_limit':m.get('allocation_binding_limit'),
                'scenario':m.get('scenario_loss_budget'),
                'sparse_history':m.get('capacity_forecast',{}).get('exit',{}).get('status')=='insufficient_history'})
        cohorts={}
        for field in ('sector','score_band','exit_reason','allocation_limit','sparse_history'):
            cohorts[field]={str(v):summary([r for r in rows if r[field]==v]) for v in sorted(set(r[field] for r in rows),key=str)}
        cohorts['risk_pct']={f'{lo}-{hi}':summary([r for r in rows if lo<=r['risk_pct']<hi]) for lo,hi in [(0,4),(4,6),(6,100)]}
        adverse={}
        for cut in (.4,.6,.8,1.,1.25):
            adverse[str(cut)]=summary([r for r in rows if r['mae_r']<=-cut])
        risk={k:{f:pctiles([r[f] for r in subset]) for f in ('risk_pct','risk_account_pct','return_pct','mae_r','mfe_r')}
              for k,subset in [('all',rows),('winners',[r for r in rows if r['net_pnl']>0]),('losers',[r for r in rows if r['net_pnl']<0])]}
        data[w]={'source':str(f),'completion_sha256':sha(f/'completion.json'),'summary':summary(rows),
                 'risk_distributions':risk,'cohorts':cohorts,'whole_trade_mae_hindsight_overlap':adverse,
                 'intraday_episode':episode(eq),'rows':rows}
    write(OUT/'static.json',data)
    print(json.dumps({w:{'summary':data[w]['summary'],'risk_median':data[w]['risk_distributions']['all']['risk_pct']['50'],
                        'mae_overlap':data[w]['whole_trade_mae_hindsight_overlap']} for w in data},indent=2))

def paths():
    output={}
    for w in ('train','oos'):
        f=ROOT/w; diag=read(OUT/(w+'_drawdown.json')); static=read(OUT/'static.json')[w]
        paths=diag['trade_execution_paths']; states=defaultdict(list); orders={}; capacity=[]
        for row in gz(f/'managed_state.jsonl.gz'): states[row['entry_order_id']].append(row)
        # The inherited state list can retain earlier-session bars. Count distinct
        # eligible bars on the current date explicitly for these new hypotheses.
        for observations in states.values():
            seen_by_day=defaultdict(set)
            for row in observations:
                day=row['timestamp'][:10]
                if row['eligible_without_pending_gate']:seen_by_day[day].add(row['timestamp'])
                row['observed_session_bar_count']=len(seen_by_day[day])
        for row in gz(f/'native_journal.jsonl.gz'):
            if row['event']=='submit': orders[row['order']['order_id']]=row['order']
            if row['event']=='capacity': capacity.append(row)
        tests={
            'close_minus_0p4R': lambda v,g: v['close_r']<=-.4 and v['mfe_r']<4,
            'close_minus_0p6R': lambda v,g: v['close_r']<=-.6 and v['mfe_r']<4,
            'close_minus_0p8R': lambda v,g: v['close_r']<=-.8 and v['mfe_r']<4,
            'entry_day_weak_no_followthrough': lambda v,g: v['timestamp'][:10]==g['buys'][0]['timestamp'][:10] and v['observed_session_bar_count']>=3 and v['close_r']<=-.4 and v['mfe_r']<.5,
            'next_day_weak_no_followthrough': lambda v,g: v['timestamp'][:10]>g['buys'][0]['timestamp'][:10] and v['observed_session_bar_count']>=3 and v['close_r']<=-.4 and v['mfe_r']<.5,
            'mature_profit_fade_1R': lambda v,g: v['timestamp'][:10]>g['buys'][0]['timestamp'][:10] and v['mfe_r']>=2 and v['close_r']<=v['mfe_r']-1,
        }
        screens={name:[] for name in tests}
        detailed=[]
        for g in paths:
            obs=states[g['entry_order_id']]
            # A hit must be available while no market liquidation is already queued.
            def actionable(v):
                oid=v['pending_exit_order_id']
                return v['eligible_without_pending_gate'] and (not oid or (orders[oid]['order_type']=='CLOSE_AUCTION' and orders[oid]['reason']=='next_close_exit'))
            native=next((v for v in obs if actionable(v) and v['reason_without_pending_gate']),None)
            native_time=stamp(native['timestamp']) if native else stamp(g['sells'][0]['submitted_at'])
            base=next(x for x in static['rows'] if x['symbol']==g['symbol'] and stamp(x['entry'])==stamp(g['buys'][0]['timestamp']))
            for name,test in tests.items():
                hit=next((v for v in obs if actionable(v) and test(v,g)),None)
                if hit:
                    screens[name].append({**base,'hit_time':hit['timestamp'],'hit_close_r':hit['close_r'],'hit_mfe_r':hit['mfe_r'],
                        'native_condition_time':native_time.isoformat(),
                        'strictly_earlier':stamp(hit['timestamp'])<native_time,
                        'advance_minutes':(native_time-stamp(hit['timestamp'])).total_seconds()/60,
                        'hit_bar':hit['bar']})
            before=[v for v in obs if v['timestamp'][:10]==g['buys'][0]['timestamp'][:10]]
            first=g['first_next_session_observation']
            order=orders[g['sells'][0]['order_id']]
            detailed.append({**base,'entry_day_last':before[-1] if before else None,
                'entry_day_min_close_r':min((v['close_r'] for v in before),default=None),
                'entry_day_max_mfe_r':max((v['mfe_r'] for v in before),default=None),
                'first_next_session':first,'first_actionable_native_condition':native,
                'exit_order':order,'sells':g['sells'],
                'next_session_capacity_events':[v for v in capacity if first and v['order']['symbol']==g['symbol'] and v['timestamp'][:10]==first['timestamp'][:10]]})
        pending=[]
        for v in diag['pending_exit_rule_observations']:
            o=orders[v['order_id']]
            pending.append({'symbol':v['symbol'],'entry':v['entry_decision_time'],'order_id':v['order_id'],
                'order_type':o['order_type'],'reason':o['reason'],'observations':v['blocked_observations'],
                'first_time':v['first']['timestamp'],'native_sells':v['all_sells']})
        output[w]={'screens':{name:{'all_hits':summary(rows),'earlier_hits':summary([r for r in rows if r['strictly_earlier']]),
                       'rows':rows} for name,rows in screens.items()},'trades':detailed,'pending_conditions':pending,
                       'interpretation':'Observed-path decision-time triggers. Affected winner PnL is exposure at risk, not forecast loss; affected losing PnL is not recoverable benefit. These are not native counterfactual returns or DD.'}
    write(OUT/'paths_and_screens.json',output)
    print(json.dumps({w:{k:v['earlier_hits'] for k,v in output[w]['screens'].items()} for w in output},indent=2))

def mechanisms():
    d=read(OUT/'oos_drawdown.json'); rows=d['contributions']; used=set(); attribution=[]
    categories=[('Shipbuilding pair on 28 May',lambda v:v['symbol'] in ('042660','329180') and v['entry_decision_time'].startswith('2026-05-27')),
        ('403870 on 16 June',lambda v:v['symbol']=='403870'),
        ('055550 on 8 June',lambda v:v['symbol']=='055550'),
        ('Three profitable peak holdings',lambda v:v['held_at_peak'] and v['final_trade_pnl']>0),
        ('042700 residual loss',lambda v:v['symbol']=='042700'),
        ('010950 temporary loss at trough',lambda v:v['held_at_trough'])]
    for title,test in categories:
        selected=[v for v in rows if test(v)]; assert not used.intersection(v['id'] for v in selected)
        used.update(v['id'] for v in selected); delta=sum(v['episode_pnl_delta'] for v in selected)
        attribution.append({'component':title,'nav_change':delta,'drawdown_points':-100*delta/d['episode']['peak_nav'],
            'share_decline_pct':-100*delta/d['episode']['decline']})
    remaining=sum(v['episode_pnl_delta'] for v in rows if v['id'] not in used)
    attribution.append({'component':'Other positions, net','nav_change':remaining,
        'drawdown_points':-100*remaining/d['episode']['peak_nav'],'share_decline_pct':-100*remaining/d['episode']['decline']})
    assert abs(sum(v['nav_change'] for v in attribution)+d['episode']['decline'])<1e-5
    obs=defaultdict(list)
    for row in gz(ROOT/'oos/managed_state.jsonl.gz'):obs[row['entry_order_id']].append(row)
    bridges=[]
    for sym,day in [('042660','2026-05-27'),('329180','2026-05-27'),('403870','2026-06-15'),('055550','2026-06-05'),('034730','2026-05-11')]:
        g=next(v for v in d['trade_execution_paths'] if v['symbol']==sym and v['entry_decision_time'].startswith(day))
        q=sum(v['qty'] for v in g['buys']);entry=sum(v['qty']*v['price'] for v in g['buys'])/q
        before=[v for v in obs[g['entry_order_id']] if v['timestamp'].startswith(day)]
        first=g['first_next_session_observation']; ec=before[-1]['bar']['close']; o=first['bar']['open']; c=first['bar']['close']
        assert first['qty']==q and all(stamp(v['timestamp'])>stamp(first['timestamp']) for v in g['sells'])
        avg=sum(v['qty']*v['price'] for v in g['sells'])/q
        fees=sum(v['fee'] for v in g['buys']+g['sells'])
        bridge={'entry_to_entry_day_close':q*(ec-entry),'overnight_gap':q*(o-ec),
                'first_bar':q*(c-o),'first_close_to_actual_exit':q*(avg-c),'fees':-fees}
        assert abs(sum(bridge.values())-g['net_pnl'])<1e-5
        bridges.append({'symbol':sym,'entry_day':day,'qty':q,'net_pnl':g['net_pnl'],
            'entry_price':entry,'entry_day_close':ec,'next_open':o,'next_first_close':c,
            'entry_day_min_close_r':min(v['close_r'] for v in before),
            'entry_day_last_close_r':before[-1]['close_r'],'first_close_r':first['close_r'],
            'first_trigger_bar':g['first_exit_rule_condition']['timestamp'],
            'actual_sells':g['sells'],'bridge':bridge,
            'interpretation':'Exact same-quantity price bridge, including actual fees. Not an executable alternative. First-close-to-exit includes market movement, not just avoidable latency. June 8 source bar overlaps the market halt.'})
    marks=read(OUT/'oos_marks.json'); exposure=[]
    for day in ['2026-05-14','2026-05-27','2026-06-05','2026-06-15']:
        row=[v for v in marks if v['timestamp'].startswith(day)][-1]
        exposure.append({'day':day,'nav':row['nav'],'gross_pct':100*row['gross_exposure']/row['nav'],
            'original_risk_pct':100*row['sum_original_risk_dollars']/row['nav'],'held':row['held']})
    write(OUT/'mechanisms.json',{'attribution':attribution,'price_bridges':bridges,'selected_close_exposures':exposure,
        'unchanged_peak_trough_improvement_required':{str(target):d['episode']['peak_nav']*(1-target/100)-d['episode']['trough_nav'] for target in (12,10,8)},
        'caveat':'Fixed original peak/trough arithmetic is not a counterfactual policy drawdown; changed exits can move both endpoints.'})
    print(json.dumps({'attribution':attribution,'bridges':[{k:v[k] for k in ['symbol','net_pnl','bridge','first_trigger_bar']} for v in bridges],
                      'exposures':exposure},indent=2))

def price_screen():
    """Restricted isolated sale arithmetic where known next-bar capacity suffices.

    This is deliberately NOT a full counterfactual replay. Unsupported cases are
    excluded, not optimistically assigned a stop-price or unlimited-size fill.
    """
    import yaml
    all_screens=read(OUT/'paths_and_screens.json');result={}
    for w in ('train','oos'):
        study=read(PRIOR/'runtime'/SELECTED/w/'baseline/study.json');code=Path(study['code'])
        halt_days={str(v['date']) for v in yaml.safe_load((code/'kis_core/data/krx_intraday_halts.yaml').read_text(encoding='utf-8'))['halts']}
        db=code/study['database']; con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
        paths=read(OUT/(w+'_drawdown.json'))['trade_execution_paths']; per_rule={}
        for name,screen in all_screens[w]['screens'].items():
            accepted=[];excluded=[]
            for row in screen['rows']:
                if not row['strictly_earlier']:continue
                g=next(v for v in paths if v['symbol']==row['symbol'] and stamp(v['buys'][0]['timestamp'])==stamp(row['entry']))
                at=stamp(row['hit_time']);next_at=at+timedelta(minutes=5);reasons=[]
                if any(stamp(v['timestamp'])>at for v in g['buys']):reasons.append('entry_not_complete')
                if any(stamp(v['timestamp'])<=at for v in g['sells']):reasons.append('already_partially_sold')
                if at.date().isoformat() in halt_days:reasons.append('halt_date_excluded')
                if not '09:05'<=next_at.strftime('%H:%M')<='15:15':reasons.append('outside_continuous_open_interval')
                nextbar=con.execute('select ts,o,h,l,c,v from bars where symbol=? and ts=?',(row['symbol'],next_at.isoformat())).fetchone()
                if nextbar is None:reasons.append('no_contiguous_next_bar')
                capacity=int(row['hit_bar']['volume']*.01)
                if capacity<row['qty']:reasons.append('insufficient_known_one_bar_capacity')
                if reasons:
                    excluded.append({'symbol':row['symbol'],'entry':row['entry'],'hit':row['hit_time'],'reasons':reasons});continue
                price=float(nextbar[1])*(1-.0005)
                hypothetical_net_sale=row['qty']*price*(1-.0022)
                actual_net_sale=sum(v['qty']*v['price']-v['fee'] for v in g['sells'])
                change=hypothetical_net_sale-actual_net_sale
                accepted.append({'symbol':row['symbol'],'entry':row['entry'],'hit':row['hit_time'],
                    'next_open_time':next_at.isoformat(),'known_capacity_shares':capacity,'qty':row['qty'],
                    'next_open':float(nextbar[1]),'hypothetical_sale_price':price,'native_pnl':g['net_pnl'],
                    'isolated_full_exit_pnl':g['net_pnl']+change,'isolated_pnl_change':change,
                    'baseline_winner':g['net_pnl']>0})
            per_rule[name]={'supported_trades':len(accepted),'excluded_trades':len(excluded),
                'isolated_total_pnl_change':sum(r['isolated_pnl_change'] for r in accepted),
                'isolated_change_on_baseline_winners':sum(r['isolated_pnl_change'] for r in accepted if r['baseline_winner']),
                'baseline_losers_made_worse':sum(r['isolated_pnl_change']<0 and not r['baseline_winner'] for r in accepted),
                'winners_changed_to_losses':sum(r['isolated_full_exit_pnl']<0 and r['baseline_winner'] for r in accepted),
                'rows':accepted,'excluded':excluded}
        con.close();result[w]=per_rule
    result['interpretation']='Restricted unchanged-quantity sale arithmetic, not a native strategy replay or predicted account return/DD. Includes only complete entries, unsold positions, non-halt dates, contiguous next continuous-session opens and enough known prior-bar capacity for all shares at 1% participation. Uses next open less 5bp slippage and 22bp sell costs. Excludes replacement/cancellation latency, changed future orders, reinvestment and compounding. Unsupported cases must be native-replayed; do not assume their effect is zero.'
    write(OUT/'isolated_price_screens.json',result)
    print(json.dumps({w:{k:{kk:v[kk] for kk in ('supported_trades','excluded_trades','isolated_total_pnl_change','isolated_change_on_baseline_winners','baseline_losers_made_worse','winners_changed_to_losses')} for k,v in result[w].items()} for w in ('train','oos')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['static','paths','mechanisms','price_screen']);a=p.parse_args()
    {'static':static,'paths':paths,'mechanisms':mechanisms,'price_screen':price_screen}[a.action]()
