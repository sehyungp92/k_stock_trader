"""Summarize observed execution mechanisms; counterfactuals are labeled arithmetic."""
from scripts.olr_drawdown_diagnosis_v1 import *


def main():
    selected=n.read(OUT/'literal_carry_guard_1p0_drawdown.json');f=ROOT/'literal_carry_guard_1p0'
    orders={};capacity=[];expired=[]
    for row in gz(f/'native_journal.jsonl.gz'):
        if row['event']=='submit':orders[row['order']['order_id']]=row['order']
        if row['event']=='capacity':capacity.append(row)
        if row['event']=='expire' and row['order']['order_type']=='CLOSE_AUCTION' and row['order']['side']=='SELL':expired.append(row)
    pending=[]
    for v in selected['pending_exit_rule_observations']:
        order=orders[v['order_id']]
        pending.append({'symbol':v['symbol'],'order_id':v['order_id'],'order_type':order['order_type'],'reason':order['reason'],
            'blocked_observations':v['blocked_observations'],'first_condition':v['first']['timestamp'],'first_condition_close_r':v['first']['close_r']})
    study=n.read(b.ENGINES/'oos/ready.json')['study'];db=Path(study['code'])/study['database'];con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    def bar(sym,stamp):
        row=con.execute('select ts,o,h,l,c,v from bars where symbol=? and ts=?',(sym,stamp)).fetchone()
        return dict(zip(('timestamp','open','high','low','close','volume'),row)) if row else None
    group=next(v for v in selected['trade_execution_paths'] if v['symbol']=='042700' and v['entry_decision_time'].startswith('2026-05-14'))
    queued=next(o for o in orders.values() if o['symbol']=='042700' and o['side']=='SELL' and o['order_type']=='CLOSE_AUCTION' and o['submitted_at'].startswith('2026-05-15'))
    trigger=bar('042700','2026-05-15T15:00:00+09:00');nextbar=bar('042700','2026-05-15T15:05:00+09:00');auction=bar('042700','2026-05-15T15:30:00+09:00')
    qty=sum(v['qty'] for v in group['buys']);price=nextbar['open']*(1-.0005);exitcash=qty*price*(1-.0022);oldcash=sum(v['qty']*v['price']-v['fee'] for v in group['sells']);benefit=exitcash-oldcash
    auction_case={'symbol':'042700','entry_date':'2026-05-14','qty':qty,'queued_order':queued,'guard_condition_bar':trigger,'next_bar':nextbar,'auction':auction,
        'next_bar_participation_budget':int(trigger['volume']*.01),'auction_participation_budget':int(auction['volume']*.01),'actual_sells':group['sells'],
        'price_limit_prevents_auction_fill':auction['close']<queued['limit_price'],'arithmetic_replacement_price':price,
        'arithmetic_replacement_pnl':group['net_pnl']+benefit,'arithmetic_improvement':benefit,'arithmetic_peak_normalized_points':100*benefit/selected['episode']['peak_nav'],
        'counterfactual_status':'Same-quantity next-eligible-bar arithmetic with original cost/participation assumptions and sufficient modeled capacity. Not a native cancel/replace policy replay or a guarantee of live liquidity; excludes downstream capital effects.'}
    critical=[]
    for sym,day in [('042660','2026-05-27'),('403870','2026-06-15'),('055550','2026-06-05'),('329180','2026-05-27')]:
        g=next(v for v in selected['trade_execution_paths'] if v['symbol']==sym and v['entry_decision_time'].startswith(day));condition=g['first_exit_rule_condition'];obs=g['first_next_session_observation'];q=sum(v['qty'] for v in g['buys']);entry=sum(v['qty']*v['price'] for v in g['buys'])/q
        entry_fee=sum(v['fee'] for v in g['buys']);o=obs['bar']['open'];close=obs['bar']['close'];weightedexit=sum(v['qty']*v['price'] for v in g['sells'])/q
        overnight=q*(o-entry);firstbar=q*(close-o);rest=q*(weightedexit-close);fees=entry_fee+sum(v['fee'] for v in g['sells'])
        assert abs(overnight+firstbar+rest-fees-g['net_pnl'])<1e-5
        critical.append({'symbol':sym,'entry_date':day,'qty':q,'risk_pct':g['risk_pct'],'one_r_account_pct':g['one_r_pct_pre_first_fill_nav'],
            'entry_price':entry,'first_next_session_bar':obs['bar'],'net_pnl':g['net_pnl'],'entry_to_first_open_pct':100*(o/entry-1),
            'first_open_to_close_pct':100*(close/o-1),'first_close_r':obs['close_r'],'trigger_bar':condition['timestamp'],'trigger_close_r':condition['close_r'],
            'sells':g['sells'],'entry_to_first_open_component':overnight,'first_bar_open_to_close_component':firstbar,'first_close_to_exit_component':rest,'fees':fees,
            'price_bridge_caveat':'Exact algebra for these four positions, all quantity still held through first bar. Not guaranteed exit prices or an executable policy counterfactual.'})
    import yaml
    halt_path=Path(study['code'])/'kis_core/data/krx_intraday_halts.yaml';halts=yaml.safe_load(halt_path.read_text(encoding='utf-8'))
    june8=next(v for v in halts['halts'] if v['date']=='2026-06-08' and v['market']=='KOSPI')
    peak=dt(selected['episode']['peak_time']);trough=dt(selected['episode']['trough_time']);late=[v for v in selected['trade_execution_paths'] if dt(v['buys'][0]['timestamp'])>peak]
    contributions=selected['contributions'];aggregate=[]
    categories=[('Two shipbuilding losses',lambda v:v['symbol'] in ('042660','329180') and v['entry_decision_time'].startswith('2026-05-27')),
        ('403870 June loss',lambda v:v['symbol']=='403870'),('055550 June loss',lambda v:v['symbol']=='055550'),
        ('Giveback on three profitable peak holdings',lambda v:v['held_at_peak'] and v['final_trade_pnl']>0),
        ('042700 auction-fallback loss',lambda v:v['symbol']=='042700'),('010950 temporary loss at trough',lambda v:v['held_at_trough'])]
    seen=set()
    for name,test in categories:
        rows=[v for v in contributions if test(v)];seen.update(v['id'] for v in rows);delta=sum(v['episode_pnl_delta'] for v in rows)
        aggregate.append({'component':name,'nav_change':delta,'drawdown_points':-100*delta/selected['episode']['peak_nav'],'positions':[v['symbol'] for v in rows]})
    rows=[v for v in contributions if v['id'] not in seen];delta=sum(v['episode_pnl_delta'] for v in rows);aggregate.append({'component':'Other positions, net','nav_change':delta,'drawdown_points':-100*delta/selected['episode']['peak_nav']})
    n.write(OUT/'mechanism_evidence.json',{'created_utc':s.utc(),'selected_episode_attribution':aggregate,'critical_execution_paths':critical,'pending_order_summary':pending,
        'auction_priority_case':auction_case,'closing_auction_expiries':expired,'june8_halt_reference':june8,'june8_halt_reference_file_sha256':n.sha(halt_path),
        'june8_official_broker_notice':'https://www.kakaopaysec.com/customer/notice/dynamicBoardPageDetail.do?id=7275',
        'june8_capacity_events':[v for v in capacity if v['order']['symbol']=='055550' and v['timestamp'].startswith('2026-06-08')],
        'post_peak_entries':{'positions':len(late),'winners':sum(v['net_pnl']>0 for v in late),'final_pnl':sum(v['net_pnl'] for v in late),'caveat':'Peak boundary is hindsight-defined and is not a proposed trading trigger.'}})
    print({'mechanism_evidence':'complete','auction_priority_arithmetic_benefit':benefit,'episode_components':aggregate},flush=True)


if __name__=='__main__':main()
