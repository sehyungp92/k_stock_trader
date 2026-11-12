"""Predefined causal-accounting review after all 0035 accounts are accepted."""
from collections import defaultdict
from datetime import datetime,timedelta
import argparse
import gzip
import json
import math
from pathlib import Path
from olr_selective_half_continuation_v1 import read,sha,write,now


def occupancy(case):
    cash=1e8;q=defaultdict(int);marks={};last=None;nav=1e8;gross=0.
    weighted=seconds=trading_weighted=trading_seconds=0.;context_fills={}
    with gzip.open(case/'minute_events.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            r=json.loads(line);kind=r['event']
            if kind in ('opening_marks','interval_marks','closing_marks'):
                at=datetime.fromisoformat(r['timestamp'])+timedelta(minutes=kind!='opening_marks')
            elif kind=='fill':at=datetime.fromisoformat(r['fill']['timestamp'])
            else:continue
            if last is not None:
                delta=(at-last).total_seconds()
                assert delta>=0,(last,at,kind)
                seconds+=delta;weighted+=delta*gross/nav
                if at.date()==last.date():trading_seconds+=delta;trading_weighted+=delta*gross/nav
            if kind=='fill':
                f=r['fill'];sign=1 if f['side']=='BUY' else -1
                assert math.isclose(cash,r['cash_before'],abs_tol=1e-5)
                cash-=sign*f['qty']*f['price']+f['metadata']['fee'];q[f['symbol']]+=sign*f['qty']
                assert q[f['symbol']]>=0 and math.isclose(cash,r['cash_after'],abs_tol=1e-5)
                if f['reason'] in ('relative_context_reduce','relative_context_exit'):
                    e=f['metadata']['relative_context_evidence'];boundary=datetime.fromisoformat(e['available_at'])
                    assert f['side']=='SELL' and at>boundary
                    key=(f['symbol'],e['available_at'])
                    row=context_fills.setdefault(key,dict(symbol=f['symbol'],available_at=e['available_at'],
                        filled_quantity=0,first_fill=at.isoformat(),last_fill=at.isoformat(),
                        expected_quantity=math.floor(e['quantity_before']*e['fraction']),order_ids=[]))
                    row['filled_quantity']+=f['qty'];row['last_fill']=at.isoformat()
                    if f['order_id'] not in row['order_ids']:row['order_ids'].append(f['order_id'])
                    assert row['filled_quantity']<=row['expected_quantity'] and len(row['order_ids'])==1
            else:marks.update(r['prices'])
            gross=sum(n*marks[s] for s,n in q.items() if n);nav=cash+gross;last=at
    assert not any(q.values())
    assert math.isclose(nav,read(case/'minute_nav_evidence.json')['final_equity'],abs_tol=.01)
    return dict(calendar_time_weighted_gross_nav=weighted/seconds,
        observed_same_day_time_weighted_gross_nav=trading_weighted/trading_seconds,
        calendar_seconds=seconds,observed_same_day_seconds=trading_seconds,actual_context_fills=list(context_fills.values()),
        scope='Previous observed mark held constant between causal events; calendar measure includes overnight/weekend occupancy. Same-day measure includes observed intraday gaps, not an invented continuous tape.')


def run(root):
    a=read(root/'assessment.json')
    out=root/'causal_review_v1'
    out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(at=now(),runner_sha256=sha(__file__),assessment_sha256=sha(root/'assessment.json'),
        method='Actual action lineage and additive account accounting; no new variants or future-labelled rule'))
    result={};occupancies={}
    for name,s in a['summaries'].items():
        case=Path(s['case']);assert sha(case/'completion.json')==s['completion_sha256']
        occupancies[name]=occupancy(case)
        if name=='control':continue
        trades=read(case/'trades.json');actions=[x for x in s['context_decisions'] if x['decision_code']=='EXIT_SUBMITTED']
        targeted={};orders=[]
        for action in actions:
            e=action['metadata']['relative_context_evidence'];at=datetime.fromisoformat(e['available_at'])
            assert at.strftime('%H:%M')=='09:30' and datetime.fromisoformat(action['timestamp'])+timedelta(minutes=5)==at
            assert e['peer_count']>=68 and e['own_return']<0 and e['market_median']>=0
            candidates=[t for t in trades if t['symbol']==action['symbol'] and datetime.fromisoformat(t['entry_fill_time'])<at<datetime.fromisoformat(t['exit_fill_time'])]
            assert len(candidates)==1,(action,len(candidates))
            t=candidates[0];key=(t['symbol'],t['entry_decision_time'][:10]);assert key not in targeted
            targeted[key]=e['available_at'][:10]
            assert action['metadata']['risk_per_share']==t['route_metadata']['risk_per_share']
            orders.append(dict(symbol=key[0],entry_day=key[1],action_day=at.date().isoformat(),qty=action['metadata']['qty'],
                quantity_before=e['quantity_before'],fraction=e['fraction'],original_risk=e['original_risk']))
        pair=a['comparisons']['control_to_'+name];per=pair['accounting']['per_opportunity'];total=pair['accounting']['additive_pnl_accounting']['total']
        groups=defaultdict(float);foreseen={}
        for r in per:
            key=(r['symbol'],r['day']);groups['targeted_opportunities' if key in targeted else 'other_opportunities']+=r['delta_pnl']
        assert math.isclose(sum(groups.values()),total,abs_tol=.01)
        base_trades=read(Path(a['summaries']['control']['case'])/'trades.json')
        for day in ['2025-10-15','2026-01-22','2026-02-04']:
            keys={key for key,value in targeted.items() if value==day}
            foreseen[day]=dict(targets=[r for r in per if (r['symbol'],r['day']) in keys],
                subsequent_same_day_entries=[r for r in per if r['day']==day],
                interpretation='Targeted total changes include quantity/price effects; subsequent entry-day deltas are descriptive and may include earlier path differences.')
        base={(t['symbol'],t['entry_decision_time'][:10]):t for t in base_trades}
        treatment={(t['symbol'],t['entry_decision_time'][:10]):t for t in trades}
        renewals=[]
        for order in orders:
            key=(order['symbol'],order['action_day'])
            if key in treatment:
                trade=treatment[key]
                assert trade['entry_fill_time']>order['action_day']+'T09:30:00+09:00'
                renewals.append(dict(symbol=key[0],day=key[1],entry_at=trade['entry_fill_time'],
                    notional=trade['qty']*trade['entry_price'],net_pnl=trade['net_pnl'],
                    existed_in_control=key in base,prior_entry_day=order['entry_day'],
                    interpretation='Fresh native daily confirmation after morning retirement; incremental account value must include prior recovery sacrificed and all costs.'))
        additional=[dict(symbol=k[0],day=k[1],notional=treatment[k]['qty']*treatment[k]['entry_price'],net_pnl=treatment[k]['net_pnl']) for k in treatment.keys()-base.keys()]
        removed=[dict(symbol=k[0],day=k[1],notional=base[k]['qty']*base[k]['entry_price'],net_pnl=base[k]['net_pnl']) for k in base.keys()-treatment.keys()]
        result[name]=dict(actual_orders=orders,actual_fills=occupancies[name]['actual_context_fills'],
            actions=len(orders),dates=len(set(targeted.values())),
            aggregate_account_delta=total,targeted_vs_other=dict(groups),foreseen_interactions=foreseen,
            added=additional,removed=removed,same_day_native_renewals=renewals,
            limits=['Targeted/other and quantity/unit-payoff decompositions are additive descriptions, not independently identified causal effects.',
                'Full funded policy evidence remains adaptively reused development; no new untouched validation.'])
    write(out/'assessment.json',dict(at=now(),actions_and_funding=result,occupancy=occupancies,
        economic_objective_achieved=False,next='Interpret failed predictions and principal readiness under the prospective criteria; no automatic new experiment.'))
    print(json.dumps(dict(output=str(out),arms=list(result))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();run(args.root)
