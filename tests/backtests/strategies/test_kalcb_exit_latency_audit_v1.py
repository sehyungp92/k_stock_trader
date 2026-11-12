import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('latency', Path(__file__).parents[3]/'backtests/strategies/kalcb/exit_latency_audit_v1.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def f(side, qty, price, at, execution):
    return {'side':side,'qty':qty,'price':price,'timestamp':at,'symbol':'A',
            'metadata':{'execution_id':execution,'fee':1.0}}


def test_partial_execution_is_one_lifetime():
    fills=[f('BUY',2,100,'2026-04-01T09:30:00+09:00','a'),f('BUY',1,101,'2026-04-01T09:35:00+09:00','b'),
           f('SELL',1,99,'2026-04-01T09:40:00+09:00','c'),f('SELL',2,98,'2026-04-01T09:45:00+09:00','d')]
    p={'position_id':'A/2026-04-01T09:30:00+09:00','qty':3,'gross_pnl_krw':-6,'fees_krw':4,'net_pnl_krw':-10}
    assert len(m.lifetimes(fills,[p]))==1
    with pytest.raises(ValueError,match='Repeated physical'):m.lifetimes(fills[:2]+[fills[1]]+fills[2:],[p])


def test_same_clock_next_bar_fill_not_in_completed_decision():
    rows=[f('BUY',2,100,'2026-04-01T09:30:00+09:00','a'),f('SELL',2,50,'2026-04-01T09:40:00+09:00','b')]
    marks={('2026-04-01T09:35:00+09:00','A'):90,('2026-04-01T09:40:00+09:00','A'):1}
    p=m.decision_snapshot({'symbol':'A'},rows,'2026-04-01T09:40:00+09:00',marks)
    assert p['held_qty']==2 and p['gross_marked_pnl_krw']==-20
    marks[('2026-04-01T09:40:00+09:00','A')]=10000
    assert m.decision_snapshot({'symbol':'A'},rows,'2026-04-01T09:40:00+09:00',marks)==p


def test_missing_completed_close_cannot_use_future():
    with pytest.raises(ValueError,match='Missing completed'):
        m.completed_mark('A','2026-04-01T09:40:00+09:00',{('2026-04-01T09:40:00+09:00','A'):90})


def test_raw_close_account_and_fee_poison():
    rows=[f('BUY',2,100,'2026-04-01T09:30:00+09:00','a'),f('SELL',2,98,'2026-04-01T09:40:00+09:00','b')]
    curve=[('2026-04-01T09:30:00+09:00',297),('2026-04-01T09:40:00+09:00',294)]
    marks={('2026-04-01T09:30:00+09:00','A'):99}
    assert m.reconcile_curve(rows,curve,marks,300)==2
    rows[1]['metadata']['fee']=2
    with pytest.raises(ValueError,match='equity mark'):m.reconcile_curve(rows,curve,marks,300)


def test_unknown_lifetime_rejected():
    with pytest.raises(ValueError,match='Unknown'):
        m.lifetimes([f('BUY',1,100,'2026-04-01T09:30:00+09:00','a')],[])


def test_unfilled_exit_intent_precedes_actual_fill_without_double_count():
    start='2026-04-01T09:30:00+09:00';end='2026-04-01T09:40:00+09:00'
    rows=[f('BUY',2,100,start,'a'),f('SELL',2,98,end,'b')]
    for i,row in enumerate(rows):
        row['order_id']=str(i);row['metadata']['base_price']=row['price']
    p={'position_id':'A/'+start,'symbol':'A','route':'first30_open_anchor','entry_fill_time':start,
       'exit_fill_time':end,'qty':2,'entry_cost_krw':200,'entry_equity_krw':100000000,
       'gross_pnl_krw':-4,'fees_krw':2,'net_pnl_krw':-6}
    orders=[{'order_id':'cancelled','symbol':'A','side':'SELL','order_type':'MARKET','qty':2,
             'reason':'signal','submitted_at':'2026-04-01T09:30:00+09:00',
             'metadata':{'order_role':'EXIT','decision_available_at':'2026-04-01T09:35:00+09:00'}},
            {'order_id':'1','symbol':'A','side':'SELL','order_type':'MARKET','qty':2,'reason':'signal',
             'submitted_at':'2026-04-01T09:35:00+09:00','metadata':{'order_role':'EXIT','decision_available_at':end}}]
    marks={(start,'A'):99,('2026-04-01T09:35:00+09:00','A'):90}
    result=m.account([p],rows,orders,[(start,99999997),(end,99999994)],marks)
    x=result['positions'][0]
    assert result['unique_sell_submissions_bound']==2
    assert x['first_full_exit']['order_id']=='cancelled'
    assert x['before_first_full_exit_gross_krw']==-2
    assert x['after_first_full_exit_gross_krw']==-2
