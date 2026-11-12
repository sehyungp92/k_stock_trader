import importlib.util
from pathlib import Path
import sys
import pytest

PATH = Path(__file__).resolve().parents[1]/'strategy_kalcb/core/risk_contract_v1.py'
spec=importlib.util.spec_from_file_location('risk_contract_test', PATH)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def exposure():
    return {'order_id':'e','decision_available_at':'2026-01-01T09:30:00+09:00',
        'action':{'symbol':'s','qty':10,'metadata':{'entry_price_ref':100.,'risk_per_share':2.,'stop_price':98.,'daily_atr':10.,'hard_stop_enabled':False}},
        'account_before':{'actual_equity':1000.}}

def fill(eid,minute,side,qty,price):
    return {'symbol':'s','timestamp':f'2026-01-01T09:{minute}:00+09:00','side':side,'qty':qty,'price':price,
            'metadata':{'execution_id':eid,'fee':1.}}

def test_reference_is_not_financial_guarantee():
    e=m.EntryRiskReference.from_exposure(exposure()).evidence()
    assert e['planned_reference_risk_krw']==20 and e['atr_to_reference_risk']==5
    assert not e['guaranteed_loss_bound']

def test_equal_next_open_fill_is_excluded_and_partial_entry_retained():
    r=m.EntryRiskReference.from_exposure(exposure())
    p=m.mark_at_completed_bar(r,[fill('a',30,'BUY',4,100),fill('b',35,'BUY',6,101)],'2026-01-01T09:30:00+09:00',97)
    assert p['held_qty']==4 and p['pending_planned_qty']==6
    assert p['net_marked_pnl_krw']==-13 and p['filled_reference_risk_krw']==8

def test_partial_exit_cash_and_fees_are_retained_once():
    r=m.EntryRiskReference.from_exposure(exposure())
    p=m.mark_at_completed_bar(r,[fill('a',30,'BUY',10,100),fill('b',35,'SELL',5,103)],'2026-01-01T09:35:00+09:00',97)
    assert p['held_qty']==5 and p['gross_marked_pnl_krw']==0 and p['net_marked_pnl_krw']==-2

def test_future_prices_do_not_change_earlier_mark():
    r=m.EntryRiskReference.from_exposure(exposure());a=[fill('a',30,'BUY',10,100)]
    assert m.mark_at_completed_bar(r,a,'2026-01-01T09:30:00+09:00',97)==m.mark_at_completed_bar(r,a+[fill('b',40,'SELL',10,500)],'2026-01-01T09:30:00+09:00',97)

def test_invalid_identity_risk_and_naive_time_fail():
    e=exposure();e['action']['metadata']['risk_per_share']=3
    with pytest.raises(ValueError):m.EntryRiskReference.from_exposure(e)
    with pytest.raises(ValueError):m.timestamp('2026-01-01T09:30:00')

def test_duplicate_execution_and_signal_bar_fail():
    r=m.EntryRiskReference.from_exposure(exposure());f=fill('a',30,'BUY',10,100)
    with pytest.raises(ValueError):m.mark_at_completed_bar(r,[f,f],'2026-01-01T09:30:00+09:00',97)
    with pytest.raises(ValueError):m.mark_at_completed_bar(r,[f],'2026-01-01T09:25:00+09:00',97)

def test_parent_same_clock_exit_is_not_independent_and_recovery_kept():
    r=m.EntryRiskReference.from_exposure(exposure());fs=[fill('a',30,'BUY',10,100),fill('b',40,'SELL',10,105)]
    p={'entry_order_ids':['e'],'entry_fill_time':fs[0]['timestamp'],'exit_fill_time':fs[1]['timestamp'],'net_pnl_krw':48}
    bars=[('2026-01-01T09:30:00+09:00',97),('2026-01-01T09:35:00+09:00',103)]
    x=m.analyze_lifetime(r,p,fs,bars,'2026-01-01T09:35:00+09:00')
    assert x['reference_stop_breach'] and x['before_parent_exit_breach'] is None
    x=m.analyze_lifetime(r,p,fs,bars,'2026-01-01T09:40:00+09:00')
    assert x['observed_after_breach_net_movement_krw']==79
