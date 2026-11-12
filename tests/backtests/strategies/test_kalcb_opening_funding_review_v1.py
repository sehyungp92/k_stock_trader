import pytest
from scripts.kalcb_opening_funding_review_v1 import opening_batches


def sample():
    rows=[]
    for i in range(3):
        rows.append({'order_id':str(i),'decision_available_at':'2026-01-01T09:30:00+09:00',
            'action':{'qty':100,'metadata':{'entry_route':'first30_open_anchor','entry_price_ref':100}},
            'account_before':{'actual_equity':10000,'actual_cash':10000,'buying_power_leverage':2,
                'positions':[],'pending_orders':[{'side':'BUY'}]*i}})
    account={'funding':{'rejected_buy_events':[{'order_id':'2','reason':'buying_power'},
        {'order_id':'2','reason':'buying_power'}]},'positions':[
        {'position_id':'a','entry_order_ids':['0','1'],'net_pnl_krw':25}]}
    return {'entry_exposures':rows},account


def test_orders_rejections_and_lifetime_pnl_are_not_duplicated():
    e,a=sample();r=opening_batches(e,a)
    assert r['summary']['planned_over_2x_batches']==1
    assert r['summary']['unique_buying_power_rejections']==1
    assert r['batches'][0]['actual_lifetime_net_krw_once']==25
    assert r['batches'][0]['filled_position_lifetimes']==1


def test_duplicate_exposure_fails():
    e,a=sample();e['entry_exposures'].append(e['entry_exposures'][0])
    with pytest.raises(ValueError,match='Duplicate'):opening_batches(e,a)


def test_nonflat_account_is_not_silently_counted_as_flat_batch():
    e,a=sample();e['entry_exposures'][1]['account_before']['actual_cash']=9000
    with pytest.raises(ValueError,match='no longer flat'):opening_batches(e,a)


def test_future_outcome_changes_do_not_change_demand():
    e,a=sample();before=opening_batches(e,a)['summary']
    a['positions'][0]['net_pnl_krw']=-1e9
    assert opening_batches(e,a)['summary']==before
