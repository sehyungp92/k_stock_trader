import copy
import pytest
from backtests.strategies.kalcb.checkpoint_hold_bridge_v1 import compare, fills_canonical, account_from_fills


def fixture():
    fills=[{'symbol':'A','side':'BUY','qty':10,'price':100.,'timestamp':'2026-01-05 09:30:00+09:00','reason':'entry','order_id':'x'},
           {'symbol':'A','side':'SELL','qty':10,'price':110.,'timestamp':'2026-01-05 10:00:00+09:00','reason':'exit','order_id':'y'}]
    costs={'commission_bps':2.,'tax_bps_on_sell':18.}
    a=account_from_fills(fills,costs)
    return fills,costs,{'live_qty':0,'fees_krw':a['fees'],'net_pnl_krw':a['net_cash_flow']}


def test_exact_bridge_ignores_order_identifier_and_timestamp_separator():
    f,c,p=fixture();other=copy.deepcopy(f)
    for r in other:r['order_id']='different';r['timestamp']=r['timestamp'].replace(' ','T')
    assert compare(f,other,'A','2026-01-05T09:45:00+09:00',c,p)['hold_lifetime_bridge']


@pytest.mark.parametrize('field,value',[('qty',9),('price',111.),('timestamp','2026-01-05T10:05:00+09:00'),('reason','other')])
def test_future_change_is_not_a_bridge_or_predecision_feature(field,value):
    f,c,p=fixture();other=copy.deepcopy(f);other[1][field]=value
    d=compare(f,other,'A','2026-01-05T09:45:00+09:00',c,p)
    assert d['exact_fill_prefix'] and not d['hold_lifetime_bridge']


def test_next_bar_is_not_in_prefix():
    f,_,_=fixture();f[1]['timestamp']='2026-01-05T09:45:00+09:00'
    assert len(fills_canonical(f,'A','2026-01-05T09:45:00+09:00'))==1


@pytest.mark.parametrize('field',['fees_krw','net_pnl_krw','live_qty'])
def test_wrong_native_account_rejected(field):
    f,c,p=fixture();p[field]+=1
    with pytest.raises(ValueError):compare(f,f,'A','2026-01-05T09:45:00+09:00',c,p)


def test_unresolved_reference_is_not_terminal_cash_pnl():
    f,c,p=fixture();d=compare(f,f[:1],'A','2026-01-05T09:45:00+09:00',c,p)
    assert not d['hold_lifetime_bridge'] and d['reference_terminal_inventory']==10


def test_tax_applies_only_to_sales():
    f,c,p=fixture();assert account_from_fills(f,c)['fees']==pytest.approx(.2+2.2)
