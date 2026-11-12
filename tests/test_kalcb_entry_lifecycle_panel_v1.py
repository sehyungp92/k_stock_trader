from copy import deepcopy
import pytest
from scripts.kalcb_entry_lifecycle_panel_v1 import cash_partition, prefix_features, state_at35

DAY='2026-04-01'
AT=DAY+'T09:35:00+09:00'


def fill(side,q,p,t,fee=1):
    return {'side':side,'qty':q,'price':p,'timestamp':DAY+'T'+t+':00+09:00',
            'order_id':'buy' if side=='BUY' else 'sell','metadata':{'fee':fee}}


def test_cash_partition_excludes_same_clock_buys_and_sells():
    fills=[fill('BUY',5,100,'09:30'),fill('BUY',5,110,'09:35'),fill('SELL',10,120,'10:00')]
    p=cash_partition(fills,AT,105)
    assert p['quantity_at35']==5
    assert p['marked_net_at35_krw']==24
    assert p['net_reconstructed_krw']==147
    assert p['remaining_net_after35_krw']==123
    assert p['terminal_inventory_reconstructed']==0
    assert cash_partition([fill('BUY',5,100,'09:30'),fill('SELL',5,110,'09:35')],AT,105)['quantity_at35']==5


def test_pending_does_not_use_future_cancellation_or_eventual_fills():
    order={'side':'BUY','qty':10,'order_id':'buy','submitted_at':DAY+'T09:25:00+09:00',
           'metadata':{'decision_available_at':DAY+'T09:30:00+09:00'}}
    e={'orders':[{'event':'submit','order':order},{'event':'cancel','order':order,'timestamp':DAY+'T10:15:00+09:00'}],
       'fills':[fill('BUY',5,100,'09:30'),fill('BUY',5,110,'09:35')]}
    assert state_at35(e,AT,5)['pending_qty_at35']==5
    e['orders'][1]['timestamp']=DAY+'T09:30:00+09:00'
    assert state_at35(e,AT,5)['pending_qty_at35']==0


def test_future_prefix_poisoning_and_distinct_remaining_value():
    bars=[{'timestamp':DAY+f'T09:{i*5:02}:00+09:00','open':100,'high':102,'low':98,'close':101,'volume':100} for i in range(7)]
    values,states=prefix_features(bars[:6],bars[6],DAY,10,1)
    assert len(values)==12
    assert not states['below_opening_low']
    future=deepcopy(bars[6]);future['timestamp']=AT
    with pytest.raises(ValueError,match='Future'):
        prefix_features(bars[:6],future,DAY,10,1)
    # A profitable whole trade can have negative remaining value after an early mark.
    p=cash_partition([fill('BUY',10,100,'09:30'),fill('SELL',10,110,'10:00')],AT,120)
    assert p['net_reconstructed_krw']>0 and p['remaining_net_after35_krw']<0
