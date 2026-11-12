import importlib.util
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/kalcb_staged_support_forward_v1.py'
s=importlib.util.spec_from_file_location('forward_split_under_test',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def fill(t,side,q,price,fee=0):
    return {'timestamp':'2025-08-08T'+t+'+09:00','side':side,'qty':q,'price':price,'metadata':{'fee':fee}}


def test_next_open_partial_is_future_and_fees_once():
    fs=[fill('09:30:00','BUY',5,100,1),fill('09:35:00','BUY',5,110,1),fill('09:40:00','SELL',10,105,2)]
    r=m.partition(fs,'2025-08-08T09:35:00+09:00',108,-4)
    assert r['quantity_at_decision']==5
    assert r['observed_net_marked_before35_krw']==39
    assert r['observed_remaining_net_after35_krw']==-43


def test_recovery_is_remaining_movement_not_saved_pnl():
    r=m.partition([fill('09:30:00','BUY',5,100),fill('09:40:00','SELL',5,110)],'2025-08-08T09:35:00+09:00',90,50)
    assert r['observed_net_marked_before35_krw']==-50 and r['observed_remaining_net_after35_krw']==100


def test_wrong_lifetime_target_fails():
    with pytest.raises(ValueError,match='identity'):
        m.partition([fill('09:30:00','BUY',1,100),fill('09:40:00','SELL',1,101)],'2025-08-08T09:35:00+09:00',100,2)
