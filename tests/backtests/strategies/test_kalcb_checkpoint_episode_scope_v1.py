import copy
import pytest
from backtests.strategies.kalcb.checkpoint_episode_scope_v1 import lifetimes,decomposition


def sample():
    return [{'side':s,'qty':q,'price':p,'timestamp':f'2026-01-05T{t}:00+09:00'}
            for s,q,p,t in [('BUY',10,100,'09:30'),('SELL',4,110,'10:00'),('SELL',6,110,'10:05'),('BUY',2,90,'10:40'),('SELL',2,80,'11:10')]]
COST={'commission_bps':2.,'tax_bps_on_sell':18.}


def test_partials_are_one_lifetime_and_reentry_is_separate():
    p=lifetimes(sample(),COST)
    assert len(p)==2 and len(p[0]['fills'])==3 and all(x['closed'] for x in p)
    assert p[0]['net_cash_flow']==pytest.approx(97.6)


def test_identical_descendants_cancel_without_double_pnl():
    h=sample();l=copy.deepcopy(h);l[1]['price']=111
    d=decomposition(h,l,COST)
    assert d['descendant_net_cash_delta']==0
    assert d['episode_net_cash_delta']==pytest.approx(d['initial_net_cash_delta'])


def test_added_reentry_is_dependent_descendant_not_initial_exit_value():
    h=sample()[:3];l=sample();d=decomposition(h,l,COST)
    assert d['initial_net_cash_delta']==0 and d['descendant_net_cash_delta']<0


def test_open_position_is_retained_and_not_closed_pnl():
    d=decomposition(sample()[:-1],sample(),COST)
    assert not d['all_closed'] and d['both_initial_closed']


@pytest.mark.parametrize('poison',['unowned','overexit','unsorted'])
def test_bad_lifetime_rejected(poison):
    f=sample()
    if poison=='unowned':f=f[1:]
    if poison=='overexit':f[1]['qty']=20
    if poison=='unsorted':f=f[::-1]
    with pytest.raises(ValueError):lifetimes(f,COST)
