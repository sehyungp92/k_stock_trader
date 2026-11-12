import pytest
from scripts.kalcb_lifecycle_readiness_v1 import conditions, select
from scripts.kalcb_timed_exit_scope_v1 import lifetimes


def observation(age, price=99., mfe=.5, clock='10:00'):
    return dict(day='2025-08-08', available_at=clock, entry_order_id='a', held_bars=age,
                current_price=price, entry_price=100., management_risk_per_share=1., reason='eligible',
                features=dict(return_atr=(price-100)/2, mfe_atr=mfe/2))


def test_clock_expiry_and_native_r_units():
    assert conditions(observation(6), 2.) == {'F': False, 'Q': False}
    assert conditions(observation(8), 2.) == {'F': True, 'Q': False}
    assert conditions(observation(10, mfe=2.), 2.) == {'F': False, 'Q': False}
    assert conditions(observation(11, mfe=2.), 2.) == {'F': False, 'Q': True}
    assert conditions(observation(11, price=99.5, mfe=2.), 2.)['Q'] is False
    with pytest.raises(ValueError, match='semantic'):
        conditions(observation(11), 3.)


def test_first_causal_state_without_outcome_selection_or_union_duplicates():
    a = {'observation': observation(8, clock='10:10'), 'future_payoff': -1000}
    b = {'observation': observation(11, clock='10:25'), 'future_payoff': 1000}
    selected = select([b, a], {'a': 2.})
    assert selected['F'] == [a]
    assert selected['Q'] == [b]
    assert selected['F_or_Q'] == [a]


def test_completed_lifetime_accounts_for_partial_fills_and_all_fees():
    def fill(side, qty, price, order, fee):
        return dict(symbol='x', side=side, qty=qty, price=price, order_id=order,
                    timestamp='2025-08-08 10:00:00+09:00', reason='quick_exit', metadata={'fee': fee})
    rows = [fill('BUY', 2, 100, 'a', 1), fill('BUY', 3, 101, 'a', 1),
            fill('SELL', 4, 102, 'b', 2), fill('SELL', 1, 103, 'b', 1)]
    result = lifetimes(rows)
    assert len(result) == 1 and result[0]['entry_order_id'] == 'a'
    assert result[0]['cash'] == 3
    with pytest.raises(ValueError, match='Nonflat'):
        lifetimes(rows[:-1])
