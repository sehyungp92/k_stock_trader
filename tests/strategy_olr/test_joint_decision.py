from dataclasses import replace
from datetime import datetime, timezone, timedelta
import pytest
from strategy_olr.core.joint_decision import Account, Action, Constraints, ScenarioAxis, decide, upper_tail_mean

NOW=datetime(2025,9,1,tzinfo=timezone.utc)
HASH='a'*64

def axis():return ScenarioAxis(('day_a','day_b'),(.5,.5),(0.,1.,2.),NOW,HASH)
def zero(group):return Action(group+'0',group,0,0,((0,0,0),(0,0,0)),axis().ids,NOW,HASH,completed_entry_probabilities=(0,0))
def buy(group,paths,cash=50,resources=()):return Action(group+'1',group,1,cash,paths,axis().ids,NOW,HASH,resources,(1,1))
def run(groups,**kwargs):return decide(Account(100,100,100,0),axis(),((0,0,0),(0,0,0)),groups,NOW,**kwargs)

def test_joint_paths_preserve_diversification_and_common_shocks():
    a=buy('a',((0,-10,10),(0,10,10)))
    hedge=buy('b',((0,10,10),(0,-10,10)))
    joint=run({'a':(zero('a'),a),'b':(zero('b'),hedge)},constraints=Constraints(maximum_tail_drawdown_fraction=.01))
    assert joint.selected.action_ids==('a1','b1')
    assert joint.selected.tail_drawdown_fraction==0
    correlated=replace(hedge,net_paths=a.net_paths)
    assert run({'a':(zero('a'),a),'b':(zero('b'),correlated)},constraints=Constraints(maximum_tail_drawdown_fraction=.01)).selected.reserve_cash==0

def test_pending_buys_reserve_cash_and_pending_sales_cannot_fund():
    a=buy('a',((0,1,5),(0,1,5)),cash=60)
    d=decide(Account(100,100,70,20),axis(),((0,0,0),(0,0,0)),{'a':(zero('a'),a)},NOW)
    assert d.selected.action_ids==('a0',)

def test_shared_sector_risk_and_physical_capacity_are_aggregated():
    a=buy('a',((0,0,4),(0,0,4)),resources=(('sector',6),('capacity',1)))
    b=buy('b',((0,0,5),(0,0,5)),resources=(('sector',6),('capacity',1)))
    d=decide(Account(100,100,100,0,(('sector',10),('capacity',2))),axis(),((0,0,0),(0,0,0)),{'a':(zero('a'),a),'b':(zero('b'),b)},NOW)
    assert d.selected.action_ids==('a0','b1')

def test_quantity_curve_can_prefer_smaller_profitable_request():
    small=buy('a',((0,-1,5),(0,-1,5)),cash=20)
    large=replace(small,action_id='a2',quantity=2,reserve_cash=60,net_paths=((0,-10,4),(0,-10,4)))
    d=run({'a':(zero('a'),small,large)})
    assert d.selected.action_ids==('a1',)

def test_full_marked_path_counts_profit_giveback():
    a=buy('a',((0,30,5),(0,30,5)))
    d=run({'a':(zero('a'),a)},constraints=Constraints(maximum_tail_drawdown_fraction=.1))
    assert d.selected.action_ids==('a0',)
    assert any(abs(x.tail_drawdown_fraction-25/130)<1e-12 for x in d.frontier)

def test_existing_book_cannot_be_repaired_by_phantom_zero_action():
    d=decide(Account(100,110,100,0),axis(),((0,-10,-5),(0,-10,-5)),{'a':(zero('a'),)},NOW,
             Constraints(maximum_tail_drawdown_fraction=.05))
    assert d.selected is None and d.reason=='no_feasible_supported_allocation'

def test_weighted_tail_fractional_boundary():
    assert upper_tail_mean([.2,.1,0],[.05,.15,.8],.1)==pytest.approx(.15)

@pytest.mark.parametrize('change',[{'available_at':NOW+timedelta(seconds=1)}, {'supported':False}, {'scenario_ids':('day_b','day_a')}, {'net_paths':((0,1),(0,1))}])
def test_future_missing_and_misaligned_evidence_fails_closed(change):
    a=replace(buy('a',((0,0,1),(0,0,1))),**change)
    with pytest.raises(ValueError):run({'a':(zero('a'),a)})

def test_no_order_is_cash_outcome_not_an_invented_alpha_label():
    with pytest.raises(ValueError):run({'a':(replace(zero('a'),net_paths=((0,1,1),(0,1,1))),)})

def test_decision_order_and_future_append_do_not_change_existing_result():
    a=buy('a',((0,-1,3),(0,-1,3)))
    b=buy('b',((0,-1,4),(0,-1,4)))
    x=run({'a':(zero('a'),a),'b':(zero('b'),b)})
    y=run({'b':(b,zero('b')),'a':(a,zero('a'))})
    assert x==y
    with pytest.raises(ValueError):decide(Account(100,100,100,0),replace(axis(),known_at=NOW+timedelta(days=1)),((0,0,0),(0,0,0)),{'a':(zero('a'),a)},NOW)

def test_enumeration_budget_and_frequency_constraints_are_explicit():
    a=buy('a',((0,0,-1),(0,0,-1)))
    assert run({'a':(zero('a'),a)}).selected.expected_completed_entries==0
    assert run({'a':(zero('a'),a)},constraints=Constraints(minimum_expected_completed_entries=1)).selected.mean_net_pnl==-1
    with pytest.raises(ValueError):run({'a':(zero('a'),a)},constraints=Constraints(maximum_combinations=1))


def test_frequency_constraint_uses_expected_completed_cycles_not_submissions():
    a=replace(buy('a',((0,0,5),(0,0,5))),completed_entry_probabilities=(.1,.1))
    result=run({'a':(zero('a'),a)},constraints=Constraints(minimum_expected_completed_entries=.5))
    assert result.selected is None
    assert run({'a':(zero('a'),a)}).selected.expected_completed_entries==pytest.approx(.1)


def test_management_competes_on_remaining_net_value_and_giveback_without_phantom_cash():
    held=((0,20,-20),(0,20,-20))
    hold=replace(zero('held'),kind='hold')
    exit_action=Action('exit','held',5,0,((0,-21,19),(0,-21,19)),axis().ids,NOW,HASH,
                       completed_entry_probabilities=(0,0),kind='exit',management_completion_deltas=(0,0))
    # Exit incurs one cash unit of execution cost; continuing gives back40.
    new=buy('new',((0,1,2),(0,1,2)),cash=10)
    result=decide(Account(100,100,0,0),axis(),held,{'held':(hold,exit_action),'new':(zero('new'),new)},NOW,
                  Constraints(maximum_tail_drawdown_fraction=.02),held_completed_entries=(1,1))
    assert result.selected.action_ids==('exit','new0')
    assert result.selected.mean_net_pnl==-1 and result.selected.reserve_cash==0


def test_cancel_uses_incremental_pending_path_and_does_not_release_reserved_cash():
    hold=replace(zero('pending'),kind='hold')
    cancel=Action('cancel','pending',0,0,((0,5,10),(0,5,10)),axis().ids,NOW,HASH,
                  completed_entry_probabilities=(0,0),kind='cancel',management_completion_deltas=(-1,-1))
    new=buy('new',((0,1,2),(0,1,2)),cash=5)
    result=decide(Account(100,100,10,10),axis(),((0,-5,-10),(0,-5,-10)),
        {'pending':(hold,cancel),'new':(zero('new'),new)},NOW,held_completed_entries=(1,1))
    assert result.selected.action_ids==('new0','cancel') and result.selected.mean_net_pnl==0
    assert result.selected.expected_completed_entries==0
    preserve=decide(Account(100,100,10,10),axis(),((0,-5,-10),(0,-5,-10)),
        {'pending':(hold,cancel),'new':(zero('new'),new)},NOW,
        Constraints(minimum_expected_completed_entries=1),held_completed_entries=(1,1))
    assert preserve.selected.action_ids==('new0','pending0')
