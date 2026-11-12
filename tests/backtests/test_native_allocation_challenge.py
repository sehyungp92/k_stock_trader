from dataclasses import replace
import pytest

from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from deployment.olr_kalcb.native_budget import NativeBudgetConfig, proportional_grants
from tests.backtests.test_native_budget_replay import account, submit
from tests.backtests.test_strategy_integrity_repairs import bar, ts


@pytest.mark.parametrize('mode,expected', [('symbol', [('000001', 8), ('000002', 2)]),
    ('native_rank', [('000002', 8), ('000001', 2)]),
    ('proportional', [('000002', 5), ('000001', 5)])])
def test_scarce_funding_order_and_reservations_agree(mode, expected):
    b = account(NativeBudgetConfig(allocation_mode=mode), max_gross_notional=1000)
    submit(b, 'KALCB', '000001', 8, candidate_rank=2)
    submit(b, 'KALCB', '000002', 8, candidate_rank=1)
    drive_timestamp_batch(b, [bar('000001'), bar('000002')], [])
    assert [(f.symbol, f.qty) for f in b.fills] == expected
    assert audit_broker(b)['execution_validity_pass']


def test_partial_fill_is_credited_in_later_proportional_callbacks():
    from backtests.engine.capacity import BarParticipationScenario
    b = account(NativeBudgetConfig(allocation_mode='proportional'), max_gross_notional=1000)
    b.capacity_policy = BarParticipationScenario(.01, 'test')
    drive_timestamp_batch(b, [bar('000001', ts(0), volume=300), bar('000002', ts(0), volume=1000)], [])
    submit(b, 'KALCB', '000001', 8, candidate_rank=1)
    submit(b, 'KALCB', '000002', 8, candidate_rank=2)
    drive_timestamp_batch(b, [bar('000001', ts(5), volume=300), bar('000002', ts(5), volume=1000)], [])
    assert [(f.symbol, f.qty) for f in b.fills] == [('000001', 3), ('000002', 5)]
    drive_timestamp_batch(b, [bar('000001', ts(10)), bar('000002', ts(10))], [])
    assert {s: b.position_qty('KALCB', s) for s in ('000001','000002')} == {'000001':5,'000002':5}
    assert audit_broker(b)['execution_validity_pass']


def test_proportional_bottleneck_does_not_strand_other_sector_resources():
    items = [{'qty':8, 'filled':0, 'unit':100, 'resources':{'gross':100,'a':100}},
             {'qty':8, 'filled':0, 'unit':100, 'resources':{'gross':100,'b':100}}]
    assert proportional_grants(items, {'gross':1000,'a':200,'b':1000}) == [2,8]


def test_proportional_no_future_order_or_sell_proceeds_are_credited():
    b = account(NativeBudgetConfig(allocation_mode='proportional'), max_gross_notional=1000)
    submit(b, 'KALCB', '000002', 8, candidate_rank=8)
    submit(b, 'KALCB', '000001', 8, ts(1), candidate_rank=1)
    drive_timestamp_batch(b, [bar('000001', ts(5)),bar('000002', ts(5))], [])
    assert [(f.symbol,f.qty) for f in b.fills] == [('000002',8),('000001',2)]
    assert audit_broker(b)['execution_validity_pass']


def test_random_order_is_reproducible_and_ignores_submission_sequence():
    outcomes=[]
    for symbols in [('000001','000002'),('000002','000001')]:
        b=account(NativeBudgetConfig(allocation_mode='random',allocation_seed=11),max_gross_notional=1000)
        for s in symbols: submit(b,'KALCB',s,8)
        drive_timestamp_batch(b,[bar(s) for s in symbols],[])
        outcomes.append([(f.symbol,f.qty) for f in b.fills])
        assert audit_broker(b)['execution_validity_pass']
    assert outcomes[0]==outcomes[1]


def test_proportional_execution_gap_and_risk_budget_remain_feasible():
    b=account(NativeBudgetConfig(allocation_mode='proportional',max_kalcb_stop_risk_fraction=.02))
    for i in (1,2):submit(b,'KALCB',f'{i:06}',8,candidate_rank=i)
    drive_timestamp_batch(b,[bar(f'{i:06}',o=125,h=130,l=120,c=125) for i in (1,2)],[])
    assert sum(f.qty for f in b.fills)<=10
    assert audit_broker(b)['execution_validity_pass']


def test_audit_reconstructs_filled_credit_instead_of_trusting_ledger():
    b=account(NativeBudgetConfig(allocation_mode='proportional'),max_gross_notional=1000)
    for i in (1,2):submit(b,'KALCB',f'{i:06}',8,candidate_rank=i)
    drive_timestamp_batch(b,[bar('000001'),bar('000002')],[])
    row=next(r for r in b.journal if r['event']=='native_budget_admission')
    row['ledger']['allocations'][row['order']['order_id']]['qty']+=1
    assert not audit_broker(b)['execution_validity_pass']


@pytest.mark.parametrize('changes',[{'allocation_mode':'oracle'},{'allocation_seed':True},{'allocation_seed':-1}])
def test_bad_allocation_configuration_fails_closed(changes):
    with pytest.raises(ValueError):NativeBudgetConfig(**changes)


@pytest.mark.parametrize('mode',['partition','daily_rebalanced'])
def test_partition_starts_with_one_account_and_independent_native_budgets(mode):
    b=account(NativeBudgetConfig(capital_mode=mode))
    assert sum(b.sleeve_cash.values())==1000
    assert b.strategy_sizing_equity()==500
    assert b.strategy_account_view('OLR')==(500,500)
    submit(b,'KALCB','000001',15);submit(b,'OLR','000002',15)
    drive_timestamp_batch(b,[bar('000001'),bar('000002')],[])
    assert [(f.strategy_id,f.qty) for f in b.fills]==[('KALCB',10),('OLR',5)]
    assert b.cash==-500 and sum(b.sleeve_cash.values())==-500
    assert audit_broker(b)['execution_validity_pass']


def test_partition_unrealized_profit_cannot_fund_a_cash_transfer():
    from datetime import timedelta
    accounts=[account(NativeBudgetConfig(capital_mode=m)) for m in ('partition','daily_rebalanced')]
    for b in accounts:
        submit(b,'KALCB','000001',5)
        drive_timestamp_batch(b,[bar('000001')],[])
        drive_timestamp_batch(b,[bar('000001',ts(10),o=200,h=200,l=200,c=200)],[])
        assert b.strategy_sizing_equity()==1000
        assert b.strategy_account_view('OLR')[0]==500
        # New day's price is 300, but wealth transfer must use yesterday's 200 mark.
        drive_timestamp_batch(b,[bar('000001',ts(10)+timedelta(days=1),o=300,h=300,l=300,c=300)],[])
        assert audit_broker(b)['execution_validity_pass']
    assert accounts[0].strategy_account_view('OLR')[0]==500
    assert accounts[1].strategy_account_view('OLR')[0]==500
    transfers=[r for r in accounts[1].journal if r['event']=='native_sleeve_transfer']
    assert transfers[-1]['requested_transfer']==-250 and transfers[-1]['kalcb_transfer']==0
    transfers[-1]['kalcb_transfer']+=1
    assert not audit_broker(accounts[1])['execution_validity_pass']


def test_cash_limited_transfer_uses_prior_marks_and_donor_cash():
    from datetime import timedelta
    b=account(NativeBudgetConfig(capital_mode='daily_rebalanced'))
    submit(b,'OLR','000001',2)
    drive_timestamp_batch(b,[bar('000001')],[])
    drive_timestamp_batch(b,[bar('000001',ts(10),o=200,h=200,l=200,c=200)],[])
    drive_timestamp_batch(b,[bar('000001',ts(10)+timedelta(days=1),o=300,h=300,l=300,c=300)],[])
    transfer=[r for r in b.journal if r['event']=='native_sleeve_transfer'][-1]
    assert transfer['requested_transfer']==100 and transfer['kalcb_transfer']==100
    assert b.sleeve_cash=={'KALCB':600,'OLR':200}
    audit=audit_broker(b)
    assert audit['execution_validity_pass']
    assert audit['native_budget_audit']['wealth_transfers'][-1]['donor_cash']==300
