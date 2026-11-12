import gzip
import json
from datetime import datetime

import pytest

from backtests.strategies.olr import native_round_diagnostics as reports


def fixture_case(tmp_path):
    from backtests.analysis.metrics import compute_trade_metrics
    from strategy_common.events import TradeOutcome
    start, end = '2026-01-02T15:30:00+09:00', '2026-01-05T15:30:00+09:00'
    window = {'start': start[:10], 'end': end[:10]}
    fills = [dict(strategy_id='OLR', side=side, qty=5, price=price, symbol='005930',
        order_id=str(index), timestamp=at, metadata={'fee':fee, 'order_date':at[:10]})
        for index,(side,price,at,fee) in enumerate((('BUY',100.,start,.1),('SELL',110.,end,1.21)))]
    journal = [dict(event='opening_marks', timestamp=start, prices={'005930':100.}),
        dict(event='fill', fill=fills[0], cash_before=1000., cash_after=499.9),
        dict(event='closing_marks', timestamp=start, prices={'005930':100.}),
        dict(event='opening_marks', timestamp=end, prices={'005930':110.}),
        dict(event='fill', fill=fills[1], cash_before=499.9, cash_after=1048.69),
        dict(event='closing_marks', timestamp=end, prices={'005930':110.})]
    trade = dict(strategy_id='OLR',symbol='005930',qty=5,entry_decision_time=start,
        entry_fill_time=start,entry_price=100.,exit_fill_time=end,exit_price=110.,
        gross_pnl=50.,net_pnl=48.69,commission=1.31,realized=True,route_metadata={},exit_reason='test')
    parsed = {**trade, **{k:datetime.fromisoformat(trade[k]) for k in ('entry_decision_time','entry_fill_time','exit_fill_time')}}
    metrics = compute_trade_metrics([TradeOutcome(**parsed)], [1000.,999.9,1048.69],
        initial_equity=1000., timestamps=[datetime.fromisoformat(start),datetime.fromisoformat(end)])
    result = dict(trades=1,sessions=2,terminal_inventory={},terminal_pending=0,return_pct=4.869,
        drawdown_pct=.01,metrics=metrics,source_generation_hash='frozen')
    audit = {'audit_receipt':{'ledger_hash':reports.stable_signature(journal),
        'daily_equity_coverage':{'status':'complete','declared_window':window,'session_count':2},
        'identity':{'input_manifest':{'manifest_hash':'input'},'qualification_snapshot_hash':'candidate'}}}
    for name,rows in [('journal',journal),('fills',fills)]:
        with gzip.open(tmp_path/(name+'.jsonl.gz'),'wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    for name,value in [('trades',[trade]),('equity',[[start,999.9],[end,1048.69]]),
        ('daily',[{'day':start[:10],'equity':999.9,'cash':499.9,'positions':{"('OLR', '005930')":5}},
                  {'day':end[:10],'equity':1048.69,'cash':1048.69,'positions':{}}])]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    return result,audit,window


def test_account_reconciliation_is_observational_and_uses_fractional_metrics(tmp_path):
    result,audit,window = fixture_case(tmp_path)
    before = {p.name:reports.digest(p) for p in tmp_path.iterdir()}
    case = reports.reconcile_exports(tmp_path,result,audit,1000.,window)
    assert case['metrics']['official_mtm_net_return_pct'] == pytest.approx(.04869)
    assert case['metrics']['max_drawdown_pct'] == pytest.approx(.0001)
    assert case['metrics']['entry_lifecycle_count'] == 1
    assert case['account_reconciliation']['cash_krw'] == pytest.approx(1048.69)
    assert before == {p.name:reports.digest(p) for p in tmp_path.iterdir()}


@pytest.mark.parametrize('field,value', [('return_pct',50.),('drawdown_pct',0.),('trades',2),('terminal_pending',1)])
def test_attractive_or_incomplete_result_cannot_pass_reporting(tmp_path,field,value):
    result,audit,window = fixture_case(tmp_path)
    result[field] = value
    with pytest.raises(ValueError):reports.reconcile_exports(tmp_path,result,audit,1000.,window)


def test_tampered_journal_is_rejected_even_if_cash_still_reconciles(tmp_path):
    result,audit,window = fixture_case(tmp_path)
    rows=reports.read_rows(tmp_path/'journal.jsonl.gz')
    rows[1]['fill']['timestamp']='2026-01-02T14:00:00+09:00'
    with gzip.open(tmp_path/'journal.jsonl.gz','wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='journal'):reports.reconcile_exports(tmp_path,result,audit,1000.,window)


def test_wrong_capital_or_omitted_mark_cannot_supply_report(tmp_path):
    result,audit,window = fixture_case(tmp_path)
    with pytest.raises(ValueError):reports.reconcile_exports(tmp_path,result,audit,10000.,window)
    p=tmp_path/'equity.json';p.write_text(json.dumps(reports.read(p)[1:]))
    with pytest.raises(ValueError,match='marks'):reports.reconcile_exports(tmp_path,result,audit,1000.,window)


def test_custom_scope_does_not_inherit_unrelated_regression_claim(tmp_path):
    from types import SimpleNamespace
    from backtests.strategies.olr.final_diagnostics import render_full_diagnostics
    result,audit,window = fixture_case(tmp_path)
    case=reports.reconcile_exports(tmp_path,result,audit,1000.,window)
    plugin=SimpleNamespace(config={'promotion_status':reports.STATUS},training_config={'date_range':window},
        diagnostic_limitations=['Support gate FAILED; consumed regression unavailable under this native contract.'],
        diagnostic_scope={'independent_oos_pass':False})
    payload=render_full_diagnostics(plugin,SimpleNamespace(cumulative_mutations={}),tmp_path,
        round_num=1,round_name='Conditional capital recycling',cases={'training':case},training_text='Research only')
    assert payload['limitations']==plugin.diagnostic_limitations
    assert payload['promotion_status']==reports.STATUS
    assert payload['diagnostic_scope']['independent_oos_pass'] is False


@pytest.mark.parametrize('fault', [None,'future_entry','same_day_exit','unfunded','oversized'])
def test_recycling_order_clock_and_funding_evidence(fault):
    at='2026-01-05T14:35:00+09:00'
    m={'decision_available_at':at,'capacity_forecast':{'asof':at,'participation':.01,
        'entry':{'samples':[['2026-01-05T14:30:00+09:00',2000]],'horizon':1,'quantity':20},
        'exit':{'samples':[['2026-01-02T15:30:00+09:00',10000]],'quantity':100}},
        'allocation_before_redistribution':10,'allocation_redistributed_qty':5,
        'allocation_native_qty':10,'candidate_rank':2,
        'capacity_recycling':{'day':at[:10],'assigned_extras':100,'displaced_request_funding':100}}
    order={'side':'BUY','qty':15,'symbol':'005930','metadata':m}
    if fault=='future_entry':m['capacity_forecast']['entry']['samples'][0][0]=at
    if fault=='same_day_exit':m['capacity_forecast']['exit']['samples'][0][0]='2026-01-05T15:30:00+09:00'
    if fault=='unfunded':m['capacity_recycling']['assigned_extras']=101
    if fault=='oversized':order['qty']=16;m['allocation_redistributed_qty']=6
    if fault:
        with pytest.raises(ValueError):reports.verify_recycling_orders([{'event':'submit','order':order}])
    else:
        assert reports.verify_recycling_orders([{'event':'submit','order':order}])['direct_extra_order_issuer_days']==1
