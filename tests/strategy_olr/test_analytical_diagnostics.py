"""Accounting, availability and economic denominators in observer-only reports."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json

import pytest

from backtests.strategies.olr.analytical_diagnostics import (
    normalize_trades, trade_stats, consistency, event_analysis, execution_summary,
    compare_cases, source_funnel, distribution, build_analysis, source_entry_calendar,
)
from backtests.strategies.olr.interval_round_diagnostics import IntervalDiagnosticsPlugin


def trade(symbol='A', qty=10, net=18, day='2026-01-05', entry=10):
    return dict(symbol=symbol, qty=qty, entry_decision_time=day+'T14:30:00+09:00',
        entry_fill_time=day+'T14:36:00+09:00', exit_fill_time='2026-01-06T09:01:00+09:00',
        entry_price=entry, exit_price=12, gross_pnl=net+2, net_pnl=net, mfe=4, mae=-2,
        exit_reason='native', route_metadata={'risk_per_share':2, 'candidate_rank':1})


def test_r_excursion_units_and_missingness_are_explicit():
    raw = trade(); t = normalize_trades([raw], 10_000)[0]
    assert t['mfe_r'] == 2 and t['mae_r'] == -1 and t['net_r'] == .9
    assert t['meaningful']  # exactly 1%, not rounded upward
    raw['route_metadata'] = {}; raw['mfe'] = None
    t = normalize_trades([raw], 100_000)[0]
    assert t['net_r'] is None and t['mfe_r'] is None and not t['meaningful']
    assert trade_stats([t])['net_r']['n'] == 0
    assert distribution([])['mean'] is None
    with pytest.raises(ValueError, match='Duplicate opportunity'):
        normalize_trades([raw, raw], 100_000)


def mark(at, price, kind='opening_marks'):
    return dict(event=kind, timestamp=at, prices={'A':price})


def submit(order_id, side, qty, at):
    return dict(event='submit', order=dict(order_id=order_id, symbol='A', side=side, qty=qty,
        submitted_at=at, reason='native', order_type='MARKET', metadata={'decision_available_at':at}))


def fill(order_id, side, qty, price, fee, at, before, after):
    return dict(event='fill', fill=dict(order_id=order_id, symbol='A', side=side, qty=qty,
        price=price, timestamp=at, metadata={'fee':fee}), cash_before=before, cash_after=after)


def test_streamed_nav_attribution_recovery_and_fragments():
    # Two buy fragments are ONE opportunity/order. Overnight 10 -> 8 loses 20;
    # intraday recovery 8 -> 12 gains 40. Explicit entry+exit fees total 2.
    events = [mark('2026-01-05T14:35:00+09:00',10), submit('b','BUY',10,'2026-01-05T14:35:00+09:00'),
        fill('b','BUY',4,10,.4,'2026-01-05T14:36:00+09:00',100,59.6),
        fill('b','BUY',6,10,.6,'2026-01-05T14:37:00+09:00',59.6,-1)]
    # Use sufficient capital so fees never imply borrowing.
    for e in events:
        if e['event']=='fill': e['cash_before']+=100; e['cash_after']+=100
    events += [mark('2026-01-06T09:00:00+09:00',8),
        mark('2026-01-06T09:00:00+09:00',12,'closing_marks'),
        submit('s','SELL',10,'2026-01-06T09:01:00+09:00'),
        fill('s','SELL',10,12,1,'2026-01-06T09:01:00+09:00',99,218)]
    result = event_analysis(iter(events),200)
    assert result['terminal_nav']==218 and result['symbol_pnl']['A']==18
    assert result['mtm_components']=={'intraday_mark':40, 'execution_vs_observable_mark':0, 'explicit_fees':-2, 'overnight_mark':-20}
    assert result['worst_percentage']['decline_krw']==21
    assert result['worst_percentage']['symbol_attribution']=={'A':-21}
    assert result['drawdown_episodes'][0]['censored'] is False
    assert result['drawdown_episodes'][-1]['censored'] is True # final fee below 219 peak, even though flat
    ex=execution_summary(result['orders'])
    assert ex['BUY']['orders']==1 and ex['BUY']['fragments']==2
    assert ex['BUY']['first_fill_delay_minutes']['median']==1


def test_known_clock_not_bar_label_and_negative_cash_rejected():
    ev=[mark('2026-01-05T14:35:00+09:00',10),submit('b','BUY',1,'2026-01-05T14:36:00+09:00'),
        fill('b','BUY',1,10,0,'2026-01-05T14:35:00+09:00',100,90)]
    with pytest.raises(ValueError,match='before decision'):
        event_analysis(ev,100)
    ev[1]['order']['metadata']['decision_available_at']='2026-01-05T14:35:00+09:00'
    ev[2]['cash_before']=5;ev[2]['cash_after']=-5
    with pytest.raises(ValueError,match='unfunded'):
        event_analysis(ev,5)


def test_percentage_and_currency_peaks_are_independent():
    ev=[mark('2026-01-05T09:00:00+09:00',10),submit('b','BUY',10,'2026-01-05T09:00:00+09:00'),
        fill('b','BUY',10,10,0,'2026-01-05T09:01:00+09:00',100,0),
        mark('2026-01-05T10:00:00+09:00',5),mark('2026-01-05T11:00:00+09:00',30),
        mark('2026-01-05T12:00:00+09:00',20)]
    r=event_analysis(ev,100)
    assert r['worst_percentage']['drawdown_fraction']==.5
    assert r['worst_percentage']['decline_krw']==50
    assert r['worst_currency']['decline_krw']==100


def test_matched_comparison_is_exact_and_counts_new_not_resize():
    a=normalize_trades([trade('A',10,18),trade('B',10,-4)],10000)
    b=normalize_trades([trade('A',20,30),trade('C',10,7)],10000)
    c=compare_cases({'trades':a},{'trades':b})
    assert c['net_difference']==23
    assert c['identities']=={'matched':1,'removed':1,'added':1}
    assert c['decomposition']==pytest.approx({'quantity_at_control_payoff':18,'net_unit_payoff_at_treatment_quantity':-6,'participation_delta':11})
    assert c['meaningful_difference']==0


def test_monthly_nav_is_not_entry_month_pnl():
    t=normalize_trades([trade()],100)
    d=[dict(day='2026-01-30',equity=90),dict(day='2026-02-02',equity=118)]
    c=consistency(t,d,100)
    assert [m['net'] for m in c['monthly']]==[-10,28]
    assert c['months']==2 and c['positive_months']==1
    assert c['rolling_windows']==0 and c['positive_rolling_windows']==0


def test_funnel_does_not_call_unselected_confirmed_or_profitable():
    snapshots=[dict(trade_date='2026-01-05',candidates=[{'symbol':'A','rank':1},{'symbol':'B','rank':2}])]
    r=source_funnel(snapshots,{},[],{'decisions':[]},{'start':'2026-01-05','end':'2026-01-05'},1)
    assert r['dispositions']=={'no_recorded_proposal':1,'outside_original_slots':1}
    assert r['recorded_confirmation_days']==0
    assert all(v['net'] is None for v in r['rows'])


def test_sealed_parent_mismatch_rejected_before_reuse(tmp_path):
    p=tmp_path/'round_final_full_diagnostics.json';p.write_text('{}')
    obj=object.__new__(IntervalDiagnosticsPlugin)
    with pytest.raises(ValueError,match='parent report bytes'):
        obj.reuse_verified_report(tmp_path,'not-the-hash')


def test_empty_analysis_does_not_invent_evidence(tmp_path):
    r,text=build_analysis({},tmp_path)
    assert r['cases']=={} and r['comparisons']=={} and not r['changed_trading_policy']


def test_equal_trade_r_does_not_masquerade_as_allocated_risk_return():
    t=normalize_trades([trade('A',10,18),trade('B',100,-10)],10000)
    s=trade_stats(t)
    assert s['net_r']['mean']==pytest.approx(.425)
    assert s['aggregate_net_per_initial_risk']==pytest.approx(8/220)


def test_missing_source_empty_source_and_overlap_tags_stay_distinct():
    sn=[dict(trade_date='2026-01-05',candidates=[],metadata={
        'afternoon_rejected_symbols':{'A':['vwap','flow'],'B':['flow']}})]
    f=source_funnel(sn,{},[],{'decisions':[]},{'start':'2026-01-05','end':'2026-01-06'},6,
        ['2026-01-05','2026-01-06'])
    assert f['missing_snapshot_sessions']==['2026-01-06']
    assert f['empty_snapshot_sessions']==['2026-01-05']
    assert f['monthly'][0]['sessions']==2 and f['monthly'][0]['snapshots']==1
    assert f['upstream_rejection_tags']['afternoon']=={'flow':2,'vwap':1}


def test_reserved_exit_dates_use_frozen_contract_not_missing_data_assumption(tmp_path):
    p=tmp_path/'replay_cache.py'
    p.write_text('_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS = 2\n'
        'def _eligible_snapshot_dates(trading_dates: Iterable[date]) -> tuple[date, ...]:\n'
        '    dates = tuple(sorted(trading_dates))\n'
        '    n = max(1, int(_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS))\n'
        '    return dates[:-n] if len(dates) > n else ()\n'
        'raise RuntimeError("Do not import/execute the rest of the engine")\n')
    days=['2026-01-05','2026-01-06','2026-01-07']
    contract=source_entry_calendar(p,days)
    assert contract['eligible_dates']==days[:1]
    sn=[dict(trade_date=days[0],candidates=[])]
    f=source_funnel(sn,{},[],{'decisions':[]},{'start':days[0],'end':days[-1]},6,days,contract['eligible_dates'])
    assert f['missing_snapshot_sessions']==[] and f['reserved_exit_sessions']==days[1:]
    assert f['absent_snapshot_sessions']==days[1:] and f['empty_snapshot_sessions']==days[:1]


@pytest.mark.parametrize('fault',[None,'artifact','python','calendar','metrics','receipt'])
def test_cached_native_report_rehashes_bound_evidence(tmp_path,fault):
    from backtests.strategies.olr.interval_round_diagnostics import digest, CASE_NAMES
    p=tmp_path/'parent';p.mkdir();native=tmp_path/'native';native.mkdir();source=tmp_path/'source';source.mkdir()
    def write(path,data):
        path.write_text(json.dumps(data),encoding='utf8')
    (source/'engine.py').write_text('engine-version-one')
    (source/'calendar.yaml').write_text('close: 15:30')
    manifest=tmp_path/'manifest.json';write(manifest,{'files':{f.name:digest(f) for f in source.iterdir()}})
    (native/'trades.json').write_text('[]')
    artifacts={'trades.json':digest(native/'trades.json')}
    write(native/'completion.json',{'artifacts':artifacts})
    study=tmp_path/'study.json';write(study,{'source':'identity'})
    specs={k:dict(path=str(native),source_generation_hash='source-v1',completion_sha256=digest(native/'completion.json'),
        study=str(study),study_sha256=digest(study),execution_root=str(source)) for k in CASE_NAMES}
    plan_path=tmp_path/'plan.json';write(plan_path,{'cases':specs})
    cases={k:dict(path=str(native),source_binding=dict(source_generation_hash='source-v1',completion_sha256=digest(native/'completion.json'),
        study_sha256=digest(study),manifest=str(manifest),manifest_sha256=digest(manifest),artifacts=artifacts),
        account_reconciliation={'passed':True},metrics={'audit_pass':True,'execution_validity_pass':True,'net_profit':18}) for k in CASE_NAMES}
    (p/'round_final_diagnostics.txt').write_text('accepted-report')
    write(p/'round_final_full_diagnostics.json',{'cases':cases,'mutations':{},'final_diagnostics_text_sha256':digest(p/'round_final_diagnostics.txt')})
    expected=digest(p/'round_final_full_diagnostics.json')
    receipt={'passed':True,'case_plan_sha256':digest(plan_path),
        'canonical_artifacts':{'round_final_full_diagnostics.json':expected},'accepted_cases':{k:{'net_profit':18} for k in CASE_NAMES}}
    if fault=='metrics':receipt['accepted_cases']['train_nominal']['net_profit']=180000
    if fault=='receipt':receipt['passed']=False
    write(p/'integrity_verification.json',receipt)
    if fault=='artifact':(native/'trades.json').write_text('["altered"]')
    if fault=='python':(source/'engine.py').write_text('engine-version-two')
    if fault=='calendar':(source/'calendar.yaml').write_text('close: 15:00')
    obj=object.__new__(IntervalDiagnosticsPlugin);obj.plan_path=plan_path;obj.plan={'cases':specs}
    obj.initial_mutations={};obj.output_dir=tmp_path/'output';obj.exported_cases={}
    if fault:
        with pytest.raises(ValueError):obj.reuse_verified_report(p,expected)
    else:
        obj.reuse_verified_report(p,expected)
        assert len(obj.exported_cases)==6 and obj.reused_validation['completion_artifacts_rehashed']==6
