"""Synthetic fixtures run inside the exact guarded native generation."""
from datetime import date,datetime,time,timedelta
import copy
import pytest
import sector_feature_runtime as m
import sector_feature_clock as clock
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.sector_intraday import build_sector_intraday_panel

SYMBOLS=['005930','000660','035420','035720']
SECTORS={s:'SEMICONDUCTORS' for s in SYMBOLS}


def bars(day,volume=100,opening=9):
    count=int(((14*60+30)-opening*60)/5)
    return {(day,s):tuple(MarketBar(s,datetime.combine(day,time(opening),KST)+timedelta(minutes=5*i),
        '5m',100,102,99,101,volume,is_completed=True) for i in range(count)) for s in SYMBOLS}


def test_history_changes_relative_volume_while_single_day_degenerates():
    a,b=date(2025,5,12),date(2025,5,13)
    prior=bars(a);current=bars(b,200)
    legacy=build_sector_intraday_panel(current,SECTORS,trade_dates=(b,))
    full=build_sector_intraday_panel({**prior,**current},SECTORS,trade_dates=(a,b))
    assert legacy.feature_for(b,SYMBOLS[0]).rel_volume==1
    assert full.feature_for(b,SYMBOLS[0]).rel_volume==2


def test_unknown_baseline_is_distinct_from_neutral_native_value():
    day=date(2025,5,12);members,details,panel=m.causal_members(bars(day),SECTORS,{},day)
    assert details[SYMBOLS[0]]['relative_volume'] is None
    expected=m.peer_expectation(members,details,SYMBOLS[0])
    assert not expected['supported'] and expected['qualified_relative_volume'] is None
    assert panel.feature_for(day,SYMBOLS[0]).rel_volume==1


def test_prior_twenty_positive_mean_not_median_or_current_day():
    day=date(2025,6,20)
    history={s:[(day-timedelta(days=25-i),float(i+1)) for i in range(25)] for s in SYMBOLS}
    _,detail,_=m.causal_members(bars(day),SECTORS,history,day)
    assert detail[SYMBOLS[0]]['prior_count']==20
    assert detail[SYMBOLS[0]]['prior_mean']==sum(range(6,26))/20
    history[SYMBOLS[0]].append((day,100))
    with pytest.raises(AssertionError):m.causal_members(bars(day),SECTORS,history,day)


def test_late_open_uses_close_anchored_decision_and_missing_bar_fails():
    day=date(2026,1,2);raw=bars(day,opening=10)
    qualified,excluded,cutoff=m.qualify_day(raw,day,clock)
    assert len(qualified)==4 and not excluded and cutoff.hour==14 and cutoff.minute==30
    raw[day,SYMBOLS[0]]=raw[day,SYMBOLS[0]][1:]
    qualified,excluded,_=m.qualify_day(raw,day,clock)
    assert (day,SYMBOLS[0]) not in qualified and SYMBOLS[0]+':'+str(day) in excluded


def test_future_and_post_cutoff_contamination_has_no_effect():
    day=date(2025,5,12);raw=bars(day);expected=m.qualify_day(raw,day,clock)
    contaminated=copy.deepcopy(raw);contaminated.update(bars(day+timedelta(days=1),1e9))
    s=SYMBOLS[0]
    contaminated[day,s]+=(MarketBar(s,datetime.combine(day,time(14,30),KST),'5m',100,1e9,1,1e8,1e12,is_completed=True),)
    contaminated[day,s]+=(MarketBar(s,datetime.combine(day,time(8,55),KST),'5m',100,1e9,1,1e8,1e12,is_completed=True),)
    assert m.qualify_day(contaminated,day,clock)==expected


def test_peer_feature_excludes_own_volume_and_preserves_native_shrinkage():
    a,b=date(2025,5,12),date(2025,5,13);history={s:[(a,6600.)] for s in SYMBOLS}
    raw=bars(b,200);members,details,panel=m.causal_members(raw,SECTORS,history,b)
    assert m.peer_expectation(members,details,SYMBOLS[0])['relative_volume']==2
    raw[b,SYMBOLS[0]]=bars(b,1e9)[b,SYMBOLS[0]]
    mm,dd,pp=m.causal_members(raw,SECTORS,history,b)
    assert pp.feature_for(b,SYMBOLS[0]).rel_volume==panel.feature_for(b,SYMBOLS[0]).rel_volume


def test_zero_volume_does_not_create_positive_history():
    day=date(2025,5,12);members,_,_=m.causal_members(bars(day,0),SECTORS,{},day);history={}
    m.update_history(history,members,day);assert not history


def test_incomplete_bar_is_missing_and_not_a_completed_observation():
    from dataclasses import replace
    day=date(2025,5,12);raw=bars(day)
    raw[day,SYMBOLS[0]]=(replace(raw[day,SYMBOLS[0]][0],is_completed=False),)+raw[day,SYMBOLS[0]][1:]
    qualified,excluded,_=m.qualify_day(raw,day,clock)
    assert (day,SYMBOLS[0]) not in qualified and SYMBOLS[0]+':'+str(day) in excluded


def test_sparse_peer_shrinkage_participation_and_unknown_support():
    a,b=date(2025,5,12),date(2025,5,13)
    sectors={s:('SEMICONDUCTORS' if i<2 else 'CHEMICALS') for i,s in enumerate(SYMBOLS)}
    history={s:[(a,13200.)] for s in SYMBOLS}
    members,details,panel=m.causal_members(bars(b,100),sectors,history,b)
    expected=m.peer_expectation(members,details,SYMBOLS[0])
    assert expected['weight']==.25 and expected['relative_volume']==.5 and expected['participation']==0
    assert panel.feature_for(b,SYMBOLS[0]).participation==0
    del history[SYMBOLS[3]]
    members,details,_=m.causal_members(bars(b),sectors,history,b)
    assert m.peer_expectation(members,details,SYMBOLS[0])['qualified_relative_volume'] is None


def test_native_import_guard_rejects_modified_module(tmp_path):
    from sector_feature_import_guard import NativeOnly,sha
    pkg=tmp_path/'sample';pkg.mkdir();source=pkg/'__init__.py';source.write_text('VALUE=1\n')
    guard=NativeOnly(tmp_path,{'sample/__init__.py':sha(source)})
    source.write_text('VALUE=2\n')
    with pytest.raises(ImportError):guard.find_spec('sample')
