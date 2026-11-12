from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from backtests.strategies.kalcb import action_value as av
from strategy_common.clock import KST
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.core_models import KALCBPortfolioView
from strategy_kalcb.core.logic import step_kalcb_core
from strategy_kalcb.core.preview import preview_entry
from strategy_kalcb.core.serializers import snapshot_state
from strategy_kalcb.core.state import KALCBState
from tests.strategy_kalcb.test_core_logic import _snapshot, _or_bars


def preview_fixture():
    day = date(2026, 1, 5)
    cfg = KALCBConfig.from_mapping({"kalcb.entry.plan_mode":"first30_open",
        "kalcb.entry.rvol_threshold":0, "kalcb.entry.cpr_threshold":0,
        "kalcb.entry.momentum_score_min":0,"kalcb.entry.entry_score_blocklist":[]})
    return cfg, _snapshot(day), _or_bars(day), KALCBPortfolioView(1_000_000,equity=1_000_000)


def test_preview_matches_core_and_ignores_future_prices():
    cfg, snapshot, bars, account = preview_fixture()
    cutoff = bars[5].timestamp+timedelta(minutes=5)
    expected = KALCBState()
    for bar in bars[:6]:
        result = step_kalcb_core(expected,bar,cfg,snapshot,account)
    observed = preview_entry(bars,snapshot,cfg,account,available_at=cutoff)
    assert observed["ready"] and observed["intents"][0]["quantity"] == result.actions[0].qty
    poisoned = bars[:6]+[replace(bars[-1],close=1e9,volume=1e12)]
    assert preview_entry(poisoned,snapshot,cfg,account,available_at=cutoff) == observed
    assert not preview_entry(bars,snapshot,cfg,account,available_at=cutoff-timedelta(minutes=1))["ready"]


def test_preview_preserves_supplied_state_and_respects_occupied_account():
    cfg, snapshot, bars, account = preview_fixture()
    state = KALCBState()
    step_kalcb_core(state,bars[0],cfg,snapshot,account)
    before = snapshot_state(state)
    preview_entry(bars[1:6],snapshot,cfg,account,state=state,
                  available_at=bars[5].timestamp+timedelta(minutes=5))
    assert snapshot_state(state) == before
    occupied = replace(account,open_positions=cfg.max_positions)
    result = preview_entry(bars,snapshot,cfg,occupied,available_at=bars[5].timestamp+timedelta(minutes=5))
    assert not result["ready"]
    assert "max_positions" in [d["reason"] for d in result["decisions"]]


@pytest.mark.parametrize("scale", [0,.25,.5,1])
def test_quantity_scale_reduces_final_capped_size_only_when_enabled(scale):
    cfg, snap, bars, account = preview_fixture()
    cutoff = bars[5].timestamp+timedelta(minutes=5)
    base = preview_entry(bars,snap,cfg,account,available_at=cutoff)["intents"][0]["quantity"]
    c = replace(snap.candidates[0],metadata={**snap.candidates[0].metadata,"action_quantity_scale":scale})
    snap = replace(snap,candidates=(c,))
    assert preview_entry(bars,snap,cfg,account,available_at=cutoff)["intents"][0]["quantity"] == base
    result = preview_entry(bars,snap,replace(cfg,entry_action_quantity_scale_enabled=True),account,available_at=cutoff)
    assert (result["intents"][0]["quantity"] if result["ready"] else 0) == int(base*scale)


def training_fixture():
    sessions = [(datetime(2025,1,1)+timedelta(days=i)).date().isoformat() for i in range(170)]
    rows = [{"opportunity_id":d+"/A","trade_date":d,"action":{"ready":True},
        "features":{k:(i%7)/10 for k in av.FEATURES}} for i,d in enumerate(sessions)]
    labels = {r["opportunity_id"]:{"label_valid":True,"label_available_at":r["trade_date"]+"T15:30:00+09:00",
        "net_value_equity_bps":i%5-2,"buy_qty":0 if i%5==2 else 10} for i,r in enumerate(rows)}
    return sessions,rows,labels


@pytest.mark.parametrize("kind", ["ridge","hgb","hurdle"])
def test_action_models_cannot_learn_future_outcomes(kind):
    from threadpoolctl import threadpool_limits
    sessions,rows,labels = training_fixture()
    spec = av.ModelSpec("test",kind)
    with threadpool_limits(limits=1):
        _, original, proof = av.fit_forecast(rows,labels,sessions,sessions[80],sessions[89],spec)
        for row in rows[79:]:
            labels[row["opportunity_id"]]["net_value_equity_bps"] = 1e9
        _, poisoned, proof2 = av.fit_forecast(rows,labels,sessions,sessions[80],sessions[89],spec)
    np.testing.assert_array_equal([p for _,p in original],[p for _,p in poisoned])
    assert proof["training_target_hash"] == proof2["training_target_hash"]


def test_rolling_window_maturity_and_no_future_fill_eligibility():
    sessions,rows,labels = training_fixture()
    labels[rows[60]["opportunity_id"]]["label_available_at"] = sessions[165]+"T15:30:00+09:00"
    idx,weights = av.training_indices(rows,labels,sessions,sessions[160],av.ModelSpec("r",window="rolling120"))
    assert min(idx)==39 and max(idx)==158 and 60 not in idx
    assert any(labels[rows[i]["opportunity_id"]]["buy_qty"]==0 for i in idx)
    assert (weights==1).all()
    _, weights = av.training_indices(rows,labels,sessions,sessions[160],av.ModelSpec("d",window="decay60"))
    assert weights[-1] > weights[0]
    with pytest.raises(ValueError,match="allowlist"):
        av.matrix(rows,["net_value_equity_bps"])


def test_quantity_policy_can_choose_smaller_profitable_action_or_cash():
    from scripts.kalcb_action_quantity_ranker import choose_quantities
    rows=[{"opportunity_id":f"2026-01-05/A/q{q}","base_opportunity_id":"2026-01-05/A",
        "feature_available_at":"2026-01-05T09:30:00+09:00",
        "action":{"quantity":int(100*q)},"opening_bars":[{"volume":4000}]*6,
        "features":{"action_quantity_scale":q,"action_low_volume_exit_slices":10}}
        for q in (.25,.5,1)]
    baseline={r['opportunity_id']:v for r,v in zip(rows,(10,40,100))}
    stress={r['opportunity_id']:v for r,v in zip(rows,(5,20,-5))}
    raw,_=choose_quantities(rows,baseline,stress)
    robust,_=choose_quantities(rows,baseline,stress,robust=True)
    assert next(iter(raw.values()))[0]['features']['action_quantity_scale']==1
    assert next(iter(robust.values()))[0]['features']['action_quantity_scale']==.5
    for key in stress:stress[key]=-1
    assert not choose_quantities(rows,baseline,stress,robust=True)[0]
    for key in baseline:baseline[key]=0
    assert not choose_quantities(rows,baseline,stress)[0]


def test_capacity_screen_never_turns_zero_share_liquidity_into_one_share():
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    row={'feature_available_at':'2026-01-05T09:30:00+09:00',
         'action':{'quantity':1},'opening_bars':[{'volume':199}]*6,
         'features':{'action_low_volume_exit_slices':1}}
    assert not capacity_ok(row)
    row['opening_bars']=[{'volume':200}]*6
    assert capacity_ok(row)
    row['action']['quantity']=0
    assert not capacity_ok(row)


def test_quantity_metadata_never_amplifies_or_accepts_nonfinite_size():
    cfg,snapshot,bars,account=preview_fixture()
    cfg=replace(cfg,entry_action_quantity_scale_enabled=True)
    for scale in (-.1,1.1,float('nan'),float('inf')):
        candidate=replace(snapshot.candidates[0],metadata={'action_quantity_scale':scale})
        with pytest.raises(ValueError,match='quantity scale'):
            preview_entry(bars,replace(snapshot,candidates=(candidate,)),cfg,account,
                available_at=bars[5].timestamp+timedelta(minutes=5))


def test_policy_calibration_ignores_future_returns_trades_and_failures():
    from scripts.kalcb_action_policy_calibration import select_policy
    sessions, _, _ = training_fixture()
    daily = [{'day':d,'equity':100_000_000*(1.001**(i+1)),'inventory':{},'pending':0}
             for i,d in enumerate(sessions)]
    trades = [{'entry_fill_time':d+'T09:30:00+09:00','exit_fill_time':d+'T15:00:00+09:00'}
              for d in sessions]
    case = {'daily':daily,'trades':trades,'residual_sessions':{},'invalid_execution_days':[]}
    cases = {'candidate':{'baseline':deepcopy(case),'combined':deepcopy(case)}}
    original = select_policy(cases,sessions,sessions[80])
    assert original[0] == 'candidate'
    for scenario in cases['candidate'].values():
        for row in scenario['daily'][79:]:
            row.update(equity=1,inventory={'A':999},pending=999)
        scenario['trades'] = scenario['trades'][:79]
        scenario['residual_sessions'] = {d:{'A':999} for d in sessions[79:]}
        scenario['invalid_execution_days'] = sessions[79:]
    assert select_policy(cases,sessions,sessions[80]) == original
    cases['candidate']['combined']['invalid_execution_days'].append(sessions[78])
    assert select_policy(cases,sessions,sessions[80])[0] == 'cash'
    # A genuinely revised action can compete separately from a failing policy.
    cases['smaller'] = {'baseline':deepcopy(case),'combined':deepcopy(case)}
    assert select_policy(cases,sessions,sessions[80])[0] == 'smaller'


def test_policy_calibration_requires_economic_and_sample_evidence():
    from scripts.kalcb_action_policy_calibration import select_policy
    sessions, _, _ = training_fixture()
    daily = [{'day':d,'equity':100_000_000,'inventory':{},'pending':0} for d in sessions]
    case = {'daily':daily,'trades':[],'residual_sessions':{},'invalid_execution_days':[]}
    cases = {'candidate':{'baseline':deepcopy(case),'combined':deepcopy(case)}}
    assert select_policy(cases,sessions,sessions[80])[0] == 'cash'
    assert select_policy(cases,sessions,sessions[20])[0] == 'cash'


def test_quantity_ridge_is_zero_at_zero_and_cannot_learn_future_labels():
    from threadpoolctl import threadpool_limits
    sessions,rows,labels=training_fixture()
    for i,row in enumerate(rows):
        row['features']['action_quantity_scale']=(.25,.5,1)[i%3]
    spec=av.ModelSpec('size','scaled_ridge')
    with threadpool_limits(limits=1):
        model,original,proof=av.fit_forecast(rows,labels,sessions,sessions[80],sessions[89],spec)
        for row in rows[79:]:labels[row['opportunity_id']]['net_value_equity_bps']=1e9
        _,poisoned,proof2=av.fit_forecast(rows,labels,sessions,sessions[80],sessions[89],spec)
    np.testing.assert_array_equal([p for _,p in original],[p for _,p in poisoned])
    assert proof['training_target_hash']==proof2['training_target_hash']
    x=av.matrix(rows[80:90],av.FEATURES)
    x[:,av.FEATURES.index('action_quantity_scale')]=0
    np.testing.assert_array_equal(model.predict(x),np.zeros(len(x)))


def test_feasibility_keeps_invalid_cash_outcomes_but_never_future_failures():
    from threadpoolctl import threadpool_limits
    sessions,rows,_=training_fixture()
    outcomes={r['opportunity_id']:{'risk_available_at':r['trade_date']+'T15:35:00+09:00',
                                  'unsafe':i%7==0,'label_valid':False if i%7==0 else True}
              for i,r in enumerate(rows)}
    outcomes[rows[50]['opportunity_id']]['risk_available_at']=sessions[165]+'T15:35:00+09:00'
    with threadpool_limits(limits=1):
        _,original,proof=av.fit_feasibility(rows,outcomes,sessions,sessions[80],sessions[89])
        for row in rows[79:]:outcomes[row['opportunity_id']]['unsafe']=not outcomes[row['opportunity_id']]['unsafe']
        _,poisoned,proof2=av.fit_feasibility(rows,outcomes,sessions,sessions[80],sessions[89])
    np.testing.assert_array_equal([p for _,p in original],[p for _,p in poisoned])
    assert proof['training_target_hash']==proof2['training_target_hash']
    assert proof['train_rows']==78 and proof['unsafe_rows']>0
    model=av.FeasibilityModel().fit(np.zeros((100,2)),np.zeros(100,dtype=bool))
    assert 0<model.predict(np.zeros((1,2)))[0]<1
