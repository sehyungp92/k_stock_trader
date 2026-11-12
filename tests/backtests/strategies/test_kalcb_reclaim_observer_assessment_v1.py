from copy import deepcopy
import json
import sqlite3
import pytest

from backtests.strategies.kalcb.reclaim_observer_assessment_v1 import (
    exposure_positions, preview_review, reconstruct_mtm, completion_gate, exposure_comparison,decision_accounts)


def fixture():
    e={'order_id':'o1','submitted_at':'2026-04-01T09:25:00+09:00','decision_available_at':'2026-04-01T09:30:00+09:00',
       'action':{'symbol':'A','qty':10,'metadata':{'entry_price_ref':100.,'daily_atr':5.,'risk_per_share':2.,'entry_route':'first30_open_anchor'}},
       'account_before':{'actual_equity':100000000.,'actual_cash':-100.,'buying_power_available':100000.,
           'positions':[],'pending_orders':[],'session_meta':{}}}
    fills=[]
    for i,(time,side,qty,price) in enumerate([('09:30','BUY',6,100.),('09:35','BUY',4,101.),('09:40','SELL',3,102.),('09:45','SELL',7,98.)]):
        fills.append({'timestamp':'2026-04-01T'+time+':00+09:00','order_id':'o1' if side=='BUY' else 's'+str(i),
            'symbol':'A','qty':qty,'price':price,'side':side,'metadata':{'execution_id':'f'+str(i),'fee':1.}})
    # Entry cost1004, exit992, four fees => net -16. Partial exits are two trade legs.
    trades=[{'symbol':'A','entry_fill_time':fills[0]['timestamp'],'qty':3,'gross_pnl':4.8,'net_pnl':3.,'commission':1.8},
            {'symbol':'A','entry_fill_time':fills[0]['timestamp'],'qty':7,'gross_pnl':-16.8,'net_pnl':-19.,'commission':2.2}]
    return {'entry_exposures':[e]},fills,trades


def test_partial_fills_and_exits_have_one_pnl_and_borrowed_cash_is_valid():
    evidence,fills,trades=fixture();rows,ids=exposure_positions(evidence,fills,trades)
    assert len(rows)==1 and len(ids)==4
    assert rows[0]['net_pnl_krw']==-16 and rows[0]['qty']==10
    assert rows[0]['entry_cash_krw']==-100


@pytest.mark.parametrize('poison',['execution','order','trade','early','excess_exit','missing_exposure'])
def test_rejects_double_count_and_unbound_evidence(poison):
    e,f,t=fixture()
    if poison=='execution':f[1]['metadata']['execution_id']=f[0]['metadata']['execution_id']
    if poison=='order':e['entry_exposures'].append(deepcopy(e['entry_exposures'][0]))
    if poison=='trade':t.append(deepcopy(t[0]))
    if poison=='early':e['entry_exposures'][0]['decision_available_at']='2026-04-01T09:35:00+09:00'
    if poison=='excess_exit':f[-1]['qty']=8
    if poison=='missing_exposure':e['entry_exposures']=[]
    with pytest.raises(ValueError):exposure_positions(e,f,t)


def test_copied_alternatives_are_not_fills_and_missing_clocks_retained():
    ids=['2026-04-01/A/minute90/q0.25','2026-04-01/B/minute90/q0.25','2026-04-02/C/minute90/q0.25']
    expected=[({'opportunity_id':i},[]) for i in ids]
    rows=[{'opportunity_id':i,'native_order_submitted':True,'new_orders':[{'symbol':i.split('/')[1],'side':'BUY'}]} for i in ids[:2]]
    rows.append({'opportunity_id':ids[2],'reason':'native_decision_clock_absent'})
    evidence={'previews':rows,'summary':{'copied_order_previews':2}}
    r=preview_review(evidence,expected)
    assert r['copied_submissions']==2 and r['copied_submission_sessions']==1
    assert r['reason_counts']['native_decision_clock_absent']==1
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):preview_review(evidence,expected)


def test_forecast_diagnostics_cannot_be_replaced():
    expected=[({'opportunity_id':'2026-04-01/A/minute90/q0.25'},[{'recipe':'raw48','admit':False}])]
    evidence={'previews':[{'opportunity_id':expected[0][0]['opportunity_id'],'forecasts':[{'recipe':'raw48','admit':True}]}],
        'summary':{'copied_order_previews':0}}
    with pytest.raises(ValueError):preview_review(evidence,expected)


def test_every_raw_mtm_mark_and_episode_reconciles(tmp_path):
    e,f,t=fixture();positions,mapping=exposure_positions(e,f,t)
    curve=[[f[0]['timestamp'],100000005.],[f[1]['timestamp'],100000004.],[f[2]['timestamp'],100000013.],[f[3]['timestamp'],99999984.]]
    (tmp_path/'fills.json').write_text(json.dumps(f));(tmp_path/'equity.json').write_text(json.dumps(curve))
    db=tmp_path/'bars.db'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE bars(symbol TEXT,ts TEXT,c REAL,day TEXT)')
        c.executemany('INSERT INTO bars VALUES(?,?,?,?)',[('A',f[i]['timestamp'],price,'2026-04-01') for i,price in enumerate([101,101,102,98])])
    episodes=[{'peak_at':curve[2][0],'trough_at':curve[3][0],'peak_equity':curve[2][1],'trough_equity':curve[3][1]}]
    r=reconstruct_mtm(tmp_path,positions,mapping,[db,db],episodes)
    assert r['all_native_marks_reconciled']==4
    assert r['episodes'][0]['position_contributions'][0]['mtm_change_krw']==-29
    curve[1][1]+=1;(tmp_path/'equity.json').write_text(json.dumps(curve))
    with pytest.raises(ValueError):reconstruct_mtm(tmp_path,positions,mapping,[db,db],episodes)


def test_gate_waits_for_exact_supervisor_and_complete_unique_cases(tmp_path):
    for name,data in [('supervisor_start.json',{'pid':1}),('status.json',{'stage':'awaiting_observer_assessment','child':None}),
        ('stage_completion.json',{'complete':True,'native_cases':8,'exact_parent_bridges':8})]:
        (tmp_path/name).write_text(json.dumps(data))
    rows=[{'name':n,'period':p,'scenario':s,'is_identity':True,'metrics':{'audit_status':'passed'}}
        for n in ('control','control_prior125') for p in ('is','oos') for s in ('baseline','combined')]
    (tmp_path/'progress.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='supervisor'):completion_gate(tmp_path,lambda _:True)
    assert len(completion_gate(tmp_path,lambda _:False))==8
    rows[-1]=rows[0];(tmp_path/'progress.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='duplicate'):completion_gate(tmp_path,lambda _:False)


def test_exposure_bins_use_is_only_and_preserve_all_positions():
    e,f,t=fixture();rows,_=exposure_positions(e,f,t)
    isrows=[{**rows[0],'atr_over_entry_reference':v} for v in (.01,.02,.03,.04)]
    oos=[{**rows[0],'atr_over_entry_reference':1.}]
    r=exposure_comparison(isrows,oos)['routes']['first30_open_anchor']['features']['atr_over_entry_reference']
    assert r['is_only_quartile_cutpoints']==pytest.approx([.0175,.025,.0325])
    assert r['oos_bins']['3']['positions']==1


def test_decision_clock_does_not_use_last_mark_or_reject_borrowing():
    row={'submitted_at':'2026-04-01T09:35:00+09:00','account_before':{
        'actual_cash':-100000001.,'actual_equity':99999999.,'buying_power_leverage':2,
        'buying_power_available':0.,'last_mark_timestamp':'2026-04-01T09:25:00+09:00',
        'positions':[{'symbol':'A','qty':2000000,'mark':100.,'marked_notional':200000000.}]}}
    fill={'timestamp':'2026-04-01T09:30:00+09:00','side':'BUY','symbol':'A','qty':2000000,'price':100.,'metadata':{'fee':1.}}
    assert decision_accounts({'entry_exposures':[row]},[fill])['decision_clock_cash_inventory_checks']==1
    row['account_before']['actual_cash']+=1
    with pytest.raises(ValueError,match='cash'):decision_accounts({'entry_exposures':[row]},[fill])
