from copy import deepcopy
import json,gzip
import pytest
from backtests.strategies.kalcb.pending_funding_assessment_v1 import (completion_gate,
    path_comparison,activity_summary,reservation_support)


def pos(key,qty,pnl,cost=1000000):
    return {'position_id':key,'symbol':key,'entry_day':'2026-04-01','qty':qty,'net_pnl_krw':pnl,
        'gross_pnl_krw':pnl+10,'fees_krw':10,'entry_cost_krw':cost,'entry_equity_krw':100000000,
        'entry_order_ids':[key]}


def test_paths_account_once_and_keep_sacrificed_winners():
    p=[pos('A',10,500),pos('B',20,-200)];c=[pos('A',5,200),pos('C',1,2)]
    r=path_comparison(c,p)
    assert r['delta_net_pnl_krw']==-98
    assert r['group_totals']['absent']['delta_net_pnl_krw']==200
    assert r['parent_top10'][0]['candidate_same_entry_net_pnl_krw']==200
    with pytest.raises(ValueError):path_comparison(c+c,p)


def test_profitable_activity_exposes_tiny_and_one_share_counts():
    rows=[pos('A',1,5,2000000),pos('B',5,4,20000),pos('C',10,8,1000000),pos('D',20,-2,5000000)]
    r=activity_summary(rows,69)
    assert r['profitable_positions']==3 and r['one_share']['profitable_positions']==1
    assert r['profitable_size_sensitivity']['0.01']['profitable_positions']==1
    assert r['profitable_size_sensitivity']['0.001']['profitable_positions']==1
    assert r['profitable_entry_notional_krw']==3020000


def test_support_collapses_partial_fills_and_ignores_order_updates(tmp_path):
    m={'original_qty':10,'capped_qty':5,'equity':100.,'reserved_gross':150.,'completed_close':10.,
        'remaining_notional':50.,'route':'first30_open_anchor','limit':2.}
    e={'entry_exposures':[{'order_id':'A','submitted_at':'2026-04-01T09:25:00',
        'decision_available_at':'2026-04-01T09:30:00','action':{'symbol':'A','qty':5,
        'metadata':{'entry_route':'first30_open_anchor','pending_funding_reservation':m}},
        'account_before':{'actual_equity':100.,'actual_cash':100.,'buying_power_available':200.}}]}
    fs=[{'side':'BUY','order_id':'A','qty':q,'price':10.} for q in (2,3)]
    d={'timestamp':'2026-04-01T09:25:00','symbol':'B','reason':'qty_zero','decision_code':'entry_rejected',
        'metadata':{'pending_funding_reservation':{**m,'capped_qty':0,'reserved_gross':200.,'remaining_notional':0.}}}
    with gzip.open(tmp_path/'decisions.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.write(json.dumps(d)+'\n');stream.write(json.dumps({**d,'decision_code':'order_update'})+'\n')
    r=reservation_support(tmp_path,e,fs,[pos('A',5,-5)])
    assert r['changed_filled_orders']==r['changed_filled_positions']==r['zero_blocks']==1
    assert r['unique_changed_position_outcomes']['net_pnl_krw']==-5
    bad=deepcopy(e);bad['entry_exposures'][0]['action']['qty']=6
    with pytest.raises(ValueError,match='quantity'):reservation_support(tmp_path,bad,fs,[pos('A',5,-5)])


def test_completion_requires_all_cases_and_exact_exit(tmp_path):
    rows=[{'name':n,'period':p,'scenario':s,'is_identity':n.startswith('control'),'metrics':{'audit_status':'passed'}}
        for n in ('control','control_prior125','pending_100','pending_125') for p in ('is','oos') for s in ('baseline','combined')]
    for n,v in [('supervisor_start.json',{'pid':1}),('status.json',{'stage':'awaiting_full_economic_assessment','child':None}),
        ('stage_completion.json',{'complete':True,'native_cases':16,'exact_parent_bridges':8}),('progress.json',rows)]:
        (tmp_path/n).write_text(json.dumps(v),encoding='utf-8')
    with pytest.raises(ValueError,match='supervisor'):completion_gate(tmp_path,lambda _:True)
    assert len(completion_gate(tmp_path,lambda _:False))==16
    rows[-1]=rows[0];(tmp_path/'progress.json').write_text(json.dumps(rows),encoding='utf-8')
    with pytest.raises(ValueError,match='duplicate'):completion_gate(tmp_path,lambda _:False)
