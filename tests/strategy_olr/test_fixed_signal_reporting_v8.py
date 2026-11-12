"""Reporting arithmetic contracts; no new market outcomes are consumed."""
from datetime import date,timedelta
import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'scripts/olr_fixed_signal_report_v8.py'
spec=importlib.util.spec_from_file_location('fixed_report_test_module',path)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def inputs():
    design=dict(blocks=[('2025-08-07','2025-10-31'),('2025-11-01','2025-12-31'),('2026-01-01','2026-03-31')],
        scenarios={'baseline':[.01,5],'combined':[.005,15]},target_budgets=[1_000_000,20_000_000])
    rows=[]
    for lo,hi in design['blocks']:
        for i in range(12):
            day=str(date.fromisoformat(lo)+timedelta(days=i))
            for symbol in ('a','b'):
                for scenario in design['scenarios']:
                    for budget in design['target_budgets']:
                        rows.append(dict(day=day,symbol=symbol,scenario=scenario,budget=budget,status='matured',
                            available_for_training_after=day+'T23:59:59+09:00',net_bps=10,net_pnl=budget*.001,
                            budget_net_bps=10,filled_notional=budget,fixed_target_risk={'max_drawdown_over_target_pct':1}))
    return rows,design


def test_complete_paired_support_uses_dates_and_all_four_lanes():
    rows,design=inputs();summary=report.summarize(rows,design)
    assert summary['separate_allocation_calibration_may_be_considered']
    assert not summary['portfolio_experiment_eligible']
    assert all(c['date_weighted_net_bps']==10 for c in summary['cells'])


def test_missing_lane_excludes_whole_opportunity_not_zero_fill():
    rows,design=inputs();rows[0].update(status='unfilled',net_pnl=None,net_bps=None,available_for_training_after=None)
    summary=report.summarize(rows,design)
    assert summary['paired_complete_opportunities']==71
    assert summary['chronological_support'][0]['paired_mature']==23
    assert all(c['date_weighted_net_bps']==10 for c in summary['cells'])


def test_labels_maturing_after_block_end_are_not_available_at_boundary():
    rows,design=inputs()
    for r in rows:
        if r['day'].startswith('2025-08'):r['available_for_training_after']='2025-11-01T23:59:59+09:00'
    result=report.summarize(rows,design)
    assert result['chronological_support'][0]['paired_mature']==0
    assert not result['separate_allocation_calibration_may_be_considered']


def test_one_negative_stressed_block_prevents_advancement():
    rows,design=inputs()
    for r in rows:
        if r['day'].startswith('2025-11') and r['scenario']=='combined' and r['budget']==1_000_000:r['net_bps']=-1
    result=report.summarize(rows,design)
    assert result['gates']['each_block_paired_support']
    assert not result['gates']['each_block_small_stressed_net_positive']


def test_date_weighting_does_not_overweight_high_candidate_count_day():
    rows,design=inputs()
    first=rows[0]['day']
    for r in rows:
        if r['day']==first:r['net_bps']=130
    extra=[{**r,'symbol':'c'} for r in rows if r['day']==first]
    result=report.summarize(rows+extra,design)
    assert all(c['date_weighted_net_bps']==20 for c in result['cells'][:4])


def test_native_fill_datetime_type_restored_without_changing_clock_or_economics():
    import copy
    rows=[{'event':'capacity','timestamp':'2025-08-29T14:35:00+09:00'},
          {'event':'fill','fill':{'timestamp':'2025-08-29 14:35:00+09:00','qty':7,'price':10.5}}]
    original=copy.deepcopy(rows)
    restored=report.restore_journal_types(rows)
    assert restored[1]['fill']['timestamp'].isoformat()==original[0]['timestamp']
    assert restored[0]==original[0]
    assert {k:v for k,v in restored[1]['fill'].items() if k!='timestamp'}=={'qty':7,'price':10.5}


def test_independent_recollection_is_exact_across_python_hash_seeds(tmp_path):
    import json,os,subprocess,sys
    rows,design=inputs()
    for i,r in enumerate(rows):
        r['net_bps']=(i%19+1)/7
        r['net_pnl']=(i%23+1)/11
    source=tmp_path/'input.json'
    source.write_text(json.dumps({'rows':rows,'design':design}),encoding='utf-8')
    code="""import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('report',sys.argv[1]);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with open(sys.argv[2],encoding='utf-8') as f:inputs=json.load(f)
print(json.dumps(mod.summarize(inputs['rows'],inputs['design']),sort_keys=True))
"""
    outputs=[subprocess.check_output([sys.executable,'-B','-c',code,str(path),str(source)],
        env={**os.environ,'PYTHONHASHSEED':str(seed)}) for seed in (1,17,71)]
    assert outputs[0]==outputs[1]==outputs[2]
