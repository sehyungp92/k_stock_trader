from pathlib import Path
import importlib.util
import math
import pytest
import numpy as np

spec=importlib.util.spec_from_file_location('information',Path(__file__).resolve().parents[2]/'scripts/olr_sector_information_v11.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def sample():
    rows=[]
    for month in ('2025-08','2025-11','2026-01'):
        for i in range(1,13):
            breadth=.8 if i%2 else .4;extra=-.3 if i%3 else 0.
            rows.append(dict(day=f'{month}-{i:02}',symbol='A',x=[breadth,extra],y=20+30*breadth+400*extra,
                available=f'{month}-{i+1:02}T23:59:59+09:00'))
    return rows


def test_closed_form_matches_independent_normal_equations():
    rows=sample()[:12];w,n=m.weights(rows)
    for dim in (1,2):
        fit=m.fit(rows,dim);x=np.array([[1.]+r['x'][:dim] for r in rows]);y=np.array([r['y'] for r in rows])
        variance=np.var(x[:,1:],axis=0)
        coef=np.linalg.solve(x.T@x+np.diag([0.]+list(12.*variance)),x.T@y)
        assert np.allclose(coef,[fit['intercept']]+fit['slopes'])


def test_duplicate_identical_same_date_does_not_change_weight():
    rows=sample()[:12];a=m.fit(rows,2);b=m.fit(rows+[dict(rows[0],symbol='B')],2)
    assert a['intercept']==pytest.approx(b['intercept']) and a['slopes']==pytest.approx(b['slopes'])


def test_future_labels_cannot_change_first_fold_fit():
    rows=sample();first=m.analyze(rows)['folds'][0]
    changed=[dict(r,y=1e12) if r['day']>='2026-01-01' else r for r in rows]
    other=m.analyze(changed)['folds'][0]
    assert first==other


def test_unmatured_training_labels_are_excluded_before_fit():
    rows=sample();rows[0]['available']='2025-12-01T00:00:00+09:00'
    _,train,_,_=m.split(rows,'2025-11-01','2025-12-31')
    assert (rows[0]['day'],rows[0]['symbol']) not in {(r['day'],r['symbol']) for r in train}


def test_missing_source_or_payoff_is_not_zero_and_support_stops_fit():
    rows=sample()
    for r in rows:
        if r['day'].startswith('2025-11') and int(r['day'][-2:])<=4:r.update(x=None,y=None,available=None)
    result=m.analyze(rows)
    assert not result['support_passed'] and not result['incremental_volume_information_supported']
    assert all('baseline' not in f for f in result['folds'])


def test_end_to_end_grouping_net_units_causality_and_source_support():
    day='2025-08-07';ledger=[];lanes=[]
    for symbol in ('A','B'):
        ledger.append(dict(day=day,symbol=symbol,own={'prior_count':20},available_at=day+'T14:30:00+09:00',
            peer_support={'supported':True,'complete_fixed_peer_coverage':True,'contributing_peers':['B' if symbol=='A' else 'A']},
            history_native={'breadth':.8,'participation':.4},legacy_native={'participation':.8}))
        for scenario,budget in ((s,b) for s in ('baseline','combined') for b in (1_000_000,20_000_000)):
            lanes.append(dict(day=day,symbol=symbol,scenario=scenario,budget=budget,status='matured',
                filled_notional=100000,net_pnl=500,net_bps=50.,native_signal_time=day+'T14:35:00+09:00',available_for_training_after='2025-08-08T23:59:59+09:00'))
    rows=m.records(lanes,ledger);assert len(rows)==2 and rows[0]['y']==50 and rows[0]['x']==[.8,-.4]
    lanes[0]['status']='unfilled';assert m.records(lanes,ledger)[0]['y'] is None
    ledger[1]['own']['prior_count']=19
    assert m.records(lanes,ledger)[0]['x'] is None
    with pytest.raises(AssertionError):m.records(lanes+[lanes[0]],ledger)


def test_constant_incremental_feature_cannot_qualify():
    rows=[dict(r,x=[r['x'][0],0.]) for r in sample()]
    result=m.analyze(rows)
    assert result['support_passed'] and not result['incremental_volume_information_supported']
    for f in result['folds']:
        assert abs(f['incremental_coefficient'])<1e-10 and abs(f['relative_mse_gain'])<1e-10


def test_units_do_not_change_predictions_or_shrinkage():
    rows=sample()[:12];scaled=[dict(r,x=[r['x'][0]*100,r['x'][1]/100]) for r in rows]
    assert m.evaluate(m.fit(rows,2),rows)==pytest.approx(m.evaluate(m.fit(scaled,2),scaled))


def test_positive_information_can_pass_but_opposite_direction_rejects():
    result=m.analyze(sample());assert result['incremental_volume_information_supported']
    negative=[dict(r,y=20+30*r['x'][0]-400*r['x'][1]) for r in sample()]
    result=m.analyze(negative);assert not result['incremental_volume_information_supported']
    assert all(not f['gates']['volume_confirmation_coefficient_positive'] for f in result['folds'])
