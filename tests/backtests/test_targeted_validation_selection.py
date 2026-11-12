from copy import deepcopy
import gzip
import json

from scripts.portfolio_targeted_validation import choose_finalist, development_checks, oos_checks, direct_support

RISK={'baseline_dd_pct':11,'adverse_dd_pct':12.5,'entries_min':515,
      'overnight_max':.8,'overnight_mean':.3,'worst_session_loss_pct':4.5}


def metric(ret=5,trades=100,dd=4):
    return {'return_pct':ret,'drawdown_pct':dd,'trades':trades,'overnight_max':.5,
            'overnight_mean':.2,'worst_session_pct':-2}


def test_short_oos_frequency_is_relative_and_negative_validation_cannot_pass():
    reference={s:metric() for s in ('baseline','combined')}
    assert oos_checks(reference,reference,RISK)['pass']
    tested=deepcopy(reference);tested['combined']['return_pct']=-1
    assert not oos_checks(tested,reference,RISK)['pass']
    tested=deepcopy(reference);tested['baseline']['trades']=97
    assert not oos_checks(tested,reference,RISK)['pass']
    tested=deepcopy(reference);tested['combined']['drawdown_pct']=4.251
    assert not oos_checks(tested,reference,RISK)['pass']


def test_no_qualifier_retains_control_and_return_tie_prefers_activity():
    assert choose_finalist({})=='control'
    portfolios={'high_return':{'baseline':metric(20,520),'combined':metric(10,520)},
                'more_active':{'baseline':metric(19,526),'combined':metric(9.1,526)},
                'outside_band':{'baseline':metric(21,600),'combined':metric(8.9,600)}}
    assert choose_finalist(portfolios)=='more_active'


def test_all_scenario_exposure_risk_is_enforced_even_with_high_return():
    m=metric(100,530,10);m['overnight_mean']=.3001
    assert not all(development_checks(m,'combined',RISK).values())


def test_changed_snapshots_without_native_requests_do_not_inflate_support(tmp_path):
    evidence={'decisions':[{'identity':f'2026-02-{d:02d}/005930','changed':True} for d in range(1,21)]}
    (tmp_path/'marginal_quantity_evidence.json').write_text(json.dumps(evidence))
    with gzip.open(tmp_path/'allocation_events.jsonl.gz','wt') as f:
        f.write(json.dumps({'timestamp':'2026-02-01T09:25:00+09:00','symbol':'005930','strategy_id':'KALCB','native_qty':1})+'\n')
    support=direct_support(tmp_path,'k_marginal')
    assert support['issuer_days']==support['sessions']==1
    assert not support['pass']
