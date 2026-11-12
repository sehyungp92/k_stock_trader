import json
from copy import deepcopy

import pytest

from scripts import portfolio_frozen_inference as frozen
from scripts import portfolio_targeted_result as result
from backtests.engine.retained_source import build_evidence


def test_frozen_model_cannot_be_applied_before_its_fit(tmp_path):
    with pytest.raises(ValueError,match='before its fit'):
        frozen.infer(tmp_path,[],[],{},{},['2026-03-09'],'test')


@pytest.mark.parametrize('poison',[
    {'train_through':'2026-03-10'},
    {'label_available_through':'2026-03-10T00:00:00+09:00'},
])
def test_forward_inference_refuses_unmatured_frozen_model_before_loading(tmp_path,poison):
    proof={'name':'interaction','scenario':'baseline','valid_from':frozen.FIT,
           'train_through':'2026-03-06','label_available_through':'2026-03-06T15:30:00+09:00',
           'fit_at':'2026-03-10T00:00:00+09:00',**poison}
    (tmp_path/'opening_chronology.json').write_text(json.dumps({'folds':[proof]}))
    row={'opportunity_id':'2026-06-01/005930/q0.5','trade_date':'2026-06-01',
         'features':{'action_quantity_scale':.5}}
    with pytest.raises(ValueError,match='Noncausal'):
        frozen.predictions(tmp_path,[row],'opening')


def test_support_offset_uses_raw_sign_even_when_calibration_flips_sign():
    raw={'A':{'quantity_scale':.5,'baseline_value_bps':-1,'combined_value_bps':2}}
    before=deepcopy(raw)
    cal={'mode':'support_offset','cohorts':{
        'baseline/q0.5/nonpositive':{'offset_bps':10},'combined/q0.5/nonpositive':{'offset_bps':20},
        'baseline/q0.5/positive':{'offset_bps':100},'combined/q0.5/positive':{'offset_bps':200}}}
    value=frozen.calibrated(raw,cal)['A']
    assert (value['baseline_value_bps'],value['combined_value_bps'])==(9,22)
    assert value['raw_score_support']=='nonpositive' and raw==before


@pytest.mark.parametrize('stamp,expected',[
    ('20260512','2026-05-12'),('20260710','2026-07-10'),('2026-07-10','2026-07-10'),
])
def test_retained_basis_tracks_declared_archive_instead_of_old_fixed_date(tmp_path,stamp,expected):
    parsers=['scripts/update_kis_intraday.py','strategy_common/daily_lrs_parquet.py',
             'backtests/core/anchored_recovery.py','backtests/core/session_replacement.py']
    for name in parsers:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture')
    contract={'daily_data_root':'data','data_end':'2026-07-10','daily_snapshot_end':stamp,
              'intraday_files':{},'files':{},'market_input_hash':'test'}
    build_evidence(tmp_path,contract,tmp_path/'evidence')
    market=json.loads((tmp_path/'evidence/market_contract.json').read_text())
    basis=market['normalization']['target_basis']
    assert basis['adjustment_as_of']==expected
    assert basis['id'].endswith(expected.replace('-',''))


def test_failed_validation_records_failure_without_publishing_or_reselecting(tmp_path,monkeypatch):
    decision={'promotion_eligible':False,'finalist':'k_marginal','acceptance':{'pass':False}}
    monkeypatch.setattr(result,'assess',lambda root:decision)
    monkeypatch.setattr(result,'make_profile',lambda root:pytest.fail('Failed validation cannot publish'))
    result.publish(tmp_path)
    assert json.loads((tmp_path/'acceptance.json').read_text())==decision
    assert not (tmp_path/'previous_selected_pointer.json').exists()


def test_paired_resampling_preserves_identical_account_paths(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    daily=[{'day':f'2026-06-{d:02d}','equity':25_000_000*(1+.001*d)} for d in range(1,21)]
    for p in (a,b):(p/'daily.json').write_text(json.dumps(daily))
    uncertainty=result.paired_uncertainty(a,b)
    for record in uncertainty['blocks_in_sessions'].values():
        assert set(record['return_difference_quantiles_pp'].values())=={0}
        assert record['fraction_positive']==0


def test_capital_export_keeps_unavailable_requests_and_realized_labels_out():
    from scripts.portfolio_opportunity_audit import decision_record
    row={'timestamp':'2026-06-01T09:30:00+09:00','order':{
        'order_id':'a','strategy_id':'KALCB','symbol':'005930','metadata':{
            'frontier_selection_score':12,'frontier_rank':2,'future_net_pnl':999}},
        'offered_qty':10,'approved_qty':0,'unit_cost':100,
        'reservation':{'binding_constraints':['physical_funding']},'config':{},
        'ledger':{'equity':1000,'held_gross':1000,'budgets':{},'remaining_funding':0,
                  'committed_planned_kalcb_risk':10,'allocations':{}}}
    submission={'submitted_order_id':'a','available_at':row['timestamp'],'native_qty':10}
    decision=decision_record(row,submission)
    assert decision['approved_quantity']==0 and decision['realized_outcomes_included'] is False
    assert 'future_net_pnl' not in json.dumps(decision)
    submission['available_at']='2026-06-01T09:35:00+09:00'
    with pytest.raises(ValueError,match='precedes availability'):decision_record(row,submission)


def test_failed_source_gate_prevents_validation_directory_or_performance_replay(tmp_path):
    from scripts.portfolio_forward_validation import freeze_replay
    folder=tmp_path/'provider_basis_audit';folder.mkdir()
    (folder/'audit.json').write_text(json.dumps({'source_gate_pass':False}))
    with pytest.raises(ValueError,match='preserve the holdout'):
        freeze_replay(tmp_path)
    assert not (tmp_path/'validation').exists()


def test_claimed_source_pass_cannot_bypass_a_warmup_basis_failure(tmp_path):
    from scripts.portfolio_forward_validation import freeze_replay
    folder=tmp_path/'provider_basis_audit';folder.mkdir()
    (folder/'audit.json').write_text(json.dumps({'source_gate_pass':True}))
    acceptance=tmp_path/'forward_inputs/files/research/forward_acceptance/acceptance.json'
    acceptance.parent.mkdir(parents=True)
    acceptance.write_text(json.dumps({'basis_failures':[{'date':'2026-05-29','symbol':'068270'}]}))
    with pytest.raises(ValueError,match='warm-up or lookback'):
        freeze_replay(tmp_path)
    assert not (tmp_path/'validation').exists()


def test_invalid_source_is_untested_validation_not_a_return_failure(tmp_path):
    pointer=tmp_path/'pointer.json';pointer.write_text('{}')
    (tmp_path/'protocol.json').write_text(json.dumps({'references':{'selected_pointer':result.bound(pointer)}}))
    (tmp_path/'development_selection.json').write_text('{}')
    folder=tmp_path/'provider_basis_audit';folder.mkdir()
    audit=folder/'audit.json'
    audit.write_text(json.dumps({'source_gate_pass':False,'original_and_adjusted_controls':{}}))
    (tmp_path/'source_basis_review.json').write_text(json.dumps({
        'source_gate_pass':False,'provider_audit':result.bound(audit),'dependencies':{}}))
    decision=result.source_blocked_decision(tmp_path,{'finalist':'k_marginal'})
    assert decision['acceptance']['pass'] is None
    assert decision['acceptance']['status']=='not_evaluated_source_gate_failed'
    assert decision['outcomes']=={} and not decision['validation_window_now_consumed']
    assert not decision['promotion_eligible'] and pointer.read_text()=='{}'
    (tmp_path/'validation').mkdir()
    with pytest.raises(ValueError,match='unopened'):
        result.source_blocked_decision(tmp_path,{'finalist':'k_marginal'})
