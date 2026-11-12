import copy

import pytest

from scripts import olr_round1_mechanism_study as study


def original_plan():
    base = {study.EXIT:{'mode':'managed','hard_stop_enabled':False,'mfe_fade_start_r':1.},
            study.ENTRY:{'mode':'confirm_next_bar'}, study.RULES:[{'max_score':325},
                {'min_score':400,'max_score':500,'allowed_sectors':['IT']},
                {'min_score':400,'max_score':500,'sector_admission':{'mode':'dynamic_confirmed_rotation'}},
                {'min_score':650}],
            'olr.allocation.max_position_pct':.65, 'olr.research.min_sector_participation':.5,
            'olr.signal.min_parent_20d_return_pct':-20}
    labels = ['current_round1','remove_capacity_recycle_liquidity_only',
        'flip_olr.trade_plan.exit/hard_stop_enabled', 'remove_olr.allocation.max_position_pct',
        'perturb_olr.allocation.max_position_pct_x0.8','remove_olr.allocation.rank_decay',
        'perturb_olr.research.min_sector_participation_x0.8','remove_olr.research.min_sector_participation',
        'perturb_olr.signal.min_parent_20d_return_pct_x1.2','remove_olr.signal.min_parent_20d_return_pct',
        'remove_band_3','remove_band_2','no_score_bands','remove_olr.trade_plan.exit/mfe_fade_start_r',
        'cumulative_legacy_round_1','historical_nested_leaf','other_leaf']
    candidates = []
    for i,label in enumerate(labels):
        mutations = copy.deepcopy(base) if i==0 else {**copy.deepcopy(base),'test_field':i}
        key=study.native.sig(mutations)
        candidates.append({'id':'c'+key[:16], 'mutations_hash':key, 'mutations':mutations,
            'label':label, 'kind':'cumulative' if 'cumulative' in label else 'leaf_ablation','aliases':[]})
    return {'baseline':base,'baseline_id':candidates[0]['id'],'candidates':candidates}


def test_revision_preserves_every_historical_definition_and_schedules_each_once():
    original=original_plan()
    before=copy.deepcopy(original)
    plan=study.build_plan(original)
    assert original==before
    known={c['id']:c for c in plan['candidates']}
    for row in original['candidates']:
        assert known[row['id']]['mutations']==row['mutations']
    ids=[cid for phase in plan['phases'] for cid in phase.get('ids',[])]
    assert len(ids)==len(set(ids))
    assert set(ids)==set(known)
    assert set(plan['original_candidate_ids'])=={c['id'] for c in original['candidates']}
    assert len(plan['activation_contrasts'])==9


def test_stops_have_matched_inactive_controls_not_just_dormant_parameter_trials():
    plan=study.build_plan(original_plan())
    known={c['id']:c for c in plan['candidates']}
    for pair in plan['activation_contrasts']:
        left=known[pair['control_id']]['mutations']; right=known[pair['treatment_id']]['mutations']
        assert study.native.mutation_edits(left,right)==[((study.EXIT,'hard_stop_enabled'),True,False)]
        assert right[study.EXIT]['stop_mode']==pair['mode']


def test_all_historical_contrasts_precede_new_targeted_mechanisms():
    plan=study.build_plan(original_plan())
    # A semantic historical restoration remains historical even when deduplicated
    # against a definition previously placed in the targeted phase.
    targeted=next(p for p in plan['phases'] if p['name']=='targeted_mechanisms')['ids'][0]
    plan['semantic_band_coverage']=[{'candidate_id':targeted}]
    phases=study.historical_first_phases(plan)
    historical=set(plan['original_candidate_ids'])|{targeted}
    assert {cid for p in phases[:2] for cid in p['ids']}==historical
    assert not historical & set(phases[2]['ids'])
    ids=[cid for p in phases for cid in p.get('ids',[])]
    assert len(ids)==len(set(ids))==len(plan['candidates'])


def test_complement_screen_keeps_is_gain_and_one_window_frequency():
    def pair(a,b,na=176,nb=47):
        return {'train':{'return_pct':a,'trades':na},'oos':{'return_pct':b,'trades':nb}}
    pairs={'control':pair(55,-3), 'is_complement':pair(60,-3),
           'is_frequency':pair(55,-3,181,47), 'oos_frequency':pair(55,-3,176,51),
           'oos_gain':pair(53,5), 'dormant':pair(55,-3), 'too_costly':pair(40,10)}
    known={cid:{'mechanism_family':'same'} for cid in pairs}
    rows=study.component_screen(known,pairs,'control')
    eligible=[r for r in rows if r['eligible_component']]
    expected={'is_complement','is_frequency','oos_frequency','oos_gain'}
    assert {r['id'] for r in eligible}==expected
    assert set(study.select_components(eligible))==expected


def test_every_factorial_retains_both_single_components():
    plan=study.build_plan(original_plan())
    known={c['id']:c for c in plan['candidates']}
    for trial in plan['factorials']:
        a,b,both=(known[trial[k]]['mutations'] for k in ('left_id','right_id','both_id'))
        assert study.native.combine_disjoint(plan['baseline'],a,b)==both


def test_cached_result_is_not_accepted_if_audit_or_metrics_are_changed(tmp_path):
    c={'id':'c1','mutations_hash':'h'}
    folder=tmp_path/'cases/c1/oos/100000000_baseline'
    result={'return_pct':1,'trades':8,'drawdown_pct':2,'audit_status':'passed','source_generation_hash':'source',
            'execution_validity_pass':True,'terminal_pending':0,'terminal_inventory':{},'sessions':2}
    study.native.write(folder/'result.json',result)
    study.native.write(folder/'execution_audit.json',{'audit_pass':True})
    study.native.write(folder/'completion.json',{'source_generation_hash':'source','result_sha256':study.native.sha(folder/'result.json'),
        'audit_sha256':study.native.sha(folder/'execution_audit.json')})
    study.native.write(folder/'assessment.json',{**result,'status':'completed','qualified':True})
    study.native.write(tmp_path/'registrations/oos/c1_100000000/candidate.json',c)
    study.native.write(tmp_path/'protocol.json',{'windows':{'oos':['2026-04-01','2026-07-10']}})
    study.native.write(tmp_path/'registrations/oos/c1_100000000/study.json',
        {'source_generation_hash':'source','configuration':{'date_range':{'start':'2026-04-01','end':'2026-07-10'}}})
    study.native.write(folder/'daily.json',[{'day':'2026-04-01'},{'day':'2026-07-10'}])
    study.validate_cached(tmp_path,c,'oos')
    study.native.write(folder/'assessment.json',{**result,'return_pct':20,'status':'completed','qualified':True})
    with pytest.raises(ValueError,match='no longer matches'):
        study.validate_cached(tmp_path,c,'oos')


def test_invalid_configuration_and_failed_native_replay_are_different(tmp_path,monkeypatch):
    c={'id':'c1','label':'c1','mutations_hash':'h'}
    def invalid(*args):
        raise ValueError('dependent feature requires allocation')
    monkeypatch.setattr(study.native,'candidate_preflight',invalid)
    assert study.evaluate(tmp_path,c,'oos')['status']=='invalid_configuration'
    monkeypatch.setattr(study.native,'candidate_preflight',lambda *args: {})
    monkeypatch.setattr(study.native,'evaluate',lambda *args:{'qualified':False,'output':'audit_failed'})
    with pytest.raises(RuntimeError,match='repair before continuing'):
        study.evaluate(tmp_path,c,'oos')


def capacity_rejection_fixture(root, extra_error=False):
    c = {'id':'c1','label':'sector_floor_removed','mutations_hash':'h'}
    folder = root/'cases/c1/oos/100000000_baseline'
    result = {'audit_status':'failed','execution_validity_pass':False,'source_generation_hash':'source',
              'return_pct':99,'trades':80,'drawdown_pct':1,'terminal_inventory':{},'terminal_pending':0,'sessions':2}
    errors = ['shared_bar_volume_exceeded'] + (['cash_reconciliation_failed'] if extra_error else [])
    audit = {'audit_pass':False,'audit_receipt':{'errors':errors},'capacity_model_audit':{
        'errors':['shared_bar_volume_exceeded'],
        'contract':{'realized_volume':'ex_post_impossibility_check_only'},
        'impossible_volume_events':[{'bar_total_filled_shares':25,'printed_shares':6}]}}
    study.native.write(folder/'result.json',result)
    study.native.write(folder/'execution_audit.json',audit)
    study.native.write(folder/'completion.json',{'complete':True,'source_generation_hash':'source',
        'result_sha256':study.native.sha(folder/'result.json'),
        'audit_sha256':study.native.sha(folder/'execution_audit.json')})
    study.native.write(folder/'assessment.json',{**result,'qualified':False,'status':'completed',
                                               'candidate_id':'c1','window':'oos'})
    study.native.write(folder/'daily.json',[{'day':'2026-04-01'},{'day':'2026-07-10'}])
    registered = root/'registrations/oos/c1_100000000'
    study.native.write(registered/'candidate.json',c)
    study.native.write(registered/'study.json',{'source_generation_hash':'source',
        'configuration':{'date_range':{'start':'2026-04-01','end':'2026-07-10'}}})
    study.native.write(root/'protocol.json',{'windows':{'oos':['2026-04-01','2026-07-10']}})
    key='c1/oos/100000000_baseline'
    review=root/'capacity_review.json'
    study.native.write(review,{'case_key':key,'mutations_hash':'h',
        'evidence_sha256':{p.name:study.native.sha(p) for p in folder.iterdir()}})
    study.native.write(root/'mechanism_plan.json',{'reviewed_capacity_rejections':{
        key:{'path':review.name,'sha256':study.native.sha(review)}}})
    return c, folder


def test_reviewed_capacity_failure_stays_unqualified_without_performance_metrics(tmp_path):
    c, folder = capacity_rejection_fixture(tmp_path)
    original = {p.name:study.native.sha(p) for p in folder.iterdir()}
    row = study.evaluate(tmp_path,c,'oos')
    assert row['status']=='execution_infeasible' and row['qualified'] is False
    assert not {'return_pct','trades','drawdown_pct'} & row.keys()
    assert original == {p.name:study.native.sha(p) for p in folder.iterdir()}
    study.native.write(tmp_path/'candidate_plan.json',{'baseline_id':'control'})
    leaders, pairs = study.native.rank_pairs(tmp_path)
    assert not leaders and 'c1' not in pairs


@pytest.mark.parametrize('defect',['unregistered','changed_evidence','extra_error','identity'])
def test_capacity_review_does_not_allow_other_failures(tmp_path, defect):
    c, folder = capacity_rejection_fixture(tmp_path, extra_error=defect=='extra_error')
    if defect=='unregistered':
        study.native.write(tmp_path/'mechanism_plan.json',{})
    elif defect=='changed_evidence':
        study.native.write(folder/'result.json',{'execution_validity_pass':True})
    elif defect=='identity':
        c['mutations_hash']='other'
    with pytest.raises(ValueError):
        study.evaluate(tmp_path,c,'oos')


def test_capacity_rejection_is_specific_to_window_capital_and_scenario(tmp_path):
    c, _ = capacity_rejection_fixture(tmp_path)
    assert study.reviewed_capacity_rejection(tmp_path,c,'train') is None
    assert study.reviewed_capacity_rejection(tmp_path,c,'oos',10_000_000) is None
    assert study.reviewed_capacity_rejection(tmp_path,c,'oos',scenario='participation') is None


def source_unavailable_fixture(root):
    c, folder = capacity_rejection_fixture(root)
    result=study.native.read(folder/'result.json')
    result.update(audit_status='unverified',execution_validity_pass=True)
    study.native.write(folder/'result.json',result)
    audit={'audit_pass':False,'audit_receipt':{'errors':[],
        'coverage':{'ledger':'pass','daily_marks':'unverified','replay_completeness':'unverified'},
        'daily_equity_coverage':{'stale_or_unknown_mark_sessions':['2026-04-01'],'status':'unverified'},
        'qualification':{'shared_scopes':{'OLR':{'details':{'replay_completeness':'management_feed_incomplete:X:2026-04-01'}}}}},
        'capacity_model_audit':{'status':'model_consistent','errors':[]}}
    study.native.write(folder/'execution_audit.json',audit)
    completion=study.native.read(folder/'completion.json')
    completion.update(result_sha256=study.native.sha(folder/'result.json'),audit_sha256=study.native.sha(folder/'execution_audit.json'))
    study.native.write(folder/'completion.json',completion)
    recovery=root/'provider_unavailable.json'
    study.native.write(recovery,{'status':'unavailable_from_current_provider','rows':0})
    receipt=audit['audit_receipt'];review=root/'source_review.json'
    study.native.write(review,{'case_key':'c1/oos/100000000_baseline','mutations_hash':'h',
        'evidence_sha256':{p.name:study.native.sha(p) for p in folder.iterdir()},
        'recovery_evidence_sha256':{recovery.name:study.native.sha(recovery)},
        'expected_coverage':receipt['coverage'],'expected_daily_coverage':receipt['daily_equity_coverage'],
        'expected_qualification_scopes':receipt['qualification']['shared_scopes']})
    study.native.write(root/'mechanism_plan.json',{'reviewed_source_unavailability':{
        'c1/oos/100000000_baseline':{'path':review.name,'sha256':study.native.sha(review)}}})
    return c,folder


def test_unavailable_source_stays_unqualified_and_is_not_ranked(tmp_path):
    c,folder=source_unavailable_fixture(tmp_path)
    before={p.name:study.native.sha(p) for p in folder.iterdir()}
    row=study.evaluate(tmp_path,c,'oos')
    assert row['status']=='source_unverified' and not row['qualified']
    assert not {'return_pct','trades','drawdown_pct'} & row.keys()
    assert before=={p.name:study.native.sha(p) for p in folder.iterdir()}
    assert study.reviewed_source_unavailability(tmp_path,c,'train') is None
    assert study.reviewed_source_unavailability(tmp_path,c,'oos',25_000_000) is None
    study.native.write(tmp_path/'candidate_plan.json',{'baseline_id':'control'})
    leaders,pairs=study.native.rank_pairs(tmp_path)
    assert not leaders and 'c1' not in pairs


@pytest.mark.parametrize('defect',['unregistered','changed_result','changed_recovery','extra_audit_error'])
def test_source_review_requires_exact_evidence_and_has_no_general_failure_bypass(tmp_path,defect):
    c,folder=source_unavailable_fixture(tmp_path)
    if defect=='unregistered':
        study.native.write(tmp_path/'mechanism_plan.json',{})
    elif defect=='changed_result':
        study.native.write(folder/'result.json',{'audit_status':'passed'})
    elif defect=='changed_recovery':
        study.native.write(tmp_path/'provider_unavailable.json',{'rows':1})
    else:
        a=study.native.read(folder/'execution_audit.json');a['audit_receipt']['errors']=['cash_error']
        study.native.write(folder/'execution_audit.json',a)
    with pytest.raises(ValueError):
        study.evaluate(tmp_path,c,'oos')


@pytest.mark.parametrize('defect', [None, 'old_review_shape', 'unverified_marks', 'extra_scope'])
def test_candidate_feed_only_source_review_remains_unqualified(tmp_path, defect):
    c, folder = source_unavailable_fixture(tmp_path)
    audit = study.native.read(folder/'execution_audit.json')
    audit['audit_receipt']['coverage']['daily_marks'] = 'pass'
    daily = {'status':'complete', 'missing_sessions':[],
             'stale_or_unknown_mark_sessions':[], 'cash_only_padded_sessions':0}
    if defect == 'unverified_marks':
        daily['stale_or_unknown_mark_sessions'] = ['2026-04-01']
    if defect == 'extra_scope':
        audit['audit_receipt']['coverage']['ledger'] = 'unverified'
    audit['audit_receipt']['daily_equity_coverage'] = daily
    study.native.write(folder/'execution_audit.json', audit)
    completion = study.native.read(folder/'completion.json')
    completion['audit_sha256'] = study.native.sha(folder/'execution_audit.json')
    study.native.write(folder/'completion.json', completion)
    review = study.native.read(tmp_path/'source_review.json')
    if defect != 'old_review_shape':
        review['reviewed_unverified_coverage'] = {'replay_completeness':'unverified'}
    review['expected_coverage'] = audit['audit_receipt']['coverage']
    review['expected_daily_coverage'] = daily
    review['evidence_sha256'] = {p.name:study.native.sha(p) for p in folder.iterdir()}
    study.native.write(tmp_path/'source_review.json', review)
    plan = study.native.read(tmp_path/'mechanism_plan.json')
    plan['reviewed_source_unavailability']['c1/oos/100000000_baseline']['sha256'] = study.native.sha(tmp_path/'source_review.json')
    study.native.write(tmp_path/'mechanism_plan.json', plan)
    if defect:
        with pytest.raises(ValueError):
            study.evaluate(tmp_path, c, 'oos')
    else:
        before = {p.name:study.native.sha(p) for p in folder.iterdir()}
        row = study.evaluate(tmp_path, c, 'oos')
        assert row['status'] == 'source_unverified' and not row['qualified']
        assert not {'return_pct','trades','drawdown_pct'} & row.keys()
        assert before == {p.name:study.native.sha(p) for p in folder.iterdir()}
        assert study.reviewed_source_unavailability(tmp_path, c, 'train') is None


def test_attribution_counts_changed_execution_even_when_quantities_match(tmp_path):
    trade={'entry_fill_time':'2026-06-01T15:00','exit_fill_time':'2026-06-02T15:30',
           'symbol':'000001','qty':10,'entry_price':100,'exit_price':80,'net_pnl':-205,'exit_reason':'close'}
    study.native.write(tmp_path/'a/trades.json',[trade])
    study.native.write(tmp_path/'b/trades.json',[{**trade,'exit_price':95,'net_pnl':-55,'exit_reason':'stop'}])
    result=study.support_comparison({'output':str(tmp_path/'a')},{'output':str(tmp_path/'b')})
    assert result['changed_issuer_days']==1 and result['pnl_delta']==150
    assert result['support_flag']=='sparse'
    assert result['new_issuer_days']==0


def test_old_command_cannot_restart_superseded_round(tmp_path):
    study.native.write(tmp_path/'superseded_run.json',{'stopped':True})
    with pytest.raises(ValueError,match='stopped by the user'):
        study.native.run(tmp_path,2)


def test_historical_high_band_does_not_restore_into_inserted_middle_branch():
    original=original_plan()
    original['baseline'][study.RULES]=[
        {'max_score':325}, {'min_score':400,'max_score':500,'allowed_sectors':['IT']},
        {'min_score':400,'max_score':500,'sector_admission':{'mode':'dynamic_confirmed_rotation'}},
        {'min_score':650}]
    original['lineage']=[{'name':'old','mutations':{study.RULES:[
        {'max_score':300},{'min_score':400,'max_score':500,'allowed_sectors':['IT']},{'min_score':700}]}}]
    plan=study.build_plan(original)
    record=next(r for r in plan['semantic_band_coverage'] if r['identity']=='high')
    assert record['historical_index']==2 and record['current_index']==3
    candidate=next(c for c in plan['candidates'] if c['id']==record['candidate_id'])
    assert candidate['mutations'][study.RULES][3]['min_score']==700
    assert candidate['mutations'][study.RULES][2]['min_score']==400


def test_capital_registration_preserves_costs_sources_and_does_not_duplicate_matches():
    policy={'costs':{'tax_bps':20},'capacity':{'participation':.01}}
    qualification={'basis':{'source':'original'},'capacity':{'phase_liquidity_verified':False,
        'registered_execution_policies':[
            {'account':'shared_current_20261007','initial_equity':100_000_000,'buying_power_leverage':2.,'policy':policy},
            {'account':'existing_25m','initial_equity':25_000_000,'buying_power_leverage':2.,'policy':policy}]}}
    before=copy.deepcopy(qualification)
    same=study.native.capital_registration(qualification,25_000_000,'parent')
    assert len(same['capacity']['registered_execution_policies'])==2
    new=study.native.capital_registration(qualification,50_000_000,'parent')
    assert new['capacity']['registered_execution_policies'][-1]['policy']==policy
    assert new['basis']==qualification['basis']
    assert new['capacity']['phase_liquidity_verified'] is False
    assert qualification==before


def test_intrabar_capability_failure_is_distinct_from_supported_completed_bar_exit():
    assert study.unsupported_exit_orders({'trade_exit_plan':{'mode':'managed','hard_stop_enabled':True}})==['intrabar_protective_stop']
    assert study.unsupported_exit_orders({'trade_exit_plan':{'mode':'managed','partial_trigger_r':1,'partial_fraction':.5}})==['intrabar_partial_limit_target']
    assert study.unsupported_exit_orders({'trade_exit_plan':{'mode':'managed','hard_stop_enabled':False,
        'failed_followthrough_bars':1,'failed_followthrough_mfe_r':4,'failed_followthrough_close_r':-.8}})==[]
    assert study.unsupported_exit_orders({'trade_exit_plan':{'mode':'next_close','hard_stop_enabled':True}})==[]


def test_completed_bar_guard_observes_close_and_retains_its_mfe_condition():
    from types import SimpleNamespace
    from strategy_olr.execution import OLRExitPlan, _completed_bar_exit_reason
    plan=OLRExitPlan(name='guard',mode='managed',failed_followthrough_bars=1,
                     failed_followthrough_mfe_r=4,failed_followthrough_close_r=-.8)
    def reason(close,high):
        return _completed_bar_exit_reason([SimpleNamespace(close=close)],0,100,10,high,plan,0)['reason']
    assert reason(93,105)==''
    assert reason(91,105)=='failed_followthrough'
    assert reason(91,145)==''  # Not misrepresented as an unconditional hard stop.
