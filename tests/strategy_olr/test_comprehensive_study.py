import copy

import pytest

from scripts import olr_round1_comprehensive_study as study


def test_wrong_window_registration_and_native_cache_are_rejected(tmp_path):
    study.write(tmp_path/'protocol.json', {'windows':{'oos':['2026-04-01','2026-07-10']}})
    june = {'configuration':{'date_range':{'start':'2026-06-01','end':'2026-07-10'}}}
    with pytest.raises(ValueError, match='Window contract mismatch'):
        study.verify_window_contract(tmp_path, 'oos', june)
    april = {'configuration':{'date_range':{'start':'2026-04-01','end':'2026-07-10'}}}
    study.write(tmp_path/'daily.json', [{'day':'2026-06-01'},{'day':'2026-07-10'}])
    study.write(tmp_path/'result.json', {'sessions':2})
    with pytest.raises(ValueError, match='account marks'):
        study.verify_window_contract(tmp_path, 'oos', april, tmp_path)
    study.write(tmp_path/'daily.json', [{'day':'2026-04-01'},{'day':'2026-07-10'}])
    study.verify_window_contract(tmp_path, 'oos', april, tmp_path)


def test_old_mechanism_root_cannot_restart_after_window_correction(tmp_path):
    from scripts import olr_round1_mechanism_study as mechanism
    study.write(tmp_path/'superseded_mechanism.json', {'new_root':'corrected'})
    with pytest.raises(ValueError, match='window is superseded'):
        mechanism.run(tmp_path, 2)


def test_nested_rule_ablation_is_single_leaf_and_does_not_change_parent():
    original = {'olr.afternoon.score_band_rules': [{'max_score': 325, 'allowed_sectors': ['IT', 'FINANCIAL']}],
                'capacity_redistribute': True}
    before = copy.deepcopy(original)
    out = study.set_leaf(original, ('olr.afternoon.score_band_rules', 0, 'allowed_sectors', 1), None, True)
    assert original == before
    assert out['olr.afternoon.score_band_rules'] == [{'max_score': 325, 'allowed_sectors': ['IT']}]
    assert out['capacity_redistribute'] is True


def test_leaf_inventory_includes_individual_sectors_and_excludes_display_names():
    rows = dict(study.flatten({'rules': [{'name': 'label', 'sectors': ['IT', 'BANK'], 'threshold': 0}]}))
    assert rows == {('rules', 0, 'sectors', 0): 'IT', ('rules', 0, 'sectors', 1): 'BANK', ('rules', 0, 'threshold'): 0}


@pytest.mark.parametrize('key', ['olr.cost.tax_bps_on_sell', 'olr.source.require_complete_prefix',
                                 'olr.universe.complete_size', 'olr.execution.auction_fill_time'])
def test_integrity_and_cost_controls_cannot_be_optimized_away(key):
    assert study.protected((key,))
    assert not study.protected(('olr.trade_plan.exit', 'hard_stop_enabled'))


def test_failed_audit_cannot_enter_ranking(tmp_path):
    study.write(tmp_path/'candidate_plan.json', {'baseline_id': 'base'})
    def row(cid, window, ret, trades=10, qualified=True):
        return {'candidate_id': cid, 'window': window, 'qualified': qualified, 'return_pct': ret,
                'trades': trades, 'drawdown_pct': 5, 'capital': 100_000_000}
    for cid, is_ret, oos_ret, count, qualified in [('base',50,-10,10,True), ('good',51,-5,11,True),
             ('low_activity',60,5,9,True), ('failed',90,20,12,False), ('damages_is',40,10,12,True)]:
        for window, ret in [('train',is_ret),('oos',oos_ret)]:
            study.write(tmp_path/'cases'/cid/window/'100000000_baseline/assessment.json',row(cid,window,ret,count,qualified))
    ranked,_=study.rank_pairs(tmp_path)
    assert [r['id'] for r in ranked] == ['good']


def test_paired_bootstrap_is_zero_for_identical_paths(tmp_path):
    path=tmp_path/'case'
    study.write(path/'daily.json',[{'day':f'2026-06-{d:02}', 'equity':1000+d} for d in range(1,21)])
    row={'output':str(path),'capital':1000}
    result=study.paired_uncertainty(row,row,draws=100)
    assert result['paired_return_gain_ci95_pp'] == [0,0]
    assert result['bootstrap_fraction_positive'] == 0


def test_paired_bootstrap_rejects_misaligned_sessions(tmp_path):
    for name,day in [('a','2026-06-01'),('b','2026-06-02')]:
        study.write(tmp_path/name/'daily.json',[{'day':day,'equity':1000}])
    with pytest.raises(ValueError,match='identical sessions'):
        study.paired_uncertainty({'output':str(tmp_path/'a'),'capital':1000}, {'output':str(tmp_path/'b'),'capital':1000})


def test_frozen_input_filter_excludes_interpreter_caches_but_retains_source_and_data():
    assert study.generated_bytecode('backtests/__pycache__/__init__.cpython-312.pyc')
    assert study.generated_bytecode('backtests/legacy.pyc')
    for name in ('backtests/__init__.py', 'kis_core/data/krx_sessions.yaml',
                 'data/ohlcv.parquet', 'research/market.sqlite'):
        assert not study.generated_bytecode(name)


def test_market_binding_rejects_mixed_calendar_generations_before_replay():
    calendar = 'kis_core/data/krx_sessions.yaml'
    contract = {'files': {calendar: 'original'}, 'train_start': '2025-08-07'}
    contract['market_input_hash'] = study.sig(contract)
    manifest = {'files': {calendar: 'new'}, 'evidence': {'shared_market_inputs': contract}}
    with pytest.raises(ValueError, match='Market input binding mismatch'):
        study.verify_market_bindings(manifest)
    manifest['files'][calendar] = 'original'
    study.verify_market_bindings(manifest)


def test_retained_certificate_detects_parser_drift_before_native_replay(tmp_path):
    study.write(tmp_path/'qualification.json', {'basis': {'acceptance_report': 'acceptance.json'}})
    study.write(tmp_path/'acceptance.json', {'parser_files': {'parser.py': 'old'}})
    manifest = {'evidence': {'qualification_path': 'qualification.json'}, 'files': {'parser.py': 'new'}}
    assert study.retained_parser_mismatches(manifest, tmp_path) == ['parser.py']
    manifest['files']['parser.py'] = 'old'
    assert study.retained_parser_mismatches(manifest, tmp_path) == []


def test_interactions_combine_distinct_nested_exit_leaves_without_clustering():
    base = {'exit': {'hard_stop': False, 'bars': 0}, 'weight': 1}
    left = study.set_leaf(base, ('exit', 'hard_stop'), True)
    right = study.set_leaf(base, ('exit', 'bars'), 12)
    assert study.combine_disjoint(base, left, right) == {'exit': {'hard_stop': True, 'bars': 12}, 'weight': 1}
    assert base['exit'] == {'hard_stop': False, 'bars': 0}


def test_interactions_reject_overlapping_leaf_or_structural_list_changes():
    base = {'rules': [{'threshold': 1}, {'threshold': 2}]}
    left = {'rules': [{'threshold': 2}]}
    right = {'rules': [{'threshold': 3}, {'threshold': 2}]}
    assert study.combine_disjoint(base, left, right) is None
    assert study.combine_disjoint(base, right, right) is None


def test_interactions_preserve_single_key_deletion_and_unrelated_addition():
    base = {'a': 1, 'b': {'x': 2}}
    assert study.combine_disjoint(base, {'b': {'x': 2}}, {**base, 'new': 3}) == {'b': {'x': 2}, 'new': 3}


def test_effect_review_distinguishes_ignored_mapping_from_dormant_configuration(tmp_path):
    study.write(tmp_path/'candidate_plan.json', {'baseline_id':'base'})
    engine = tmp_path/'engine'
    study.write(engine/'snapshots.json', [
        {'trade_date':'2026-06-01','candidates':[{'symbol':'000001','rank':1,'selection_score':10}]},
        {'trade_date':'2026-05-01','candidates':[{'symbol':'000002','rank':1,'selection_score':50}]}])
    for cid,config in [('base',{'stop':False}),('ignored',{'stop':False}),('dormant',{'stop':True})]:
        case = tmp_path/'registrations/oos'/(cid+'_100000000')
        out = tmp_path/'cases'/cid/'oos/100000000_baseline'
        study.write(case/'study.json', {'code':str(engine),'olr_snapshots':'snapshots.json'})
        study.write(case/'effective_config.json',config)
        row={'candidate_id':cid,'window':'oos','status':'completed','qualified':True,'output':str(out)}
        study.write(out/'assessment.json',row)
        study.write(out/'result.json',row)
        study.write(out/'trades.json',[{'symbol':'000001','qty':1,'net_pnl':0,'source_artifact_hash':cid}])
        study.write(out/'daily.json',[{'day':'2026-06-01','equity':100}])
    study.candidate_effects(tmp_path)
    effects={r['candidate_id']:r for r in study.read(tmp_path/'candidate_effects.json')['cases']}
    assert not effects['ignored']['effective_configuration_changed']
    assert effects['dormant']['effective_configuration_changed']
    assert not effects['dormant']['executions_changed']
    assert not effects['ignored']['executions_changed']
    assert effects['base']['candidate_rows']==1


def test_reconstruction_uses_prior_context_but_never_future_database_dates(tmp_path):
    from datetime import date
    import sqlite3
    database=tmp_path/'market.sqlite'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE bars(day TEXT)')
        connection.executemany('INSERT INTO bars VALUES (?)', [('2026-05-28',),('2026-05-29',),
            ('2026-06-01',),('2026-06-02',),('2026-07-10',)])
    days=[date(2026,6,1),date(2026,6,2)]
    assert study.reconstruction_days(database,days)==[date(2026,5,28),date(2026,5,29),*days]
    assert study.reconstruction_days(database,[date(2026,5,28)])==[date(2026,5,28)]


def test_is_oos_boundary_does_not_reset_sector_volume_history(tmp_path):
    from datetime import date, timedelta
    import sqlite3
    from strategy_common.sector_intraday import _relative_cutoff_volume_map
    days=[date(2026,3,1)+timedelta(days=i) for i in range(34)]
    requested=days[31:]
    db=tmp_path/'market.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE bars(day TEXT)')
        c.executemany('INSERT INTO bars VALUES (?)',[(str(d),) for d in days])
    volumes={(d,'000001'):100.+i*10 for i,d in enumerate(days)}
    continuous=_relative_cutoff_volume_map(volumes)
    context=study.reconstruction_days(db,requested)
    resumed=_relative_cutoff_volume_map({k:v for k,v in volumes.items() if k[0] in context})
    cold=_relative_cutoff_volume_map({k:v for k,v in volumes.items() if k[0]>=days[30]})
    assert all(resumed[d,'000001']==continuous[d,'000001'] for d in requested)
    assert cold[requested[0],'000001']!=continuous[requested[0],'000001']


@pytest.mark.parametrize('changed_band',[False,True])
def test_completed_bridge_authenticates_and_checks_execution_metadata(tmp_path,changed_band):
    import sqlite3
    study.write(tmp_path/'protocol.json',{'generation_directories':{'oos':'g'}})
    files=tmp_path/'g/oos/inputs/files'
    original=[{'trade_date':'2026-06-01','candidates':[{'symbol':'000001','rank':1,'selection_score':1,
        'metadata':{'afternoon_score_band_rule':'low','afternoon_features':{'ret':.1},'prior_day_date':'2026-05-29'}}]}]
    study.write(files/'original.json',original)
    with sqlite3.connect(files/'market.sqlite') as conn:
        conn.execute('CREATE TABLE bars(day TEXT)')
        conn.execute("INSERT INTO bars VALUES ('2026-06-01')")
    study.write(tmp_path/'g/oos/ready.json',{'files':str(files),'study':{'source_generation_hash':'source',
        'olr_snapshots':'original.json','database':'market.sqlite'}})
    rebuilt=copy.deepcopy(original)
    if changed_band:
        rebuilt[0]['candidates'][0]['metadata']['afternoon_score_band_rule']='high'
    output=tmp_path/'builder_bridge/oos/snapshots.json'
    study.write(output,rebuilt)
    study.write(output.with_suffix('.receipt.json'),{'snapshots_sha256':study.sha(output),
        'candidate_hash':'candidate','source_hash':'source','context_only_dates':[]})
    if changed_band:
        with pytest.raises(ValueError,match='reconstruction differs'):
            study.reconstruction_bridge(tmp_path,'oos',{'mutations_hash':'candidate'})
    else:
        study.reconstruction_bridge(tmp_path,'oos',{'mutations_hash':'candidate'})
    assert study.read(output.parent/'bridge.json')['pass'] is not changed_band
