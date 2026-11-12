"""Register the measured, parity-verified reconstruction/scheduling repair."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import olr_round1_mechanism_study as m

n = m.native
v = ROOT/'revisions/revision_18_reconstruction_cache'
plan = n.read(ROOT/'mechanism_plan.json')
assert plan['version'] == 17
assert n.read(ROOT/'STOP_MECHANISM.json')['owner'] == 'performance_revision18'
assert '82 passed' in (v/'validation_tests_release.log').read_text(encoding='utf-8-sig')
known = m.catalog(ROOT)
authenticated = []
for p in sorted((ROOT/'cases').glob('*/*/*/assessment.json')):
    a = n.read(p)
    cid, window, scenario_dir = p.parts[-4:-1]
    capital, scenario = scenario_dir.split('_', 1)
    capped = scenario.endswith('_capped')
    if capped:
        scenario = scenario.removesuffix('_capped')
    m.validate_cached(ROOT, known[cid], window, int(capital), scenario, capped)
    authenticated.append({'candidate':cid, 'window':window, 'scenario':scenario_dir,
                          'qualified':a['qualified'], 'status':a['status'],
                          'assessment_sha256':n.sha(p)})
proofs, source_hashes, dependencies = {}, {}, {}
for window in ('train','oos'):
    benchmark = n.read(v/f'benchmarks/{window}_prefix_warm_control.json')
    assert benchmark['parity_pass'] and benchmark['prefix_cache']['hits'] == benchmark['output_sessions']
    study = n.read(v/f'native_control_{window}/study.json')
    replay_input = Path(study['code'])/study['olr_snapshots']
    rebuilt = v/f'benchmarks/{window}_prefix_warm_control.snapshots.json'
    assert replay_input.read_bytes() == rebuilt.read_bytes()
    comparison = v/('native_control_train_economic_comparison.json' if window == 'train' else 'native_control_oos_comparison.json')
    proof = n.read(comparison)
    assert proof['native_full_audit_pass'] and proof.get('pass_reviewed', proof['pass'])
    audit_path = v/f'native_control_{window}/replay/execution_audit.json'
    assert n.sha(audit_path) == proof['audit_sha256'] and n.read(audit_path)['audit_pass']
    code = "from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())"
    actual = subprocess.check_output([sys.executable,'-B','-c',code],cwd=study['code'],text=True).strip()
    assert actual == study['source_generation_hash'] == proof['source_hash']
    source_hashes[window] = actual
    preflight = n.read(ROOT/f'revision10_preflight_{window}.json')
    supported = [known[c['id']] for c in preflight['candidates'] if c['status']=='supported']
    payload = json.dumps([c['mutations'] for c in supported])
    cfg_code = "import json,sys; from strategy_olr.config import OLRConfig; a=json.load(sys.stdin); print(json.dumps(sorted(set((max(100,int(OLRConfig.from_mapping({},x).research_min_history_days)+5),OLRConfig.from_mapping({},x).require_complete_feature_prefix) for x in a))))"
    resolved = json.loads(subprocess.check_output([sys.executable,'-B','-c',cfg_code],input=payload,cwd=study['code'],text=True))
    assert resolved == [[100, True]], resolved
    dependencies[window] = {'supported':len(supported), 'resolved_cache_dependencies':resolved,
        'preflight_counts':dict(Counter(c['status'] for c in preflight['candidates']))}
    proofs[window] = {'benchmark':benchmark,'native_proof':str(comparison),'native_proof_sha256':n.sha(comparison),
        'final_optimized_snapshots_byte_identical_to_audited_native_input':True,
        'snapshot_sha256':n.sha(rebuilt), 'audit_sha256':n.sha(audit_path), 'metrics':proof['metrics'],
        'provenance_note':proof.get('explanation'), 'native_replay_seconds':proof['native_replay_seconds']}

now = datetime.now(timezone.utc).isoformat()
scripts = Path(__file__).parent
files = ['olr_snapshot_streaming.py','olr_snapshot_acceleration.py','olr_prefix_memo.py',
         'olr_round1_comprehensive_study.py','olr_round1_mechanism_study.py']
hashes = {f:n.sha(scripts/f) for f in files}
validation = {'revision':18,'validated_utc':now,'passed':True,'tests_passed':82,
    'tests_log_sha256':n.sha(v/'validation_tests_release.log'),'implementation_sha256':hashes,
    'control_proofs':proofs,'supported_cache_dependency_coverage':dependencies,
    'reauthenticated_assessments':authenticated,'qualified_count':sum(x['qualified'] for x in authenticated),
    'status_counts':dict(Counter(x['status'] for x in authenticated)),
    'source_hashes_unchanged':source_hashes,'native_engine_and_audits_unchanged':True,
    'source_data_and_continuous_warmup_unchanged':True,'strategy_trials_added':0,
    'workers':4,'preparation_lock':'one reconstruction/registration per window; native replay outside lock',
    'timeout_seconds_per_process':172800,'overall_timeout':None,
    'failure_policy':'Stop new dispatch on any unreviewed failure; finish active children; exact reviewed dispositions unchanged.'}
n.write(ROOT/'revision18_validation.json', validation)
change = {'at_utc':now,'reason':'Authenticated native feature memoization on SSD, session streaming, omitted discarded warm-up selections, exact native prefix memoization, and four-worker bounded shared queue. Full native replay/audits unchanged.'}
plan.update(version=18, runner_sha256=hashes['olr_round1_mechanism_study.py'],
            native_evaluator_sha256=hashes['olr_round1_comprehensive_study.py'], revision18=change)
plan['snapshot_acceleration'] = {'version':3,'module':'olr_snapshot_streaming.py',
    'implementation_sha256':hashes['olr_snapshot_streaming.py'],
    'dependencies':{f:hashes[f] for f in ('olr_snapshot_acceleration.py','olr_prefix_memo.py')},
    'cache_root':'C:/Users/sehyu/.codex/research/olr_invariant_cache/continuous_history_v10',
    'prefix_memo':True,'validated_source_hashes':source_hashes,
    'validation':'revision18_validation.json','validation_sha256':n.sha(ROOT/'revision18_validation.json')}
plan['source_evidence']['revision18_validation.json'] = n.sha(ROOT/'revision18_validation.json')
n.write(ROOT/'mechanism_plan.json',plan)
protocol = n.read(ROOT/'protocol.json')
protocol.update(harness_revision=18,harness_sha256=hashes['olr_round1_comprehensive_study.py'],
    orchestration_sha256=hashes['olr_round1_mechanism_study.py'],mechanism_plan_sha256=n.sha(ROOT/'mechanism_plan.json'),revision18=change)
n.write(ROOT/'protocol.json',protocol)
n.write(ROOT/'revision18_registration.json',{'revision':18,'registered_utc':now,'change':change,
    'registered_candidates':len(known),'native_source_unchanged':True,'strategy_and_gates_unchanged':True,
    'qualified_results_reauthenticated':validation['qualified_count'],'status_counts':validation['status_counts'],
    'validation_sha256':n.sha(ROOT/'revision18_validation.json'),'plan_sha256':n.sha(ROOT/'mechanism_plan.json'),
    'source_unverified_cases':len(plan['reviewed_source_unavailability']),
    'physical_execution_failures':len(plan['reviewed_capacity_rejections']), 'workers':4})
print(json.dumps({'registered_revision':18,'counts':validation['status_counts'],'qualified':validation['qualified_count'],'proofs':{w:p['metrics'] for w,p in proofs.items()}}))
