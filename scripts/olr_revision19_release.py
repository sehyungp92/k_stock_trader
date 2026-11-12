"""Fail-closed release of isolated, fully audited native performance revision19."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import olr_round1_mechanism_study as m
from scripts import olr_native_acceleration as a

n=m.native
r=Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10')
v=r/'revisions/revision_19_native_overheads'
assert n.read(r/'STOP_MECHANISM.json')['owner']=='performance_revision19'
plan=n.read(r/'mechanism_plan.json');assert plan['version']==18
proofs={}
for label in ('control_train','control_oos','allocation_train','allocation_oos','capacity_rejection_oos','source_gap_train'):
    p=v/'native_proofs'/label/'comparison.json'
    proof=n.read(p);assert proof['passed'] and proof['full_audit']
    for name,digest in proof['artifact_sha256'].items():
        assert n.sha(p.parent/'replay'/name)==digest
    completion=n.read(p.parent/'replay/completion.json')
    assert completion['complete']
    for name,digest in completion['artifacts'].items():
        assert n.sha(p.parent/'replay'/name)==digest
    assert proof['native_audit_pass']==(label not in ('capacity_rejection_oos','source_gap_train'))
    proofs[label]={'path':str(p),'sha256':n.sha(p),'proof':proof}
for window in ('train','oos'):
    assert n.read(v/f'source_rows_{window}_parity.json')['pass']
assert '120 passed' in (v/'native_tests.log').read_text(encoding='utf-8-sig')
assert '22 passed' in (v/'harness_tests.log').read_text(encoding='utf-8-sig')
assert '31 passed' in (v/'mechanism_tests.log').read_text(encoding='utf-8-sig')
assert 'failed' not in (v/'mechanism_tests.log').read_text(encoding='utf-8-sig').lower()
generations={}
for window in ('train','oos'):
    p=Path('C:/Users/sehyu/.codex/research/olr_native_r19')/window/'inputs/generation.json'
    g=n.read(p)
    for name,change in g['changed_files'].items():
        assert n.sha(Path(g['original_engine'])/name)==change['before']
        assert n.sha(Path(g['engine'])/name)==change['after']
        assert (Path(g['engine'])/name).read_text(encoding='utf-8')==a.optimized_source(name,(Path(g['original_engine'])/name).read_text(encoding='utf-8'))
    actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'],cwd=g['engine'],text=True).strip()
    assert actual==g['source_generation_hash']==proofs['control_'+window]['proof']['source_generation_hash']
    generations[window]={'path':str(p),'sha256':n.sha(p),'source_generation_hash':actual,
                         'original_source_hash':g['original_source_hash']}
known=m.catalog(r);authenticated=[]
for p in sorted((r/'cases').glob('*/*/*/assessment.json')):
    assessment=n.read(p)
    cid,window,scenario_dir=p.parts[-4:-1]
    capital,scenario=scenario_dir.split('_',1);capped=scenario.endswith('_capped')
    if capped:scenario=scenario.removesuffix('_capped')
    m.validate_cached(r,known[cid],window,int(capital),scenario,capped)
    authenticated.append({'candidate':cid,'window':window,'scenario':scenario_dir,
                          'qualified':assessment['qualified'],'status':assessment['status'],'sha256':n.sha(p)})
now=datetime.now(timezone.utc).isoformat()
validation={'revision':19,'passed':True,'validated_utc':now,'proofs':proofs,'generations':generations,
    'test_counts':{'native':120,'harness':22,'mechanism_and_registration':31,'overlapping_registration_tests':3,'distinct_tests':170},
    'native_tests_log_sha256':n.sha(v/'native_tests.log'),'harness_tests_log_sha256':n.sha(v/'harness_tests.log'),
    'mechanism_tests_log_sha256':n.sha(v/'mechanism_tests.log'),
    'source_row_parity':{w:n.read(v/f'source_rows_{w}_parity.json') for w in ('train','oos')},
    'source_data_changed':False,'source_history_changed':False,'strategy_trials_added':0,
    'strategy_costs_audit_rules_and_acceptance_gates_changed':False,
    'full_audit_verdicts_cached':False,'previous_qualified_results_reauthenticated':sum(x['qualified'] for x in authenticated),
    'reauthenticated_assessments':authenticated,'status_counts':dict(Counter(x['status'] for x in authenticated)),
    'maximum_workers':2,'additional_slot_minimum_available_mb':5120,
    'snapshot_reconstruction_and_warmed_cache_bytes_unchanged':True,
    'implementation_sha256':{name:n.sha(Path(__file__).parent/name) for name in
        ('olr_native_acceleration.py','olr_round1_comprehensive_study.py','olr_round1_mechanism_study.py')}}
n.write(r/'revision19_validation.json',validation)
change={'at_utc':now,'reason':'Isolated SSD replay engines: reduce repeated path resolution, timestamp conversion, audit object creation, unused OHLCV retention, temporary watch-bar storage and canonical hash overhead. Complete native audits preserved; bounded memory-aware dispatch.'}
plan.update(version=19,runner_sha256=validation['implementation_sha256']['olr_round1_mechanism_study.py'],
            native_evaluator_sha256=validation['implementation_sha256']['olr_round1_comprehensive_study.py'],revision19=change)
plan['native_replay_acceleration']={'version':1,'implementation_sha256':n.sha(a.__file__),
    'generations':generations,'validation':'revision19_validation.json','validation_sha256':n.sha(r/'revision19_validation.json')}
plan['resource_admission']={'minimum_available_mb_for_additional_job':5120,
    'description':'Maximum two jobs; reserve for two future audit peaks plus host headroom. One continues under host load; additional dispatch waits for 5 GiB available. Never interrupt an active job.'}
plan['source_evidence']['revision19_validation.json']=n.sha(r/'revision19_validation.json')
n.write(r/'mechanism_plan.json',plan)
protocol=n.read(r/'protocol.json')
protocol.update(harness_revision=19,harness_sha256=plan['native_evaluator_sha256'],orchestration_sha256=plan['runner_sha256'],
                mechanism_plan_sha256=n.sha(r/'mechanism_plan.json'),revision19=change)
n.write(r/'protocol.json',protocol)
n.write(r/'revision19_registration.json',{'revision':19,'registered_utc':now,'change':change,
    'qualified_results_reauthenticated':validation['previous_qualified_results_reauthenticated'],
    'status_counts':validation['status_counts'],'native_audit_semantics_unchanged':True,
    'snapshot_sources_unchanged':True,'isolated_replay_source_generations':generations,
    'validation_sha256':n.sha(r/'revision19_validation.json'),'plan_sha256':n.sha(r/'mechanism_plan.json'),
    'new_strategy_trials':0,'maximum_workers':2})
print({'revision':19,'counts':validation['status_counts'],'registered':True},flush=True)
