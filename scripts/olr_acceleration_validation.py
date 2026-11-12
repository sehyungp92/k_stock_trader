"""Full native control replay for the revision-18 reconstruction accelerator."""
import argparse
import copy
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import olr_round1_comprehensive_study as n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--window', required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--reuse-label')
    args = parser.parse_args()
    r, window = args.root, args.window
    v = r/'revisions/revision_18_reconstruction_cache'
    label = args.reuse_label or window+'_stream_warm_control'
    baseline = n.read(r/'candidate_plan.json')['baseline_id']
    cmd = [sys.executable,'-B','-u',str(Path(__file__).with_name('olr_acceleration_benchmark.py')),
           '--root',str(r),'--window',window,'--candidate',baseline,'--label',label,
           '--cache-root',str(args.cache_root),'--streaming']
    if not args.reuse_label and n.run_child(cmd, n.REPO, v/(label+'.log')):
        raise ValueError('Streaming baseline snapshot parity failed')
    benchmark = n.read(v/'benchmarks'/(label+'.json'))
    assert benchmark['parity_pass']
    rows = n.read(v/'benchmarks'/(label+'.snapshots.json'))
    normalize = lambda values:[{k:value for k,value in row.items() if k!='generated_at'} for row in values]
    assert n.sha(benchmark['reference_path'])==benchmark['reference_sha256']
    assert normalize(rows)==normalize(n.read(benchmark['reference_path']))
    old_registration = r/'registrations'/window/(baseline+'_100000000')/'study.json'
    study = n.read(old_registration)
    original_study = copy.deepcopy(study)
    engine = Path(study['code'])
    manifest = n.read(study['manifest'])
    manifest.pop('manifest_hash')
    prefix = 'research/olr_comprehensive/revision18_control_proof/'+window+'/'
    study['olr_snapshots'] = prefix+'snapshots.json'
    n.write(engine/study['olr_snapshots'], n.read(v/'benchmarks'/(label+'.snapshots.json')))
    study['spec_path'] = prefix+'spec.json'
    mp = engine.parent/('revision18_'+window+'_control.manifest.json')
    study['manifest'] = str(mp)
    study['configuration']['input_manifest_path'] = str(mp)
    spec = {k:value for k,value in study.items() if k not in ('manifest','code','spec_path','source_generation_hash')}
    n.write(engine/study['spec_path'], spec)
    for name in (study['olr_snapshots'],study['spec_path']):
        manifest['files'][name] = n.sha(engine/name)
    n.write(mp,{**manifest,'manifest_hash':n.sig(manifest)})
    registration = v/('native_control_'+window)/'study.json'
    n.write(registration,study)
    out = registration.parent/'replay'
    if out.exists():
        raise ValueError('Control proof output already exists; preserve before rerunning')
    start = time.perf_counter()
    cmd = [sys.executable,'-B','-u',str(engine/'scripts/portfolio_synergy_study.py'),'replay',
           '--study',str(registration),'--recipe','candidate','--scenario','baseline','--standalone','OLR',
           '--output',str(out),'--full-audit']
    if n.run_child(cmd,engine,v/('native_control_'+window+'.log')):
        raise ValueError('Native control replay failed')
    elapsed = time.perf_counter()-start
    original = r/'cases'/baseline/window/'100000000_baseline'
    result, audit, completion = [n.read(out/name) for name in ('result.json','execution_audit.json','completion.json')]
    expected = n.read(original/'result.json')
    assert audit['audit_pass'] and result['audit_status']=='passed' and result['execution_validity_pass']
    assert completion['result_sha256']==n.sha(out/'result.json') and completion['audit_sha256']==n.sha(out/'execution_audit.json')
    n.verify_window_contract(r,window,study,out)
    assert not result['terminal_inventory'] and result['terminal_pending']==0
    metrics = ['return_pct','trades','win_rate','drawdown_pct','final_equity']
    metrics = [k for k in metrics if k in expected]
    metric_equal = {k:result[k]==expected[k] for k in metrics}
    content_equal = {name:n.read(out/name)==n.read(original/name)
                     for name in ('daily.json','trades.json')}
    comparison = {'window':window,'native_full_audit_pass':True,'metrics_identical':metric_equal,
                  'native_content_identical':content_equal,'metrics':{k:result[k] for k in metrics},
                  'native_replay_seconds':elapsed,'benchmark':benchmark,'result_sha256':n.sha(out/'result.json'),
                  'audit_sha256':n.sha(out/'execution_audit.json'), 'source_hash':result['source_generation_hash'],
                  'original_registration_sha256':n.sha(old_registration),
                  'pass':all(metric_equal.values()) and all(content_equal.values())}
    n.write(v/('native_control_'+window+'_comparison.json'),comparison)
    print(comparison,flush=True)
    if not comparison['pass']:
        raise ValueError('Native control economic parity failed')


if __name__ == '__main__':
    main()
