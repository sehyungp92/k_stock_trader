"""Repair stale audit declarations while proving forecasts/candidates unchanged."""
import copy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
BASE = Path('C:/Users/sehyu/.codex/research/portfolio_source_repair_20261009')


def annotate(old, new):
    runtime = Path(json.loads((old / 'inference_runtime.json').read_text())['root'])
    sys.path.insert(0, str(runtime))
    import yaml
    from scripts.kalcb_executed_value_study import connect, load_day
    from strategy_common.source_quality import complete_feature_bars, POLICY
    from strategy_common.sector_intraday import AFTERNOON_CUTOFF
    from backtests.strategies.kalcb.executed_value import digest, write_json, signature
    from scripts.portfolio_targeted_validation import bound
    original = json.loads((old / 'forward_olr_snapshots.json').read_text())
    snapshots = copy.deepcopy(original)
    symbols = tuple(str(s).zfill(6) for s in yaml.safe_load((new / 'forward_inputs/files/config/universe_103.yaml').read_text())['symbols'])
    database = new / 'forward_features/market.sqlite'; source = digest(database)
    conn = connect({'database': str(database)})
    checks = []
    for snap in snapshots:
        day = date.fromisoformat(snap['trade_date'])
        rows = {(day, s): [] for s in symbols}
        for bar in load_day(conn, str(day), source, set(symbols)): rows[day, bar.symbol].append(bar)
        _, exclusions = complete_feature_bars(rows, AFTERNOON_CUTOFF)
        excluded = [s.split(':')[0] for s in exclusions]
        if set(excluded) & {c['symbol'] for c in snap['candidates']}:
            raise ValueError('Existing OLR candidates violate their frozen prefix rule')
        if any(n in snap['metadata'] for n in ('feature_prefix_policy', 'feature_prefix_exclusions')):
            raise ValueError('Do not overwrite an existing fallback declaration')
        snap['metadata'].update(feature_prefix_policy=POLICY, feature_prefix_exclusions=excluded)
        checks.append({'day':str(day), 'excluded':excluded, 'candidates_unchanged':True})
    conn.close()
    for a, b in zip(original, snapshots):
        copy_b = copy.deepcopy(b)
        for key in ('feature_prefix_policy', 'feature_prefix_exclusions'): del copy_b['metadata'][key]
        if copy_b != a: raise ValueError('Changed OLR signal, feature or forecast')
    write_json(new / 'forward_olr_snapshots.json', snapshots)
    write_json(new / 'metadata_annotation.json', {'pass': True, 'snapshots': len(snapshots),
        'only_added_fields': ['feature_prefix_policy','feature_prefix_exclusions'], 'candidates_changed': False,
        'source': bound(old / 'forward_olr_snapshots.json'), 'output':bound(new / 'forward_olr_snapshots.json'),
        'database':bound(database), 'checks':checks, 'frozen_runtime':bound(old / 'inference_runtime.json'),
        'original_candidate_digest':signature([s['candidates'] for s in original]),
        'corrected_candidate_digest':signature([s['candidates'] for s in snapshots]), 'program':bound(Path(__file__))})


def build():
    sys.path.insert(0, str(REPO))
    from scripts.portfolio_targeted_validation import read, bound
    from backtests.strategies.kalcb.executed_value import digest, write_json
    old = BASE / 'sector_complete_v2'; new = BASE / 'oos'
    new.mkdir(exist_ok=False)
    for folder in ('forward_inputs','forward_features','forward_feature_bridge','forward_policies','repaired_sessions'):
        for path in (old / folder).rglob('*'):
            if path.is_file():
                target = new / path.relative_to(old); target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
    for name in ('protocol.json','screen.json','bridge.json','source_audit.json','development_selection.json',
                 'frozen_inference_bridge.json','olr_inference_bridge.json','development/study.json','battery/study.json',
                 'flow_basis_audit.json','volume_revision_audit.json','lookback_overlap_probe.json','lookback_overlap_verified.json',
                 'software_tests.xml','forward_dependency_tests.xml','sector_dependency_tests.xml','real_merge_regression.json',
                 'inference_runtime.json','inference_import_audit.json'):
        target = new / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(old / name, target)
    generation = read(old / 'repair_generation.json');generation['new_manifest'] = bound(new / 'forward_inputs/manifest.json')
    generation['audit_predecessor'] = bound(old / 'validation/qualified/control_baseline/execution_audit.json')
    write_json(new / 'repair_generation.json', generation)
    sector = read(old / 'sector_dependency_repair.json')
    sector['restored'] = bound(new / 'forward_inputs/files/data/krx_daily_parquet/tables/sector_map.parquet')
    write_json(new / 'sector_dependency_repair.json', sector)
    overlap = read(old / 'lookback_development_verified.json')
    overlap['repaired_manifest'] = bound(new / 'forward_inputs/manifest.json')
    overlap['prior_completed_audit'] = bound(old / 'lookback_development_verified.json')
    write_json(new / 'lookback_development_verified.json', overlap)
    result = subprocess.run([sys.executable,'-B',str(Path(__file__)), 'annotate', str(old), str(new)], check=True)
    review = read(old / 'forward_program_dependency_review.json')
    review['audit_metadata_annotation'] = bound(new / 'metadata_annotation.json')
    review['qualification_repair_scope'] = 'No inference repeated or changed; source audit metadata added from identical original observations'
    write_json(new / 'forward_program_dependency_review.json', review)
    completion = read(old / 'forward_prediction_completion.json')
    for key, path in [('kalcb_policy','forward_policies/policies_manifest.json'),('olr_snapshots','forward_olr_snapshots.json'),('feature_completion','forward_features/completion.json')]:
        completion[key] = bound(new / path)
    completion['unmodified_inference_completion'] = bound(old / 'forward_prediction_completion.json')
    completion['audit_metadata_annotation'] = bound(new / 'metadata_annotation.json')
    write_json(new / 'forward_prediction_completion.json', completion)
    runtime = Path(read(old / 'inference_runtime.json')['root'])
    subprocess.run([sys.executable,'-B',str(runtime / 'scripts/portfolio_forward_validation.py'),'audit-split','--root',str(new)],
                   cwd=runtime,check=True)
    repaired = read(old / 'repaired_inference_completion.json')
    repaired.update(predictions=bound(new / 'forward_prediction_completion.json'), split=bound(new / 'split_audit_verified.json'),
                    original_inference=bound(old / 'repaired_inference_completion.json'), metadata_annotation=bound(new / 'metadata_annotation.json'))
    write_json(new / 'repaired_inference_completion.json', repaired)
    write_json(new / 'qualification_repair.json', {'pass':True, 'old_audit':generation['audit_predecessor'],
        'new_annotation':bound(new / 'metadata_annotation.json'), 'portfolio_parameters_changed':False,
        'market_prices_volumes_features_models_changed':False,
        'capacity_change':'Refresh declaration to the already-qualified target basis when freezing replay; registered account/cost/participation policies unchanged',
        'economic_identity_to_verify':'control_baseline against the predecessor replay', 'program':bound(Path(__file__))})
    print(json.dumps({'audit_evidence_repaired':True,'root':str(new)}),flush=True)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='annotate': annotate(Path(sys.argv[2]),Path(sys.argv[3]))
    else: build()
