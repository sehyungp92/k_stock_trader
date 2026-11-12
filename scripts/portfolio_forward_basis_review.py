"""Bind and classify the reserved-window source audit without strategy outcomes."""
from collections import Counter
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=Path.home()/'.codex/research/portfolio_targeted_validation_20261009'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bound(path):return {'path':str(Path(path).resolve()),'sha256':digest(path)}
def write(path,payload):Path(path).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def review(root):
    audit=read(root/'provider_basis_audit/audit.json');manifest=read(root/'forward_inputs/manifest.json')
    if digest(audit['program']['path'])!=audit['program']['sha256']:raise ValueError('Changed provider collection program')
    for item in audit['original_and_adjusted_controls'].values():
        if digest(item['path'])!=item['sha256']:raise ValueError('Changed raw provider receipt')
    dependencies={'frozen_manifest':bound(root/'forward_inputs/manifest.json')}
    daily=[]
    for name,sha in manifest['files'].items():
        if name.startswith('data/krx_daily_parquet/daily_ohlcv/'):
            live=REPO/name;frozen=root/'forward_inputs/files'/name
            if digest(live)!=sha or digest(frozen)!=sha:raise ValueError('Provider comparison did not use the frozen daily source')
            daily.append(name);dependencies[name]=bound(live)
    if len(daily)!=103:raise ValueError('Incomplete daily source coverage')
    exceptions=[]
    for row in audit['sessions']:
        fields=[k for k in row['archive'] if row['archive'][k]!=row['original'][k]]
        if fields:
            category='volume_only_revision' if fields==['volume'] else (
                'adjusted_price_share_basis' if row['matches_adjusted'] else 'other_ohlcv_difference')
            exceptions.append({**row,'differing_fields':fields,'classification':category})
    if exceptions!=[{**r,'differing_fields':[k for k in r['archive'] if r['archive'][k]!=r['original'][k]],
                      'classification':e['classification']} for r,e in zip(audit['non_original_sessions'],exceptions)]:
        raise ValueError('Audit exception list differs from its source rows')
    for folder in ('warmup_sources','warmup_basis_controls'):
        for path in (root/folder).rglob('*'):
            if path.is_file():dependencies[str(path.relative_to(root))]=bound(path)
    for path in (REPO/'data/kis_intraday_parquet/068270').glob('068270_*_20250529_20260[57]*.parquet'):
        dependencies[str(path.relative_to(REPO))]=bound(path)
    old_delivery=Path('E:/CodexResearch/k_stock_trader/olr_forward_validation_20261007/delivery_result.json')
    dependencies['prior_olr_forward_delivery']=bound(old_delivery)
    for name in ('split_audit_verified.json','forward_prediction_completion.json','forward_inputs/files/research/forward_acceptance/acceptance.json'):
        dependencies[name]=bound(root/name)
    split_program=read(root/'split_audit_verified.json')['program']
    if digest(split_program['path'])!=split_program['sha256']:raise ValueError('Changed split verification program')
    dependencies['split_verification_program']=split_program
    ratios={'archived_boundary_return_pct':100*(182722/192900-1),
            'original_boundary_return_pct':100*(191700/192900-1)}
    ratios['artificial_difference_pp']=ratios['archived_boundary_return_pct']-ratios['original_boundary_return_pct']
    result={'version':'reserved-source-basis-review-v1','provider_audit':bound(root/'provider_basis_audit/audit.json'),
        'source_gate_pass':False,'strategy_performance_evaluated':False,
        'chronological_split_pass':read(root/'split_audit_verified.json')['pass'],
        'all_103_daily_sources_match_frozen_generation':True,'non_original_sessions':len(exceptions),
        'classification_counts':dict(Counter(r['classification'] for r in exceptions)),
        'exceptions':exceptions,'boundary_price_diagnostic':ratios,
        'provider_action_marker':{'symbol':'068270','date':'2026-06-04','flng_cls_code':'01','prtt_rate':'-4.68',
            'scope':'Raw provider fields; no inferred entitlement or account position adjustment.'},
        'decisive_failure':'Mixed original/adjusted price and share basis in 068270 May29 warm-up and June1–2; consistent lookback and corporate-action account treatment unqualified.',
        'volume_revision_scope':'24 July10 daily-volume differences are separate from the price-basis failure. July10 is liquidation-only; no return effect is inferred from their existence.',
        'repair_requirements':[
            'One documented dated price/share convention for all daily, minute and investor-flow lookbacks.',
            'Source-backed daily/minute conservation and point-in-time corporate-action share/cash treatment.',
            'A new immutable generation; repeat feature/inference bridges with the frozen March models.',
            'Evaluate only the sealed development finalist and the two fixed controls; preserve window and thresholds.'],
        'dependencies':dependencies,'program':bound(Path(__file__))}
    if len(exceptions)!=26 or result['classification_counts']!={'volume_only_revision':24,'adjusted_price_share_basis':2}:
        raise ValueError('Source finding changed; review before issuing the registered incident receipt')
    write(root/'source_basis_review.json',result)
    quarantine={'version':'forward-inference-source-quarantine-v1','status':'not_qualified_for_replay',
        'reason':result['decisive_failure'],'source_review':bound(root/'source_basis_review.json'),
        'prediction_completion':bound(root/'forward_prediction_completion.json'),
        'chronology_pass_does_not_imply_source_compatibility':True,
        'strategy_performance_evaluated':False,'original_artifacts_preserved':True}
    write(root/'forward_inference_quarantine.json',quarantine)
    print(json.dumps({k:result[k] for k in ('source_gate_pass','chronological_split_pass','classification_counts','boundary_price_diagnostic')}),flush=True)


if __name__=='__main__':review(ROOT)
