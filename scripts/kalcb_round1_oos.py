"""Locked standalone KALCB validation; no fitting or candidate search."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.qualified_artifacts import read, digest, require, validate_case, diagnostic_result
from backtests.strategies.kalcb.executed_value import write_json, signature

ROOT = Path('C:/Users/sehyu/.codex/research/kalcb_round1_oos_20261009')
FORWARD = Path('C:/Users/sehyu/.codex/research/portfolio_source_repair_20261009/oos')
SCENARIOS = ('baseline', 'participation', 'cost', 'combined')


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True, default=str, allow_nan=False) + '\n')


def prepare(root):
    from scripts.portfolio_frozen_inference import infer, FIT
    from scripts.kalcb_executed_value_replay import verify_policy, timed_policy
    from scripts.kalcb_executed_value_freeze import freeze
    from backtests.strategies.kalcb import executed_value as ev
    from strategy_kalcb.config import KALCBConfig
    canonical = REPO/'data/backtests/output/kalcb/round_1'
    selected = read(canonical/'selected_research_profile.json')
    root.mkdir(parents=True, exist_ok=False)
    protocol = {
        'version':'kalcb-round1-standalone-oos-v1',
        'registered_utc':datetime.now(timezone.utc).isoformat(),
        'canonical_profile':bound(canonical/'selected_research_profile.json'),
        'canonical_config':bound(canonical/'optimized_config.json'),
        'period':{'start':'2026-06-01','end':'2026-07-10','sessions':29},
        'period_basis':bound(FORWARD/'protocol.json'),
        'excluded_consumed_validation':{'start':'2026-04-01','end':'2026-05-12'},
        'capital_krw':100_000_000, 'scenarios':list(SCENARIOS),
        'strategy_parameters':'Exactly current round_1, full standalone sizing; no portfolio budget multiplier.',
        'model_policy':'Last pre-OOS model and calibration held fixed; no OOS model fitting or parameter selection.',
        'adaptation_limit':'Measures frozen-model generalization, not the separate prospective expanding-refit protocol.',
        'entry_sessions':'All 29 sessions, including July 9 and 10; normal intraday flattening.',
        'independence':'Historically reserved window already inspected in a related portfolio study; not newly untouched.',
        'outcome_policy':'Report every scenario and all failures; do not tune, truncate dates, or change the canonical round.',
        'live_enabled':False, 'program':bound(__file__)}
    write_json(root/'protocol.json', protocol)
    for name in ('frozen_inference_bridge.json','forward_feature_bridge/comparison.json'):
        require(read(FORWARD/name)['pass'], 'Failed frozen inference or feature bridge')
    source = Path(selected['execution_source']['code'])
    for name in ('strategy_kalcb/config.py','strategy_kalcb/core/logic.py','strategy_kalcb/core/candidate_updates.py',
                 'backtests/engine/sim_broker.py'):
        require(digest(source/name)==digest(REPO/name), 'Changed economic execution component: '+name)
    # Keep the selected economic source and supply the already-qualified OOS
    # calendar plus the audit-context loader; neither changes trading decisions.
    execution_source=root/'execution_source'
    shutil.copytree(source,execution_source)
    shutil.copy2(REPO/'scripts/kalcb_executed_value_replay.py',execution_source/'scripts/kalcb_executed_value_replay.py')
    for name in ('krx_sessions.yaml','krx_intraday_halts.yaml'):
        shutil.copy2(FORWARD/'validation/inputs/files/kis_core/data'/name,execution_source/'kis_core/data'/name)
    # Archived prices were ingested with this exact parser generation. Retain
    # that provenance even though the downloader is never invoked by replay.
    parser='scripts/update_kis_intraday.py'
    shutil.copy2(FORWARD/'validation/inputs/files'/parser,execution_source/parser)
    features=FORWARD/'forward_features'
    completion=read(features/'completion.json')
    for item in completion['stages'].values():
        require(digest(item['path'])==item['sha256'], 'Changed prepared feature stage')
    development=read(FORWARD/'development/study.json')
    original=Path(development['code'])/development['kalcb_policies']
    opening=features/'continuation/opportunities.jsonl'
    checkpoint=features/'checkpoint/opportunities.jsonl'
    sessions=[d for d in read(features/'checkpoint/prepared.json')['sessions'] if '2026-06-01'<=d<='2026-07-10']
    require(len(sessions)==29, 'Wrong reserved period coverage')
    inference_program=bound(REPO/'scripts/portfolio_frozen_inference.py')
    result=infer(original, ev.read_jsonl(opening), ev.read_jsonl(checkpoint),
        read(features/'plan.json'), read(features/'checkpoint_plan.json'), sessions,
        digest(opening), tail=(), timed_source=digest(checkpoint))
    require(bound(inference_program['path'])==inference_program, 'Inference source changed')
    # Reconstruct the same frozen choices on the common 27-session cohort.
    parent_policy, parent_opening=verify_policy(FORWARD/'forward_policies',selected['policy']['name'])
    parent_updates=timed_policy(FORWARD/'forward_policies',parent_policy,parent_opening)
    def choices(s):
        return {c.symbol:(float(c.metadata.get('action_quantity_scale',1)),c.selection_score) for c in s.candidates}
    for day,snap in result['snapshots'].items():
        if str(day) in sessions[-2:]:continue
        require(choices(snap)==choices(parent_opening[day]),'Opening forward reconstruction mismatch')
        require(choices(result['updates'].get(day,[snap])[-1])==
                choices(parent_updates.get(day,[parent_opening[day]])[-1]),'Checkpoint forward reconstruction mismatch')
    policies=root/'policies'
    shutil.copytree(FORWARD/'forward_policies',policies)
    name=selected['policy']['name']
    write_json(policies/(name+'.snapshots.json'),[s.to_json_dict() for s in result['snapshots'].values()])
    jsonl(policies/(name+'.admission.jsonl'),result['admissions'])
    jsonl(policies/'timed.snapshots.jsonl',(s.to_json_dict() for rows in result['updates'].values() for s in rows))
    jsonl(policies/'timed.admission.jsonl',result['timed_admissions'])
    lineage=read(policies/'forward_lineage.json')
    lineage.update(no_entry_tail=[], standalone_protocol=bound(root/'protocol.json'),
        common_27_session_decision_parity=True, full_window_sessions=sessions)
    write_json(policies/'forward_lineage.json',lineage)
    pm={**parent_policy,'scope':'standalone_reserved_validation_fixed_inference_all_29_sessions'}
    pm['files']={p.name:digest(p) for p in policies.iterdir() if p.is_file() and p.name not in ('policies_manifest.json','completion.json')}
    write_json(policies/'policies_manifest.json',pm)
    verify_policy(policies,name);timed_policy(policies,pm,result['snapshots'])
    require(asdict(KALCBConfig.from_mapping(read(selected['configuration'])))==
        asdict(KALCBConfig.from_mapping(read(canonical/'optimized_config.json')['mutations'])), 'Canonical config drift')
    plan=read(selected['plan'])
    plan.update(start=sessions[0],end=sessions[-1],database=completion['database']['path'],
        config=selected['configuration'])
    context=read(plan['historical_reference'])['metrics']['audit_receipt']['identity']['configuration']
    context.update(start=sessions[0],end=sessions[-1],holdout_start=sessions[0],data_snapshot_end=sessions[-1],
        date_range={'start':sessions[0],'end':sessions[-1]},initial_mutations=read(selected['configuration']),
        baseline={'name':'locked_round1_standalone_oos','evaluation_start':sessions[0],'evaluation_end':sessions[-1]})
    write_json(root/'audit_configuration.json',context)
    plan['audit_configuration']=str(root/'audit_configuration.json')
    paths=set(plan['input_sha256'])-{read(selected['plan'])['database'],read(selected['plan'])['config']}
    paths.update((plan['database'],plan['config'],plan['audit_configuration']))
    plan['input_sha256']={p:digest(p) for p in sorted(paths)}
    write_json(root/'plan.json',plan)
    # Retain all market provenance, but replace stale derived KALCB research and
    # executable Python bindings with this round's frozen execution generation.
    parent_path=FORWARD/'validation/inputs/manifest.json'
    parent=read(parent_path); identity=parent.pop('manifest_hash')
    require(signature(parent)==identity,'Changed qualified forward parent')
    parent_root=parent_path.parent/'files'
    target=root/'market_parent/files';files={}
    for relative,expected in parent['files'].items():
        if (relative.endswith('.py') and not relative.startswith('research/')) or relative.startswith(('tests/','research/kalcb_executed_value/')):continue
        src=parent_root/relative
        require(digest(src)==expected,'Changed forward source: '+relative)
        dst=target/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        os.link(src,dst)
        files[relative]=expected
    filtered={**parent,'files':files}
    write_json(root/'market_parent/manifest.json',{**filtered,'manifest_hash':signature(filtered)})
    freeze(root/'market_parent/manifest.json',execution_source,policies,features/'checkpoint',[],root/'plan.json',root/'inputs')
    from backtests.core.input_manifest import InputManifest
    from backtests.engine.retained_source import verify_acceptance
    generation=InputManifest(root/'inputs/manifest.json')
    qualification=json.loads(generation.read(generation.root/generation.payload['evidence']['qualification_path']))
    verify_acceptance(lambda name:json.loads(generation.read(generation.root/name)),qualification,generation=generation)
    write_json(root/'preparation.json',{'passed':True,'protocol':bound(root/'protocol.json'),
        'source_manifest':bound(root/'inputs/manifest.json'),'source_generation_hash':read(root/'inputs/manifest.json')['source_generation_hash'],
        'economic_source_unchanged':True,'common_27_session_decision_parity':True,'sessions':sessions,
        'opening_admissions':len(result['admissions']),'timed_admissions':len(result['timed_admissions']),
        'tail_opening_admissions':sum(r['trade_date'] in sessions[-2:] for r in result['admissions']),
        'model_proofs':result['model_proofs'],'inference_program':inference_program})
    print(json.dumps({'prepared':str(root),'sessions':len(sessions),'opening_admissions':len(result['admissions']),
                      'timed_admissions':len(result['timed_admissions'])}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','evaluate'])
    p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    else:
        from backtests.strategies.kalcb.oos_validation import evaluate
        evaluate(a.root)


if __name__=='__main__':main()
