"""Freeze C18 label programs/protocol before pilot outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,gzip,collections
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(p),sha256=sha(p))
def main():
    out=ROOT/'labels_run_v5';out.mkdir(exist_ok=False);programs=out/'programs';programs.mkdir()
    files={}
    for name,source in {'worker':REPO/'scripts/kalcb_research_decision_label_worker_v4.py',
        'controller':REPO/'scripts/kalcb_research_decision_labels_run_v5.py',
        'horizon':REPO/'backtests/strategies/kalcb/research_decision_horizon_v2.py'}.items():
        dest=programs/source.name;shutil.copyfile(source,dest);files[name]=bound(dest)
    timing=read(ROOT/'timing_run_v2/registration.json');prep=read(ROOT/'native_v1/preparation.json')
    counts=collections.Counter();days=collections.defaultdict(set)
    cohorts=ROOT/'cohorts_v1/episodes.jsonl.gz'
    with gzip.open(cohorts,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);days[r['period']].add(r['trade_date'])
            if r['candidate'] and r['original_forecasts']:counts[r['period']]+=1
    reg=dict(version='C18-labels-v5',at_utc=datetime.now(timezone.utc).isoformat(),
        study_start_epoch=timing['study_start_epoch'],financial_deadline_epoch=timing['financial_deadline_epoch'],report_deadline_epoch=timing['report_deadline_epoch'],
        timing_registration=bound(ROOT/'timing_run_v2/registration.json'),base_controller=bound(ROOT/'timing_run_v2/controller.py'),
        native_preparation=bound(ROOT/'native_v1/preparation.json'),saved_evidence=bound(ROOT/'saved_evidence_v1/completion.json'),
        horizon_unit=bound(ROOT/'horizon_unit_v3/receipt.json'),worker_program=files['worker'],horizon_program=files['horizon'],
        controller_program=files['controller'],programs=list(files.values()),profiles={p:prep['profiles'][p]['off'] for p in ['is','oos']},
        cohorts=bound(cohorts),decision_scores=bound(ROOT/'scores_v1/decision_scores.jsonl.gz'),
        finer_sources=bound(ROOT.parent.parent/'kalcb_round2_assessment_20261017/v1/capital_minute_registration.json'),
        horizons=[0,30,60,120],scenarios=['baseline','combined'],pilot_sessions={p:sorted(ds)[:2] for p,ds in days.items()},
        expected_episode_counts=counts,max_distinct_episodes=24000,worker_seconds_guard=7200,aggregate_label_seconds=14400,max_workers=2,
        timing_completion_required=True,all_labels_diagnostic_only=True,holdout_accesses=0,
        bindings=[bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043.md'),
            bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043-label-protocol.md'),bound(__file__),
            bound(ROOT/'horizon_unit_v3/registration.json'),bound(ROOT/'horizon_unit_v3/receipt.json'),
            bound(ROOT/'horizon_cache_unit_v1/registration.json'),bound(ROOT/'horizon_cache_unit_v1/receipt.json'),bound(ROOT/'labels_run_v4/not_launched.json'),
            bound(ROOT/'cohorts_v1/completion.json'),bound(ROOT/'scores_v1/completion.json'),
            bound(ROOT/'timing_run_v2/controller.py'),bound(ROOT/'timing_run_v2/registration.json'),
            bound(ROOT/'labels_run_v2/pre_dispatch_withdrawal.json'),bound(ROOT/'labels_run_v3/pre_dispatch_withdrawal.json'),bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043-counterfactual-protocol.md'),bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043-label-clock-amendment.md')])
    with (out/'registration.json').open('x',encoding='utf-8') as f:json.dump(reg,f,indent=2)
    print(json.dumps(dict(registration=bound(out/'registration.json'),expected_episodes=counts,pilot_sessions=reg['pilot_sessions'])))
if __name__=='__main__':main()
