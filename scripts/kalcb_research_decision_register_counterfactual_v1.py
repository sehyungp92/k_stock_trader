"""Bind accepted C18 labels to previously frozen diagnostics; no evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,gzip,argparse
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path(__file__).resolve().parents[1]
OLD=ROOT.parent.parent/'kalcb_round3_20261017/attempt_2/round_run_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def verify(b):assert bound(b['path'])['sha256']==b['sha256']
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--labels',default=str(ROOT/'labels_run_v5'));a=ap.parse_args()
    lab=Path(a.labels);lr=read(lab/'registration.json');tr=read(ROOT/'timing_run_v2/registration.json')
    assert read(lab/'labels_complete.json')['passed'] and read(ROOT/'timing_run_v2/timing_complete.json')['passed']
    frozen=read(ROOT/'counterfactual_programs_v1/freeze.json');programs=frozen['programs']
    for b in programs.values():verify(b)
    labels={p:{} for p in ['is','oos']};info={p:{} for p in ['is','oos']};bindings=[]
    for p in labels:
        for s in ['baseline','combined']:
            receipt=lab/f'full/{p}/{s}_acceptance.json'
            # The existing label controller owns this storage layout.
            if not receipt.exists():
                candidates=[f for f in lab.rglob('*_acceptance.json') if (r:=read(f))['process']['period']==p and r['process']['scenario']==s and not r['process']['pilot']]
                assert len(candidates)==1;receipt=candidates[0]
            r=read(receipt);assert r['passed'] and not r['eligible_for_scoring']
            assert r['completion']['distinct_episodes']==lr['expected_episode_counts'][p]
            for b in r['files'].values():verify(b)
            labels[p][s]=r['files']['labels.jsonl.gz']
            info[p][s]=dict(labels=r['files']['labels.jsonl.gz'],details=r['files']['native_records.jsonl.gz'],receipt=bound(receipt))
            bindings.append(bound(receipt))
    controls={p:{} for p in labels}
    for r in read(OLD/'evaluations/v6_h0.seal.json')['receipts']:
        if r['process']['scenario'] in ['baseline','combined']:controls[r['item']['period']][r['process']['scenario']]={'folder':r['folder']}
    days={p:sorted(d['day'] for d in read(Path(controls[p]['baseline']['folder'])/'daily.json')) for p in labels}
    common=dict(at_utc=datetime.now(timezone.utc).isoformat(),study_start_epoch=tr['study_start_epoch'],
        financial_deadline_epoch=tr['financial_deadline_epoch'],report_deadline_epoch=tr['report_deadline_epoch'],
        timing_registration=bound(ROOT/'timing_run_v2/registration.json'),base_controller=bound(ROOT/'timing_run_v2/controller.py'),
        profiles=lr['profiles'],cohorts=lr['cohorts'],controls=controls,holdout_accesses=0,max_workers=2,
        bindings=bindings+[bound(__file__),bound(ROOT/'counterfactual_programs_v1/freeze.json'),
            bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043-counterfactual-protocol.md'),
            bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043-resource-amendment.md'),
            bound(ROOT/'oracle_unit_v1/receipt.json'),bound(ROOT/'horizon_unit_v3/receipt.json'),bound(lab/'registration.json')])
    out=ROOT/'oracle_run_v1';out.mkdir(exist_ok=False)
    names=['kalcb_research_decision_oracle_run_v1.py','kalcb_research_decision_oracle_worker_v1.py','research_decision_oracle_v1.py','research_decision_horizon_v2.py','kalcb_research_decision_audit_v1.py']
    reg=dict(**common,labels=labels,programs=[programs[n] for n in names],
        controller_program=programs[names[0]],worker_program=programs[names[1]],oracle_program=programs[names[2]],horizon_program=programs[names[3]],
        independent_accounting_program=programs[names[4]],smoke_through={p:ds[9] for p,ds in days.items()},planning_native_seconds=900,
        required_completions=[bound(ROOT/'timing_run_v2/timing_complete.json'),bound(lab/'labels_complete.json')],
        prerequisite_launches=[bound(ROOT/'timing_run_v2/launch.json'),bound(lab/'launch.json')],eligible_for_scoring=False)
    save(out/'registration.json',reg)
    # Participation is read directly from the frozen scenario declaration, never inferred from results.
    code=Path(lr['profiles']['is']['manifest']['path']).parent/'files'
    import ast
    tree=ast.parse((code/'backtests/strategies/kalcb/executed_value.py').read_text(encoding='utf-8'))
    values=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCENARIOS' for t in n.targets))
    source=programs['kalcb_research_decision_information_v1.py'];infoout=ROOT/'information_v1';infoout.mkdir(exist_ok=False)
    databases={p:bound(read(lr['profiles'][p]['plan']['path'])['database']) for p in labels}
    assert read(ROOT/'information_unit_v3/receipt.json')['passed']
    assert read(ROOT/'information_unit_v3/registration.json')['analysis']['sha256']==source['sha256']
    infocommon={**common,'controls':{p:controls[p]['baseline'] for p in controls},
        'bindings':common['bindings']+[source,bound(lab/'registration.json'),bound(ROOT/'information_unit_v3/receipt.json'),bound(ROOT/'information_unit_v3/registration.json')]}
    save(infoout/'registration.json',dict(**infocommon,labels=info,program=source,databases=databases,calendars=days,label_registration=bound(lab/'registration.json'),
        participation={s:values[s][0] for s in ['baseline','combined']},financial_evaluations=0,output=str(infoout/'results')))
    print(json.dumps(dict(oracle=bound(out/'registration.json'),information=bound(infoout/'registration.json'))))
if __name__=='__main__':main()
