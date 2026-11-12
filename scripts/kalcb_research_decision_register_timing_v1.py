"""Bind the finite C18 timing diagnosis before any new account outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,time

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OLD=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2/round_run_v2')
def read(p):return json.loads(Path(p).read_text())
def bound(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p.resolve()),sha256=h)

def main():
    root=ROOT/'timing_run_v1';root.mkdir(exist_ok=False)
    shutil.copyfile(REPO/'scripts/kalcb_research_decision_run_v1.py',root/'controller.py')
    audit=ROOT/'saved_evidence_v1';assert read(audit/'completion.json')['passed']
    unit=ROOT/'unit_v2/receipt.json';assert read(unit)['passed']
    prep=ROOT/'native_v1/preparation.json';assert read(prep)['complete']
    start=datetime.fromisoformat(read(audit/'registration.json')['at_utc']).timestamp()
    sources=[root/'controller.py',Path(__file__),REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043.md',
        prep,unit,ROOT/'unit_v2/registration.json',ROOT/'unit_v2/stdout.log',audit/'completion.json',
        audit/'authentication.json',audit/'reproduction.json',audit/'accounting.csv',
        OLD/'round_artifacts/phase_state.json',OLD/'evaluations/v6_h0.seal.json',OLD/'registration.json']
    receipts=read(OLD/'evaluations/v6_h0.seal.json')['receipts']
    seconds=[r['process']['seconds'] for r in receipts]
    reg=dict(version='C18-timing-v1',at_utc=datetime.now(timezone.utc).isoformat(),
        study_start_epoch=start,financial_deadline_epoch=start+13*3600,report_deadline_epoch=start+16*3600,
        native_preparation=bound(prep),saved_evidence=bound(audit/'completion.json'),unit_receipt=bound(unit),
        bindings=[bound(p) for p in sources],max_workers=2,financial_accounts=18,
        arms=['off','entry','exit'],scenarios=['baseline','participation','cost','combined'],
        off_cells=['is/baseline','oos/combined'],delay_cells='both periods, all four scenarios',
        economics='Only selected ordinary market action activation delayed 5min; all original policies/configs/native execution unchanged',
        launch_rule='Phase1 disabled path/hydration must pass before Phase2; measured cost gate before each delayed arm',
        decision_basis='Saved matched-price identity does not independently identify entry versus exit effects; both price terms are material',
        resource=dict(chosen_total_wall_hours=16,aggregate_process_hours=24,report_reserve_hours=3,
            preparation_seconds=read(prep)['seconds'],authentication_seconds=read(audit/'completion.json')['elapsed_seconds'],
            historical_account_seconds=seconds,projected_timing_wall_seconds=18*max(seconds)/2*1.5,
            projected_total_study_hours=12,estimate_status='Planning estimate; label/oracle feasibility/runtime still required'),
        no_policy_selection=True,no_treatment_optimisation=True,stopped_operations=False,holdout_accesses=0)
    with (root/'registration.json').open('x') as f:json.dump(reg,f,indent=2)
    print(json.dumps(bound(root/'registration.json')))

if __name__=='__main__':main()
