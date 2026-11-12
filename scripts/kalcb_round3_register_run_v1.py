"""Freeze the controller and complete bounded workload before full treatments."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,os,shutil,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import ROOT,read,write,bound
REPO=Path(__file__).resolve().parents[1];OUT=ROOT/'round_run_v1';start=time.monotonic()

def main():
    OUT.mkdir(exist_ok=False)
    reach=read(ROOT/'reach_review_v1.json');arms=read(ROOT/'native_v5/arms.json')
    for p in ('preflight_v2/bridge_completion.json','reporting_preflight_v1/completion.json','reused_controls_v2/completion.json','capital_controls_v1/completion.json'):
        if read(ROOT/p).get('passed') is not True:raise ValueError('Missing preflight '+p)
    for p in ('is','oos'):
        for label in ('pending','inventory'):
            if read(ROOT/f'restart_v3/{p}_{label}.json')['passed'] is not True:raise ValueError('Fresh restart missing')
    if not reach['complete'] or not reach['delay_reach']['changed_physical_fills']:raise ValueError('Native reach missing')
    # All earlier attempts, including failures, count toward this chosen ceiling.
    processes=[]
    for directory in ('bridge_v1','bridge_v2','preflight_v1','preflight_v2','reach_v1'):
        for path in (ROOT/directory).rglob('*_process.json'):processes.append((path,read(path)))
    seconds=sum(p.get('seconds',0) for _,p in processes);equivalents=0
    for _,p in processes:
        if p.get('smoke'):equivalents+=(3/156 if p['period']=='is' else 34/69)
        else:equivalents+=1 # Conservative: even failed startup counts as a full account.
    # Fresh-process verifier continuations and two failed restart verifier attempts.
    equivalents+=6;seconds+=6*30
    selectable=[n for n,a in arms.items() if a['selectable'] and reach['arms'][n]['eligible_for_full_evaluation']]
    diagnostics=[n for n in arms if n.startswith('remove_') and n not in ('remove_cap','remove_veto') and reach['arms'][n]['eligible_for_full_evaluation']]
    # Qualification neighbours remain within the original eight-policy grid, even if reach excludes one from selection.
    max_accounts=8*len(diagnostics)+8*8+(16+16+8)*len(selectable)+8+2
    if equivalents+max_accounts>512 or seconds+1200*max_accounts>720000:raise ValueError('Prospectively reduce scope before launch; registered budget exceeded')
    timings={p:[r['seconds'] for _,r in processes if not r.get('smoke') and r.get('exit_code')==0 and r.get('period')==p] for p in ('is','oos')}
    per_account=sum(sum(v)/len(v) for v in timings.values())/2
    # Finer-marks cost is measured inside reporting preflight, conservatively assigned entirely to every qualification marking.
    reporting_seconds=read(ROOT/'reporting_preflight_v1/completion.json')['seconds']
    finer_cases=8+(24+8)*len(selectable);preparation_elapsed=6*60
    estimate=dict(complete_max_additional_accounts=max_accounts,selectable=selectable,diagnostic_new=diagnostics,reused_control_accounts=16,
        mean_native_seconds_by_period={p:sum(v)/len(v) for p,v in timings.items()},
        central_native_account_hours=max_accounts*per_account/3600,guarded_native_account_hours=max_accounts/3,
        preparation_allowance_hours=preparation_elapsed/3600,qualification_reporting_hours=(finer_cases+8)*reporting_seconds/3600,
        typical_wall_hours=max_accounts*per_account/7200+(finer_cases+8)*reporting_seconds/3600+preparation_elapsed/3600,
        worst_native_plus_serial_qualification_hours=(seconds+1200*max_accounts+(finer_cases+8)*reporting_seconds+preparation_elapsed)/3600,
        estimate_limitation='Conservative all-survivor branch; logically rejected candidates stop qualification. Shared scenarios and capitals are not independent samples.',
        chosen_ceiling_not_resource_exhaustion=True)
    if estimate['worst_native_plus_serial_qualification_hours']>200:raise ValueError('Complete cost above chosen budget')
    write(OUT/'cost_estimate.json',estimate)
    # Snapshot source, never hard-link mutable working files. No secrets or new market data.
    code=OUT/'controller/files';names=subprocess.check_output(['rg','--files','-g','*.py','-g','!data/**','-g','!logs/**','-g','!docs/**'],cwd=REPO,text=True).splitlines()
    source_bindings=[];programs=[]
    for name in names+['kis_core/data/krx_holidays.yaml']:
        source=REPO/name;dest=code/name;before=bound(source);content=source.read_bytes();dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(content)
        after=bound(source);saved=bound(dest)
        if before['sha256']!=after['sha256'] or before['sha256']!=saved['sha256']:raise ValueError('Source changed during snapshot')
        if dest.suffix=='.py':ast.parse(content)
        source_bindings.append(dict(original=before,saved=saved));programs.append(saved)
    write(OUT/'controller_manifest.json',dict(files=source_bindings,seconds=time.monotonic()-start,mutable_workspace_not_execution_source=True))
    documents=['kalcb-exp-0042.md','kalcb-exp-0042-executable-binding.md','kalcb-exp-0042-qualification-binding.md']
    for name in documents:
        source=REPO/'docs/optimisation/kalcb/experiments'/name;dest=OUT/'documents'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest);programs.append(bound(dest))
    registration=dict(version='C17-full-native-round-v1',at_utc=datetime.now(timezone.utc).isoformat(),repository=str(REPO),
        programs=programs,controller_manifest=bound(OUT/'controller_manifest.json'),freeze=bound(ROOT/'freeze/manifest.json'),
        reach=bound(ROOT/'reach_review_v1.json'),reused_controls=bound(ROOT/'reused_controls_v2/completion.json'),capital_controls=bound(ROOT/'capital_controls_v1/completion.json'),
        resource_envelope=dict(spent_account_equivalents=equivalents,spent_native_seconds=seconds,max_account_equivalents=512,max_total_account_hours=200),
        cost_estimate=bound(OUT/'cost_estimate.json'),native_runtime=bound(ROOT/'native_v5/preparation.json'),
        max_workers=2,nested_native_workers=0,family_size=8,scenario_weights={s:.25 for s in ('baseline','participation','cost','combined')},
        period_weights={'is':.5,'oos':.5},score='C16 unchanged',holdout='None untouched; zero holdout access',
        scope_exclusions=['User-stopped exp0031 and exp0039/C15','New ranker/entry mechanism','Live activation','Data acquisition'],
        measurement_gaps=['All-rejected standalone labels','Global funded S/E/SE oracles','Matched-entry placebo'],
        inactive_prefix_coverage_gaps=[n for n,a in reach['arms'].items() if not a['eligible_for_full_evaluation']],
        no_full_treatment_outcomes_inspected=True)
    write(OUT/'registration.json',registration)
    print(json.dumps(dict(registered=str(OUT/'registration.json'),controller_files=len(programs),cost=estimate,spent_equivalents=equivalents),indent=2))

if __name__=='__main__':main()
