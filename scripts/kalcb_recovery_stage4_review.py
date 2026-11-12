"""Review the completed finite recency/calibration/sizing round, retaining costs."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,read,write,bound,rows
from scripts.kalcb_recovery_mechanism_round import DEST,ROUND,RECIPES,SIZING,check
from scripts.kalcb_recovery_stage_review import paired
from scripts.kalcb_recovery_report import segments


def review():
    if (ROOT/'stage4_review.json').exists():raise ValueError('Preserve completed review')
    check(DEST);done=read(DEST/'replay_completion.json')
    if not done['complete']:raise ValueError('Incomplete native round')
    prior=read(ROOT/'stage3_review.json');records={p.parent.name:read(p) for p in (ROOT/ROUND/'cases').glob('*/record.json')}
    rejected={p.parent.name:read(p) for p in (ROOT/ROUND/'cases').glob('*/rejected_record.json')}
    if set(records)|set(rejected)!={'control',*RECIPES,*SIZING}:raise ValueError('Incomplete registered trial inventory')
    details={};components=[]
    for name,r in records.items():
        cells={}
        for key,ref in r['cases'].items():
            directory=Path(ref['path']).parent;result=read(directory/'result.json');trades=read(directory/'trades.json')
            if not result['execution_validity_pass'] or result['residual_sessions'] or result['terminal_inventory'] or result['terminal_pending']:
                raise ValueError('Invalid admitted screening record')
            gross=sum(t['gross_pnl'] for t in trades);fees=sum(t['commission'] for t in trades);net=sum(t['net_pnl'] for t in trades)
            if abs(gross-fees-net)>.01 or abs(net-result['return_pct']*1e6)>.01:raise ValueError('Cash mismatch')
            cells[key]={'gross_after_slippage':gross,'commission_tax':fees,'net':net,'audit_status':result['audit_status'],'attribution':segments(directory)}
        item={**r,'cells':cells,'paired_to_canonical':{s:paired(Path(r['cases']['oos_'+s]['path']).parent,ROOT/'control'/s) for s in ('baseline','combined')}}
        if name!='control':
            base='original_target_full_history' if name.startswith('original_') else 'selected_target_all'
            parent=prior['trials'][base]
            item['component_control']=base
            components.append({'name':name,'control':base,
                'metric_delta':{k:r['metrics'][k]-parent['metrics'][k] for k in r['metrics'] if k.endswith(('_return','_dd','_trades','_active'))},
                'paired':{s:paired(Path(r['cases']['oos_'+s]['path']).parent,Path(parent['cases']['oos_'+s]['path']).parent) for s in ('baseline','combined')}})
        details[name]=item
    calibration={}
    for name in ('original_affine','selected_affine'):
        calibration[name]=read(DEST/'fit/policies'/name/'oos/chronology.json')['calibration']
    # Diagnostic population support is separate from economic improvement;
    # these counts do not claim broker-effective share interventions.
    support={}
    for name in RECIPES:
        forecast=rows(DEST/'fit/forecasts'/name/'opening.jsonl')
        scope=[r for r in forecast if r['trade_date']>='2026-04-01']
        support[name]={'opening_quantity_actions':len(scope),'capacity_ok':sum(r['capacity_ok'] for r in scope),
            'capacity_and_physical_risk_ok':sum(r['capacity_ok'] and r['failure_probability']<.001 for r in scope),
            'unique_issuer_days':len({r['opportunity_id'].rsplit('/q',1)[0] for r in scope})}
    all_records=[*read(ROOT/'stage2_review.json')['trials'],*[r for n,r in prior['trials'].items() if n!='control'],*[r for n,r in details.items() if n!='control']]
    objectives=[(f'oos_{s}_{m}',-1 if m=='dd' else 1) for s in ('baseline','combined') for m in ('return','dd','trades')]
    dominates=lambda a,b:all(sign*a[k]>=sign*b[k] for k,sign in objectives) and any(sign*a[k]>sign*b[k] for k,sign in objectives)
    frontier=[r['recipe'] for r in all_records if not any(dominates(t['metrics'],r['metrics']) for t in all_records)]
    findings=[
        'Original decay60 is the first newly tested full-history recipe in this stage with positive baseline OOS return:0.6747%/10.4047% DD/271 trades; combined remains-11.2496%/14.9299%/281. Its fresh IS47.9022%/7.3775%/382 and combined27.9645%/9.0040%/383 materially sacrifice canonical IS return. A useful tradeoff, not a qualified robust recovery.',
        'Both hard rolling120 variants deteriorate OOS return relative to their expanding component controls. Selected decay60 lowers drawdown and increases activity but does not resolve negative OOS expectancy.',
        'Both affine variants deteriorate full-window OOS return and drawdown markedly. Their fitted score slopes reach zero for the June8 fold in both scenarios: historical residual calibration can erase discrimination. Do not turn this observed date into an exclusion rule.',
        'Current capacity screen alone passes4770/4827 OOS opening quantity actions. Capacity-only filtering of model training is therefore a weak next hypothesis; the stricter learned physical-risk screen creates most of the eligibility reduction. No failed labels will be dropped to manufacture safer training returns.',
        'Retain all eight comparisons, paired uncertainty, full-period fees and IS costs. The OOS Pareto frontier is descriptive reused-development evidence, includes previous OOS-only schedules with explicitly historical IS references, and is not a promotion list.',
        'Next finite hypotheses: retain long-history information via fixed25/50/75percent expanding/decay forecast blends and recalibrate each blend on its own earlier OOF forecasts; test prior promising ATR4/5 sizing on decay60 plus ATR5 on the central blend; separately test past-only checkpoint residual calibration on original expanding and decay60. These are eight specific contrasts, not an open parameter search.',
        'Checkpoint calibration keeps native action timing, half-size cap and48bps margin; only earlier matched-target residuals can change forecasts. No date/symbol filters or relaxed execution assumptions.',
        'Screening is distinct from finalist qualification. Canonical remains unchanged. Every finalist needs full source/config/model/label lineage and four native stress scenarios plus support, concentration, parameter stability and account reconciliation.'
    ]
    payload={'created_at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),'prior':bound(ROOT/'stage3_review.json'),
        'completion':bound(DEST/'replay_completion.json'),'trials':details,'rejected':rejected,'component_comparisons':components,
        'oos_frontier_including_prior_trials':frontier,'affine_calibration':calibration,'opportunity_support':support,'findings':findings,
        'scope':'All April1-July10 results reused development. Paired resampling is descriptive and unadjusted for selection. All eight new trials have actual new IS native replays.'}
    write(ROOT/'stage4_review.json',payload)
    lines=['KALCB RECENCY/CALIBRATION/SIZING REVIEW',payload['scope'],'',*findings,'',
        'Recipe | IS baseline return/DD/trades | IS combined | OOS baseline | OOS combined']
    for name,r in details.items():
        m=r['metrics'];cells=[name]
        for p in ('is','oos'):
            for s in ('baseline','combined'):
                k=p+'_'+s;cells.append(f"{m[k+'_return']:.4f}/{m[k+'_dd']:.4f}/{m[k+'_trades']}")
        lines.append(' | '.join(cells))
    (ROOT/'stage4_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'review':str(ROOT/'stage4_review.txt'),'records':len(details),'rejected':list(rejected),'frontier':frontier}),flush=True)


if __name__=='__main__':review()
