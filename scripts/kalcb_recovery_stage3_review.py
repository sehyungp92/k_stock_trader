"""Keep target-alignment results and their component costs alongside prior trials."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT, read, write, bound
from scripts.kalcb_alignment_walk_forward_v3 import DEST, ROUND, RECIPES
from scripts.kalcb_recovery_stage_review import paired
from scripts.kalcb_recovery_report import segments


def run():
    if (ROOT/'stage3_review.json').exists():
        raise ValueError('Preserve prior review')
    done=read(DEST/'replay_completion.json')
    if not done['complete']:raise ValueError('Incomplete main contrast')
    records={p.parent.name:read(p) for p in (ROOT/ROUND/'cases').glob('*/record.json')}
    if set(records)!={'control',*RECIPES}:raise ValueError('Missing comparison records')
    details={}
    for name,r in records.items():
        economics={}
        for key,ref in r['cases'].items():
            directory=Path(ref['path']).parent; result=read(directory/'result.json');trades=read(directory/'trades.json')
            if not result['execution_validity_pass'] or result['terminal_inventory'] or result['terminal_pending'] or result['residual_sessions']:
                raise ValueError('Invalid screening execution')
            gross=sum(t['gross_pnl'] for t in trades);fees=sum(t['commission'] for t in trades);net=sum(t['net_pnl'] for t in trades)
            if abs(gross-fees-net)>.01 or abs(net-result['return_pct']*1e6)>.01:
                raise ValueError('Portfolio cash/fee reconciliation failed')
            economics[key]={'gross_pnl_after_fill_slippage_krw':gross,'recorded_commission_and_tax_krw':fees,
                'net_pnl_krw':net,'audit_status':result['audit_status'],'attribution':segments(directory)}
        details[name]={**r,'economics':economics,'paired_to_canonical':{s:paired(Path(r['cases']['oos_'+s]['path']).parent,
            ROOT/'control'/s) for s in ('baseline','combined')}}
    components=[]
    for a,b in [('original_target_full_history','control'),('selected_target_value_only','original_target_full_history'),('selected_target_all','selected_target_value_only')]:
        x,y=details[a],details[b]
        components.append({'candidate':a,'component_control':b,
            'metric_delta':{k:x['metrics'][k]-y['metrics'][k] for k in x['metrics'] if k.endswith(('_return','_dd','_trades','_active'))},
            'paired':{s:paired(Path(x['cases']['oos_'+s]['path']).parent,Path(y['cases']['oos_'+s]['path']).parent) for s in ('baseline','combined')}})
    findings=[
        'The fresh original-target continuous-history control exactly matches canonical IS return/DD/trade metrics and earlier continuous_all_models OOS metrics. Full-history continuity and the new IS replay are now explicitly evidenced.',
        'Original continuous OOS: baseline -3.9635%/14.2625% DD/211 trades; combined -13.9874%/17.8648% DD/220. Frozen canonical OOS remains -15.5603%/18.2971%/290 and -26.2801%/27.2333%/300.',
        'Selected economic targets yield IS54.0828%/9.1951%/385 and combined34.1895%/11.5675%/390; OOS-2.5449%/14.6065%/236 and combined-16.3118%/20.8178%/243. Activity and baseline OOS improve versus continuous original, but stressed OOS, drawdown and IS returns deteriorate. This is not a clear improvement.',
        'Changing selected physical-risk targets as well produces identical observed portfolios in all four cells; retain it as a zero incremental-impact contrast, not another success.',
        'Matched selected-management action audit: admitted original/selected opening core baseline means -0.456/-1.843 reference-equity bps versus forecasts23.310/23.275. Selected admitted checkpoint mean-23.332 versus21.466 forecast, only18 observations. Weak/negative rank relationships survive target alignment. These fixed-reference isolated action outcomes are not portfolio returns.',
        'Common q1 feasible opening actions have Apr-May baseline mean13.230 bps and Jun-Jul-34.275; current selection does not capture the early opportunity efficiently and has large late overprediction. This motivates past-only recency and slope calibration, never retrospective date filters.',
        'All prior42 trial entries are retained through immutable stage2 linkage. The three new contrasts include a freshly rebuilt control, not three independent sources of evidence. All remain reused development.',
        'Next finite stage: four global-clock rolling120/decay60 economic fits (original/selected targets), two existing past-OOF affine opening calibrations, two ATR4/5 sizing perturbations on selected-target policies. Every trial gets actual IS/OOS native replay, with component/backout comparisons and full diagnostics.',
        'No canonical promotion. Native screening and cash reconciliation passed; full source/model/label and four-stress finalist qualification remains required.'
    ]
    payload={'created_at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),
        'prior42':bound(ROOT/'stage2_review.json'),'main_contrast_completion':bound(DEST/'replay_completion.json'),
        'outcome_audit':bound(ROOT/'alignment_outcome_review_v1.json'),'trials':details,'component_comparisons':components,'findings':findings,
        'independence':'All April1-July10 reused development; paired bootstrap is descriptive and unadjusted for adaptive selection.',
        'canonical_config':bound(Path('config/research/kalcb_intraday_advantage_20261007/selected_config.json').resolve())}
    write(ROOT/'stage3_review.json',payload)
    lines=['KALCB TARGET-ALIGNMENT REVIEW',payload['independence'],'',*findings,'',
        'Recipe | IS baseline return/DD/trades | IS combined | OOS baseline | OOS combined']
    for n,r in details.items():
        cells=[n];m=r['metrics']
        for p in ('is','oos'):
            for s in ('baseline','combined'):
                k=p+'_'+s;cells.append(f"{m[k+'_return']:.4f}/{m[k+'_dd']:.4f}/{m[k+'_trades']}")
        lines.append(' | '.join(cells))
    (ROOT/'stage3_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(ROOT/'stage3_review.txt'),'comparisons':len(details)}),flush=True)


if __name__=='__main__':run()
