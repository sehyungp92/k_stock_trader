"""One training-only incremental-information assay, never an allocation policy."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import math
import random

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/structural_v6')
ROOT=BASE/'sector_information_v1'
OUT=REPO/'data/backtests/analysis/olr_structural_20261012/sector_information_v1'
FEATURE=BASE/'sector_feature_audit_v1'
FEATURE_OUT=OUT.parent/'sector_feature_audit_v1'
SIGNALS=OUT.parent/'fixed_signal_v1'
FILES=['scripts/olr_sector_information_v11.py','tests/strategy_olr/test_sector_information_v11.py']
BLOCKS=[('2025-08-07','2025-10-31'),('2025-11-01','2025-12-31'),('2026-01-01','2026-03-31')]
PENALTY=12.


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def utc():return datetime.now(timezone.utc).isoformat()


def records(lanes,ledger):
    features={(r['day'],r['symbol']):r for r in ledger}
    assert len(features)==len(ledger)
    groups=defaultdict(dict)
    for lane in lanes:
        key=(lane['scenario'],lane['budget']);g=groups[lane['day'],lane['symbol']]
        assert key not in g;g[key]=lane
    needed={(s,b) for s in ('baseline','combined') for b in (1_000_000,20_000_000)}
    out=[]
    for (day,symbol),g in sorted(groups.items()):
        assert set(g)==needed
        f=features[day,symbol];own=f['own'];peer=f['peer_support']
        reasons=[]
        if not own or not peer:reasons.append('missing_source_prefix')
        else:
            if own['prior_count']!=20:reasons.append('short_own_history')
            if not peer['supported']:reasons.append('unknown_peer_baseline')
            if not peer['complete_fixed_peer_coverage']:reasons.append('incomplete_fixed_peer_coverage')
            for s in peer['contributing_peers']:
                other=features[day,s]
                if not other['own'] or other['own']['prior_count']!=20:
                    reasons.append('short_peer_history');break
        mature=all(x['status']=='matured' for x in g.values())
        small=g['combined',1_000_000]
        if not reasons:
            breadth=float(f['history_native']['breadth']);participation=float(f['history_native']['participation'])
            assert 0<=participation<=breadth+1e-12<=1+1e-12
            # Native single-day participation degenerates to price breadth.
            assert math.isclose(float(f['legacy_native']['participation']),breadth,abs_tol=1e-12)
            if small['native_signal_time']:
                assert datetime.fromisoformat(f['available_at'])<=datetime.fromisoformat(small['native_signal_time'])
            x=[breadth,participation-breadth]
        else:x=None
        if mature:
            assert small['filled_notional']>0 and small['net_pnl'] is not None
            y=10000*small['net_pnl']/small['filled_notional']
            assert math.isclose(y,small['net_bps'],rel_tol=1e-10,abs_tol=1e-10)
            available=max(v['available_for_training_after'] for v in g.values())
        else:y=available=None
        out.append(dict(day=day,symbol=symbol,x=x,y=y,available=available,
            source_exclusions=reasons,source_available_at=f['available_at']))
    return out


def weights(rows):
    counts=defaultdict(int)
    for r in rows:counts[r['day']]+=1
    return [1/counts[r['day']] for r in rows],len(counts)


def fit(rows,dimensions,penalty=PENALTY):
    w,n=weights(rows);assert n>=10 and dimensions in (1,2)
    mx=[math.fsum(a*r['x'][j] for r,a in zip(rows,w))/n for j in range(dimensions)]
    my=math.fsum(a*r['y'] for r,a in zip(rows,w))/n
    variance=[math.fsum(wt*(r['x'][j]-mx[j])**2 for r,wt in zip(rows,w))/n for j in range(dimensions)]
    scales=[math.sqrt(v) if v>0 else 1. for v in variance]
    # Algebraically equivalent to training-only standardization followed by
    # equal ridge penalties, with coefficients returned in original units.
    a=[[math.fsum(wt*(r['x'][j]-mx[j])*(r['x'][k]-mx[k]) for r,wt in zip(rows,w))+(penalty*scales[j]**2 if j==k else 0)
        for k in range(dimensions)] for j in range(dimensions)]
    b=[math.fsum(wt*(r['x'][j]-mx[j])*(r['y']-my) for r,wt in zip(rows,w)) for j in range(dimensions)]
    if dimensions==1:slopes=[b[0]/a[0][0]]
    else:
        det=a[0][0]*a[1][1]-a[0][1]*a[1][0];assert det>0
        slopes=[(b[0]*a[1][1]-a[0][1]*b[1])/det,(a[0][0]*b[1]-b[0]*a[1][0])/det]
    return dict(intercept=my-math.fsum(v*x for v,x in zip(slopes,mx)),slopes=slopes,
        train_dates=n,train_opportunities=len(rows),training_keys=[[r['day'],r['symbol']] for r in rows],penalty=penalty,training_scales=scales)


def daily_means(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[r['day']].append(r[key])
    return {d:math.fsum(v)/len(v) for d,v in sorted(groups.items())}


def evaluate(model,test):
    return [model['intercept']+math.fsum(v*x for v,x in zip(model['slopes'],r['x'])) for r in test]


def split(data,lo,hi):
    current=[r for r in data if lo<=r['day']<=hi];assert current
    cutoff=min(r['day'] for r in current)+'T00:00:00+09:00'
    train=[r for r in data if r['day']<lo and r['available'] and r['available']<cutoff and r['x'] is not None]
    test=[r for r in current if r['x'] is not None and r['available'] and r['available'][:10]<=hi]
    return current,train,test,cutoff


def bootstrap(values,seed):
    rng=random.Random(seed);n=len(values)
    means=sorted(math.fsum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return [means[99],means[1899]]


def analyze(data):
    folds=[];splits=[]
    for lo,hi in BLOCKS[1:]:
        current,train,test,cutoff=split(data,lo,hi);splits.append((train,test))
        nt=len({r['day'] for r in test});ntrain=len({r['day'] for r in train})
        folds.append(dict(start=lo,end=hi,cutoff=cutoff,registered_opportunities=len(current),
            source_supported_opportunities=sum(r['x'] is not None for r in current),
            test_opportunities=len(test),test_dates=nt,training_dates=ntrain,coverage=len(test)/len(current),
            support_gate=nt>=10 and ntrain>=10 and len(test)>=.8*len(current)))
    enough=all(f['support_gate'] for f in folds)
    if enough:
        for i,(f,(train,test)) in enumerate(zip(folds,splits)):
            base=fit(train,1);aug=fit(train,2);pb=evaluate(base,test);pa=evaluate(aug,test)
            predictions=[dict(day=r['day'],symbol=r['symbol'],actual=r['y'],x=r['x'],baseline=b,augmented=a,
                baseline_error=(r['y']-b)**2,augmented_error=(r['y']-a)**2,
                error_gain=(r['y']-b)**2-(r['y']-a)**2) for r,b,a in zip(test,pb,pa)]
            bm=math.fsum(daily_means(predictions,'baseline_error').values())/f['test_dates']
            am=math.fsum(daily_means(predictions,'augmented_error').values())/f['test_dates']
            gain=1-am/bm if bm else None
            daily=daily_means(predictions,'error_gain');positive=sorted((x for x in daily.values() if x>0),reverse=True)
            f.update(baseline=base,augmented=aug,predictions=predictions,baseline_mse=bm,augmented_mse=am,
                relative_mse_gain=gain,incremental_coefficient=aug['slopes'][1],
                error_gain_descriptive_90pct_interval=bootstrap(list(daily.values()),27110+i),
                top3_positive_error_gain_share=math.fsum(positive[:3])/math.fsum(positive) if positive else None,
                gates=dict(error_reduction_at_least_5pct=gain is not None and gain>=.05,volume_confirmation_coefficient_positive=aug['slopes'][1]>0))
    return dict(folds=folds,support_passed=enough,
        incremental_volume_information_supported=enough and all(all(f['gates'].values()) for f in folds),
        candidate_policy_created=False,allocation_design_eligible=False,portfolio_experiment_eligible=False,
        promotion=False,new_native_replays=0,independent_validation=False,
        interpretation='Small-target combined-stress net return on actual acquired notional for original confirmed incumbents only. Tests volume confirmation beyond price breadth; not rank-to-capital calibration, added frequency or a self-funding portfolio. Outcomes were already inspected; this is chronological development evidence.')


def authenticate():
    p=read(ROOT/'protocol.json')
    for path,h in p['bindings'].items():assert sha(path)==h,path
    return p


def freeze():
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)>=9
    assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    assert sha(FEATURE_OUT/'scope_delivery.json')=='a7bcf36f894edbdd42179431b57287f8485cb49165fe4cae4fcce6c5c93ff479'
    assert sha(SIGNALS/'results.json')=='931ea3e3879462043d63f93f36ee0aae127c4b270e00563e9e38ad3013970be5'
    assert sha(SIGNALS/'delivery_checks.json')=='aa65c05d0da19ec688a97000294ad6b5709b0de230b47a76c8cd245dee05ef8c'
    assert read(FEATURE/'completion.json')['ledger_sha256']==sha(FEATURE/'ledger.json')
    paths=[REPO/n for n in FILES]+[ROOT/'tests.xml',ROOT/'tests.log']
    paths += [FEATURE/n for n in ('ledger.json','completion.json','scope_complete.json','scope_repair.json','protocol.json')]
    paths += [FEATURE_OUT/n for n in ('summary.json','report_receipt.json','delivery_checks.json','scope_delivery.json')]
    paths += [SIGNALS/n for n in ('results.json','report_receipt.json','delivery_checks.json')]
    paths += [BASE/'fixed_signal_v1/design.json',OUT.parent/'allocation_calibration_v1/results.json']
    p=dict(utc=utc(),version='incremental-sector-information-v1',
        justification='Source audit confirms causal historical volume with full20 own histories in all supported training rows; old static values match exactly. Legacy current participation degenerates to price breadth. Confirmed small-target stressed payoff is positive in all three fixed diagnostic blocks. Test whether volume adds predictive information beyond price breadth before any corrected-score policy.',
        scope='Only83 original outcome-blind sampled incumbents on36 dates, existing332 lanes. No new labels, OOS, rank model, changed selections or new trades. Rejected score-only cohort excluded.',
        features='At original14:30 source clock: price breadth; incremental feature=volume-confirmed participation minus breadth, in[-1,0]. Own and every contributing peer require20 prior positive complete-prefix observations, and full fixed-universe peer coverage. Missing remains missing; no imputation or value cutoff.',
        target='Combined-stress KRW1m lane net PNL divided by actual acquired notional, bps. All four paired lanes must mature to preserve the original coverage contract. No quantity scaling or rank variable.',
        estimators='Exactly baseline intercept+breadth versus augmented intercept+breadth+(participation-breadth). Features are standardized using date-weighted training-only means/standard deviations; constant feature scale1. Same ridge:12 date units on each standardized slope, intercept unpenalized. Returned coefficients in original units. This makes penalty invariant to feature units; no held-out scaling or penalty tuning.',
        chronology='Exactly Aug-Oct->Nov-Dec and Aug-Dec->Jan-Mar. Fit once per block; training all paired labels mature before first test date, test maturity byblockend. No within-block updates. Date weights equal total mass per date.',
        gates='Both folds must have>=10 mature source-supported training and test dates,>=80% coverage of all registered test opportunities. If either fails, do not fit either fold. Otherwise require>=5% date-weighted MSE reduction versus breadth-only AND positive incremental coefficient in BOTH folds. This nontrivial prediction hurdle rejects marginal gains. No replacement predictor, sign flip, penalty tuning, threshold search or case exclusions after rejection.',
        uncertainty='2000 date-grouped bootstrap resamples of error gain, seeds27110/27111, descriptive90% interval. Report concentration; neither bootstrap nor chronological fits erase serial dependence, small sample or prior research selection.',
        limitations='Even passing is information evidence only: no allocation design or portfolio eligibility. Full return/frequency/notional/DD/stress/capped/history gates remain mandatory. Native source/execution mechanics unchanged. Sparse auctions, fixed universe/taxonomy, inferred clocks, conservative proxies, five-session labels and repeatedly inspected April-July remain explicit.',
        before_joined_calibration_outcomes=True,saved_payoff_outcomes_previously_inspected=True,bindings={str(x.resolve()):sha(x) for x in paths})
    write(ROOT/'protocol.json',p);write(OUT/'protocol.json',p)
    print('Incremental-information protocol frozen before joined calibration outcomes.')


def run():
    authenticate();lanes=read(SIGNALS/'results.json')['lanes'];assert len(lanes)==332
    data=records(lanes,read(FEATURE/'ledger.json'));assert len(data)==83
    result=analyze(data);write(OUT/'records.json',data);write(OUT/'results.json',result)
    write(OUT/'selected_research_policy.json',dict(selected_id='c4f49be01ed09b8dc',promoted=False,live_activation=False,results_sha256=sha(OUT/'results.json')))
    lines=['OLR incremental sector-volume information assay',result['interpretation'],
        'Frozen comparison: breadth-only versus breadth plus volume confirmation. Training-only standardized features; same12-date-unit ridge, two chronological folds; no parameter selection.',
        'Source-qualified history was already available; this tests its incremental net information, not a new raw-feature substitution.']
    for f in result['folds']:
        lines += ['',f"{f['start']} to {f['end']}: {f['test_opportunities']}/{f['registered_opportunities']} paired supported opportunities on {f['test_dates']} dates; {f['coverage']:.2%} coverage; support {f['support_gate']}."]
        if 'relative_mse_gain' in f:
            lines += [f"Date-weighted MSE reduction: {f['relative_mse_gain']:.3%}; incremental coefficient {f['incremental_coefficient']:.4f} bps per feature unit.",
                'Gates: '+json.dumps(f['gates']),
                'Descriptive90% date-bootstrap interval for squared-error gain: '+str(f['error_gain_descriptive_90pct_interval']),
                'Top3 positive error-gain dates / all positive error-gain: '+str(f['top3_positive_error_gain_share'])]
    lines += ['', 'Incremental volume information supported: '+str(result['incremental_volume_information_supported']),
        'No candidate, allocation rule, additional trading frequency or promotion. Keep c4f49be01ed09b8dc. Rejected rank calibration and score-only cohort remain closed.',
        'Missing/unfilled/unsupported labels remain missing. If rejected, do not replace the predictor, flip its sign, tune penalties or remove known losing observations.',
        'Limits:36 sampled dates, overlapping signals and five-session horizon; selection on original confirmed incumbents; source/peer coverage restrictions; fixed universe/taxonomy; sparse auctions; inferred clocks; conservative execution proxies. Payoff outcomes and April-July previously inspected development; no independent-validation claim.']
    content='\n'.join(lines)+'\n'
    from html import escape
    for name,text in [('report.txt',content),('report.html','<!doctype html><meta charset="utf-8"><title>OLR sector information</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui;line-height:1.6}pre{white-space:pre-wrap}</style><pre>'+escape(content)+'</pre>')]:
        with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
    bundle=OUT/'review_bundle';bundle.mkdir();patch=[]
    for name in FILES:
        src=REPO/name;dest=bundle/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
        patch.extend(difflib.unified_diff([],src.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    (bundle/'implementation.patch').write_text(''.join(patch),encoding='utf-8')
    paths=[OUT/n for n in ('records.json','results.json','selected_research_policy.json','report.txt','report.html','review_bundle/implementation.patch')]+list(bundle.rglob('*.py'))
    write(OUT/'report_receipt.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'protocol.json'),files={str(p.resolve()):sha(p) for p in paths}))
    print(json.dumps({k:result[k] for k in ('support_passed','incremental_volume_information_supported','promotion')}))


def verify():
    import numpy as np
    authenticate();assert not (OUT/'delivery_checks.json').exists()
    receipt=read(OUT/'report_receipt.json');review=read(OUT/'delivery_review.json')
    assert receipt['protocol_sha256']==sha(ROOT/'protocol.json') and review['report_receipt_sha256']==sha(OUT/'report_receipt.json')
    for path,h in {**receipt['files'],**review['files']}.items():assert sha(path)==h,path
    data=records(read(SIGNALS/'results.json')['lanes'],read(FEATURE/'ledger.json'));assert data==read(OUT/'records.json')
    result=analyze(data);assert result==read(OUT/'results.json')
    checked=0
    if result['support_passed']:
        for f in result['folds']:
            _,train,test,_=split(data,f['start'],f['end']);w,_=weights(train)
            for dim,name in ((1,'baseline'),(2,'augmented')):
                x=np.array([[1.]+r['x'][:dim] for r in train]);y=np.array([r['y'] for r in train]);ww=np.array(w)
                mx=np.average(x[:,1:],axis=0,weights=ww)
                scales=np.sqrt(np.average((x[:,1:]-mx)**2,axis=0,weights=ww));scales[scales==0]=1.
                penalty=np.diag([0.]+list(PENALTY*scales**2))
                coef=np.linalg.solve(x.T@(ww[:,None]*x)+penalty,x.T@(ww*y))
                expected=[f[name]['intercept']]+f[name]['slopes']
                assert np.allclose(coef,expected,rtol=1e-10,atol=1e-9)
                for row,pred in zip(test,f['predictions']):assert math.isclose(float(coef@np.array([1.]+row['x'][:dim])),pred[name],rel_tol=1e-10,abs_tol=1e-9)
                checked+=1
    write(OUT/'delivery_checks.json',dict(utc=utc(),passed=True,independent_normal_equation_fits=checked,
        protocol_sha256=sha(ROOT/'protocol.json'),results_sha256=sha(OUT/'results.json'),report_receipt_sha256=sha(OUT/'report_receipt.json'),
        profile_sha256=sha(OUT/'selected_research_policy.json'),code_sha256=sha(__file__),promotion=False))
    print('Independent model equations, chronology, saved-data reproduction and artifact bindings passed.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','verify']);globals()[p.parse_args().phase]()
