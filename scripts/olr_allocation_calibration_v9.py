"""One preregistered chronological allocation-information check; never a policy."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import math
import random

REPO = Path(__file__).resolve().parents[1]
PARENT = REPO/'data/backtests/analysis/olr_structural_20261012/fixed_signal_v1'
OUT = PARENT.parent/'allocation_calibration_v1'
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/structural_v6/allocation_calibration_v1')
FILES = ['scripts/olr_allocation_calibration_v9.py', 'tests/strategy_olr/test_allocation_calibration_v9.py']
BLOCKS = [('2025-08-07','2025-10-31'),('2025-11-01','2025-12-31'),('2026-01-01','2026-03-31')]
RIDGE_DATE_UNITS = 12.


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def paired_records(lanes):
    groups=defaultdict(dict)
    for row in lanes:
        key=(row['scenario'],row['budget']);group=groups[row['day'],row['symbol']]
        if key in group:raise ValueError('Duplicate lane')
        group[key]=row
    needed={(s,b) for s in ('baseline','combined') for b in (1_000_000,20_000_000)}
    records=[]
    for (day,symbol),group in sorted(groups.items()):
        if set(group)!=needed:raise ValueError('Missing registered lane')
        ranks={r['rank'] for r in group.values()}
        if len(ranks)!=1 or not 1<=next(iter(ranks))<=6:raise ValueError('Rank mismatch')
        mature=all(r['status']=='matured' for r in group.values())
        small,big=group['combined',1_000_000],group['combined',20_000_000]
        # Net value per requested incremental capital includes quantity-dependent
        # fill opportunity; actual acquired notional is not the decision input.
        records.append(dict(day=day,symbol=symbol,rank=next(iter(ranks)),
            x=1-math.log(next(iter(ranks)))/math.log(6),
            y=10000*(big['net_pnl']-small['net_pnl'])/19_000_000 if mature else None,
            available=max(r['available_for_training_after'] for r in group.values()) if mature else None))
    return records


def date_means(rows, field):
    grouped=defaultdict(list)
    for row in rows:grouped[row['day']].append(row[field])
    return {day:math.fsum(values)/len(values) for day,values in sorted(grouped.items())}


def fit(rows, penalty=RIDGE_DATE_UNITS):
    counts=defaultdict(int)
    for row in rows:counts[row['day']]+=1
    if len(counts)<10:raise ValueError('Insufficient matured training dates')
    total=len(counts)
    mx=math.fsum(r['x']/counts[r['day']] for r in rows)/total
    my=math.fsum(r['y']/counts[r['day']] for r in rows)/total
    variance=math.fsum((r['x']-mx)**2/counts[r['day']] for r in rows)
    covariance=math.fsum((r['x']-mx)*(r['y']-my)/counts[r['day']] for r in rows)
    slope=covariance/(variance+penalty)
    return dict(intercept=my-slope*mx,slope=slope,pooled_prediction=my,
        training_dates=total,training_opportunities=len(rows),ridge_date_units=penalty,
        training_keys=[[r['day'],r['symbol']] for r in rows])


def confidence(values, seed):
    rng=random.Random(seed);n=len(values)
    means=sorted(math.fsum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return dict(date_resampling_repetitions=2000,seed=seed,
        descriptive_90pct_interval=[means[99],means[1899]],
        caveat='Date grouping retains same-date dependence; serial dependence and research selection remain. Not independent validation.')


def analyze(lanes):
    records=paired_records(lanes);folds=[]
    for i,(lo,hi) in enumerate(BLOCKS[1:],1):
        current=[r for r in records if lo<=r['day']<=hi]
        if not current:raise ValueError('Missing chronological block')
        cutoff=min(r['day'] for r in current)+'T00:00:00+09:00'
        train=[r for r in records if r['day']<lo and r['available'] and r['available']<cutoff]
        test=[r for r in current if r['available'] and r['available'][:10]<=hi]
        model=fit(train)
        predictions=[dict(day=r['day'],symbol=r['symbol'],rank=r['rank'],actual=r['y'],
            model=model['intercept']+model['slope']*r['x'],pooled=model['pooled_prediction']) for r in test]
        for p in predictions:
            p['model_squared_error']=(p['model']-p['actual'])**2
            p['pooled_squared_error']=(p['pooled']-p['actual'])**2
        daily=date_means(predictions,'actual')
        mse=lambda field: math.fsum(date_means(predictions,field).values())/len(daily)
        mm,pm=mse('model_squared_error'),mse('pooled_squared_error')
        mean=math.fsum(daily.values())/len(daily)
        positives=sorted((v for v in daily.values() if v>0),reverse=True)
        gates=dict(paired_support=len(daily)>=10 and len(test)>=.8*len(current),
            incremental_net_value_positive=mean>0,rank_information_improves_pooled=mm<pm,
            higher_rank_coefficient_nonnegative=model['slope']>=0)
        folds.append(dict(start=lo,end=hi,fit_cutoff=cutoff,model=model,
            test_opportunities=len(test),registered_opportunities=len(current),test_dates=len(daily),
            paired_coverage=len(test)/len(current),date_weighted_incremental_net_bps=mean,
            model_date_weighted_mse=mm,pooled_date_weighted_mse=pm,
            relative_mse_improvement=1-mm/pm if pm else None,
            positive_date_top3_share=math.fsum(positives[:3])/math.fsum(positives) if positives else None,
            uncertainty=confidence(list(daily.values()),19073+i),gates=gates,predictions=predictions))
    return dict(folds=folds,calibrated_rank_allocation_design_eligible=all(all(f['gates'].values()) for f in folds),
        candidate_policy_created=False,portfolio_experiment_eligible=False,promotion=False,
        new_native_replays=0,independent_validation=False,
        interpretation='Tests incremental stressed net value per extra KRW19m requested. It neither estimates a self-funding portfolio nor adopts removal of forecast request limits. Fixed-signal outcomes informed this hypothesis; chronological splits are development checks, not untouched validation.')


def authenticate():
    protocol=read(ROOT/'protocol.json')
    for path,digest in protocol['bindings'].items():assert sha(path)==digest,path
    assert read(PARENT/'delivery_checks.json')['passed']
    assert read(PARENT/'results.json')['summary']['separate_allocation_calibration_may_be_considered']
    return protocol


def freeze():
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
    count=sum(int(t.get('tests',0)) for t in suites)
    assert count>=6 and not any(int(t.get(k,0)) for t in suites for k in ('failures','errors','skipped'))
    assert sha(PARENT/'delivery_checks.json')=='aa65c05d0da19ec688a97000294ad6b5709b0de230b47a76c8cd245dee05ef8c'
    assert read(PARENT/'delivery_checks.json')['results_sha256']==sha(PARENT/'results.json')
    assert read(PARENT/'results.json')['summary']['separate_allocation_calibration_may_be_considered']
    paths=[REPO/n for n in FILES]+[ROOT/'tests.xml',ROOT/'tests.log']
    paths += [PARENT/n for n in ('results.json','delivery_checks.json','report_receipt.json','selected_research_policy.json')]
    protocol=dict(utc=utc(),version='rank-incremental-value-calibration-v1',
        motivation='Three positive stressed fixed-signal blocks justify testing allocation information; quantity effects vary by block and winners are concentrated. Do not infer rank exclusions from inspected subgroup averages.',
        input_scope='Only already authenticated 83 incumbent opportunities/332 lanes; excluded score-map cohort stays excluded; no new native replay or OOS data.',
        model='One weighted ridge regression: incremental stressed net PNL / KRW19m requested increment in bps, on intercept plus 1-log(rank)/log(6). Intercept unpenalized; slope penalty 12 date units (one diagnostic block). Equal total weight per entry date; no tuning.',
        fits='Exactly two expanding chronological fits: Aug-Oct -> Nov-Dec; Aug-Dec -> Jan-Mar. Training uses only four-lane paired labels matured before the first test date. Fit once per block. Test maturity is capped at that block end.',
        comparator='Training-date-weighted pooled constant predicts the identical target. This tests whether original ordinal rank adds incremental sizing information, not whether rank selection has alpha.',
        gates='Both folds: >=10 paired mature evaluation dates and >=80% coverage, positive date-weighted incremental stressed net value, strictly lower date-weighted MSE than pooled predictor, nonnegative higher-rank coefficient. Failure rejects this single model; no replacement, exponent search or gate relaxation.',
        uncertainty='2000 date bootstrap resamples, seed19074/19075, descriptive 90% interval; report concentration. Not a promotion or significance claim; serial dependence and development selection remain.',
        no_policy='No trade exclusion, allocation rule, portfolio promotion or live change. Even passing only permits separate bounded allocation design; every original portfolio gate remains mandatory, including meaningful frequency and both DD grids.',
        sample_outcomes_already_inspected=True,before_model_calibration_outcomes=True,
        tests_passed=count,bindings={str(p.resolve()):sha(p) for p in paths})
    write(ROOT/'protocol.json',protocol);write(OUT/'protocol.json',protocol)
    print('Separate calibration protocol frozen; no model outcomes computed.')


def run():
    protocol=authenticate();result=analyze(read(PARENT/'results.json')['lanes'])
    write(OUT/'results.json',result)
    write(OUT/'selected_research_policy.json',dict(selected_id='c4f49be01ed09b8dc',promoted=False,live_activation=False,results_sha256=sha(OUT/'results.json')))
    lines=['OLR incremental allocation-information calibration',result['interpretation'],
        'One fixed model, two chronological fits. No policy change or new trading-frequency claim.',
        'The model predicts extra stressed net PNL per KRW19m requested, not return on actual fills or account return.']
    for f in result['folds']:
        lines += ['',f"{f['start']} to {f['end']}: {f['test_opportunities']} paired opportunities / {f['test_dates']} dates; coverage {f['paired_coverage']:.2%}.",
            f"Incremental stressed net value: {f['date_weighted_incremental_net_bps']:.3f} bps; descriptive 90% interval {f['uncertainty']['descriptive_90pct_interval']}.",
            f"Prediction MSE reduction versus pooled training mean: {f['relative_mse_improvement']:.2%}; rank slope {f['model']['slope']:.3f}.",
            'Gates: '+json.dumps(f['gates']),f"Three strongest positive dates / all positive-date value: {f['positive_date_top3_share']:.2%}."]
    lines += ['', 'Separate calibrated-rank allocation design eligible: '+str(result['calibrated_rank_allocation_design_eligible']),
        'No candidate policy created. Keep c4f49be01ed09b8dc. Do not retune a rejected model or remove observed losers.',
        'Limitations: only 36 sampled dates; overlapping signals; fixed universe/taxonomy; sparse auction observations; inferred archive clocks and conservative execution proxies; five-session label horizon; outcomes and April-July already inspected development. No independent-validation claim.']
    text='\n'.join(lines)+'\n';OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'report.txt').open('x',encoding='utf-8') as stream:stream.write(text)
    from html import escape
    with (OUT/'report.html').open('x',encoding='utf-8') as stream:stream.write('<!doctype html><meta charset="utf-8"><title>OLR allocation calibration</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui;line-height:1.6}pre{white-space:pre-wrap}</style><pre>'+escape(text)+'</pre>')
    bundle=OUT/'review_bundle';bundle.mkdir();patch=[]
    for name in FILES:
        source=REPO/name;target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        patch.extend(difflib.unified_diff([],source.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    (bundle/'implementation.patch').write_text(''.join(patch),encoding='utf-8')
    paths=[OUT/n for n in ('results.json','report.txt','report.html','selected_research_policy.json','review_bundle/implementation.patch')]
    write(OUT/'report_receipt.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'protocol.json'),files={str(p.resolve()):sha(p) for p in paths}))
    print(json.dumps({'calibrated_rank_allocation_design_eligible':result['calibrated_rank_allocation_design_eligible']}))


def verify():
    authenticate();assert not (OUT/'delivery_checks.json').exists()
    receipt=read(OUT/'report_receipt.json');review=read(OUT/'delivery_review.json')
    assert receipt['protocol_sha256']==sha(ROOT/'protocol.json')
    assert review['report_receipt_sha256']==sha(OUT/'report_receipt.json')
    for path,digest in {**receipt['files'],**review['files']}.items():assert sha(path)==digest,path
    assert read(OUT/'results.json')==analyze(read(PARENT/'results.json')['lanes'])
    write(OUT/'delivery_checks.json',dict(utc=utc(),passed=True,protocol_sha256=sha(ROOT/'protocol.json'),
        report_receipt_sha256=sha(OUT/'report_receipt.json'),results_sha256=sha(OUT/'results.json'),
        profile_sha256=sha(OUT/'selected_research_policy.json'),code_sha256=sha(__file__),promotion=False))
    print('Calibration reproduction and artifact bindings verified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run','verify'])
    globals()[parser.parse_args().phase]()
