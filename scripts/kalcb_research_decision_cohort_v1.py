"""C18 complete declared opening/checkpoint coverage, no outcomes or replay."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,gzip,hashlib,time

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OLD=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(p),sha256=sha(p))
def rows(p):
    with Path(p).open() as f:
        for n,l in enumerate(f,1):
            if l.strip():yield n,json.loads(l)
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)

def main():
    start=time.monotonic();out=ROOT/'cohorts_v1';out.mkdir(exist_ok=False)
    controls=read(OLD/'round_run_v2/evaluations/v6_h0.seal.json')['receipts']
    parents=read(ROOT/'opening_parent_authentication_v1.json');inputs=[];sets={}
    for period,parent in zip(['is','oos'],parents):
        control=next(r for r in controls if r['item']['period']==period and r['process']['scenario']=='baseline')
        frozen=Path(control['item']['manifest']['path']).parent/'files';pol=Path(control['item']['policy']['path']).parent
        lineage=read(pol/'lineage.json')['label_manifests_and_receipts']
        original={}
        for clock in ['opening','timed']:
            matches=[p for p in lineage if 'original_target_full_history' in p and Path(p).name==clock+'.jsonl']
            assert len(matches)==1
            p=Path(matches[0]);assert sha(p)==lineage[matches[0]];original[clock]=bound(p);inputs.append(original[clock])
        op=Path(parent['opportunities.jsonl']['path']);assert sha(op)==parent['opportunities.jsonl']['sha256']
        cp=frozen/'research/kalcb_executed_value/prepared/opportunities.jsonl'
        fm=read(control['item']['manifest']['path']);rel=cp.relative_to(frozen).as_posix();assert sha(cp)==fm['files'][rel]
        inputs += [bound(op),bound(cp),bound(pol/'interaction_cost_margin.admission.jsonl'),bound(pol/'timed.admission.jsonl')]
        days=[x['day'] for x in read(Path(control['folder'])/'daily.json')]
        universe=read(pol/'policies_manifest.json')['observation_symbols'];assert len(universe)==103
        sets[period]=dict(control=control,pol=pol,days=days,universe=universe,opening=str(op),checkpoint=str(cp),forecasts=original)
    save(out/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),program=bound(__file__),
        design=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043.md'),inputs=inputs,
        cohort='Full declared103universe ×225sessions × opening/checkpoint clocks; no outcome membership filters',
        forbidden_causal_fields=['quality_target'],no_labels_or_account_evaluations=True,
        forecast_source='Frozen original expanding original_target_full_history; missing downstream forecasts remain unknown',
        no_stopped_operations=True))
    summaries=[]
    with gzip.open(out/'episodes.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as stream:
        for period,info in sets.items():
            days=set(info['days']);counts=Counter();admissions={}
            for route,file in [('opening','interaction_cost_margin.admission.jsonl'),('checkpoint','timed.admission.jsonl')]:
                index={}
                for n,r in rows(info['pol']/file):
                    symbol=r.get('symbol') or r['opportunity_id'].split('/')[1]
                    index[r['trade_date'],symbol]=(n,r)
                admissions[route]=index
            for route,forecast in [('opening','opening'),('checkpoint','timed')]:
                originals={};predictions=defaultdict(list)
                source=Path(info[route])
                for n,row in rows(source):
                    if row['trade_date'] not in days:continue
                    key=(row['trade_date'],row['symbol']);assert key not in originals
                    originals[key]=(n,row)
                for n,row in rows(info['forecasts'][forecast]['path']):
                    key=row['opportunity_id'].split('/')[:2]
                    if key[0] not in days:continue
                    assert row['valid_from']<=key[0] and key[0]<=row.get('valid_through',key[0])
                    predictions[tuple(key)].append(dict(source=info['forecasts'][forecast]['path'],source_row=n,record=row))
                for day in info['days']:
                    for symbol in info['universe']:
                        key=(day,symbol);counts[route+':declared']+=1
                        rec=dict(period=period,trade_date=day,symbol=symbol,cohort=route,issuer_day=day+'/'+symbol,
                            episode_id=period+'/'+day+'/'+symbol+'/'+route,source=str(source),source_row=None,
                            feature_available_at=None,status='source_context_unavailable',original_guard_at_rank1=None,
                            candidate=None,features=None,historical_action_preview=None,
                            original_forecasts=predictions.get(key,[]),original_admission=None)
                        if key in originals:
                            n,row=originals[key];rec.update(source_row=n,feature_available_at=row['feature_available_at'],
                                status='source_context_available',original_guard_at_rank1=row.get('original_guard_at_rank1',row.get('guard_at_rank1')),
                                candidate=row['candidate'],features=row['features'],historical_action_preview=row.get('action'))
                            assert 'quality_target' not in rec['features']
                            counts[route+':context_available']+=1
                        if key in admissions[route]:
                            n,row=admissions[route][key];rec['original_admission']=dict(source=str(info['pol']/('interaction_cost_margin.admission.jsonl' if route=='opening' else 'timed.admission.jsonl')),source_row=n,record=row)
                            counts[route+':admitted']+=1
                        if not rec['original_forecasts']:counts[route+':forecast_unavailable']+=1
                        # Absence from an admitted-only file does not identify which gate rejected it.
                        rec['admission_disposition']='admitted' if rec['original_admission'] else 'not_admitted_exact_downstream_reason_unavailable'
                        stream.write(json.dumps(rec,allow_nan=False,default=str)+'\n')
            summaries.append(dict(period=period,sessions=len(days),symbols=len(info['universe']),counts=counts))
    save(out/'completion.json',dict(passed=True,seconds=time.monotonic()-start,cohorts=summaries,episodes=bound(out/'episodes.jsonl.gz'),
        native_event_census=bound(ROOT/'saved_evidence_v1/entry_events.csv'),labels_generated=0,
        limitations=['Historical action previews have older source/rank1 semantics; not current native action certificates',
        'Forecast missingness and short-circuited rejection reasons preserved; no inferred passed gates',
        'Later native repeated events remain in separately bound complete entry_events census; not independent entries']))
    print(json.dumps(summaries))

if __name__=='__main__':main()
