"""Extract only existing decision-time scores; never carry old outcomes forward."""
from pathlib import Path
import json,gzip,hashlib,time

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OLD=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2/native_v5/runtime')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    start=time.monotonic();out=ROOT/'scores_v1';out.mkdir(exist_ok=False)
    policy=OLD/'is/inputs/files/research/kalcb_executed_value/policies'
    lineage=read(policy/'lineage.json')['label_manifests_and_receipts']
    source=next(Path(p) for p in lineage if p.endswith('decision_audit.jsonl.gz'))
    assert sha(source)==lineage[str(source)]
    with (out/'registration.json').open('x') as f:json.dump(dict(program=dict(path=__file__,sha256=sha(__file__)),
        source=dict(path=str(source),sha256=lineage[str(source)]),fields=['action_id','available_at','calibrated_core','calibrated_support','capacity_pass','day','failure_probability','family','physical_risk_pass','quantity','raw','symbol','valid_from'],
        explicit_exclusions=['physical_outcome','selected_management_labels'],scope='Decision-time provenance only; no new fits or labels'),f,indent=2)
    index={}
    with gzip.open(source,'rt') as stream,gzip.open(out/'decision_scores.jsonl.gz','xt',compresslevel=1) as target:
        fields=read(out/'registration.json')['fields']
        for n,line in enumerate(stream,1):
            r=json.loads(line)
            if r['family']!='expanding':continue
            assert r['valid_from']<=r['day']
            key=r['action_id'];row={k:r[k] for k in fields};row.update(source_row=n)
            assert key not in index;index[key]=row;target.write(json.dumps(row)+'\n')
    bridge=[]
    for period in ['is','oos']:
        p=OLD/period/'inputs/files/research/kalcb_executed_value/policies/interaction_cost_margin.admission.jsonl'
        for n,line in enumerate(p.open(),1):
            r=json.loads(line);x=index[r['action_id']]
            origin='calibrated_core' if r['forecast_origin']=='frozen_core' else 'calibrated_support'
            predicted=x[origin]['baseline'];err=abs(r['score']-predicted)
            bridge.append(dict(period=period,source_row=n,action_id=r['action_id'],origin=origin,error=err))
    passed=all(x['error']<1e-8 for x in bridge)
    with (out/'completion.json').open('x') as f:json.dump(dict(passed=passed,seconds=time.monotonic()-start,
        rows=len(index),admission_bridge=bridge,max_error=max(x['error'] for x in bridge),
        causal_scores=dict(path=str(out/'decision_scores.jsonl.gz'),sha256=sha(out/'decision_scores.jsonl.gz'))),f,indent=2)
    print('Score bridge',passed,len(bridge),'max_error',max(x['error'] for x in bridge))

if __name__=='__main__':main()
