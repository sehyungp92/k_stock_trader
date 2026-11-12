"""Verify a completed immutable response census; reconstruct all cash paths.

Usage: python -B scripts/olr_joint_response_census.py --root <generation>
No fitting, return selection, gate relaxation, or portfolio claims occur here.
"""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,gzip,hashlib,json,sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.olr.decision_response import reconstruct,paired_cells


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)


def collect(root):
    p=read(root/'protocol.json');out=root/'verified_response'
    if out.exists():raise FileExistsError('Do not overwrite verified evidence')
    proofs={};lanes=[];completions=[]
    for shard in range(p['shards']):
        recovered=root/'recovered'/str(shard)
        folder=recovered if (recovered/'labels_complete.json').exists() else root/'shards'/str(shard)
        done=read(folder/'labels_complete.json')
        if not done['complete'] or sha(folder/'source_proofs.json')!=done['source_proofs_sha256']:raise ValueError('Incomplete or changed source generation')
        if sha(folder/'frozen_harness.json')!=done['frozen_harness_sha256']:raise ValueError('Changed native harness')
        source=read(folder/'source_proofs.json')
        if not source['origins_independently_reparsed']:raise ValueError('Unverified source origins')
        for key,value in source['sessions'].items():
            if key in proofs and proofs[key]!=value:raise ValueError('Different source proof for one session')
            proofs[key]=value
        lanes.extend(done['lanes']);completions.append(dict(path=str(folder/'labels_complete.json'),sha256=sha(folder/'labels_complete.json')))
    # Old controls' exit sessions were sourced by their original worker.
    old=Path(p['frozen_parent_design']['source_ready']).parent
    oldroot=root.parent/'olr_round1_optimization_20261010/structural_v6/fixed_signal_v1'
    original=read(oldroot/'labels_complete.json')
    if sha(oldroot/'source_proofs.json')!=original['source_proofs_sha256']:raise ValueError('Changed original source proofs')
    for key,value in read(oldroot/'source_proofs.json')['sessions'].items():
        if key in proofs and proofs[key]!=value:raise ValueError('Changed old/new source proof')
        proofs[key]=value
    expected={(r['day'],r['symbol'],s,b) for r in p['cohort'] for s in p['scenarios'] for b in p['budgets']}
    if len(lanes)!=len(expected):raise ValueError('Incomplete response census')
    cohort={(r['day'],r['symbol']):r for r in p['cohort']};records=[];seen=set();invalid=[];bindings={}
    # All source and completed receipt identities checked before creating output.
    for item in lanes:
        lane=Path(item['path']);receipt=read(lane/'receipt.json')
        if sha(lane/'receipt.json')!=item['receipt_sha256']:raise ValueError('Receipt mismatch')
        if receipt['result_sha256']!=item['result_sha256']:raise ValueError('Result receipt mismatch')
        for name,key in [('result.json','result_sha256'),('execution_audit.json','audit_sha256'),('journal.jsonl.gz','journal_sha256')]:
            if sha(lane/name)!=receipt[key]:raise ValueError('Lane changed: '+str(lane/name))
        r=read(lane/'result.json');audit=read(lane/'execution_audit.json')
        key=(r['day'],r['symbol'],r['scenario'],r['budget'])
        if key not in expected or key in seen:raise ValueError('Unexpected/duplicate response')
        seen.add(key)
        if signature({k:proofs[k] for k in r['source_sessions']})!=r['source_proof_hash']:raise ValueError('Response source proof mismatch')
        valid=audit['execution_validity_pass']
        if valid!=r['execution_arithmetic_pass'] or (not valid and r['status']!='invalid_execution'):
            raise ValueError('Invalid execution qualification')
        with gzip.open(lane/'journal.jsonl.gz','rt',encoding='utf-8') as f:z=reconstruct(r,(json.loads(x) for x in f))
        z.update({k:r[k] for k in ('day','symbol','rank','scenario','budget','native_signal_time','requested_quantity','filled_quantity')})
        z.update(incumbent_disposition=cohort[key[:2]]['incumbent_disposition'],receipt_sha256=sha(lane/'receipt.json'),
            lane=str(lane),execution_valid=valid,source_proof_hash=r['source_proof_hash'])
        records.append(z);bindings[str(lane/'receipt.json')]=z['receipt_sha256']
        if not valid:
            invalid.append(dict(key=list(key),errors=audit['audit_receipt']['errors'],
                impossible_volume_events=audit.get('capacity_model_audit',{}).get('impossible_volume_events',[])))
        if len(records)%200==0:print(json.dumps(dict(verified=len(records),total=len(expected))),flush=True)
    if seen!=expected:raise ValueError('Missing response keys')
    out.mkdir()
    with gzip.open(out/'paths.jsonl.gz','xt',encoding='utf-8') as f:
        for r in sorted(records,key=lambda r:(r['day'],r['symbol'],r['scenario'],r['budget'])):
            f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    rows=[{k:v for k,v in r.items() if k not in ('path','path_columns')} for r in records]
    write(out/'records.json',rows)
    cells=[]
    for s in p['scenarios']:
        for b in p['budgets']:
            subset=[r for r in rows if r['scenario']==s and r['budget']==b]
            cells.append(dict(scenario=s,budget=b,status_counts=dict(Counter(r['status'] for r in subset)),
                cash_labels=sum(r['policy_cash_pnl'] is not None for r in subset),dates=len({r['day'] for r in subset if r['known_at']})))
    blocks=p['blocks']
    if blocks and isinstance(blocks[0],dict):blocks=[(b['start'],b['end']) for b in blocks]
    summary=dict(census_lanes=len(rows),opportunities=len(cohort),entry_dates=len({r['day'] for r in rows}),
        disposition_counts=dict(Counter(r['incumbent_disposition'] for r in p['cohort'])),cells=cells,
        complete_paired_blocks=paired_cells(rows,blocks,p['budgets'],p['scenarios']),invalid_execution=invalid,
        objective_achieved=False,portfolio_replays=0,oos_read=False,selected=p['selected'],promotion=False,
        interpretation='Native isolated quantity responses, with costs, maturity, source proofs and full marked paths. These are not funded portfolio returns. Invalid execution, unresolved orders/inventory and missing sources remain untrainable.',
        next_decision_requirements=['Causal forecasts for both new actions and remaining held/pending exposures on the same dates/time axes',
            'Out-of-time calibration, paired support and marginal economic value at actual request quantities',
            'Funded native portfolio comparison with meaningful completed entries, returns and both drawdown grids',
            'Unchanged full promotion gates; honest coverage accounting for invalid/censored responses'])
    write(out/'summary.json',summary)
    receipt=dict(utc=datetime.now(timezone.utc).isoformat(),complete=True,protocol_sha256=sha(root/'protocol.json'),
        completions=completions,bindings=bindings,files={x.name:sha(x) for x in out.iterdir() if x.is_file()},
        collector_sha256=sha(Path(__file__)),reconstruction_sha256=sha(REPO/'backtests/strategies/olr/decision_response.py'))
    write(out/'receipt.json',receipt)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);collect(parser.parse_args().root)
