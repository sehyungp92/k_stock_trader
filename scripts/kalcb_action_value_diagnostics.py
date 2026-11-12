"""Authenticate opening previews against saved full execution episodes."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def verify_previews(prepared, labels):
    provenance=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'opportunities.jsonl')!=provenance['opportunities_sha256']:
        raise ValueError('Changed previews')
    expected={}
    with (prepared/'opportunities.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            expected[r['opportunity_id']]=(r['feature_available_at'],r['action']['ready'],r['action']['quantity'],r['action']['routes'])
    counts=Counter();mismatches=[];seen=set();sources={}
    for folder in labels:
        manifest=json.loads((folder/'labels_manifest.json').read_text())
        for name,digest in manifest['event_files'].items():
            path=folder/name
            if ev.digest(path)!=digest:raise ValueError('Changed saved execution evidence')
            sources[str(path)]=digest
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for line in stream:
                    detail=json.loads(line);key=detail['opportunity_id']
                    if key not in expected:continue
                    if key in seen:raise ValueError('Duplicate execution episode')
                    seen.add(key);cutoff,ready,qty,routes=expected[key]
                    orders=[o['order'] for o in detail['orders'] if o['event']=='submit' and
                        o['order']['side']=='BUY' and o['order'].get('metadata',{}).get('decision_available_at','9999')<=cutoff]
                    observed=(bool(orders),sum(o['qty'] for o in orders),[o['metadata'].get('entry_route') for o in orders])
                    counts['checked']+=1
                    counts['opening_ready']+=ready
                    if observed!=(ready,qty,routes):
                        mismatches.append({'opportunity_id':key,'preview':[ready,qty,routes],'execution':observed})
    if seen!=set(expected):raise ValueError('Missing saved episodes for preview comparison')
    return {'counts':counts,'exact_parity':not mismatches,'mismatches':mismatches,'sources':sources,
        'comparison':'opening intent existence, requested quantity and route; no future fill-based readiness'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--labels',required=True,type=Path,action='append')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=verify_previews(a.prepared,a.labels)
    ev.write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
    if not result['exact_parity']:raise ValueError('Opening preview/ordinary execution mismatch')


if __name__=='__main__':main()
