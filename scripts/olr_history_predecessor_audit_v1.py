"""Recover already-tested broad-history contrasts, preserving original scope."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from olr_predecessor_audit_v1 import BASE, read, sha

OUT=Path(__file__).resolve().parents[1]/'docs/optimisation/olr/evidence/earlier-history-predecessors.json'
inputs={str(Path(__file__)):sha(Path(__file__))};rows=[];profiles={}
for generation,recipes in [(7,['median','broad','broad_entry','volume_sqrt','volume_linear']),
                            (11,['median','broad','recycle_top2','broad_recycle_top2'])]:
    root=BASE/f'generation_{generation}'
    for name in ['registration.json','screen_verdict.json']:inputs[str(root/name)]=sha(root/name)
    registration=read(root/'registration.json');profiles[str(generation)]=registration['profiles']
    assert read(root/'screen_verdict.json')['complete'] and not read(root/'screen_verdict.json')['eligible']
    for recipe in recipes:
        for scenario in ['baseline','combined']:
            case=root/'cases'/f'{recipe}_100000000_{scenario}'
            c,r,t=[read(case/name) for name in ['completion.json','result.json','trades.json']]
            assert c['complete'] and sha(case/'result.json')==c['result_sha256']
            assert not r['terminal_inventory'] and r['terminal_pending']==0
            net=sum(x['net_pnl'] for x in t)
            assert math.isclose(net,r['return_pct']*1e6,abs_tol=.01) and len(t)==r['trades']
            for name in ['completion.json','result.json','trades.json']:inputs[str(case/name)]=sha(case/name)
            rows.append(dict(generation=generation,recipe=recipe,scenario=scenario,return_pct=r['return_pct'],
                drawdown_pct=r['drawdown_pct'],trades=r['trades'],net_pnl=net,audit_status_reported=r['audit_status']))
assert all(sha(Path(p))==h for p,h in inputs.items())
with OUT.open('x',encoding='utf8') as f:
    json.dump(dict(at=datetime.now(timezone.utc).isoformat(),inputs=inputs,profiles=profiles,rows=rows,
        new_replays=0,limits=['Saved-result/ledger authentication; old audit_status remains unverified.',
            'Older execution/population, not N0; no pooling or fresh physical qualification.']),f,indent=2)
print(json.dumps(dict(path=str(OUT),accounts=len(rows))))
