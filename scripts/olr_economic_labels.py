"""Build matured entry and remaining-native-value labels from audited lanes.

No performance selection or hyperparameter search. Holding targets estimate
remaining native value, not a causal estimate of an immediate market exit.
Actual exit interventions are tested in the funded native experiment.
"""
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
import argparse,bisect,gzip,hashlib,json,sqlite3,sys


def build(root,repo):
    sys.path.insert(0,str(repo))
    from backtests.strategies.olr.economic_policy import features,OFFSETS
    parent=root.parent/'olr_joint_decision_20261013'
    ready=json.loads((parent.parent/'olr_round1_optimization_20261010/structural_v6/drawdown_reserve_v1/native_engines/train/ready.json').read_text())
    files=Path(ready['files']);study=ready['study']
    snapshots=json.loads((files/study['olr_snapshots']).read_text())
    candidates={(s['trade_date'],c['symbol']):c for s in snapshots for c in s['candidates']}
    db=files/study['database'];con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    cache={}
    def bars(symbol,day,at):
        key=(symbol,day)
        if key not in cache:
            cache[key]=[dict(timestamp=datetime.fromisoformat(t),open=o,high=h,low=l,close=c,volume=v)
                for t,o,h,l,c,v in con.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',key)]
        return [b for b in cache[key] if b['timestamp']+timedelta(minutes=5)<=at]
    rows=[];missing=Counter();seen=set()
    with gzip.open(parent/'verified_response/paths.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['status'] not in ('matured','unfilled') or not r['execution_valid']:
                missing[r['status']]+=1;continue
            candidate=candidates.get((r['day'],r['symbol']))
            if candidate is None:missing['candidate']+=1;continue
            at=datetime.fromisoformat(r['native_signal_time']);prefix=bars(r['symbol'],r['day'],at)
            if not prefix:missing['prefix']+=1;continue
            path=r['path'];points=sorted(path,key=lambda p:(p[6],p[1]))
            clocks=[datetime.fromisoformat(p[6]) for p in points]
            def point(t):
                i=bisect.bisect_right(clocks,t)-1
                return points[max(0,i)]
            known=datetime.fromisoformat(r['known_at'])
            native=json.loads((Path(r['lane'])/'result.json').read_text())
            orders=native.get('native_requests',[])
            reserved=(orders[0]['metadata']['reserved_notional'] if orders else r['budget'])
            # Final flat cash may be carried forward, never an open/censored mark.
            def future(t,base,den):
                return [(float(point(t+timedelta(minutes=m))[3])-base)/den for m in OFFSETS[1:]]
            key=f"{r['day']}:{r['symbol']}:{r['scenario']}:{r['budget']}"
            assert key not in seen;seen.add(key)
            rows.append(dict(id=key,day=r['day'],decision_at=at.isoformat(),kind='entry',scenario=r['scenario'],known_at=r['known_at'],
                x=features(candidate,prefix,r['requested_quantity']),
                y=future(at,0,reserved)+[float(r['actual_acquired']>=1_000_000)],
                source=r['receipt_sha256']))
            if not r['filled_quantity']:continue
            # Three predeclared opening checkpoints per lifecycle, not selected
            # because the observed trade was losing or at its hindsight peak.
            next_days=sorted({t.date() for t in clocks if t.date()>at.date()})
            if not next_days:continue
            for hh,mm in ((9,5),(9,30),(10,0)):
                t=at.replace(year=next_days[0].year,month=next_days[0].month,day=next_days[0].day,hour=hh,minute=mm,second=0)
                available=[p for p,c in zip(points,clocks) if c<t or c==t and p[2]=='closing_marks']
                if not available:continue
                p=available[-1];qty=p[5]
                if qty<=0 or t>=known:continue
                prefix=bars(r['symbol'],t.date().isoformat(),t)
                if not prefix:continue
                close=prefix[-1]['close'];den=qty*close
                # Only use completed acquisition cost when every acquisition
                # fragment is already in the decision prefix. A future fill
                # invalidates this row; its later average is never a predictor.
                buys=[];previous=0
                for pp,clock in zip(points,clocks):
                    if pp[2]=='fill' and pp[5]>previous:buys.append(clock)
                    previous=pp[5]
                if not buys or max(buys)>=t:continue
                entry=r['actual_acquired']/r['filled_quantity']
                ep=bars(r['symbol'],at.date().isoformat(),t)
                peak=max([entry]+[b['high'] for b in ep if b['timestamp']>=at]+[b['high'] for b in prefix])
                position=dict(entry_price=entry,entry_time=min(buys).isoformat(),max_favorable_price=peak)
                rows.append(dict(id=key+f':{hh}:{mm}',day=r['day'],decision_at=t.isoformat(),kind='hold',scenario=r['scenario'],known_at=r['known_at'],
                    x=features(candidate,prefix,qty,position=position,at=t),y=future(t,float(p[3]),den),source=r['receipt_sha256']))
    root.mkdir(parents=True,exist_ok=True)
    target=root/'labels.json'
    with target.open('x',encoding='utf-8') as f:json.dump(rows,f,separators=(',',':'))
    receipt=dict(rows=len(rows),kinds=dict(Counter(r['kind'] for r in rows)),missing=dict(missing),
        labels_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        estimand='Future native policy net marked PNL; holding intervention evaluated by funded replay, not assumed execution at current mark.',
        source=hashlib.sha256((parent/'verified_response/paths.jsonl.gz').read_bytes()).hexdigest())
    with (root/'labels_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    build(a.root,Path(__file__).resolve().parents[1])
