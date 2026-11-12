"""Independently audit the completed-clock PnL partition from raw bars/fills."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bind(p):
    p=Path(p).resolve()
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':h}
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def near(x,y):
    if abs(x-y)>max(.01,abs(y)*1e-10):raise ValueError('Independent forward mismatch')


def run(root):
    import psutil
    identity=read(root/'supervisor_start.json')
    try:
        q=psutil.Process(identity['pid'])
        if q.create_time()==identity['process_created'] and q.cmdline()==identity['command']:raise ValueError('Producer active')
    except psutil.NoSuchProcess:pass
    done=read(root/'completion.json');p=read(root/'protocol.json')
    bindings=[p['program']]+p['bindings']+done['outputs']+[done['assessment']]
    for b in bindings:
        if bind(b['path'])['sha256']!=b['sha256']:raise ValueError('Changed binding '+b['path'])
    write(root/'review_registration_v1.json',{'program':bind(__file__),'inputs':bindings,'read_only':True,
        'at_utc':datetime.now(timezone.utc).isoformat()})
    out=root/'review_v1';out.mkdir(exist_ok=False);me=psutil.Process()
    write(out/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    parent=read(root.parent/'protocol.json');dbs=[sqlite3.connect('file:'+Path(d).as_posix()+'?mode=ro',uri=True) for d in parent['databases']]
    checks=[];count=0;cache={}
    try:
        for case in parent['cases']:
            original=read(root.parent/'accounts'/Path(case['key']).with_suffix('.json'))
            needed={r['position_id']:r for r in original['rows'] if r['position_id']}
            result=read(root/'accounts'/Path(case['key']).with_suffix('.json'))
            assert len(result['rows'])==len(needed) and set(r['position_id'] for r in result['rows'])==set(needed)
            opened={};quantity=defaultdict(int);allcash=defaultdict(float);priorcash=defaultdict(float);priorqty=defaultdict(int);priorfees=defaultdict(float)
            for f in read(Path(case['folder'])/'fills.json'):
                s=f['symbol'];t=datetime.fromisoformat(f['timestamp']);sign=1 if f['side']=='BUY' else -1
                if sign>0 and not quantity[s]:opened[s]=s+'/'+t.isoformat()
                key=opened[s];delta=-sign*f['qty']*f['price']-f['metadata']['fee']
                if key in needed:
                    allcash[key]+=delta
                    if t<datetime.fromisoformat(needed[key]['completion_at']):
                        priorcash[key]+=delta;priorqty[key]+=sign*f['qty'];priorfees[key]+=f['metadata']['fee']
                quantity[s]+=sign*f['qty']
                if not quantity[s]:del opened[s]
            for row in result['rows']:
                key=row['position_id'];originalrow=needed[key];pair=row['symbol'],row['day']
                if pair not in cache:
                    raw=[]
                    for c in dbs:raw.extend(c.execute('SELECT ts,c FROM bars WHERE symbol=? AND day=?',pair))
                    raw=[x for x in raw if datetime.fromisoformat(x[0]).hour==9 and datetime.fromisoformat(x[0]).minute==30]
                    if len(raw)!=1:raise ValueError('Missing/duplicate source close')
                    cache[pair]=raw[0][1]
                marked=priorcash[key]+priorqty[key]*cache[pair]
                near(allcash[key],originalrow['parent_net_pnl_krw'])
                near(marked,row['observed_net_marked_before35_krw']);near(allcash[key]-marked,row['observed_remaining_net_after35_krw'])
                near(priorfees[key],row['paid_fee_before_decision_krw']);assert priorqty[key]==row['quantity_at_decision']
                count+=1
            for category,g in result['groups'].items():
                rs=[r for r in result['rows'] if r['category']==category]
                near(sum(r['observed_remaining_net_after35_krw'] for r in rs),g['after35_remaining_krw'])
                near(sum(r['observed_net_marked_before35_krw'] for r in rs),g['before35_marked_krw'])
            checks.append({'account':case['key'],'unique_positions':len(result['rows']),'passed':True})
            print(json.dumps(checks[-1]),flush=True)
    finally:
        for c in dbs:c.close()
    review={'passed':True,'positions':count,'accounts':checks,'scope':'Independent rawclose and physical cashflow time partition. Observed remaining native movement is not attainable newentry PnL. No native rerun, fitting or policy qualification.','study_complete':False}
    write(out/'review.json',review);write(out/'completion.json',{'complete':True,'passed':True,'review':bind(out/'review.json'),'study_complete':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();run(a.root)
