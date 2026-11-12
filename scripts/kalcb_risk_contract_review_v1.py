"""Independent raw-bar/fill reconstruction of V46; no application imports."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sqlite3

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(b):
    if sha(b['path'])!=b['sha256']:raise ValueError('Changed '+b['path'])
def dt(x):return datetime.fromisoformat(x)
def near(a,b):
    if abs(a-b)>max(.01,abs(b)*1e-10):raise ValueError(f'Reconstruction mismatch {a} {b}')

def run(root):
    import psutil
    identity=read(root/'supervisor_start.json')
    try:
        p=psutil.Process(identity['pid'])
        if abs(p.create_time()-identity['process_created'])<.01 and p.cmdline()==identity['command']:
            raise ValueError('Exact producer still running')
    except psutil.NoSuchProcess:pass
    done=read(root/'completion.json');verify(done['assessment'])
    protocol=read(root/'protocol.json')
    for b in protocol['programs']+protocol['bindings']+done['outputs']:verify(b)
    out=root/'review_v1';out.mkdir(exist_ok=False)
    me=psutil.Process();(out/'supervisor_start.json').write_text(json.dumps({'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()}),encoding='utf-8')
    connections=[sqlite3.connect('file:'+Path(p).as_posix()+'?mode=ro',uri=True) for p in protocol['databases']]
    checks=[];marks=0
    for case in protocol['cases']:
        folder=Path(case['folder']);result=read(root/'accounts'/Path(case['key']).with_suffix('.json'))
        obs=read(folder/'observer_evidence.json');exp={e['order_id']:e for e in obs['entry_exposures']}
        active={};grouped={}
        for f in read(folder/'fills.json'):
            symbol=f['symbol']
            if f['side']=='BUY' and symbol not in active:
                key=symbol+'/'+dt(f['timestamp']).isoformat();active[symbol]=[key,0];grouped[key]=[]
            key,qty=active[symbol];grouped[key].append(f)
            qty+=f['qty']*(1 if f['side']=='BUY' else -1)
            assert qty>=0
            if qty:active[symbol][1]=qty
            else:del active[symbol]
        assert not active
        n=0;aggregate=0.;after=0.;winners=0
        latency=read(Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v29_exit_latency_audit/accounts')/Path(case['key']).with_suffix('.json'))
        exits={r['position_id']:r.get('first_full_exit',{}).get('available_at') for r in latency['positions']}
        for r in result['rows']:
            n+=1;fs=grouped[r['position_id']];e=exp[r['entry_order_ids'][0]];m=e['action']['metadata']
            gross=sum((1 if f['side']=='SELL' else -1)*f['qty']*f['price'] for f in fs)
            fees=sum(f['metadata']['fee'] for f in fs);near(gross-fees,r['net_pnl_krw']);aggregate+=gross-fees
            risk=m['risk_per_share'];near(risk*e['action']['qty'],r['planned_reference_risk_krw'])
            bars=[]
            for c in connections:
                bars.extend(c.execute('SELECT ts,c FROM bars WHERE symbol=? AND day=? ORDER BY ts',(r['symbol'],r['entry_day'])).fetchall())
            first=None;max_r=0.;worst=0.;count=0;any_breach=False
            for ts,price in sorted(bars):
                at=dt(ts)
                if not dt(r['entry_fill_time'])<=at<dt(r['exit_fill_time']):continue
                past=[f for f in fs if dt(f['timestamp'])<=at]
                held=sum(f['qty']*(1 if f['side']=='BUY' else -1) for f in past)
                if not held:continue
                bought=sum(f['qty'] for f in past if f['side']=='BUY')
                cash=sum(f['qty']*f['price']*(1 if f['side']=='SELL' else -1)-f['metadata']['fee'] for f in past)
                net=cash+held*price;worst=min(worst,net);max_r=max(max_r,-net/(bought*risk));count+=1
                available=at+timedelta(minutes=5)
                if price<=m['stop_price']:
                    any_breach=True
                    exit_at=exits[r['position_id']]
                    if first is None and (exit_at is None or available<dt(exit_at)):
                        first=(available,net,held)
            assert count==r['raw_marks'] and any_breach==r['reference_stop_breach'];marks+=count
            near(max_r,r['max_marked_net_loss_reference_r'])
            # Worst may be positive for uninterrupted winners; compare separately.
            if worst<0:near(worst,r['worst_marked_net_pnl_krw'])
            if first:
                point=r['before_parent_exit_breach'];assert dt(point['available_at'])==first[0] and point['held_qty']==first[2]
                near(point['net_marked_pnl_krw'],first[1]);near(r['net_pnl_krw']-first[1],r['observed_after_breach_net_movement_krw'])
                after+=r['net_pnl_krw']-first[1];winners+=r['net_pnl_krw']>0
            else:assert r['before_parent_exit_breach'] is None
        near(aggregate,result['summary']['net_krw']);near(after,result['summary']['after_breach_observed_net_movement_krw'])
        assert winners==result['summary']['breach_eventual_winners']
        checks.append({'key':case['key'],'unique_opening_lifetimes':n,'passed':True})
        print(json.dumps(checks[-1]),flush=True)
    for c in connections:c.close()
    report={'passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'program':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
            'accounts':checks,'raw_position_marks':marks,'scope':'Independent position cash/fees/raw-close/boundary reconstruction. Historical whole-source qualification reused, no counterfactual economic claim.','study_complete':False}
    (out/'review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'completion.json').write_text(json.dumps({'complete':True,'passed':True,'review_sha256':sha(out/'review.json'),'study_complete':False}),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
