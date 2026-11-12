"""Bounded0039 timing/coverage diagnosis; marks are not trading-policy PnL."""
import argparse
from collections import defaultdict
from datetime import datetime
import math
from pathlib import Path
import sqlite3
from olr_selective_half_continuation_v1 import read,write,sha,now,status


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    design=Path(__file__).resolve().parents[1]/'docs/optimisation/olr/experiments/olr-exp-0039.md'
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files'])
    census=root.parent/'opportunity_continuity_v2/train_rows.json'
    db=files/ready['study']['database'];snap=files/ready['study']['olr_snapshots']
    inputs={str(p):sha(p) for p in [Path(__file__),census,db,snap,root/'dependency_review_v1/assessment.json']}
    assert inputs[str(db)]==ready['manifest']['files'][ready['study']['database']]
    write(out/'protocol.json',dict(at=now(),inputs=inputs,design_sha256=sha(design),question='Earlier native confirmation lost by expiry: opportunity quality, reach, and original priority conflict. Existing marks only; no alternative trading return.'))
    (out/'design_at_freeze.md').write_bytes(design.read_bytes());(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    snapshots={r['trade_date']:r for r in read(snap)};allrows=read(census);days=sorted(snapshots)
    groups=defaultdict(list)
    for r in allrows:groups[r['day']].append(r)
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
    rows=[]
    for r in allrows:
        if r['disposition']!='outside_slots':continue
        first=next((x for x in r['timeline'] if x['passes']),None);final=r['timeline'][-1]
        result=dict(day=r['day'],symbol=r['symbol'],rank=r['rank'],first_confirmation=r['first_confirmation'],final_passes=final['passes'])
        if first is None:
            result['status']='no_native_confirmation';rows.append(result);continue
        c=next(c for c in snapshots[r['day']]['candidates'] if c['symbol']==r['symbol']);b=first['book']
        held=bool(b['quantities'].get(r['symbol']) or any(o['symbol']==r['symbol'] and o['side']=='BUY' for o in b['pending']))
        originals=[x for x in groups[r['day']] if x['rank']<=6]
        price=first['close']*1.015;unit=price*1.0007;risk=max(c['daily_atr'],first['close']*.005,1.)
        loss=risk*1.25+price*.0034;weight=1.2*r['rank']**-1.5/sum(x['rank']**-1.5 for x in originals)
        limits=dict(weight=math.floor(b['equity']*weight/unit),cash=math.floor(max(0,b['deployable'])/unit),
            gross=math.floor(max(0,b['equity']-b['gross']-b['reserve'])/unit),
            name=math.floor(max(0,.04*b['equity']-b['risk'].get(r['symbol'],0))/loss),
            sector=math.floor(max(0,.06*b['equity']-b['sector_risk'].get(c['sector'],0))/loss),
            account=math.floor(max(0,.08*b['equity']-sum(b['risk'].values()))/loss))
        q=min(limits.values()) if not held else 0
        later=[dict(symbol=o['symbol'],at=p['at'],approved=p['decision']['approved_qty']) for o in originals for p in o['proposals'] if p['at']>first['at']]
        futureday=next(d for d in days if d>r['day'])
        bars=list(con.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',(r['symbol'],futureday)))
        assert bars and all(datetime.fromisoformat(z[0]).date().isoformat()==futureday for z in bars)
        op=bars[0][1];cl=bars[-1][4];hi=max(z[2] for z in bars);lo=min(z[3] for z in bars)
        result.update(status='held_or_pending' if held else 'first_native_confirmation',first_close=first['close'],final_close=final['close'],
            original_r=risk,book=b,limits=limits,quantity_upper_bound=q,notional_upper_bound=q*price,
            later_original_proposals=later,next_day=futureday,next_bar_count=len(bars),next_first_label=bars[0][0],next_last_label=bars[-1][0],
            next_open_mark=op,next_close_mark=cl,next_high_mark=hi,next_low_mark=lo,
            first_to_next_open_bps=1e4*(op/first['close']-1),first_to_next_close_bps=1e4*(cl/first['close']-1),
            final_to_next_close_bps=1e4*(cl/final['close']-1),next_high_from_first_r=(hi-first['close'])/risk,next_low_from_first_r=(lo-first['close'])/risk,
            scope='Parent funding upper bound excludes fresh capacity, competing extras and displaced future original requests. Future OHLC marks omit fills/costs/management; no alternative-policy return.')
        rows.append(result)
    assert len(rows)==24
    con.close()
    write(out/'assessment.json',dict(at=now(),rows=rows,earlier_confirmed_final_invalid=[r for r in rows if r['first_confirmation'] and not r['final_passes']],
        new_financial_accounts=0,economic_objective_achieved=False))
    for p,h in inputs.items():assert sha(Path(p))==h
    status(out,'complete_awaiting_disposition')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.root,a.out)
