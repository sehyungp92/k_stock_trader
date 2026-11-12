"""Inspect fixed structural half/access interaction without a financial replay."""
import argparse
from collections import defaultdict
from datetime import datetime,timedelta
import math
from pathlib import Path
import sqlite3,sys
from types import SimpleNamespace
from olr_selective_half_continuation_v1 import read,write,sha,now,status,frozen_imports
from olr_opportunity_continuity_v1 import ledger


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files']);study=ready['study']
    reference=root.parent/'selective_half_v1/protocol.json';old=read(reference);partial=Path(old['nominal_reference'])
    review=read(root/'causal_review_v1/assessment.json')['windows']['train'];extras=review['extra_traces']
    census=root.parent/'opportunity_continuity_v2/train_rows.json';allrows=read(census)
    db=files/study['database'];definition=files/study['minute_source_definition'];fine=files/read(definition)['canonical_database']
    design=Path(__file__).resolve().parents[1]/'docs/optimisation/olr/experiments/olr-exp-0039.md'
    rules=[files/'strategy_olr/core'/n for n in ['weak_partial.py','precision_v3.py','logic.py']]
    inputs={str(p):sha(p) for p in [Path(__file__),reference,partial/'completion.json',root/'assessment.json',root/'causal_review_v1/assessment.json',census,db,fine,definition,*rules]}
    write(out/'protocol.json',dict(at=now(),inputs=inputs,design_sha256=sha(design),question='Does exact selective-half management reach the new access losses, and does its existing account change extra eligibility/funding? No component PnL sum or joint replay.'))
    (out/'design_at_freeze.md').write_bytes(design.read_bytes());(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    assert sha(partial/'completion.json')==old['nominal_completion_sha256']
    checks={n:sha(partial/n)==h for n,h in read(partial/'completion.json')['artifacts'].items()};assert all(checks.values())
    source_hash,completed_case,_=frozen_imports(root);assert completed_case(partial)['valid']
    from strategy_common.market import MarketBar
    from strategy_olr.core.precision_v3 import structural_evidence
    from strategy_olr.execution import OLRExitPlan,_completed_bar_exit_reason
    assert Path(sys.modules['strategy_olr.core.precision_v3'].__file__).resolve()==rules[1].resolve()
    oldfiles=partial.parents[2]/'native_engines/train/files'
    rule_checks={p.name:sha(p)==sha(oldfiles/'strategy_olr/core'/p.name) for p in rules[:2]};assert all(rule_checks.values())
    assert sha(db)==ready['manifest']['files'][study['database']]
    assert sha(fine)==ready['manifest']['files'][read(definition)['canonical_database']]
    coarse=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);minute=sqlite3.connect(fine.as_uri()+'?mode=ro',uri=True)
    coarse.execute('PRAGMA query_only=ON');minute.execute('PRAGMA query_only=ON')
    points=set();paths=[]
    for x in extras:
        t=x['trade'];s=x['symbol'];entry=datetime.fromisoformat(t['entry_fill_time']);end=datetime.fromisoformat(t['exit_fill_time'])
        bars=[MarketBar(symbol=s,timestamp=datetime.fromisoformat(ts),open=o,high=h,low=l,close=c,volume=v,timeframe='5m',is_completed=True) for ts,o,h,l,c,v in coarse.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day>=? AND day<=? ORDER BY ts',(s,entry.date().isoformat(),end.date().isoformat()))]
        lows=[b.low for b in bars if b.timestamp.date()==entry.date()];assert lows
        finebars=list(minute.execute('SELECT ts,h FROM bars WHERE symbol=? AND day>=? AND day<=? ORDER BY ts',(s,entry.date().isoformat(),end.date().isoformat())))
        finebars=[(datetime.fromisoformat(ts),h) for ts,h in finebars]
        firstfill=next(e for o in x['orders'] if o['side']=='BUY' for e in o['events'] if e['kind']=='fill')
        risk=x['decision']['proposal']['metadata']['risk_per_share'];plan=OLRExitPlan(**t['route_metadata']['trade_exit_plan'])
        rows=[]
        for b in bars:
            at=b.timestamp+timedelta(minutes=5)
            if b.timestamp.date()<=entry.date() or at>end:continue
            high=max([firstfill['price']]+[h for ts,h in finebars if entry<=ts and ts+timedelta(minutes=1)<=at]+[z.high for z in bars if z.timestamp>=entry and z.timestamp+timedelta(minutes=5)<=at])
            pos=SimpleNamespace(entry_time=entry,entry_price=t['entry_price'],max_favorable_price=high,metadata={'structural_entry_day_low':min(lows)})
            session=SimpleNamespace(session_bars=[z for z in bars if z.timestamp.date()==b.timestamp.date() and z.timestamp<=b.timestamp])
            evidence=structural_evidence(session,pos,b,risk)
            held=[z for z in bars if entry<z.timestamp<=b.timestamp]
            full=_completed_bar_exit_reason(held,len(held)-1,t['entry_price'],risk,high,plan,0)['reason']
            rows.append(dict(at=at.isoformat(),close=b.close,high_water=high,mfe_r=(high-t['entry_price'])/risk,structural=evidence,full_priority_reason=full))
            points.add(at)
        paths.append(dict(symbol=s,day=x['day'],trade=t,orders=x['orders'],entry_day_low=min(lows),rows=rows))
    outside=[r for r in allrows if r['disposition']=='outside_slots']
    points.update(datetime.fromisoformat(r['timeline'][-1]['at']) for r in outside)
    status(out,'reconstructing_access_and_existing_partial_books')
    case=Path(read(root/'assessment.json')['summaries']['train_access']['case'])
    book=ledger(case,sorted(points));pb=ledger(partial,sorted(points))
    results=[]
    for p in paths:
        for z in p['rows']:
            b=book['snapshots'][z['at']];q=b['quantities'].get(p['symbol'],0);pending=[o for o in b['pending'] if o['symbol']==p['symbol']]
            z.update(held=q,pending=pending,actionable=bool(z['structural'] and not z['full_priority_reason'] and q>=2 and not pending))
        first=next((z for z in p['rows'] if z['actionable']),None)
        if first:
            legs=[e for o in p['orders'] if o['side']=='SELL' for e in o['events'] if e['kind']=='fill' and e['at']>first['at']]
            assert sum(e['fill_qty'] for e in legs)==first['held']
            proceeds=sum(e['fill_qty']*e['price']-e['fee'] for e in legs);sell=first['held']//2
            first['half_shares']=sell;first['hold_minus_half_mark_sale_proxy']=sell*(proceeds/first['held']-first['close']*(1-.0027))
        results.append({**p,'first_actionable':first})
    byday=defaultdict(list)
    for r in allrows:byday[r['day']].append(r)
    snapshots={s['trade_date']:s for s in read(files/study['olr_snapshots'])};funding=[]
    for day in sorted({r['day'] for r in outside}):
        group=byday[day];original=[r for r in group if r['rank']<=6];at=original[0]['timeline'][-1]['at'];b=pb['snapshots'][at]
        held=lambda s:bool(b['quantities'].get(s) or any(o['symbol']==s and o['side']=='BUY' for o in b['pending']))
        expired=[r['symbol'] for r in original if r['timeline'][-1]['source_complete'] and not r['first_confirmation'] and not held(r['symbol'])]
        extras2=sorted((r for r in group if r['rank']>6 and r['timeline'][-1]['source_complete'] and r['timeline'][-1]['passes'] and not held(r['symbol'])),key=lambda r:r['rank'])
        admitted={r['symbol'] for r in extras2[:len(expired)]}
        for r in group:
            if r['disposition']!='outside_slots':continue
            c=next(x for x in snapshots[day]['candidates'] if x['symbol']==r['symbol']);z=r['timeline'][-1]
            price=z['close']*1.015;unit=price*1.0007;risk=max(c['daily_atr'],z['close']*.005,1);loss=risk*1.25+price*.0034
            weight=1.2*r['rank']**-1.5/sum(x['rank']**-1.5 for x in original)
            limits=dict(weight=math.floor(b['equity']*weight/unit),cash=math.floor(max(0,b['deployable'])/unit),gross=math.floor(max(0,b['equity']-b['gross']-b['reserve'])/unit),name=math.floor(max(0,.04*b['equity']-b['risk'].get(r['symbol'],0))/loss),sector=math.floor(max(0,.06*b['equity']-b['sector_risk'].get(c['sector'],0))/loss),account=math.floor(max(0,.08*b['equity']-sum(b['risk'].values()))/loss))
            q=min(limits.values()) if r['symbol'] in admitted else 0
            funding.append(dict(symbol=r['symbol'],day=day,admitted=r['symbol'] in admitted,held=held(r['symbol']),expired=expired,book=b,limits=limits,quantity_upper_bound=q,notional_upper_bound=q*price))
    write(out/'assessment.json',dict(at=now(),partial_artifacts=checks,exact_rule_bytes=rule_checks,extra_lifecycles=results,partial_funding=funding,
        scope='Existing parent path, exact frozen structural predicate, native-priority and minute high-water checks. No alternative fills, dynamic R change, combined policy or funded return. No0032 worker launched.',new_financial_accounts=0,economic_objective_achieved=False))
    coarse.close();minute.close()
    for p,h in inputs.items():assert sha(Path(p))==h
    status(out,'complete_awaiting_disposition')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.root,a.out)
