"""Source-qualified all-universe entry-to-liquidation paths; isolated generation."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import date,datetime,timedelta
import hashlib,json
from pathlib import Path
import sqlite3,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.market import MarketBar
from strategy_common.sector_map import load_canonical_sector_map
from strategy_common.execution_session import session_cutoff
from strategy_olr.execution import DECISION_CUTOFF
from strategy_olr.core.capacity import phase
from strategy_olr.core.liquidation_tail_v3 import empty,advance

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(study_path,output):
    study=json.loads(Path(study_path).read_text(encoding='utf-8'));files=Path.cwd()
    manifest=json.loads(Path(study['manifest']).read_text(encoding='utf-8'))
    def auth(name):
        p=files/name;assert sha(p)==manifest['files'][name],name;return p
    db=auth(study['database']);assert sha(db)==study['database_hash']
    uname=study['configuration']['universe']
    if uname not in manifest['files']:uname='config/'+uname
    universe=sorted(str(x).zfill(6) for x in yaml.safe_load(auth(uname).read_text(encoding='utf-8'))['symbols'])
    assert len(universe)==len(set(universe))==103
    sector_name='config/olr/sector_map.yaml';auth(sector_name);sectors=load_canonical_sector_map(default_path=sector_name)
    assert all(x in sectors for x in universe)
    ex={(r['symbol'],r['ex_date']) for r in json.loads(auth(study['corporate_actions']).read_text(encoding='utf-8'))['events']} if study.get('corporate_actions') else set()
    con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    lo,hi=con.execute('select min(day),max(day) from bars').fetchone()
    cal=get_trading_calendar();days=[];day=date.fromisoformat(lo)
    while day<=date.fromisoformat(hi):
        if cal.is_trading_day(day):days.append(day.isoformat())
        day+=timedelta(days=1)
    rows=[];excluded=Counter();inspected=0
    for symbol in universe:
        refs={};paths=defaultdict(list);previous=None;current_day=None
        for stamp,o,h,l,c,v in con.execute('select ts,o,h,l,c,v from bars where symbol=? order by ts',(symbol,)):
            inspected+=1;at=datetime.fromisoformat(stamp);d=stamp[:10]
            if current_day!=d:previous=None;current_day=d
            bar=MarketBar(symbol,at,'5m',o,h,l,c,v,source_fingerprint=study['database_hash']);kind=phase(bar)
            if kind=='continuous' and (at+timedelta(minutes=5)).time().replace(tzinfo=None)<=session_cutoff(at.date(),DECISION_CUTOFF):
                refs[d]=(c,stamp)
            if kind=='continuous' and previous and previous[1]=='continuous' and previous[0].timestamp+timedelta(minutes=5)==at:
                paths[d].append([stamp,o,previous[0].volume,previous[0].timestamp.isoformat()])
            previous=bar,kind
        for d in days:
            prev=cal.previous_trading_day(date.fromisoformat(d)).isoformat()
            if (symbol,d) in ex:excluded['corporate_action']+=1;continue
            ref=refs.get(prev)
            if ref is None:excluded['missing_adjacent_decision_reference']+=1;continue
            # Require the exact final predecision bar, not a stale earlier quote.
            rt=datetime.fromisoformat(ref[1]);cutoff=session_cutoff(rt.date(),DECISION_CUTOFF)
            if (rt+timedelta(minutes=5)).time().replace(tzinfo=None)!=cutoff:
                excluded['incomplete_decision_prefix']+=1;continue
            closing=cal.session_hours(date.fromisoformat(d))[1]
            available=datetime.combine(date.fromisoformat(d),closing,tzinfo=rt.tzinfo)+timedelta(minutes=5)
            rows.append({'symbol':symbol,'sector':sectors[symbol],'day':d,'previous_day':prev,
                'reference_price':ref[0],'reference_at':ref[1],'available_at':available.isoformat(),
                'liquidation':paths[d]})
            if not paths[d]:excluded['included_as_total_liquidity_impairment']+=1
    con.close();rows.sort(key=lambda r:(r['day'],r['symbol']));grouped=defaultdict(list)
    for row in rows:grouped[row['day']].append(row)
    state=empty()
    for day in days:
        state=advance(state,grouped[day],day,[day])
        if day==days[-1] or day=='2026-03-31':
            assert state==advance(empty(),[r for r in rows if r['day']<=day],day,[d for d in days if d<=day])
    result={'version':'liquidation-panel-v3','database_hash':study['database_hash'],'builder_sha256':sha(__file__),
      'universe':universe,'universe_sha256':sha(files/uname),'sector_map_sha256':sha(files/sector_name),
      'sessions':days,'observations':rows,'source_rows_inspected':inspected,'exclusions':dict(excluded),
      'prefix_invariance_passed':True,'coverage':dict(sorted(Counter(r['day'] for r in rows).items()))}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
    print({'rows':len(rows),'sessions':len(days),'exclusions':dict(excluded),'sha256':sha(output)},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.study,a.output)
