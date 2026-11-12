"""Build a frozen, all-universe observation panel; run inside its source generation."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.market import MarketBar
from strategy_common.sector_map import load_canonical_sector_map
from strategy_olr.core.capacity import phase
from strategy_olr.core.opening_tail import empty,advance


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build(study_path,output):
    study=json.loads(Path(study_path).read_text(encoding='utf-8'))
    files=Path.cwd();manifest=json.loads(Path(study['manifest']).read_text(encoding='utf-8'))
    def authenticated(name):
        p=files/name
        assert sha(p)==manifest['files'][name],name
        return p
    db=authenticated(study['database']);assert sha(db)==study['database_hash']
    uname=study['configuration']['universe']
    if uname not in manifest['files']:uname='config/'+uname
    universe=sorted(str(x).zfill(6) for x in yaml.safe_load(authenticated(uname).read_text())['symbols'])
    assert len(universe)==len(set(universe))==103
    sector_name='config/olr/sector_map.yaml';authenticated(sector_name)
    sectors=load_canonical_sector_map(default_path=sector_name)
    assert all(x in sectors for x in universe)
    snapshots=json.loads(authenticated(study['olr_snapshots']).read_text(encoding='utf-8'))
    assert all(c['sector']==sectors[c['symbol']] for snap in snapshots for c in snap['candidates'])
    ex=set()
    if study.get('corporate_actions'):
        ex={(e['symbol'],e['ex_date']) for e in json.loads(authenticated(study['corporate_actions']).read_text())['events']}
    con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    lo,hi=con.execute('SELECT min(day),max(day) FROM bars').fetchone()
    cal=get_trading_calendar();days=[];d=date.fromisoformat(lo)
    while d<=date.fromisoformat(hi):
        if cal.is_trading_day(d):days.append(d.isoformat())
        d+=timedelta(days=1)
    closes={};opens={};excluded=Counter();source_rows=0
    for symbol in universe:
        previous=None;current_day=None;found=False
        for stamp,o,h,l,c,v in con.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? ORDER BY ts',(symbol,)):
            source_rows+=1;day=stamp[:10]
            if day!=current_day:previous=None;found=False;current_day=day
            # Only first eligible continuous open and the official match are needed.
            if found and not o==h==l==c:continue
            at=datetime.fromisoformat(stamp)
            if found and at.time().replace(tzinfo=None)!=cal.closing_match_time(symbol,at.date()):continue
            bar=MarketBar(symbol,at,'5m',o,h,l,c,v,source_fingerprint=study['database_hash'])
            kind=phase(bar)
            if kind=='auction' and v>0:closes[(symbol,day)]=(c,stamp)
            if not found:
                # At least one share at both registered participation scenarios.
                if (kind=='continuous' and previous is not None and previous[1]=='continuous'
                        and previous[0].timestamp+timedelta(minutes=5)==at and previous[0].volume*.005>=1):
                    opens[(symbol,day)]=(o,stamp,previous[0].timestamp.isoformat(),previous[0].volume)
                    found=True
                previous=(bar,kind)
    con.close();rows=[]
    for day in days:
        prev=cal.previous_trading_day(date.fromisoformat(day)).isoformat()
        for symbol in universe:
            if (symbol,day) in ex:excluded['declared_corporate_action']+=1;continue
            close=closes.get((symbol,prev));opening=opens.get((symbol,day))
            if close is None:excluded['missing_adjacent_official_close']+=1;continue
            if opening is None:excluded['missing_eligible_continuous_open']+=1;continue
            loss=max(0.,1-opening[0]/close[0])
            if loss>.31:raise ValueError('Unexplained large price discontinuity: '+str((symbol,day,loss)))
            rows.append({'symbol':symbol,'sector':sectors[symbol],'day':day,'previous_day':prev,
                'previous_close':close[0],'previous_close_at':close[1],'eligible_open':opening[0],
                'loss_fraction':loss,'available_at':opening[1],
                'capacity_bar':opening[2],'capacity_volume':opening[3]})
    # Every prefix must match incremental replay ingestion, regardless of future rows.
    grouped=defaultdict(list)
    for row in rows:grouped[row['day']].append(row)
    state=empty()
    for day in days:
        state=advance(state,grouped[day],day,[day])
        if day==days[-1] or day=='2026-03-31':
            direct=advance(empty(),[r for r in rows if r['day']<=day],day,[d for d in days if d<=day])
            assert state==direct
    coverage=Counter(r['day'] for r in rows)
    result={'version':'opening-tail-panel-v2','database_hash':study['database_hash'],
      'builder_hash':sha(__file__),'universe_sha256':sha(files/uname),'sector_map_sha256':sha(files/sector_name),
      'universe':universe,'sessions':days,'observations':rows,
      'coverage':dict(sorted(coverage.items())),'exclusions':dict(excluded),'source_rows_inspected':source_rows,
      'prefix_invariance_passed':True,'minimum_previous_volume':200,
      'limitations':['Frozen inherited sector classification is not a point-in-time classification archive.',
        'Opening-price observation is not a full-size execution forecast or a guaranteed loss bound.',
        'Only declared corporate-action discontinuities can be excluded with the retained evidence.']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('version','exclusions','source_rows_inspected')}),flush=True)
    print({'rows':len(rows),'days':len(days),'through':days[-1],'sha256':sha(output)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.study,a.output)
