"""Finest retained observation inventory and strict five-minute compatibility census."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,sqlite3
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')
OUT=ROOT/'minute_source_audit'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str)
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['none']);w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=False);wanted=defaultdict(set);period_days={};sources={}
    for p in ('is','oos'):
        period_days[p]={r['day'] for r in read(SOURCE/'cases'/f'combined_{p}_baseline/daily.json')}
        for s in ('baseline','participation','cost','combined'):
            for t in read(SOURCE/'cases'/f'combined_{p}_{s}/trades.json'):wanted[t['symbol']].add(t['entry_fill_time'][:10])
    for sym in wanted:
        files=list((REPO/'data/kis_intraday_parquet'/sym).glob(sym+'_1m_*.parquet'))
        if files:sources[sym]=max(files,key=lambda p:(p.stem.split('_')[-1],p.stem.split('_')[-2]))
    write(OUT/'registration.json',dict(program={'path':__file__,'sha256':sha(Path(__file__))},
        design={'path':'docs/optimisation/kalcb/experiments/kalcb-exp-0041-minute-binding.md','sha256':sha(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-minute-binding.md')},
        sources={k:{'path':str(p),'sha256':sha(p)} for k,p in sources.items()},missing_instruments=sorted(set(wanted)-set(sources))))
    conn={p:sqlite3.connect('file:'+Path(read(SOURCE/'runtime'/f'{p}_cap2/plan.json')['database']).as_posix()+'?mode=ro',uri=True) for p in ('is','oos')}
    days=[];bad=[];counts=defaultdict(int)
    for sym,dates in sorted(wanted.items()):
        if sym not in sources:continue
        df=pd.read_parquet(sources[sym]);df['day']=df['timestamp'].dt.strftime('%Y-%m-%d');df=df[df['day'].isin(dates)].copy()
        df['bin']=df['timestamp'].dt.floor('5min');df['minute']=df['timestamp'].dt.minute
        ag=df.groupby(['day','bin']).agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum'),minutes=('minute','nunique'),rows=('minute','size')).reset_index()
        ag['ts']=ag['bin'].apply(lambda x:x.isoformat());lookup={(r.day,r.ts):r for r in ag.itertuples()}
        for day in sorted(dates):
            p=next(p for p in period_days if day in period_days[p]);native=conn[p].execute('SELECT ts,o,h,l,c,v FROM bars WHERE day=? AND symbol=? ORDER BY ts',(day,sym)).fetchall()
            c=defaultdict(int)
            for ts,o,h,l,cl,v in native:
                ts=pd.Timestamp(ts).isoformat();row=lookup.get((day,ts));c['native_bins']+=1
                if row is None:status='missing'
                elif row.minutes!=5 or row.rows!=5:status='incomplete_or_duplicate'
                elif any(abs(float(getattr(row,k))-float(x))>1e-7 for k,x in zip(('o','h','l','c','v'),(o,h,l,cl,v))):status='mismatch'
                else:status='exact'
                c[status]+=1;counts[status]+=1
                if status!='exact':bad.append(dict(period=p,day=day,symbol=sym,timestamp=ts,status=status,native_open=o,native_high=h,native_low=l,native_close=cl,native_volume=v,
                    minute_open=row.o if row else None,minute_high=row.h if row else None,minute_low=row.l if row else None,minute_close=row.c if row else None,minute_volume=row.v if row else None,
                    retained_file=str(sources[sym])))
            days.append(dict(period=p,day=day,symbol=sym,**{k:c[k] for k in ('native_bins','exact','missing','incomplete_or_duplicate','mismatch')},fully_matched=bool(c['native_bins'] and c['exact']==c['native_bins'])))
    for c in conn.values():c.close()
    csvwrite(OUT/'issuer_days.csv',days);csvwrite(OUT/'disagreements.csv',bad)
    write(OUT/'completion.json',dict(complete=True,instruments=len(sources),issuer_days=len(days),counts=dict(counts),fully_matched_issuer_days=sum(d['fully_matched'] for d in days),
        full_finer_account_qualified=not bad and len(sources)==len(wanted),scope='Read-only retained raw minute versus accepted corrected five-minute OHLCV; no source replacement or financial replay.'))
    print(dict(counts),flush=True)


if __name__=='__main__':main()
