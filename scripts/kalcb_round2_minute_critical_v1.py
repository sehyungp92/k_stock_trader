"""Inspect source age at already observed finer-MTM peak/trough, no new execution."""
import csv
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read,write,bound,SOURCE,SCENARIOS

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')


def main():
    reg=read(ROOT/'minute_source_audit/registration.json');cache={};allrows=[]
    for p in ('is','oos'):
        for sc in SCENARIOS:
            peak=1e8;worst=0;peakrow=None;chosen=None
            for index,r in enumerate(csv.DictReader((ROOT/'minute_marks'/f'{p}_{sc}_marks.csv').open()),2):
                nav=float(r['nav']);r={**r,'csv_row':index}
                if nav>=peak:peak=nav;peakrow=r
                dd=100*(1-nav/peak)
                if dd>worst:worst=dd;chosen=(peakrow,r)
            fills=read(SOURCE/'cases'/f'combined_{p}_{sc}'/'fills.json')
            for kind,r in zip(('peak','trough'),chosen):
                at=datetime.fromisoformat(r['at']);after=r['phase']=='open_after_fill';q=Counter();cash=1e8
                for f in fills:
                    ft=datetime.fromisoformat(f['timestamp'])
                    if ft>at or ft==at and not after:continue
                    sign=1 if f['side']=='BUY' else -1
                    q[f['symbol']]+=sign*f['qty'];cash-=sign*f['qty']*f['price']+f['metadata']['fee']
                source=[];nav=cash
                for sym,n in q.items():
                    if not n:continue
                    if sym not in cache:cache[sym]=pd.read_parquet(reg['sources'][sym]['path'])
                    df=cache[sym];past=df[df.timestamp<=at];candidates=[]
                    for row in past.tail(3).itertuples():
                        t=row.timestamp.to_pydatetime()
                        if t<at or t==at and after:candidates.append((t,1,float(row.open),'open',row.Index))
                        if t+timedelta(minutes=1)<=at:candidates.append((t+timedelta(minutes=1),0,float(row.close),'close',row.Index))
                    observed,order,price,field,idx=max(candidates)
                    age=(at-observed).total_seconds()/60;nav+=n*price
                    source.append(dict(symbol=sym,qty=n,price=price,available_at=str(observed),age_minutes=age,field=field,
                        parquet_index=int(idx),source=reg['sources'][sym]))
                assert abs(nav-float(r['nav']))<.02,(p,sc,kind,nav,r)
                allrows.append(dict(period=p,scenario=sc,kind=kind,at=r['at'],phase=r['phase'],minute_marks_csv_row=r['csv_row'],
                    nav=nav,maximum_held_price_age_minutes=max([s['age_minutes'] for s in source],default=0),source_marks=source))
    write(ROOT/'minute_marks/critical_mark_review.json',dict(program=bound(__file__),passed=True,points=allrows))
    print([(r['period'],r['scenario'],r['kind'],r['maximum_held_price_age_minutes']) for r in allrows])


if __name__=='__main__':main()
