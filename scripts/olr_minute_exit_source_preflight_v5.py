"""Read-only minute/5m source reconciliation over the entire declared universe.

No candidate outcomes, trade P&L, exclusions based on outcomes, or backtest
performance are consumed. This inventory is not a qualified execution model.
"""
from collections import Counter
from pathlib import Path
import json,sqlite3
import numpy as np
import pandas as pd
import yaml
from scripts import olr_participation_study_v5 as t
n=t.n
OUT=t.OUT/'minute_exit_source'
FIELDS=['open','high','low','close','volume']


def compare(one,native):
    """Return exact mismatches, including one-sided missing source bars."""
    five=one.resample('5min').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna()
    union=five.index.union(native.index)
    a=five.reindex(union)[FIELDS];b=native.reindex(union)[FIELDS]
    bad=((a-b).abs()>1e-7).any(axis=1)|a.isna().any(axis=1)|b.isna().any(axis=1)
    return union,bad,a,b


def main():
    if (OUT/'complete.json').exists():
        receipt=n.read(OUT/'complete.json')
        for path,digest in receipt['sources'].items():assert n.sha(path)==digest
        for path,digest in receipt['outputs'].items():assert n.sha(OUT/path)==digest
        print('Authenticated completed minute source inventory');return
    OUT.mkdir(parents=True,exist_ok=True);sources={str(Path(__file__).resolve()):n.sha(__file__)};windows={};cache={}
    for w in ('train','oos'):
        study_path=t.PARENT/'runtime'/t.REFERENCE/w/'baseline/study.json';study=n.read(study_path);files=Path(study['code'])
        sources[str(study_path)]=n.sha(study_path)
        universe=Path(study['configuration']['universe'])
        universe=next(p for p in (files/universe,files/'config'/universe) if p.is_file())
        sources[str(universe)]=n.sha(universe)
        symbols=sorted(str(x).zfill(6) for x in yaml.safe_load(universe.read_text())['symbols'])
        db=files/study['database'];assert n.sha(db)==study['database_hash'];sources[str(db)]=study['database_hash']
        conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);period=study['configuration']['date_range']
        records=[];bad_records=[];totals=Counter()
        for symbol in symbols:
            native=pd.DataFrame(conn.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day>=? AND day<=? ORDER BY ts',
                (symbol,period['start'],period['end'])).fetchall(),columns=['timestamp',*FIELDS])
            if native.empty:
                records.append(dict(symbol=symbol,native_rows=0,minute_status='no_native_rows'));continue
            native['timestamp']=pd.to_datetime(native.timestamp);native=native.set_index('timestamp')
            path=t.REPO/f'data/kis_intraday_parquet/{symbol}/{symbol}_1m_20250710_20260710.parquet'
            totals['native_bars']+=len(native)
            if not path.is_file():
                records.append(dict(symbol=symbol,native_rows=len(native),minute_status='missing_primary_archive'))
                totals['missing_archives']+=1;continue
            before=n.sha(path);sources[str(path)]=before
            one=pd.read_parquet(path).set_index('timestamp').sort_index()[FIELDS]
            assert one.index.tz is not None and not one.index.duplicated().any()
            assert np.isfinite(one.to_numpy()).all() and (one.volume>=0).all()
            assert ((one.low<=one[['open','close']].min(axis=1))&(one.high>=one[['open','close']].max(axis=1))).all()
            one=one.loc[period['start']:period['end']]
            union,bad,a,z=compare(one,native)
            details=[]
            for stamp in union[bad]:
                clean=lambda row:{k:None if pd.isna(v) else float(v) for k,v in row.items()}
                details.append(dict(symbol=symbol,timestamp=stamp.isoformat(),minute_resampled=clean(a.loc[stamp]),native=clean(z.loc[stamp])))
            bad_records.extend(details)
            days=sorted(set(native.index.strftime('%Y-%m-%d')))
            bad_days=sorted(set(union[bad].strftime('%Y-%m-%d')))
            adjacency=int(((one.index.to_series().diff()==pd.Timedelta(minutes=1)) & (one.index.to_series().dt.date==one.index.to_series().shift().dt.date)).sum())
            records.append(dict(symbol=symbol,native_rows=len(native),minute_rows=len(one),minute_status='exact' if not details else 'mismatches',
                compared_bins=len(union),mismatch_bins=len(details),native_days=days,mismatch_days=bad_days,contiguous_predictor_pairs=adjacency,
                source_path=str(path),source_sha256=before))
            totals['minute_rows']+=len(one);totals['compared_bins']+=len(union);totals['mismatch_bins']+=len(details)
            assert n.sha(path)==before,'Source changed during read'
        conn.close()
        windows[w]=dict(window=w,declared_symbols=len(symbols),period=period,counts=dict(totals),symbols=records,
            mismatch_days=sorted({r['timestamp'][:10] for r in bad_records}),mismatches=bad_records,
            interpretation='Inventory only. Do not convert archive gaps into hindsight-based trading exclusions or fabricate minute prices/volume. Resolve source coverage and causal fallback in a separately frozen execution-clock model before native performance qualification.')
        n.write(OUT/f'{w}.json',windows[w]);print(json.dumps(dict(window=w,counts=dict(totals),mismatch_days=windows[w]['mismatch_days'])),flush=True)
    n.write(OUT/'complete.json',dict(sources=sources,outputs={w+'.json':n.sha(OUT/(w+'.json')) for w in windows},
        no_pnl_consumed=True,no_performance_claim=True,no_source_files_modified=True))


if __name__=='__main__':main()
