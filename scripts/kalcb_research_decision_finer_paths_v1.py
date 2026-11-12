"""Source-bound descriptive minute paths; never a financial execution simulator."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timedelta,timezone
import argparse,json,gzip,hashlib,time,os
import pandas as pd
import pyarrow
import psutil
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path(__file__).resolve().parents[1]
OLD=ROOT.parent.parent/'kalcb_round3_20261017/attempt_2/round_run_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def verify(b):assert bound(b['path'])['sha256']==b['sha256']
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for n,line in enumerate(f,1):yield n,json.loads(line)
def emit(f,x):f.write(json.dumps(x,default=str,allow_nan=False)+'\n')
def calculate(fills,frame):
    if any(f['metadata'].get('execution_phase')!='open' for f in fills):return dict(supported=False,reason='nonopen_fill_phase'),[]
    if not fills:return dict(supported=False,reason='no_fill'),[]
    first=min(datetime.fromisoformat(f['timestamp']) for f in fills);last=max(datetime.fromisoformat(f['timestamp']) for f in fills)
    relevant=frame[(frame.timestamp>=first)&(frame.timestamp<last)]
    records={r.timestamp.to_pydatetime():r for r in relevant.itertuples()};events=defaultdict(list)
    for f in fills:events[datetime.fromisoformat(f['timestamp'])].append(f)
    cash=1e8;q=0;curve=[];highs=[];lows=[];prices=[];seen=0
    for clock in sorted(set(records)|set(events)):
        for f in events[clock]:
            fee=f['metadata']['fee'];sign=1 if f['side']=='BUY' else -1
            cash-=sign*f['qty']*f['price']+fee;q+=sign*f['qty'];assert q>=0;seen+=1
        r=records.get(clock)
        if r is not None and q:
            curve.append([clock.isoformat(),int(r.source_row),q,cash+q*r.close-1e8])
            highs.append((cash+q*r.high-1e8,clock,int(r.source_row)))
            lows.append((cash+q*r.low-1e8,clock,int(r.source_row)))
            prices.append((r.high,r.low,clock,int(r.source_row)))
    assert seen==len(fills)
    initial_buy=next(f for f in fills if f['side']=='BUY');entry=initial_buy['price']
    hi=max(highs,key=lambda p:(p[0],-p[1].timestamp()),default=None)
    lo=min(lows,key=lambda p:(p[0],p[1].timestamp()),default=None)
    pricehi=max(prices,key=lambda p:(p[0],-p[2].timestamp()),default=None)
    result=dict(supported=True,terminal_inventory=q,terminal_cash_change=cash-1e8,first_fill=str(first),last_fill=str(last),
        observed_held_minutes=len(curve),expected_span_minutes=max(0,int((last-first).total_seconds()/60)),
        missing_span_minutes=max(0,int((last-first).total_seconds()/60)-len(relevant)),
        first_fill_price=entry,price_mfe_pct=100*(pricehi[0]/entry-1) if pricehi else None,
        price_mae_pct=100*(min(p[1] for p in prices)/entry-1) if prices else None,
        time_to_price_mfe_minutes=(pricehi[2]-first).total_seconds()/60 if pricehi else None,
        mtm_mfe_krw=hi[0] if hi else None,mtm_mae_krw=lo[0] if lo else None,
        mtm_mfe_source_row=hi[2] if hi else None,mtm_mae_source_row=lo[2] if lo else None,
        scope='own filled quantities; already incurred costs; retained intraminute marks, not executable exits; no carry over missing marks')
    return result,curve
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--labels',default=str(ROOT/'labels_run_v5'));a=ap.parse_args()
    lab=Path(a.labels);assert read(lab/'labels_complete.json')['passed'];lr=read(lab/'registration.json')
    regpath=Path(lr['finer_sources']['path']);verify(lr['finer_sources']);sources=read(regpath)['sources']
    out=ROOT/'finer_paths_v1';out.mkdir(exist_ok=False);start=time.monotonic();pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1)
    save(out/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),program=bound(__file__),labels=bound(lab/'registration.json'),
        sources=bound(regpath),protocol=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043-finer-path-protocol.md'),seconds_guard=3600,max_bytes=2*1024**3,financial_evaluations=0))
    grouped=defaultdict(list);timing=defaultdict(list);bindings=[];counts=Counter();max_cash_error=0
    for receipt in sorted(lab.rglob('*_acceptance.json')):
        r=read(receipt)
        if r['process']['pilot']:continue
        assert r['passed'];bindings.append(bound(receipt))
        for b in r['files'].values():verify(b)
        summaries={x['episode_id']+'/'+str(x['horizon_minutes']):(n,x) for n,x in rows(r['files']['labels.jsonl.gz']['path'])}
        for n,x in rows(r['files']['native_records.jsonl.gz']['path']):
            ln,label=summaries[x['episode_id']+'/'+str(x['horizon_minutes'])]
            counts['all_labels']+=1
            if not x['fills']:counts['unfilled_labels']+=1;continue
            grouped[label['symbol']].append(dict(label=label,fills=x['fills'],label_source=r['files']['labels.jsonl.gz']['path'],label_row=ln,
                detail_source=r['files']['native_records.jsonl.gz']['path'],detail_row=n))
    for r in read(OLD/'evaluations/v6_h0.seal.json')['receipts']:
        folder=Path(r['folder']);fillfile=folder/'fills.json';bindings.append(bound(fillfile))
        for n,f in enumerate(read(fillfile),1):timing[f['symbol']].append(dict(period=r['item']['period'],scenario=r['process']['scenario'],source=str(fillfile),row=n,fill=f))
    with gzip.open(out/'label_paths.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as paths,gzip.open(out/'fill_windows.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as windows:
        for index,symbol in enumerate(sorted(set(grouped)|set(timing))):
            if time.monotonic()-start>3600 or psutil.Process().memory_info().rss>2*1024**3:raise RuntimeError('Chosen finer path resource bound reached')
            binding=sources.get(symbol)
            if binding is None:
                counts['missing_source_labels']+=len(grouped[symbol]);continue
            verify(binding);bindings.append(binding);frame=pd.read_parquet(binding['path']);frame['source_row']=range(1,len(frame)+1)
            assert not frame.timestamp.duplicated().any();frame['day']=frame.timestamp.dt.strftime('%Y-%m-%d');by_day=frame.groupby('day',sort=False);days=defaultdict(list)
            for item in grouped[symbol]:days[item['label']['day']].append(item)
            for day,items in days.items():
                dayframe=by_day.get_group(day) if day in by_day.groups else frame.iloc[:0]
                for item in items:
                    label=item.pop('label');fills=item.pop('fills');result,curve=calculate(fills,dayframe)
                    if result.get('supported'):
                        error=abs(result['terminal_cash_change']-label['net_cash_krw']);max_cash_error=max(max_cash_error,error);assert error<.02
                    counts['supported_paths' if result['supported'] else 'unsupported_paths']+=1
                    emit(paths,dict(episode_id=label['episode_id'],scenario=label['scenario'],horizon=label['horizon_minutes'],source=binding,
                        label_valid=label['label_valid'],**item,**result,curve_columns=['minute_open','source_row_1based','quantity','close_mtm_change_krw'],curve=curve))
            lookup=frame.set_index('timestamp')
            for item in timing[symbol]:
                fill=item.pop('fill');clock=datetime.fromisoformat(fill['timestamp']);obs=[]
                for offset in [0,1,2,5]:
                    at=clock+timedelta(minutes=offset)
                    if at in lookup.index:
                        row=lookup.loc[at];obs.append(dict(offset=offset,at=str(at),source_row=int(row.source_row),open=float(row.open),close=float(row.close)))
                emit(windows,dict(**item,symbol=symbol,side=fill['side'],fill_time=fill['timestamp'],fill_price=fill['price'],quantity=fill['qty'],source=binding,observations=obs))
                counts['fill_windows']+=1
            if index%10==0:print(json.dumps(dict(instruments=index+1,counts=counts,seconds=time.monotonic()-start)),flush=True)
    save(out/'source_bindings.json',bindings)
    save(out/'completion.json',dict(passed=True,seconds=time.monotonic()-start,counts=counts,max_terminal_cash_error=max_cash_error,
        financial_evaluations=0,external_validation=False,holdout_accesses=0,files={p.name:bound(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':main()
