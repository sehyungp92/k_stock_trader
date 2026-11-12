"""Observational minute marks of fixed saved fills, with native-close reconciliation."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timedelta
import csv,hashlib,json
import pandas as pd

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')
OUT=ROOT/'minute_marks'
def read(p):return json.loads(p.read_text())
def stamp(x):return datetime.fromisoformat(x)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str)
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    reg=read(ROOT/'minute_source_audit/registration.json');wanted=defaultdict(set)
    for p in ('is','oos'):
        for s in ('baseline','participation','cost','combined'):
            for t in read(SOURCE/'cases'/f'combined_{p}_{s}/trades.json'):wanted[t['symbol']].add(t['entry_fill_time'][:10])
    write(OUT/'binding.json',dict(program=__file__,sha256=sha(Path(__file__)),source_registration_sha256=sha(ROOT/'minute_source_audit/registration.json'),
        design='kalcb-exp-0041-minute-marks.md',scope='Observational saved-execution mark refinement; incomplete minute coverage remains.'))
    quotes=defaultdict(lambda:defaultdict(dict))
    for sym,dates in wanted.items():
        source=Path(reg['sources'][sym]['path']);assert sha(source)==reg['sources'][sym]['sha256']
        df=pd.read_parquet(source);df=df[df['timestamp'].dt.strftime('%Y-%m-%d').isin(dates)]
        assert not df['timestamp'].duplicated().any()
        for r in df.itertuples():quotes[r.timestamp.strftime('%Y-%m-%d')][r.timestamp.to_pydatetime()][sym]=(float(r.open),float(r.close))
    results=[]
    for period in ('is','oos'):
        for scenario in ('baseline','participation','cost','combined'):
            folder=SOURCE/'cases'/f'combined_{period}_{scenario}';fills=read(folder/'fills.json');eq=read(folder/'equity.json');daily=read(folder/'daily.json')
            assert all(f['metadata']['execution_phase']=='open' for f in fills)
            bystamp=defaultdict(list)
            for f in fills:bystamp[stamp(f['timestamp'])].append(f)
            native={stamp(t)+timedelta(minutes=5):v for t,v in eq};cash=1e8;q=Counter();marks={};observed={};peak=1e8;peak_at=None;worst=0;dd=None;errors=[];allmarks=[];stale_max=0
            def mark(at,phase):
                nonlocal peak,peak_at,worst,dd,stale_max
                nav=cash+sum(n*marks[s] for s,n in q.items() if n)
                for s,n in q.items():
                    if n:stale_max=max(stale_max,(at-observed[s]).total_seconds()/60)
                if nav>=peak:peak=nav;peak_at=at
                d=100*(peak-nav)/peak
                if d>worst:worst=d;dd=dict(peak_at=peak_at,trough_at=at,phase=phase,peak_equity=peak,trough_equity=nav,dd_pct=d)
                allmarks.append((at,phase,nav))
                return nav
            for day in daily:
                d=day['day'];base=stamp(d+'T09:00:00+09:00');last=stamp(d+'T16:40:00+09:00')
                # Cash-only pre/post session marks do not add price observations.
                at=base
                while at<=last:
                    here=quotes[d].get(at,{})
                    for s,(o,c) in here.items():marks[s]=o;observed[s]=at
                    for f in bystamp.get(at,[]):
                        sign=1 if f['side']=='BUY' else -1;cash+=-sign*f['qty']*f['price']-f['metadata']['fee'];q[f['symbol']]+=sign*f['qty']
                        assert q[f['symbol']]>=0
                        if sign>0 and f['symbol'] not in marks:raise ValueError('Unobserved entry open')
                    mark(at,'open_after_fill')
                    for s,(o,c) in here.items():marks[s]=c;observed[s]=at+timedelta(minutes=1)
                    closeat=at+timedelta(minutes=1);nav=mark(closeat,'close_before_next_open')
                    if closeat in native:errors.append(abs(nav-native[closeat]))
                    at=closeat
                assert not any(q.values()) and abs(cash-day['equity'])<.02
            assert len(errors)==len(eq) and max(errors)<.02,(len(errors),len(eq),max(errors))
            recover=next((t for t,phase,nav in allmarks if t>dd['trough_at'] and nav>=dd['peak_equity']),None)
            result=dict(period=period,scenario=scenario,retained_minute_dd_pct=worst,native_dd_pct=read(folder/'result.json')['drawdown_pct'],
                peak_at=str(dd['peak_at']),trough_at=str(dd['trough_at']),recovery_at=str(recover),money_decline_krw=dd['peak_equity']-dd['trough_equity'],
                maximum_held_mark_staleness_minutes=stale_max,checked_native_marks=len(errors),max_native_nav_error_krw=max(errors),
                execution_phase='all_open',strict_finer_source_qualified=False)
            results.append(result);print(result,flush=True)
            csvwrite(OUT/(period+'_'+scenario+'_marks.csv'),[dict(at=str(t),phase=p,nav=v) for t,p,v in allmarks])
    csvwrite(OUT/'summary.csv',results);write(OUT/'completion.json',dict(complete=True,cells=8,all_native_closes_reconciled=True,strict_finer_source_qualified=False))


if __name__=='__main__':main()
