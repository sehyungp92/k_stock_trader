"""Descriptive request-cap comparison; deliberately does not estimate PnL."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import hashlib,json,math
from pathlib import Path
import sqlite3,sys

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/continuous_state_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    window=sys.argv[1];protocol=read(ROOT/'protocol.json');info=read(Path(protocol['generation_directories'][window])/window/'ready.json')
    files=Path(info['files']);sys.path.insert(0,str(files))
    from strategy_common.market import MarketBar
    from strategy_olr.core.capacity import phase
    evidence=REPO/'data/backtests/analysis/olr_round1_oos_review_20261010/evidence'/window
    trades=read(evidence/'trades.json');database=files/info['study']['database']
    with database.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==info['study']['database_hash']
    histories=defaultdict(list)
    with sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True) as c:
        rows=c.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day<=? AND o=h AND h=l AND l=c ORDER BY ts,symbol',(max(t['entry_fill_time'][:10] for t in trades),))
        for symbol,stamp,*values in rows:
            b=MarketBar(symbol,datetime.fromisoformat(stamp),'5m',*values,source_fingerprint=info['study']['database_hash'])
            if phase(b)=='auction':histories[symbol].append((b.timestamp,float(b.volume)))
    details=[]
    for t in trades:
        meta=t['route_metadata'];old=meta['capacity_forecast'];at=datetime.fromisoformat(old['asof'])-timedelta(minutes=5)
        prior=[(stamp,v) for stamp,v in histories[t['symbol']] if stamp.date()<at.date()][-20:]
        prior=[(stamp,v) for stamp,v in prior if stamp>=at-timedelta(days=60)]
        values=sorted(v for _,v in prior);point=(len(values)-1)*old['exit']['quantile'];lo=math.floor(point)
        qty=math.floor((values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(point-lo))*old['participation']) if len(values)>=3 else None
        details.append({'symbol':t['symbol'],'entry':t['entry_fill_time'],'old_samples':len(old['exit']['samples']),
            'old_exit_cap':old['exit']['quantity'],'full_history_samples':len(prior),'full_history_exit_cap':qty,
            'original_native_quantity':meta.get('capacity_native_qty'),'original_filled_quantity':t['qty'],
            'full_history_below_original_filled_quantity':qty is not None and qty<t['qty'],
            'full_history_below_original_native_request':qty is not None and qty<meta.get('capacity_native_qty',t['qty']),
            'original_net_pnl':t['net_pnl'],'full_history_sources':[[stamp.isoformat(),v] for stamp,v in prior]})
    result={'window':window,'database_sha256':info['study']['database_hash'],'original_trade_episodes':len(trades),
        'old_sample_counts':dict(Counter(x['old_samples'] for x in details)),
        'new_sample_counts':dict(Counter(x['full_history_samples'] for x in details)),
        'new_bound_below_original_filled_quantity_count':sum(x['full_history_below_original_filled_quantity'] for x in details),
        'new_bound_below_original_native_request_count':sum(x['full_history_below_original_native_request'] for x in details),
        'interpretation':'One-factor forecast comparison at original decisions only. Changes in cash, entries, fills, marks, exits, and subsequent allocations require the paired native replay; no rescaled-PnL claim.',
        'details':details}
    out=REPO/'data/backtests/analysis/olr_round1_optimization_20261010'/f'capacity_request_counterfactual_{window}.json'
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print({k:v for k,v in result.items() if k!='details'})


if __name__=='__main__':main()
