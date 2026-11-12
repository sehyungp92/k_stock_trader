"""Read-only postmortem of exp0030's rejected intraday residual path."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
import math
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound


def main():
    root=Path('E:/CodexResearch/k_stock_trader/kalcb_capital_sensitivity_20261016/v1')
    folder=root/'cases/compound_participation';target=root/'residual_diagnosis_v1.json'
    if target.exists():raise ValueError('Preserve previous diagnosis')
    symbol='240810';day='2026-06-04'
    fills=[f for f in read(folder/'fills.json') if f['symbol']==symbol and day<=f['timestamp'][:10]<='2026-06-05']
    buys=[f for f in fills if f['side']=='BUY'];sales=[f for f in fills if f['side']=='SELL']
    same=[f for f in sales if f['timestamp'][:10]==day];nextday=[f for f in sales if f['timestamp'][:10]>day]
    orders=[];decisions=[]
    with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);order=row.get('order',{})
            if order.get('order_id') in ('sim-000000303','sim-000000304'):orders.append(row)
    with gzip.open(folder/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('symbol')==symbol and row['timestamp'][:10]==day and row.get('actions'):decisions.append(row)
    plan=read(root/'plans/compound_participation/plan.json');database=Path(plan['database'])
    conn=sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True)
    bars=conn.execute('select ts,o,h,l,c,v from bars where symbol=? and day=? order by ts',(symbol,day)).fetchall();conn.close()
    capacities=[]
    for previous,current in zip(bars,bars[1:]):
        at=datetime.fromisoformat(current[0]);before=datetime.fromisoformat(previous[0])
        if '09:45'<=current[0][11:16]<='15:15' and (at-before).total_seconds()==300:
            capacities.append(dict(at=current[0],predictor=previous[0],budget=math.floor(previous[5]*.005)))
    bytime={f['timestamp'].replace(' ','T'):f['qty'] for f in same}
    assert sum(x['budget'] for x in capacities)==sum(f['qty'] for f in same)==680
    assert all(bytime.get(x['at'],0)==x['budget'] for x in capacities)
    assert sum(f['qty'] for f in buys)==716 and sum(f['qty'] for f in nextday)==36
    audit=read(folder/'execution_audit.json')
    assert audit['audit_pass'] and read(folder/'result.json')['residual_sessions']=={day:{'KALCB:240810':36}}
    previous=read('E:/CodexResearch/k_stock_trader/kalcb_frontier_completion_20261016/v3/account_index.json')
    comparators={}
    for name in ('compound_prior100','fixed_prior100'):
        old=Path(previous[name+'/oos/participation'])
        comparators[name]=[t for t in read(old/'trades.json') if t['symbol']==symbol and t['entry_decision_time'][:10]==day]
    code=root/'inputs/files'
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0030',case='compound_participation',
        qualification='rejected: intraday residual; unchanged independent account checks retained',
        detected_failure_not_exhaustive_independent_validation=True,entry_quantity=716,same_day_sold=680,next_day_sold=36,
        every_available_continuous_budget_consumed=True,capacity_quotes=capacities,
        closing_mark_notional_krw=36*bars[-1][4],next_day_execution=nextday,
        closing_auction_observation=bars[-1],
        auction_caveat='Existing MARKET exit uses continuous open phase; no CLOSE_AUCTION order was submitted. Auction volume is not an executable fill for this unchanged policy. A different phase/order policy needs a separate prospective intervention and preserves historical model liquidity limitations.',
        orders_event_counts=dict(Counter(x['event'] for x in orders)),orders=orders,action_decisions=decisions,
        prior100m_comparators=comparators,
        diagnosis='Larger entry exceeds subsequent modeled continuous-session liquidation capacity. Existing exit stays active, timer skips exit_in_flight, and MARKET remainder persists into next day. This is an adverse executable-policy/capacity boundary, not a validator schema error or proof of an accounting arithmetic defect.',
        next='Finish the five unattempted registered cases in a new orchestration generation; preserve this rejection. No forced fill, looser validator, smaller capital or retrospective quantity cap.',
        bindings=[bound(folder/f) for f in ('result.json','completion.json','request.json','execution_audit.json','daily.json','trades.json','fills.json','orders.jsonl.gz','decisions.jsonl.gz')]+
            [bound(code/f) for f in ('backtests/engine/capacity.py','backtests/engine/sim_broker.py','strategy_kalcb/core/logic.py')]+[bound(__file__)],
        database=dict(path=str(database),sha256=plan['input_sha256'][str(database)]),economic_objective_achieved=False)
    write(target,report)
    print(json.dumps({k:report[k] for k in ('qualification','entry_quantity','same_day_sold','next_day_sold','every_available_continuous_budget_consumed','closing_mark_notional_krw')}))


if __name__=='__main__':main()
