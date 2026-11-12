"""Summarize actual funded native orders without inventing counterfactual fills."""
from collections import Counter
import gzip
from pathlib import Path
import json
from backtests.strategies.kalcb.retention_research_io_v1 import read,bound


def funding_review(folder):
    folder=Path(folder)
    if not read(folder/'independent_account_verification.json')['passed']:
        raise ValueError('Unverified funded account')
    submitted={};rejects=[];events=Counter()
    with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);events[row['event']]+=1
            order=row.get('order',{})
            if order.get('side')!='BUY':continue
            if row['event']=='submit':submitted[order['order_id']]=order
            if row['event']=='reject':
                rejects.append({'order_id':order['order_id'],'symbol':order['symbol'],
                    'reason':row['reason'],'timestamp':row['timestamp'],
                    'decision_available_at':order.get('metadata',{}).get('decision_available_at'),
                    'actual_cash':row.get('cash'),'attempted_qty':row.get('attempted_qty')})
    cash=100_000_000.;minimum=cash;executions=set();filled=set()
    for fill in read(folder/'fills.json'):
        eid=fill['metadata']['execution_id']
        if eid in executions:raise ValueError('Duplicate physical execution')
        executions.add(eid)
        buy=fill['side']=='BUY';cash+=(-1 if buy else 1)*fill['qty']*fill['price']-fill['metadata']['fee']
        minimum=min(minimum,cash)
        if buy:
            if fill['order_id'] not in submitted:raise ValueError('Unsubmitted physical buy')
            filled.add(fill['order_id'])
    if abs(cash-read(folder/'daily.json')[-1]['equity'])>.01:raise ValueError('Funding cash mismatch')
    return {'verified':True,'submitted_buy_orders':len(submitted),'physically_filled_buy_orders':len(filled),
        'submitted_buy_orders_without_fill':len(set(submitted)-filled),'events':dict(events),
        'rejected_buy_events':rejects,'unique_rejected_buy_orders':len({r['order_id'] for r in rejects}),
        'rejection_reasons':dict(Counter(r['reason'] for r in rejects)),
        'minimum_funded_cash_krw':minimum,'terminal_cash_krw':cash,
        'scope':'Actual broker journals and physical fills only. Borrowed cash is allowed within original audited leverage; negative cash alone is not an invalid fill. Retries/partial fills are dependent. These observations neither isolate gate PnL nor prove a hypothetical order was fundable.',
        'bindings':[bound(folder/f) for f in ('orders.jsonl.gz','fills.json','daily.json','independent_account_verification.json')]}
