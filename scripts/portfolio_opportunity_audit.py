"""Export decision-time capital constraints separately from realized trade labels.

Rejected orders have no invented counterfactual PNL. Native strategy scores are
retained for diagnosis and explicitly cannot be compared as common expected KRW.
"""
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path

from scripts.portfolio_targeted_validation import read,bound,compressed_rows
from scripts.portfolio_native_budget_study import checked
from backtests.strategies.kalcb.executed_value import write_json


def decision_record(row,submission):
    order=row['order'];metadata=order.get('metadata',{})
    at=datetime.fromisoformat(row['timestamp'])
    available=datetime.fromisoformat(submission['available_at'])
    if available>at:raise ValueError('Admission precedes availability of native request')
    if order['order_id']!=submission['submitted_order_id']:
        raise ValueError('Capital record attached to another native request')
    sid=order['strategy_id']
    score=metadata.get('frontier_selection_score') if sid=='KALCB' else metadata.get('candidate_score')
    return {'order_id':order['order_id'],'strategy':sid,'symbol':order['symbol'],
        'available_at':available.isoformat(),'admission_at':at.isoformat(),
        'native_requested_quantity':submission['native_qty'],'offered_quantity':row['offered_qty'],
        'approved_quantity':row['approved_qty'],'unit_cost_at_admission':row['unit_cost'],
        'native_score':score,'score_comparability':'Within original strategy only; not common expected net KRW',
        'native_rank':metadata.get('frontier_rank',metadata.get('candidate_rank')),
        'binding_constraints':row['reservation']['binding_constraints'],
        'decision_state':{k:row['ledger'][k] for k in
            ('equity','held_gross','budgets','remaining_funding','committed_planned_kalcb_risk')},
        'same_event_reservations':row['ledger']['allocations'],
        'native_budget_config':row['config'],
        'value_forecast_status':'Not a cross-strategy marginal-value forecast',
        'realized_outcomes_included':False}


def export_case(case,destination,*,reserved_validation=False):
    case=Path(case);destination=Path(destination)
    checked(case)
    destination.mkdir(parents=True,exist_ok=False)
    submissions={r['submitted_order_id']:r for r in compressed_rows(case/'allocation_events.jsonl.gz')}
    counts=Counter();order_ids=set();seen=set()
    with gzip.open(destination/'decisions.jsonl.gz','wt',encoding='utf-8') as f:
        for row in compressed_rows(case/'native_budget_admissions.jsonl.gz'):
            oid=row['order']['order_id']
            if oid not in submissions:raise ValueError('Admission without a native submission')
            record=decision_record(row,submissions[oid]);key=(oid,record['admission_at'])
            if key in seen:raise ValueError('Duplicate capital decision identity')
            seen.add(key);order_ids.add(oid)
            counts['events']+=1
            counts['fully_blocked_events']+=row['offered_qty']>0 and row['approved_qty']==0
            counts['resized_events']+=0<row['approved_qty']<row['offered_qty']
            f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    # Economic trade outcomes are a separate, explicitly ex-post table. No
    # missing trade or rejected order is assigned zero PNL as a model target.
    trades=read(case/'trades.json')
    with gzip.open(destination/'realized_trades.jsonl.gz','wt',encoding='utf-8') as f:
        for t in trades:
            f.write(json.dumps({'strategy':t['strategy_id'],'symbol':t['symbol'],
                'entry_decision_at':t['entry_decision_time'],'entry_fill_at':t['entry_fill_time'],
                'label_available_at':t['exit_fill_time'],'quantity':t['qty'],'net_pnl_krw':t['net_pnl'],
                'label_scope':'Actual filled lifecycle only; not a rejected-order counterfactual'},sort_keys=True)+'\n')
    receipt={'version':'capital-opportunity-audit-v1','case':bound(case/'completion.json'),
        'tables':{name:bound(destination/name) for name in ('decisions.jsonl.gz','realized_trades.jsonl.gz')},
        'counts':dict(counts),'distinct_native_orders':len(order_ids),'economic_trade_labels':len(trades),
        'reserved_validation':reserved_validation,
        'training_use':'Consumed validation; any future use requires a new untouched evaluation window' if reserved_validation
                       else 'Reused development; purging, label maturity and independent evaluation still required',
        'limitations':['Funding admissions omit signals rejected upstream by the native strategy.',
            'Reservation counts and partial-fill events are not economic trade frequency.',
            'Rejected orders have no identified execution or outcome counterfactual.',
            'Available capital and rank are endogenous to the portfolio path.']}
    write_json(destination/'manifest.json',receipt)
    return receipt
