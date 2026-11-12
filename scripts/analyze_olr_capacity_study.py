"""Economic/mechanism verification of completed native OLR capacity cases."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import argparse
import gzip
import json
from pathlib import Path

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def analyze(case):
    result=read(case/'result.json');request=read(case/'request.json')
    capital=read(Path(request['arguments']['study']))['initial_equity']
    trades=read(case/'trades.json');days=[x['day'] for x in read(case/'daily.json')]
    fills=list(rows(case/'fills.jsonl.gz'));orders=defaultdict(list);sells=defaultdict(list)
    for f in fills:
        orders[f['order_id']].append(f)
        if f['side']=='SELL':sells[f['symbol']].append(f)
    late_notional=entry_notional=0.;late=0
    for t in trades:
        first=datetime.fromisoformat(t['entry_fill_time']);last=datetime.fromisoformat(t['exit_fill_time'])
        next_day=days[days.index(first.date().isoformat())+1]
        legs=[f for f in sells[t['symbol']] if first<datetime.fromisoformat(f['timestamp'])<=last]
        assert sum(f['qty'] for f in legs)==t['qty']
        q=sum(f['qty'] for f in legs if datetime.fromisoformat(f['timestamp']).date().isoformat()>next_day)
        late+=q>0;late_notional+=q*t['entry_price'];entry_notional+=t['qty']*t['entry_price']
    entries=shortfall=clipped=joint_resized=guards=0;support=set();forecast_checks=0
    actual_requests={}
    entry_requested=entry_filled=0;expiry=Counter();forecast_counts=Counter()
    for row in rows(case/'journal.jsonl.gz'):
        order=row.get('order',{});m=order.get('metadata',{})
        if row['event']=='expire' and order.get('side')=='SELL':expiry[row.get('reason','unspecified')]+=1
        if row['event']!='submit':continue
        at=datetime.fromisoformat(order['submitted_at'])
        issuer_day=(order['symbol'],at.date().isoformat())
        if order['side']=='BUY':
            actual_requests[order['symbol']+'|'+order['submitted_at']]=order['qty']
            entries+=1;q=sum(f['qty'] for f in orders[order['order_id']])
            entry_requested+=order['qty'];entry_filled+=q;shortfall+=q<order['qty']
            if m.get('capacity_approved_qty',order['qty'])<m.get('capacity_native_qty',order['qty']):
                clipped+=1;support.add(issuer_day)
            if order['qty']<m.get('allocation_native_qty',order['qty']):
                joint_resized+=1;support.add(issuer_day)
        if order.get('reason') in ('capacity_exit_before_auction','capacity_exit_slice'):
            guards+=1;support.add(issuer_day)
        evidence=m.get('capacity_forecast')
        if evidence:
            asof=datetime.fromisoformat(evidence['asof'])
            assert asof<=at+timedelta(minutes=5),(case,order['order_id'],'future_forecast')
            for ts,_ in evidence.get('entry',{}).get('samples',[]):
                assert datetime.fromisoformat(ts)+timedelta(minutes=5)<=asof
            for ts,_ in evidence.get('exit',{}).get('samples',[]):
                assert datetime.fromisoformat(ts).date()<asof.date()
            forecast_checks+=1
            forecast_counts[evidence.get('exit',{}).get('status','unspecified')]+=1
    skipped=Counter()
    for event in rows(case/'decisions.jsonl.gz'):
        if event.get('decision_code')=='ENTRY_SKIPPED':skipped[event['reason']]+=1
    return {'capital':capital,'recipe':result['recipe'],'scenario':result['scenario'],
        'source_generation_hash':result['source_generation_hash'],'case_path':str(case),
        'return_pct':result['return_pct'],'drawdown_pct':result['drawdown_pct'],'trades':result['trades'],
        'execution_validity_pass':result['execution_validity_pass'],'audit_status':result['audit_status'],
        'terminal_inventory':result['terminal_inventory'],'terminal_pending':result['terminal_pending'],
        'entry_orders':entries,'entry_orders_shortfall':shortfall,
        'quantity_fill_fraction':entry_filled/entry_requested if entry_requested else 0,
        'delayed_trades':late,'late_entry_notional_fraction':late_notional/entry_notional if entry_notional else 0,
        'entry_turnover':entry_notional/capital,'capacity_clipped_entries':clipped,
        'joint_resized_entries':joint_resized,'exit_guard_orders':guards,
        'intervention_issuer_days':len(support),'forecast_causality_checks':forecast_checks,
        'exit_forecast_support':dict(forecast_counts),'entry_skips':dict(skipped),
        'sell_expiry_reasons':dict(expiry),
        'tiny_tickets':sum(t['qty']*t['entry_price']<capital*.0025 for t in trades),
        'actual_entry_requests':actual_requests}

def main(root):
    results={}
    paths={}
    if (root/'case_references.json').exists():
        paths.update({k:Path(v) for k,v in read(root/'case_references.json').items()})
    paths.update({p.name:p for p in (root/'cases').glob('*') if (p/'completion.json').exists()})
    for path in sorted(paths.values()):
        if (path/'completion.json').exists():results[path.name]=analyze(path)
    for row in results.values():
        control=results.get(f"control_{row['capital']}_{row['scenario']}")
        if control:
            a=row['actual_entry_requests'];b=control['actual_entry_requests']
            row['changed_submitted_issuer_days_vs_control']=len({(k.split('|')[0],k.split('|')[1][:10])
                for k in set(a)|set(b) if a.get(k,0)!=b.get(k,0)})
    for row in results.values():row.pop('actual_entry_requests')
    (root/'mechanism_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    decisions={}
    for recipe in read(root/'registration.json')['profiles']:
        if recipe=='control':continue
        reasons=[];complete=True
        for scenario in ('baseline','combined'):
            key=f'{recipe}_100000000_{scenario}';control=results.get(f'control_100000000_{scenario}')
            row=results.get(key)
            if not row or not control:complete=False;continue
            if row['return_pct']<=control['return_pct']:reasons.append(scenario+'_return')
            if row['drawdown_pct']>control['drawdown_pct']+1e-10:reasons.append(scenario+'_drawdown')
            if row['trades']<.95*control['trades']:reasons.append(scenario+'_frequency')
            if row['intervention_issuer_days']<10:reasons.append(scenario+'_support')
        decisions[recipe]={'complete':complete,'reasons':reasons,'eligible':complete and not reasons}
    eligible=[k for k,v in decisions.items() if v['eligible']]
    winner=max(eligible,key=lambda k:(results[f'{k}_100000000_combined']['return_pct'],
        results[f'{k}_100000000_baseline']['return_pct'])) if eligible else None
    selection={'provisional_winner':winner,'decisions':decisions,
        'qualification_required':True,'independent_performance_evidence':False}
    (root/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps({'cases':[{k:v for k,v in row.items() if k in ('recipe','capital','scenario','return_pct',
        'drawdown_pct','trades','entry_orders_shortfall','delayed_trades','late_entry_notional_fraction',
        'intervention_issuer_days','exit_guard_orders')} for row in results.values()], 'selection':selection},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)
