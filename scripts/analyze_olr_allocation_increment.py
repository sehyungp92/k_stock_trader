"""Describe native allocation gains without inventing a counterfactual replay."""
from collections import defaultdict
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def lines(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def analyze(root,profile,capital,scenario):
    case=root/'cases'/f'{profile}_{capital}_{scenario}'
    control=root/'cases'/f'median_{capital}_{scenario}'
    results=[read(p/'result.json') for p in (case,control)]
    if any(r['terminal_inventory'] for r in results):raise ValueError('P&L decomposition requires flat terminal accounts')
    def pnl(path):
        values=defaultdict(float)
        for trade in read(path/'trades.json'):
            values[(trade['entry_fill_time'][:10],trade['symbol'])]+=trade['net_pnl']
        return values
    candidate,baseline=pnl(case),pnl(control)
    pairs=sorted([{'day':d,'symbol':s,'candidate_pnl':candidate[(d,s)],'control_pnl':baseline[(d,s)],
        'delta_won':candidate[(d,s)]-baseline[(d,s)],
        'delta_return_pp':100*(candidate[(d,s)]-baseline[(d,s)])/capital}
        for d,s in candidate.keys()|baseline.keys()],key=lambda r:r['delta_won'],reverse=True)
    delta=results[0]['return_pct']-results[1]['return_pct']
    if not math.isclose(sum(p['delta_return_pp'] for p in pairs),delta,abs_tol=1e-6):
        raise ValueError('Native P&L does not reconcile to whole-account return difference')
    interventions={}
    for event in lines(case/'decisions.jsonl.gz'):
        m=event.get('metadata') or {}
        if event['decision_code']=='ENTRY_SUBMITTED' and m.get('allocation_redistributed_qty',0)>0:
            key=(event['timestamp'][:10],event['symbol'])
            if key in interventions:raise ValueError('Ambiguous repeated intervention lifecycle')
            interventions[key]={'day':key[0],'symbol':key[1],'rank':m['candidate_rank'],
                'native_funded_qty':m['allocation_before_redistribution'],
                'requested_extra_qty':m['allocation_redistributed_qty'],'filled_qty':0}
    entry_ids=set()
    for fill in lines(case/'fills.jsonl.gz'):
        if fill['side'].upper()!='BUY':continue
        entry_ids.add(fill['order_id'])
        key=(str(fill['metadata'].get('entry_submission_time') or fill['timestamp'])[:10],fill['symbol'])
        if key in interventions:interventions[key]['filled_qty']+=fill['qty']
    for row in interventions.values():
        row['fills_above_native_qty']=max(0,row['filled_qty']-row['native_funded_qty'])
    if len(entry_ids)!=results[0]['trades']:raise ValueError('Economic trade count differs from filled entry-order count')
    return {'profile':profile,'capital':capital,'scenario':scenario,
        'whole_account_return_gain_pp':delta,
        'largest_paired_contribution_pp':pairs[0]['delta_return_pp'],
        'gain_excluding_largest_paired_contribution_pp':delta-pairs[0]['delta_return_pp'],
        'top_gains':pairs[:5],'top_losses':list(reversed(pairs[-5:])),
        'direct_interventions':list(interventions.values()),
        'intervention_issuer_days':len(interventions),
        'intervention_dates':len({r['day'] for r in interventions.values()}),
        'intervention_issuers':len({r['symbol'] for r in interventions.values()}),
        'interventions_filled_above_native_qty':sum(r['fills_above_native_qty']>0 for r in interventions.values()),
        'filled_entry_order_count':len(entry_ids),'reported_trades':results[0]['trades'],
        'case_result_hash':digest(case/'result.json'),'control_result_hash':digest(control/'result.json'),
        'interpretation':'Descriptive differences between complete realized account paths, including compounding and later allocation. Excluding a paired contribution is arithmetic, not a replay with that trade removed. Fills above native quantity describe actual orders, not an independent counterfactual fill path.',
        'independent_evidence':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--profile',required=True);p.add_argument('--capital',type=int,default=100000000)
    a=p.parse_args();rows=[analyze(a.root,a.profile,a.capital,s) for s in ('baseline','combined')]
    target=a.root/f'{a.profile}_{a.capital}_incremental_analysis.json'
    target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('scenario','whole_account_return_gain_pp',
        'largest_paired_contribution_pp','gain_excluding_largest_paired_contribution_pp',
        'intervention_issuer_days','intervention_dates','intervention_issuers',
        'interventions_filled_above_native_qty')} for r in rows],indent=2))
