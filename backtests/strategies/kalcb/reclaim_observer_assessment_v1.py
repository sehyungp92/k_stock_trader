"""Read-only, unique-position attribution of the unchanged V21 parent observers.

No policy construction, native replay, optimization or promotion is performed.
IS exposure quartiles are descriptive and reused unchanged in OOS, not gates.
"""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import sqlite3

from .retention_research_io_v1 import read, write, bound, verify, PERIODS
from .late_artifact_validation_v2 import validate_native
from .population_research_io_v1 import bridge_accounts
from .exit_qualification_assessment_v1 import account_attribution, activity
from .fixed_allocation_funding_v1 import funding_review


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, message, tolerance=.01):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b) <= tolerance, message)


def stamp(value):
    return datetime.fromisoformat(value).isoformat()


def exact_process(identity):
    import psutil
    try:
        p = psutil.Process(identity['pid'])
        return abs(p.create_time()-identity['process_created']) < .01 and p.cmdline() == identity['command']
    except psutil.NoSuchProcess:
        return False


def completion_gate(root, alive=exact_process):
    """Fail closed before any registration or analysis of the full study."""
    root = Path(root)
    identity = read(root/'supervisor_start.json')
    require(not alive(identity), 'Exact native supervisor still active')
    state = read(root/'status.json')
    require(not state.get('child') or not alive(state['child']), 'Exact native child still active')
    require(state['stage'] == 'awaiting_observer_assessment', 'Native study not successfully complete')
    done = read(root/'stage_completion.json')
    require(done.get('complete') is True and done.get('native_cases') == done.get('exact_parent_bridges') == 8,
            'Missing native completion and eight bridges')
    rows = read(root/'progress.json')
    expected = {(n,p,s) for n in ('control','control_prior125') for p in PERIODS for s in ('baseline','combined')}
    keys = [(r['name'],r['period'],r['scenario']) for r in rows]
    require(len(keys) == 8 and set(keys) == expected, 'Missing or duplicate native case')
    require(all(r.get('is_identity') is True and r['metrics']['audit_status'] == 'passed' for r in rows),
            'Unaudited parent identity')
    return rows


def exposure_positions(evidence, fills, trades):
    """Reconcile partial entry/exit executions into one position lifetime.

    Net PnL is attached once per lifetime, never once per quantity/fill row.
    Multiple orders in a lifetime remain dependent; first submission defines
    entry-context features and every contributing order ID is retained.
    """
    submitted = {}
    for e in evidence['entry_exposures']:
        oid = e['order_id']
        require(oid and oid not in submitted, 'Duplicate or absent submitted order ID')
        submitted[oid] = e
    actual, active, seen, fill_positions = {}, {}, set(), {}
    last = None
    for f in fills:
        ts = stamp(f['timestamp']); eid = f['metadata']['execution_id']
        require(eid not in seen and (last is None or ts >= last), 'Duplicate or unordered execution')
        seen.add(eid); last = ts
        symbol = f['symbol']; q = f['qty']; fee = float(f['metadata']['fee'])
        require(q > 0 and int(q) == q and f['side'] in ('BUY','SELL'), 'Invalid execution quantity/side')
        if f['side'] == 'BUY':
            require(f['order_id'] in submitted, 'Missing actual pre-submission exposure')
            e = submitted[f['order_id']]; a = e['action']; meta = a['metadata']; account = e['account_before']
            require(a['symbol'] == symbol and ts >= stamp(e['decision_available_at']), 'Wrong symbol or early fill')
            if symbol not in active:
                key = symbol + '/' + ts
                require(key not in actual, 'Ambiguous position lifetime')
                equity = account['actual_equity']; price = float(meta['entry_price_ref'])
                require(equity > 0 and price > 0, 'Invalid entry denominator')
                notional = a['qty'] * price
                marked = sum(p['marked_notional'] for p in account['positions'])
                sector = meta.get('sector','UNKNOWN')
                sector_marked = sum(p['marked_notional'] for p in account['positions'] if p.get('sector') == sector)
                atr = meta.get('daily_atr')
                actual[key] = {'position_id':key,'symbol':symbol,'entry_fill_time':ts,'entry_day':ts[:10],
                    'entry_order_ids':[],'first_submitted_at':e['submitted_at'],
                    'decision_available_at':e['decision_available_at'], 'last_mark_timestamp':account.get('last_mark_timestamp'),
                    'route':meta.get('entry_route','unknown'),'sector':sector,
                    'entry_equity_krw':equity,'entry_cash_krw':account['actual_cash'],
                    'buying_power_available_krw':account['buying_power_available'],
                    'pending_order_count':len(account['pending_orders']),
                    'checkpoint_success_count':account.get('session_meta',{}).get('entry_route_session_counts',{}).get('checkpoint_continuation',0),
                    'atr_over_entry_reference':atr/price if atr is not None else None,
                    'planned_notional_over_equity':notional/equity,
                    'marked_gross_exposure_before':marked/equity,
                    'same_sector_marked_exposure_before':sector_marked/equity,
                    'planned_stop_risk_over_equity':float(meta.get('risk_per_share',0))*a['qty']/equity,
                    'planned_notional_over_buying_power':notional/account['buying_power_available'] if account['buying_power_available']>0 else None,
                    'qty':0,'live_qty':0,'entry_cost_krw':0.,'gross_pnl_krw':0.,'fees_krw':0.,'net_pnl_krw':0.,'fill_count':0}
                active[symbol] = key
            r = actual[active[symbol]]
            if f['order_id'] not in r['entry_order_ids']:r['entry_order_ids'].append(f['order_id'])
            r['qty'] += q; r['live_qty'] += q; r['entry_cost_krw'] += q*f['price']
            cash_delta = -q*f['price']
        else:
            require(symbol in active, 'Sell without position')
            r = actual[active[symbol]]
            require(q <= r['live_qty'], 'Duplicate or excess exit')
            r['live_qty'] -= q; cash_delta = q*f['price']; r['exit_fill_time'] = ts
        r['gross_pnl_krw'] += cash_delta; r['fees_krw'] += fee; r['net_pnl_krw'] += cash_delta-fee; r['fill_count'] += 1
        fill_positions[eid] = r['position_id']
        if not r['live_qty']:active.pop(symbol)
    require(not active, 'Unclosed observed position')
    grouped = defaultdict(list)
    for t in trades:grouped[t['symbol']+'/'+stamp(t['entry_fill_time'])].append(t)
    require(set(grouped) == set(actual), 'Trades and position lifetimes disagree')
    for key, rows in grouped.items():
        r = actual[key]
        close(sum(t['qty'] for t in rows),r['qty'],'Partial exits duplicate quantity')
        close(sum(t['net_pnl'] for t in rows),r['net_pnl_krw'],'Position PnL join mismatch')
        close(sum(t['gross_pnl'] for t in rows),r['gross_pnl_krw'],'Position gross PnL mismatch')
        close(sum(t['commission'] for t in rows),r['fees_krw'],'Position costs mismatch')
        r['net_return_on_filled_entry_notional_bps'] = 10000*r['net_pnl_krw']/r['entry_cost_krw']
    return list(actual.values()), fill_positions


def decision_accounts(evidence, fills):
    """Reconcile pre-submission cash/inventory at the actual submission clock.

    Actual entry decisions follow the current bar's executions in this parent.
    Observer last-mark time is explicitly not substituted for that clock.
    """
    cash=100000000.;inventory=defaultdict(int);cursor=0
    rows=evidence['entry_exposures'];previous=None
    for row in rows:
        ts=stamp(row['submitted_at'])
        require(previous is None or ts>=previous,'Nonchronological entry census')
        previous=ts
        while cursor<len(fills) and stamp(fills[cursor]['timestamp'])<=ts:
            f=fills[cursor];buy=f['side']=='BUY'
            cash+=(-1 if buy else 1)*f['qty']*f['price']-f['metadata']['fee']
            inventory[f['symbol']]+=f['qty'] if buy else -f['qty'];cursor+=1
        account=row['account_before'];positions=account['positions']
        require(len({p['symbol'] for p in positions})==len(positions),'Duplicate exposure position')
        require({k:v for k,v in inventory.items() if v}=={p['symbol']:p['qty'] for p in positions},
                'Decision-clock inventory mismatch')
        close(cash,account['actual_cash'],'Decision-clock cash mismatch')
        marked=sum(p['qty']*p['mark'] for p in positions)
        for p in positions:close(p['marked_notional'],p['qty']*p['mark'],'Exposure mark denominator mismatch')
        close(cash+marked,account['actual_equity'],'Observed account equity mismatch')
        lev=account['buying_power_leverage']
        available=max(cash,0.) if lev<=1 else max((cash+marked)*lev-marked,0.)
        close(available,account['buying_power_available'],'Observed native funding formula mismatch')
    return {'decision_clock_cash_inventory_checks':len(rows),
        'scope':'Pre-submission cash/inventory independently reconstructed from actual fills. Marked equity and native leverage funding reconcile within each captured snapshot; no future or hypothetical funding claim.'}


def preview_review(evidence, expected):
    """Classify conditional orders; no realized outcome or recoverable PnL sum."""
    expected = {r['opportunity_id']:scores for r,scores in expected}
    rows = evidence['previews']; keys = [r['opportunity_id'] for r in rows]
    require(len(keys) == len(expected) and set(keys) == set(expected), 'Missing or duplicate observed opportunity')
    result = []; reasons = Counter(); failed_filters = Counter()
    for r in rows:
        oid = r['opportunity_id']; orders = r.get('new_orders',[])
        reasons_set = {r['reason']} if r.get('reason') else set()
        require(r.get('forecasts',expected[oid]) == expected[oid], 'Changed diagnostic forecasts')
        require(all(o['symbol'] == oid.split('/')[1] and o['side'] == 'BUY' for o in orders), 'Unexpected copied order')
        submitted = r.get('native_order_submitted',False)
        require(bool(orders) == submitted, 'Copied order count mismatch')
        require(not (submitted and r.get('already_candidate')), 'Incumbent labeled a new issuer')
        filters = set()
        for d in r.get('decisions',[]):
            if d.get('reason'):reasons_set.add(d['reason'])
            for f in d.get('metadata',{}).get('filter_decisions',[]):
                if f.get('applicable',True) and f.get('passed') is False:filters.add(f['filter_name'])
        reasons.update(reasons_set); failed_filters.update(filters)
        a = r.get('account_before',{})
        result.append({'opportunity_id':oid,'copied_submission':submitted,'order_count':len(orders),
            'reasons':sorted(reasons_set),'failed_filters':sorted(filters),'forecast_diagnostics':expected[oid],
            'actual_cash_krw':a.get('actual_cash'),'actual_equity_krw':a.get('actual_equity'),
            'buying_power_available_krw':a.get('buying_power_available'),
            'pending_order_count':len(a.get('pending_orders',[])),
            'checkpoint_success_count':a.get('session_meta',{}).get('entry_route_session_counts',{}).get('checkpoint_continuation'),
            'position_count':len(a.get('positions',[]))})
    by_day = Counter(r['opportunity_id'][:10] for r in result if r['copied_submission'])
    require(sum(by_day.values()) == evidence['summary']['copied_order_previews'], 'Observer summary mismatch')
    return {'opportunities':len(rows),'copied_submissions':sum(by_day.values()),'copied_submission_sessions':len(by_day),
        'competing_alternatives_by_session':dict(by_day),'reason_counts':dict(reasons),'failed_filter_counts':dict(failed_filters),
        'rows':result,'scope':'Separately copied last-ranked orders before parent timer, not changed-policy fills or funded PnL. Same-session alternatives compete; scenarios and parents are dependent.'}


def summarize_positions(rows):
    winners = [r for r in rows if r['net_pnl_krw']>0]
    cost = sum(r['entry_cost_krw'] for r in rows)
    return {'positions':len(rows),'sessions':len({r['entry_day'] for r in rows}),'issuers':len({r['symbol'] for r in rows}),
        'profitable_positions':len(winners),'net_pnl_krw':sum(r['net_pnl_krw'] for r in rows),
        'gross_pnl_krw':sum(r['gross_pnl_krw'] for r in rows),'fees_krw':sum(r['fees_krw'] for r in rows),
        'filled_entry_notional_krw':cost,'notional_weighted_net_bps':10000*sum(r['net_pnl_krw'] for r in rows)/cost if cost else None,
        'top5_position_net_pnl_krw':sum(sorted((r['net_pnl_krw'] for r in winners),reverse=True)[:5]),
        'worst5_position_net_pnl_krw':sum(sorted(r['net_pnl_krw'] for r in rows if r['net_pnl_krw']<0)[:5])}


EXPOSURE_FEATURES = ('atr_over_entry_reference','planned_notional_over_equity','marked_gross_exposure_before',
    'same_sector_marked_exposure_before','planned_stop_risk_over_equity','planned_notional_over_buying_power')


def exposure_comparison(insample, outsample):
    import numpy as np
    result = {}
    for route in sorted({r['route'] for r in insample+outsample}):
        one = [r for r in insample if r['route']==route]; two = [r for r in outsample if r['route']==route]
        result[route] = {'is':summarize_positions(one),'oos':summarize_positions(two),'features':{}}
        for feature in EXPOSURE_FEATURES:
            values = [r[feature] for r in one if r[feature] is not None]
            cuts = sorted(set(float(v) for v in np.quantile(values,[.25,.5,.75]))) if values else []
            def group(rows):
                bins = defaultdict(list)
                for r in rows:
                    value = r[feature]
                    key = 'missing' if value is None else str(sum(value>c for c in cuts))
                    bins[key].append(r)
                return {key:summarize_positions(rs) for key,rs in bins.items()}
            result[route]['features'][feature] = {'is_only_quartile_cutpoints':cuts,'is_bins':group(one),'oos_bins':group(two)}
    return {'routes':result,'scope':'Descriptive associations; IS-only route-specific exposure quartiles, tied cutpoints deduplicated, identical bins applied OOS. No causal budget benefit, policy threshold, independent validation or rescaled portfolio PnL.'}


def reconstruct_mtm(folder, positions, fill_positions, databases, episodes):
    """Position MTM contributions from physical cash flows and raw bar closes.

    Reconciles EVERY native equity mark before attributing peak-to-trough changes.
    This is additive accounting, not an executable removal/sizing counterfactual.
    """
    folder=Path(folder); fills=read(folder/'fills.json'); curve=read(folder/'equity.json')
    symbols=sorted({r['symbol'] for r in positions}); marks={}
    start,end=curve[0][0][:10],curve[-1][0][:10]
    for index,path in enumerate(databases):
        lo,hi = (start,min(end,'2026-03-31')) if index==0 else (max(start,'2026-04-01'),end)
        if lo>hi or not symbols:continue
        sql='SELECT symbol,ts,c FROM bars WHERE day>=? AND day<=? AND symbol IN ('+','.join('?' for _ in symbols)+')'
        with sqlite3.connect('file:'+Path(path).as_posix()+'?mode=ro',uri=True) as conn:
            for symbol,ts,price in conn.execute(sql,[lo,hi,*symbols]):
                key=(stamp(ts),symbol)
                require(key not in marks, 'Duplicate source close')
                marks[key]=price
    flows=defaultdict(float); held=defaultdict(int); last_marks={}; pending=0; captured={}
    required={stamp(e[k]) for e in episodes for k in ('peak_at','trough_at') if e[k]!='initial'}
    mapping={r['position_id']:r for r in positions}
    for when,value in curve:
        ts=stamp(when)
        while pending<len(fills) and stamp(fills[pending]['timestamp'])<=ts:
            f=fills[pending];key=fill_positions[f['metadata']['execution_id']]; buy=f['side']=='BUY'
            flows[key]+=(-1 if buy else 1)*f['qty']*f['price']-f['metadata']['fee']
            held[key]+=f['qty'] if buy else -f['qty'];pending+=1
        for symbol in symbols:
            if (ts,symbol) in marks:last_marks[symbol]=marks[(ts,symbol)]
        components={}
        for key,cash in flows.items():
            symbol=mapping[key]['symbol']
            require(not held[key] or symbol in last_marks,'Unmarked held position')
            components[key]=cash+held[key]*last_marks.get(symbol,0)
        close(100000000+sum(components.values()),value,'Raw-source position MTM does not reconcile')
        if ts in required:captured[ts]=components
    require(pending==len(fills) and not any(held.values()),'Unconsumed fill or terminal holdings')
    result=[]
    for episode in episodes:
        peak={} if episode['peak_at']=='initial' else captured[stamp(episode['peak_at'])]
        trough=captured[stamp(episode['trough_at'])]
        contributions=[{'position_id':key,'symbol':mapping[key]['symbol'],'route':mapping[key]['route'],
            'mtm_change_krw':trough.get(key,0)-peak.get(key,0)} for key in set(peak)|set(trough)]
        contributions=[r for r in contributions if abs(r['mtm_change_krw'])>1e-7]
        close(sum(r['mtm_change_krw'] for r in contributions),episode['trough_equity']-episode['peak_equity'],
              'Drawdown position attribution mismatch')
        result.append({**episode,'position_contributions':sorted(contributions,key=lambda r:r['mtm_change_krw'])})
    return {'all_native_marks_reconciled':len(curve),'episodes':result,
        'scope':'Additive raw-close MTM and physical cash-flow attribution, not recoverable PnL or causal allocation benefit.'}


def assess(root, output):
    root,output=Path(root),Path(output);rows=completion_gate(root)
    registration=read(output/'protocol.json');verify(registration['programs']+registration['bindings'])
    protocol=read(root/'protocol.json');spec=read(root/'opportunities.json');accounts={};bindings=[]
    for row in rows:
        name,period,scenario=(row[k] for k in ('name','period','scenario'));prior=protocol['recipes'][name]['prior']
        folder=root/'cases'/name/period/scenario;item=read(folder.parent/'input_profile.json')
        metrics=validate_native(folder,item,scenario,save=False)
        bridge=bridge_accounts(folder,protocol['profiles'][str(prior)][period]['references'][scenario])
        require(bridge['passed'] and read(folder/'identity_bridge.json')['passed'],'Parent identity failed')
        done=read(folder/'completion.json');verify([done['evidence'],done['raw_source']]+done['origins_before']+done['origins_after'])
        proof=read(folder/'reclaim_source_proof.json');daily=read(folder/'daily.json')
        require(proof['passed'] and [d['day'] for d in proof['days']]==[d['day'] for d in daily], 'Missing raw90 source day')
        require(all(d['sector_membership_verified'] and d['geometry_verified'] and d['complete_consumed_prefixes']>0 for d in proof['days']),
                'Incomplete raw90 source proof')
        evidence=read(folder/'observer_evidence.json');fills=read(folder/'fills.json');trades=read(folder/'trades.json')
        entries,mapping=exposure_positions(evidence,fills,trades)
        decisions=decision_accounts(evidence,fills)
        expected=[(r,[s for s in scores if s['prior']==prior]) for r,scores in spec['opportunities'] if PERIODS[period][0]<=r['trade_date']<=PERIODS[period][1]]
        previews=preview_review(evidence,expected);attribution=account_attribution(folder)
        mtm=reconstruct_mtm(folder,entries,mapping,spec['databases'],attribution['mtm_drawdown_episodes'])
        account={'metrics':metrics,'bridge':bridge,'source90_complete_prefixes':sum(d['complete_consumed_prefixes'] for d in proof['days']),
            'previews':previews,'positions':entries,'activity':activity(folder),'attribution':attribution,
            'funding':funding_review(folder),'decision_account_reconciliation':decisions,'independent_position_mtm':mtm}
        key='/'.join((name,period,scenario));accounts[key]=account
        files=('observer_evidence.json','completion.json','reclaim_source_proof.json','identity_bridge.json','result.json',
            'request.json','execution_audit.json','fills.json','trades.json','equity.json','daily.json','orders.jsonl.gz','independent_account_verification.json')
        bindings += [bound(folder/f) for f in files]+[bound(folder.parent/'input_profile.json')]
        write(output/'accounts'/name/period/(scenario+'.json'),account)
        print(json.dumps({'assessed':key,'count':len(accounts),'marks_reconciled':mtm['all_native_marks_reconciled']}),flush=True)
    comparisons={}
    for name in protocol['recipes']:
        for scenario in ('baseline','combined'):
            comparisons[name+'/'+scenario]=exposure_comparison(accounts[name+'/is/'+scenario]['positions'],accounts[name+'/oos/'+scenario]['positions'])
    verify(registration['programs']+registration['bindings']+bindings)
    result={'accounts':{key:{'file':bound(output/'accounts'/Path(key).parent/(Path(key).name+'.json')),
        'metrics':value['metrics'],'activity':value['activity'],'previews':{k:v for k,v in value['previews'].items() if k!='rows'}} for key,value in accounts.items()},
        'exposure_comparisons':comparisons,'bindings':bindings,'registered':bound(output/'protocol.json'),
        'new_economic_variants':0,'study_complete':False,'promotion':False,
        'limitations':['Copied orders are not future fills or funded incremental PnL.','Position-level returns and MTM attribution are descriptive on unchanged parent paths.',
            'IS bins are descriptive; no volatility/concentration budget is qualified.','April-July remains reused development.','No whole-study success or new optimization parent is established.']}
    write(output/'assessment.json',result)
    write(output/'completion.json',{'complete':True,'study_complete':False,'assessment':bound(output/'assessment.json'),'native_cases':8,'exact_bridges':8})
