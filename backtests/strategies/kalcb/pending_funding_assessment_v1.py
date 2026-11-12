"""Read-only V25 comparison. No policy creation, replay, or automatic next trial."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gzip
import json
import numpy as np

from .retention_research_io_v1 import read, write, bound, verify, PERIODS
from .late_artifact_validation_v2 import validate_native
from .population_research_io_v1 import bridge_accounts
from .fixed_allocation_funding_v1 import funding_review
from .exit_qualification_assessment_v1 import account_attribution, paired_blocks, GLOBAL
from .reclaim_observer_assessment_v1 import (require, close, exact_process,
    exposure_positions, decision_accounts, reconstruct_mtm, summarize_positions, exposure_comparison)

RECIPES = ('control', 'control_prior125', 'pending_100', 'pending_125')
SCENARIOS = ('baseline', 'combined')


def completion_gate(root, alive=exact_process):
    root = Path(root)
    require(not alive(read(root/'supervisor_start.json')), 'Exact native supervisor still active')
    status = read(root/'status.json')
    require(not status.get('child') or not alive(status['child']), 'Exact native child still active')
    require(status['stage'] == 'awaiting_full_economic_assessment', 'Native stage incomplete or failed')
    done = read(root/'stage_completion.json')
    require(done.get('complete') is True and done['native_cases'] == 16 and done['exact_parent_bridges'] == 8,
            'Missing sixteen completed cases and eight bridges')
    rows = read(root/'progress.json')
    keys = [(r['name'], r['period'], r['scenario']) for r in rows]
    expected = {(n,p,s) for n in RECIPES for p in PERIODS for s in SCENARIOS}
    require(len(keys) == 16 and set(keys) == expected, 'Missing or duplicate native case')
    for r in rows:
        require(r['metrics']['audit_status'] == 'passed', 'Unaudited case')
        require(r['is_identity'] == r['name'].startswith('control'), 'Wrong identity role')
    return rows


def activity_summary(positions, sessions):
    """Report economic size continuously and at disclosed descriptive floors.

    Floors are report sensitivities, not strategy gates or optimized thresholds.
    They do not turn a higher count into a claim of profitable incremental alpha.
    """
    result = summarize_positions(positions)
    winners = [r for r in positions if r['net_pnl_krw'] > 0]
    result.update(profitable_positions_per_session=len(winners)/sessions,
        profitable_entry_notional_krw=sum(r['entry_cost_krw'] for r in winners),
        gross_winning_net_pnl_krw=sum(r['net_pnl_krw'] for r in winners),
        gross_losing_net_pnl_krw=sum(r['net_pnl_krw'] for r in positions if r['net_pnl_krw']<0),
        unique_issuer_days=len({(r['symbol'],r['entry_day']) for r in positions}),
        one_share=summarize_positions([r for r in positions if r['qty']==1]),
        filled_notional_quantiles_krw=np.quantile([r['entry_cost_krw'] for r in positions],[0,.1,.25,.5,.75,.9,1]).tolist() if positions else [],
        profitable_size_sensitivity={})
    for floor in (.001,.01,.05):
        selected=[r for r in positions if r['qty']>1 and r['entry_cost_krw']/r['entry_equity_krw']>=floor]
        result['profitable_size_sensitivity'][str(floor)]=summarize_positions(selected)
    result['size_scope']='0.1%, 1%, 5% of actual pre-submission equity and >1 share are descriptive report floors, not selection rules or acceptance cutoffs. All positions and full notional distributions remain reported.'
    return result


def reservation_support(folder, evidence, fills, positions):
    fills_by_order=defaultdict(list)
    for f in fills:
        if f['side']=='BUY':fills_by_order[f['order_id']].append(f)
    position_by_order={oid:p['position_id'] for p in positions for oid in p['entry_order_ids']}
    changed=[]
    for e in evidence['entry_exposures']:
        action=e['action'];m=action['metadata'].get('pending_funding_reservation')
        if not m:continue
        require(action['metadata']['entry_route']=='first30_open_anchor','Reservation escaped opening scope')
        cap=min(m['original_qty'],int(max(0.,2*m['equity']-m['reserved_gross'])/m['completed_close']))
        require(cap==m['capped_qty']==action['qty'] and cap>0,'Reservation formula/actual submitted quantity mismatch')
        close(m['equity'],e['account_before']['actual_equity'],'Reservation equity differs from actual submission')
        if cap>=m['original_qty']:continue
        fs=fills_by_order[e['order_id']]
        changed.append({'order_id':e['order_id'],'symbol':action['symbol'],'submitted_at':e['submitted_at'],
            'decision_available_at':e['decision_available_at'],**m,'filled_qty':sum(f['qty'] for f in fs),
            'filled_entry_notional_krw':sum(f['qty']*f['price'] for f in fs),
            'position_id':position_by_order.get(e['order_id']),
            'actual_cash':e['account_before']['actual_cash'],'actual_buying_power':e['account_before']['buying_power_available']})
    zero=[];seen=set()
    with gzip.open(Path(folder)/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            d=json.loads(line);m=d.get('metadata',{}).get('pending_funding_reservation')
            if not m or d['decision_code']!='entry_rejected' or d['reason']!='qty_zero':continue
            if m['original_qty']<=0:continue
            require(m['capped_qty']==0,'Nonzero reservation labeled zero')
            require(int(max(0.,2*m['equity']-m['reserved_gross'])/m['completed_close'])==0,'Unsupported zero reservation')
            key=(d['timestamp'],d['symbol'])
            require(key not in seen,'Duplicate zero-reservation decision');seen.add(key)
            zero.append({'timestamp':key[0],'symbol':key[1],**m})
    ids={r['position_id'] for r in changed if r['filled_qty']>0}
    require(None not in ids,'Changed filled order missing position')
    actual=[p for p in positions if p['position_id'] in ids]
    return {'changed_submitted_orders':len(changed),'changed_filled_orders':sum(r['filled_qty']>0 for r in changed),
        'changed_filled_positions':len(ids),'zero_blocks':len(zero),'zero_block_rows':zero,'orders':changed,
        'unique_changed_position_outcomes':summarize_positions(actual),
        'scope':'Original quantity means uncapped shared-core quantity in the SAME candidate pre-decision state. Actual changed fills establish intervention support, not isolated causal PnL; position outcome attached once, not per partial/retry/order.'}


def path_comparison(candidate, parent):
    c={r['position_id']:r for r in candidate};p={r['position_id']:r for r in parent}
    require(len(c)==len(candidate) and len(p)==len(parent),'Duplicate position lifetime')
    groups=defaultdict(list)
    for key in sorted(set(c)|set(p)):
        a,b=c.get(key),p.get(key)
        group='new' if b is None else 'absent' if a is None else 'common'
        groups[group].append({'position_id':key,'candidate_qty':a['qty'] if a else 0,'parent_qty':b['qty'] if b else 0,
            'candidate_net_pnl_krw':a['net_pnl_krw'] if a else 0.,'parent_net_pnl_krw':b['net_pnl_krw'] if b else 0.,
            'delta_net_pnl_krw':(a['net_pnl_krw'] if a else 0.)-(b['net_pnl_krw'] if b else 0.)})
    delta=sum(r['delta_net_pnl_krw'] for rows in groups.values() for r in rows)
    close(delta,sum(r['net_pnl_krw'] for r in candidate)-sum(r['net_pnl_krw'] for r in parent),'Position difference double counted')
    top=sorted(parent,key=lambda r:r['net_pnl_krw'],reverse=True)[:10]
    return {'groups':dict(groups),'delta_net_pnl_krw':delta,
        'group_totals':{k:{'positions':len(v),'delta_net_pnl_krw':sum(r['delta_net_pnl_krw'] for r in v)} for k,v in groups.items()},
        'parent_top10':[{ 'position_id':r['position_id'],'parent_net_pnl_krw':r['net_pnl_krw'],
            'candidate_same_entry_net_pnl_krw':c.get(r['position_id'],{}).get('net_pnl_krw',0)} for r in top],
        'scope':'Exact entry symbol/time keys, unique lifetimes. New/absent/common account-path attribution includes changed allocations and descendants; not mutually independent evidence or recoverable isolated profits.'}


def account(root, row, protocol):
    name,period,scenario=(row[k] for k in ('name','period','scenario'))
    folder=root/'cases'/name/period/scenario;item=read(folder.parent/'input_profile.json')
    metrics=validate_native(folder,item,scenario,save=False)
    require(metrics==row['metrics'],'Progress metrics drift')
    verify([row['result'],row['exposure_mtm']])
    done=read(folder/'completion.json');verify([done['evidence']]+done['origins_before']+done['origins_after'])
    bridge=None
    if row['is_identity']:
        prior=protocol['recipes'][name]['prior']
        bridge=bridge_accounts(folder,protocol['profiles'][str(prior)][period]['references'][scenario])
        require(bridge['passed'] and read(folder/'identity_bridge.json')['passed'],'Parent bridge failed')
    e=read(folder/'observer_evidence.json');f=read(folder/'fills.json');t=read(folder/'trades.json')
    require(not e['previews'],'Unexpected copied action in V25')
    positions,mapping=exposure_positions(e,f,t);decisions=decision_accounts(e,f)
    attr=account_attribution(folder)
    mtm=reconstruct_mtm(folder,positions,mapping,protocol['databases'],attr['mtm_drawdown_episodes'])
    close(sum(r['net_pnl_krw'] for r in positions)/1e6,metrics['return_pct'],'Unique PnL/account mismatch')
    evidence={'metrics':metrics,'bridge':bridge,'positions':positions,'activity':activity_summary(positions,PERIODS[period][2]),
        'attribution':attr,'mtm':mtm,'decision_accounts':decisions,'funding':funding_review(folder),
        'support':reservation_support(folder,e,f,positions),
        'route_attribution':{r:summarize_positions([p for p in positions if p['route']==r]) for r in sorted({p['route'] for p in positions})}}
    files=('completion.json','result.json','execution_audit.json','independent_account_verification.json','independent_exposure_mtm.json',
        'daily.json','equity.json','fills.json','trades.json','orders.jsonl.gz','decisions.jsonl.gz','observer_evidence.json','request.json')
    bindings=[bound(folder/f) for f in files]+[bound(folder.parent/'input_profile.json')]+[item[k] for k in ('manifest','plan','policy')]
    if bridge:bindings.append(bound(folder/'identity_bridge.json'))
    return evidence,bindings


def compare(c,p,period,scenario,candidate_folder,parent_folder):
    require(c['attribution']['daily_dates']==p['attribution']['daily_dates'],'Unpaired session calendar')
    delta={k:c['metrics'][k]-p['metrics'][k] for k in ('return_pct','drawdown_pct','trades','active_days')}
    delta.update({k:c['activity'][k]-p['activity'][k] for k in ('positions','profitable_positions','unique_issuer_days',
        'profitable_entry_notional_krw','gross_winning_net_pnl_krw','gross_losing_net_pnl_krw','filled_entry_notional_krw','fees_krw')})
    delta['profitable_positions_ge_1pct_equity']=c['activity']['profitable_size_sensitivity']['0.01']['profitable_positions']-p['activity']['profitable_size_sensitivity']['0.01']['profitable_positions']
    try:
        aliases=bridge_accounts(candidate_folder,parent_folder)
    except ValueError as error:
        require(str(error).startswith(('Native account bridge drift','Native fill bridge drift')),'Unexpected alias verification error')
        aliases={'passed':False,'first_difference':str(error)}
    paths=path_comparison(c['positions'],p['positions'])
    close(paths['delta_net_pnl_krw']/1e6,delta['return_pct'],'Net return/PnL difference mismatch')
    bands=None
    if period=='is':
        global_delta=c['metrics']['return_pct']-GLOBAL[scenario]
        bands={'local_return_delta_pp':delta['return_pct'],'global_return_delta_pp':global_delta,
            'local_sacrifice_exceeds_pp':{str(x):delta['return_pct'] < -x for x in (2,5,10)},
            'global_sacrifice_exceeds_pp':{str(x):global_delta < -x for x in (2,5,10)},'scope':'Descriptive bands, not acceptance cutoffs.'}
    return {'delta':delta,'paths':paths,'exact_account_alias':aliases,'is_bands':bands,
        'paired_uncertainty':[paired_blocks(c['attribution']['daily_returns'],p['attribution']['daily_returns'],b) for b in (5,10)],
        'month_delta_krw':{m:c['attribution']['months'][m]['net_change_krw']-v['net_change_krw'] for m,v in p['attribution']['months'].items()},
        'issuer_delta_krw':{s:c['attribution']['issuers'].get(s,{}).get('net_pnl_krw',0)-p['attribution']['issuers'].get(s,{}).get('net_pnl_krw',0)
            for s in sorted(set(c['attribution']['issuers'])|set(p['attribution']['issuers']))}}


def assess(root,output):
    rows=completion_gate(root);registration=read(output/'protocol.json');verify(registration['programs']+registration['bindings'])
    protocol=read(root/'protocol.json');accounts={};bindings=[]
    for row in rows:
        key='/'.join(row[k] for k in ('name','period','scenario'))
        accounts[key],proof=account(root,row,protocol);bindings+=proof
        print('assessed '+key,flush=True)
    comparisons={};exposures={}
    for name,parent in (('pending_100','control'),('pending_125','control_prior125')):
        for period in PERIODS:
            for scenario in SCENARIOS:
                key='/'.join((name,period,scenario));pk='/'.join((parent,period,scenario))
                comparisons[key]=compare(accounts[key],accounts[pk],period,scenario,root/'cases'/name/period/scenario,root/'cases'/parent/period/scenario)
    for name in RECIPES:
        for scenario in SCENARIOS:
            exposures[name+'/'+scenario]=exposure_comparison(accounts[name+'/is/'+scenario]['positions'],accounts[name+'/oos/'+scenario]['positions'])
    verify(registration['programs']+registration['bindings']+bindings)
    result={'version':'pending-funding-assessment-v1','complete':True,'native_cases':16,'exact_parent_bridges':8,
        'accounts':accounts,'comparisons':comparisons,'exposure_descriptive':exposures,'bindings':bindings,
        'scope':'Read-only full baseline/combined screening assessment, not all-four-stress/funded finalist qualification. Existing full diagnostic reports follow. April-July reused development; no independent validation.',
        'next_experiments_authorized':False,'study_complete':False,'canonical_replaced':False}
    write(output/'assessment.json',result)
    write(output/'completion.json',{'complete':True,'assessment':bound(output/'assessment.json'),'native_cases':16,'exact_parent_bridges':8,
        'study_complete':False,'next_experiments_authorized':False,'full_reports_pending':True})
    return result
