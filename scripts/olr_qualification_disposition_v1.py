"""Read-only final qualification verification and registered mechanism postmortem."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')


def key(t):
    return t['symbol'],t['entry_decision_time'][:10]


def grouped(rows, field):
    groups=defaultdict(list)
    for r in rows:groups[r[field]].append(r)
    result={}
    for name, members in groups.items():
        ds=defaultdict(float)
        for r in members:ds[r['action_day']]+=r['delta_pnl']
        result[name]=dict(n=len(members),dates=len(ds),delta_pnl=sum(ds.values()),
            positive=sum(r['delta_pnl']>0 for r in members),negative=sum(r['delta_pnl']<0 for r in members),
            positive_wealth=sum(max(r['delta_pnl'],0) for r in members),
            negative_wealth=sum(min(r['delta_pnl'],0) for r in members),by_date=dict(ds),
            excluding_largest_absolute_date=sum(ds.values())-max(ds.values(),key=abs))
    return result


def run(root):
    out=root/'qualification_disposition_v1';out.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    design=repo/'docs/optimisation/olr/experiments/olr-exp-0036.md'
    (out/'design_at_freeze.md').write_bytes(design.read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    # Bind the accepted campaign and all source receipts before grouped extraction.
    sources=[root/'assessment.json',root/'preflight.json',root/'repair_receipt.json']
    sources += list(root.glob('pairs/*/assessment.json'))+list(root.glob('pairs/*/causal_review_v1/assessment.json'))
    write(out/'protocol.json',dict(at=datetime.now(timezone.utc).isoformat(),design_sha256=sha(design),
        runner_sha256=sha(__file__),sources={str(p):sha(p) for p in sources},
        method='Final hard-validity checks, original criteria, fixed action anchor groups and all acquired OOS morning-state census. No replay, candidate selection or cutoff search.'))
    a=read(root/'assessment.json');checks={};summaries=a['summaries'];causals={};trade_maps={}
    for name,s in summaries.items():
        case=Path(s['case']);c=read(case/'completion.json')
        assert c['complete'] and sha(case/'completion.json')==s['completion_sha256']
        artifacts={n:sha(case/n)==h for n,h in c['artifacts'].items()};assert all(artifacts.values())
        audit=read(case/'execution_audit.json');capacity=audit['capacity_model_audit']
        assert audit['audit_pass'] and audit['execution_validity_pass'] and not capacity['errors']
        assert capacity['terminal_open_shares']==capacity['terminal_pending_shares']==0
        assert capacity['capital']['initial_equity']==1e8
        assert read(case/'funded_population_receipt.json')['audit']['pass_']
        assert read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
        window,scenario,policy=name.split('_');assert c['source_generation_hash']==read(root/'preflight.json')['source_by_window'][window]
        restart=None
        if window=='oos':
            r=read(case/'capacity_restart_receipt.json');parent=Path(summaries['train_'+scenario+'_'+policy]['case'])
            assert sha(parent/'capacity_terminal_state.json')==r['parent_checkpoint_sha256']
            assert sha(parent/'completion.json')==r['parent_completion_sha256']
            assert r['same_configuration_and_scenario'] and r['all_samples_verified_against_oos_retained_ohlcv']
            restart={k:v for k,v in r.items() if k!='restored_state'}
        trades=read(case/'trades.json');trade_maps[name]={key(t):t for t in trades};assert len(trades)==len(trade_maps[name])
        assert math.isclose(sum(t['net_pnl'] for t in trades),s['return_pct']*1e6,abs_tol=.01)
        checks[name]=dict(complete=True,artifacts=artifacts,source=c['source_generation_hash'],
            execution_valid=True,checkpoint_parity=True,population_valid=True,flat_terminal=True,
            restart=restart,phase_liquidity_verified=capacity['phase_liquidity_verified'])
    for name in a['comparisons']:
        causal=read(root/'pairs'/name/'causal_review_v1/assessment.json');causals[name]=causal
        act=causal['actions_and_funding']['full']
        assert len(act['actual_fills'])==act['actions']
        for fill in act['actual_fills']:
            assert fill['filled_quantity']==fill['expected_quantity']
            assert fill['first_fill']>fill['available_at']
        for order in act['actual_orders']:
            t=trade_maps[name+'_full'][(order['symbol'],order['entry_day'])]
            assert t['route_metadata']['risk_per_share']==order['original_risk']
        assert not [d for d in summaries[name+'_control']['context_decisions'] if d['decision_code']=='EXIT_SUBMITTED']
    # Recompute original numerical criteria, without changing requirements.
    cmp=a['comparisons'];rules={w+'_nominal_material_return':cmp[w+'_nominal']['delta']['return_pct']>=.5 for w in ['train','oos']}
    for scenario in ['stress','capped']:
        values=[cmp[w+'_'+scenario]['delta']['return_pct'] for w in ['train','oos']]
        rules[scenario+'_return_support']=sum(values)>0 and min(values)>=-.5
    changes=[cmp[w+'_nominal']['delta']['meaningful'] for w in ['train','oos']]
    rules['meaningful_participation']=sum(changes)>=2 and changes[1]>=-1
    rules['positive_changed_participation_value']=sum(x['net_pnl_attribution'] for x in a['activity'].values())>0
    assert rules==a['necessary_checks']
    anchor_rows=[]
    for window in ['train','oos']:
        name=window+'_nominal';per={ (r['symbol'],r['day']):r for r in cmp[name]['accounting']['per_opportunity'] }
        submitted=[x for x in summaries[name+'_full']['context_decisions'] if x['decision_code']=='EXIT_SUBMITTED']
        orders=causals[name]['actions_and_funding']['full']['actual_orders']
        assert len(submitted)==len(orders)
        for d in submitted:
            e=d['metadata']['relative_context_evidence'];day=e['available_at'][:10]
            order=next(x for x in orders if x['symbol']==d['symbol'] and x['action_day']==day)
            k=(d['symbol'],order['entry_day']);t=trade_maps[name+'_full'][k];risk=order['original_risk']
            close_r=(e['close']-t['entry_price'])/risk;open_r=(e['opening']-t['entry_price'])/risk
            anchor_rows.append(dict(window=window,symbol=k[0],entry_day=k[1],action_day=day,entry_price=t['entry_price'],
                original_risk=risk,opening_r=open_r,close_r=close_r,quantity=order['quantity_before'],
                anchor='below_minus_one_R' if close_r<=-1 else ('below_entry_above_minus_one_R' if close_r<0 else 'at_or_above_entry'),
                own_return=e['own_return'],market_median=e['market_median'],
                entry_rank=t['route_metadata']['candidate_rank'],entry_score=t['route_metadata']['candidate_score'],
                delta_pnl=per[k]['delta_pnl'],control_pnl=per[k]['control_pnl'] if 'control_pnl' in per[k] else trade_maps[name+'_control'].get(k,{}).get('net_pnl',0),
                full_pnl=t['net_pnl']))
    anchor_groups={w:grouped([x for x in anchor_rows if x['window']==w],'anchor') for w in ['train','oos']}
    # Exact recorded 09:30 held states, including ordinary periods and winners.
    ready=read(root/'native_engines/oos/ready.json');features_path=Path(ready['files'])/ready['study']['relative_context_features']
    features=read(features_path);control_case=Path(summaries['oos_nominal_control']['case'])
    books=read(control_case/'marked_book_observations.json');at_by_time={}
    for book in books:
        if book['observed_at'][11:19]=='09:30:00':at_by_time[book['observed_at']]=book
    morning=[]
    for at,book in sorted(at_by_time.items()):
        for p in book['positions']:
            ts=[t for t in trade_maps['oos_nominal_control'].values() if t['symbol']==p['symbol'] and t['entry_fill_time']<at<t['exit_fill_time']]
            assert len(ts)==1,(at,p['symbol'],len(ts))
            t=ts[0];k=key(t);f=features['rows'].get(at,{}).get(p['symbol'])
            pending=[x for x in book['pending'] if x['symbol']==p['symbol']]
            if k[1]>=at[:10]:reason='same_day_entry'
            elif p['pending_exit'] or any(x['side']=='SELL' for x in pending):reason='native_exit_pending'
            elif any(x['side']=='BUY' for x in pending):reason='native_acquisition_pending'
            elif f is None:reason='unsupported_prefix'
            elif f['own_return']>=0:reason='own_nondecline'
            elif f['market_median']<0:reason='shared_decline'
            else:reason='relative_trigger'
            if f:assert math.isclose(p['mark'],f['close'],abs_tol=.01),(at,p,f)
            morning.append(dict(at=at,symbol=k[0],entry_day=k[1],reason=reason,position=p,pending=pending,feature=f,
                entry_rank=t['route_metadata']['candidate_rank'],entry_score=t['route_metadata']['candidate_score'],
                final_net_pnl=t['net_pnl'],exit_at=t['exit_fill_time'],exit_reason=t['exit_reason']))
    by_lifecycle=defaultdict(list)
    for m in morning:by_lifecycle[(m['symbol'],m['entry_day'])].append(m)
    census=[]
    for k,t in trade_maps['oos_nominal_control'].items():
        rows=by_lifecycle[k];reason='no_recorded_0930_holding' if not rows else ('relative_trigger' if any(x['reason']=='relative_trigger' for x in rows) else rows[0]['reason'])
        census.append(dict(symbol=k[0],entry_day=k[1],qty=t['qty'],net_pnl=t['net_pnl'],entry_price=t['entry_price'],
            entry_at=t['entry_fill_time'],exit_at=t['exit_fill_time'],exit_reason=t['exit_reason'],
            entry_rank=t['route_metadata']['candidate_rank'],entry_score=t['route_metadata']['candidate_score'],
            morning_states=len(rows),reach_class=reason,
            episode_overlap=t['exit_fill_time']>='2026-05-15T09:04:00+09:00' and t['entry_fill_time']<='2026-07-09T09:01:00+09:00'))
    reach={}
    for group in sorted({x['reach_class'] for x in census}):
        members=[x for x in census if x['reach_class']==group]
        reach[group]=dict(n=len(members),net_pnl=sum(x['net_pnl'] for x in members),
            winners=sum(x['net_pnl']>0 for x in members),losers=sum(x['net_pnl']<0 for x in members))
    result=dict(at=datetime.now(timezone.utc).isoformat(),checks=checks,artifact_checks=sum(len(x['artifacts']) for x in checks.values()),
        original_necessary_checks=rules,qualified=False,economic_objective_achieved=False,
        all_actions_filled=True,nominal_action_anchor_rows=anchor_rows,nominal_anchor_groups=anchor_groups,
        oos_features_sha256=sha(features_path),oos_morning_rows=morning,oos_acquired_census=census,oos_reach_groups=reach,
        oos_drawdown_overlap=[x for x in census if x['episode_overlap']],
        limits=['Anchor groups are post-outcome explanation, not funded selective-policy returns.',
                'Morning reach uses actual N0 books. It does not reconstruct untraded outcomes or assert control and treatment path parity.',
                'Whole lifecycle PNL on drawdown-overlapping trades is not exact peak-to-trough currency attribution.',
                'All history reused; exchange queue and independent forward qualification remain unavailable.'])
    write(out/'assessment.json',result)
    print(json.dumps(dict(output=str(out),artifact_checks=result['artifact_checks'],necessary_checks=rules,
        anchor_groups=anchor_groups,reach=reach,morning_states=len(morning),drawdown_overlap_count=len(result['oos_drawdown_overlap']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();run(args.root)
