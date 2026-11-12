"""Read-only, independent collection of the bounded structural native cases."""
from collections import Counter
from datetime import date,datetime,timedelta
from pathlib import Path
from math import floor
import argparse,gzip,json
from scripts import olr_structural_native_v6 as experiment
from scripts import olr_minute_exit_qualify_v5 as previous_metrics
from scripts.olr_structural_v6 import write,sha,read,utc,REPO
n,s,b=experiment.n,experiment.s,experiment.b

def audit_case(candidate,window,scenario,capped):
    experiment.configure();folder=b.output(candidate,window,scenario,capped);s.forensic.authenticate(folder)
    ready=read(experiment.t.ENGINES/window/'ready.json');files=Path(ready['files'])
    panel=read(files/'research/structural_v6/source_panel.json')
    panel_hash=sha(files/'research/structural_v6/source_panel.json')
    native=read(folder/'execution_audit.json');result=read(folder/'result.json')
    assert native['audit_pass'] and result['execution_validity_pass'] and result['audit_status']=='passed'
    assert not result['terminal_inventory'] and not result['terminal_pending']
    clock=native['audit_receipt']['minute_clock_audit']
    assert clock['status']=='pass' and clock['unchanged_thresholds'] and not clock['errors'] and clock['recorded_signals']==0
    def source_audits(value):
        if not isinstance(value,dict):return []
        return ([value['minute_sources']] if 'minute_sources' in value else [])+[x for v in value.values() for x in source_audits(v)]
    sources=source_audits(native);assert len(sources)==1
    source=sources[0]
    assert all(source[k] for k in ('passed','origins_reparsed','minute_source_scope_complete','coarse_basis_projection_exact'))
    assert source['closing_clock_corrections']['clock_corrections_verified']
    assert source['closing_clock_corrections']['corrected_sessions']==(99 if window=='oos' else 0)
    projection=read(folder/'structural_projection_audit.json');requests=read(folder/'structural_request_audit.json')
    assert projection['passed'] and projection['rows_recomputed']==len(panel['rows']) and requests['passed']
    if window=='oos':
        previous=b.output(candidate,'train',scenario,capped);s.forensic.authenticate(previous)
        restart=read(folder/'capacity_restart_receipt.json');terminal=read(previous/'capacity_terminal_state.json')
        assert restart['parent_checkpoint_sha256']==sha(previous/'capacity_terminal_state.json')
        assert restart['parent_completion_sha256']==sha(previous/'completion.json')
        assert restart['restored_state']==terminal['capacity_history_v1'] and projection['actual_is_prefix_restored']
        assert terminal['structural_mode_v6']==candidate['structural_mode']
        fields=('day','symbol','prefix_volume','prefix_available_at','prefix_rows','prefix_complete','auction_volume','auction_available_at','auction_disposition')
        expected=[{k:r[k] for k in fields} for r in sorted(panel['rows'],key=lambda x:(x['day'],x['symbol'])) if r['day']<='2026-03-31']
        assert terminal['structural_observed_prefix_v6']==expected
    # Verify the exact source samples and forecast identity as well as quantity.
    # A coincidentally equal median from the wrong samples cannot pass.
    groups={symbol:[] for symbol in panel['universe']};lookup={(r['day'],r['symbol']):r for r in panel['rows']}
    for row in panel['rows']:
        if row['auction_volume'] is not None:groups[row['symbol']].append(row)
    for rows in groups.values():rows.sort(key=lambda r:r['day'])
    counts=Counter();bindings=Counter();records=[]
    with gzip.open(folder/'native_decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line);meta=event.get('metadata',{});forecast=meta.get('capacity_forecast');code=event['decision_code']
            if code.startswith('ENTRY_'):
                counts[event['reason']]+=1
                if meta.get('allocation_binding_limit'):bindings[meta['allocation_binding_limit']]+=1
                records.append({k:event[k] for k in ('timestamp','symbol','decision_code','reason','metadata')})
            if not forecast:continue
            evidence=forecast['exit'].get('structural_forecast')
            if candidate['structural_mode']=='off':assert evidence is None;continue
            assert evidence is not None and evidence['model']=='broad_median' and evidence['version']=='olr-causal-liquidity-v6'
            # Fills/expiry events retain their original decision forecast. A
            # next-session expiry must not relabel that forecast as current.
            symbol=event['symbol'];at=datetime.fromisoformat(forecast['asof']);day=at.date().isoformat()
            assert evidence['symbol']==symbol and evidence['day']==day and at.date().isoformat()==day
            assert evidence['available_at']==lookup[(day,symbol)]['prefix_available_at']
            assert datetime.fromisoformat(evidence['available_at'])<=at
            cutoff=(date.fromisoformat(day)-timedelta(days=60)).isoformat()
            rows=[r for r in groups[symbol] if cutoff<=r['day']<day][-20:]
            expected=[[(datetime.fromisoformat(r['auction_available_at'])-timedelta(minutes=5)).isoformat(),r['auction_volume']] for r in rows]
            assert forecast['exit']['samples']==expected and forecast['exit']['panel_sha256']==panel_hash
            assert evidence['own_observations']==len(rows) and evidence['latest_own_day']==max((r['day'] for r in rows),default=None)
            values=sorted(r['auction_volume'] for r in rows);length=len(values)
            median=(values[length//2] if length%2 else (values[length//2-1]+values[length//2])/2) if length else None
            assert evidence['predicted_volume']==median
            assert forecast['exit']['quantity']==(floor(median*forecast['participation']) if median is not None else 0)
    nav=read(folder/'minute_nav_evidence.json')
    original_grid=read(files/ready['study']['minute_original_grids'][scenario+('_capped' if capped else '')])
    assert [r['timestamp'] for r in nav['common_original_grid']]==[r[0] for r in original_grid]
    measured=experiment.metrics.episode([(r['timestamp'],r['equity']) for r in nav['common_original_grid']])
    assert abs(measured['drawdown_pct']-nav['common_original_drawdown_pct'])<1e-8
    trades=read(folder/'trades.json');assert len(trades)==len({(x['symbol'],x['entry_decision_time']) for x in trades})
    assert abs(sum(t['net_pnl'] for t in trades)/1e6-result['return_pct'])<1e-7
    row=experiment.describe(candidate,window,scenario,capped)
    row.update(unique_entries=len(trades),entry_days=len({t['entry_fill_time'][:10] for t in trades}),
        entry_notional=sum(t['qty']*t['entry_price'] for t in trades),admission_reasons=dict(counts),binding_limits=dict(bindings),
        completion_sha256=sha(folder/'completion.json'),all_exact_forecast_samples_verified=True,
        fine_drawdown_episode=nav['maximum'],source_authentication=source,
        actual_is_history_restored=window=='oos',original_grid_sha256=sha(files/ready['study']['minute_original_grids'][scenario+('_capped' if capped else '')]))
    return row,records

def collect():
    experiment.configure();experiment.verify_frozen();catalog=read(experiment.ROOT/'candidate_plan.json')['candidates']
    done=read(experiment.ROOT/'qualification_complete.json');rows=[]
    for c in catalog:
        row={k:c[k] for k in ('id','label','mutations','structural_mode')}
        for scenario,capped,group in experiment.GROUPS:
            row[group]={}
            for window in ('train','oos') if done['oos_evaluated'] else ('train',):
                row[group][window],_=audit_case(c,window,scenario,capped)
        rows.append(row)
    assert sha(experiment.PARENT_OUT/'results.json')=='883f3d76d8cf1a0fea0c22660f104c8466c8840623ca7934bd0f8590337a8415'
    parent=read(experiment.PARENT_OUT/'results.json');refs=parent['references'];control,migration=rows
    migration['comparisons']={};migration['scenario_gates']={}
    for _,_,group in experiment.GROUPS:
        migration['comparisons'][group]={w:previous_metrics.comparison(control[group][w],migration[group][w]) for w in migration[group]}
        # The original gate implementation reads t.ROOT/registration.gates. The
        # new protocol deliberately stores the same numeric gates under
        # prior_gates, so evaluate explicitly rather than rebinding that file.
        if 'oos' in migration[group]:
            a,z=migration[group]['train'],migration[group]['oos'];ca,cz=control[group]['train'],control[group]['oos'];old=refs['retained']
            registration=read(experiment.ROOT/'registration.json');g=registration['prior_gates']
            checks=dict(matched_is_return=a['return_pct']>=ca['return_pct']-1.5,matched_oos_return=z['return_pct']>=cz['return_pct'],
                matched_oos_common_drawdown=z['drawdown_pct']<cz['drawdown_pct'],matched_oos_complete_fine_drawdown=z['complete_fine_drawdown_pct']<cz['complete_fine_drawdown_pct'],
                matched_is_frequency=a['trades']>=ca['trades'],matched_oos_frequency=z['trades']>=cz['trades'],
                matched_is_entry_days=a['entry_days']>=ca['entry_days'],matched_oos_entry_days=z['entry_days']>=cz['entry_days'],
                is_selected_return=a['return_pct']>=old[group]['train']['return_pct']-g['is_return_loss_max_pp_selected'])
            if group=='capped':checks.update(oos_return=z['return_pct']>=old[group]['oos']['return_pct'],oos_drawdown=z['drawdown_pct']<old[group]['oos']['drawdown_pct'])
            else:
                original=refs['original_stress' if group=='stress' else 'original']['train'];guard=refs['guard'][group]['train']
                checks.update(is_original_return=a['return_pct']>=original['return_pct']-g['is_return_loss_max_pp_original'],
                    is_guard_return=a['return_pct']>=guard['return_pct']-g['is_return_loss_max_pp_guard'],is_guard_drawdown=a['drawdown_pct']<=guard['drawdown_pct']+g['is_dd_increase_max_pp_guard'],
                    oos_return=z['return_pct']>=old[group]['oos']['return_pct']+(g['oos_return_improvement_min_pp_selected'] if group=='windows' else 0),
                    oos_drawdown=z['drawdown_pct']<=old[group]['oos']['drawdown_pct']-g['oos_dd_reduction_min_pp_selected'] if group=='windows' else z['drawdown_pct']<old[group]['oos']['drawdown_pct'])
            if group=='windows':
                economic=previous_metrics.repair.frequency_economics(z,old[group]['oos'])
                checks.update(is_frequency=a['trades']>=g['is_trades_min'],oos_frequency=z['trades']>=g['oos_trades_min'],
                    is_entry_days=a['entry_days']>=g['is_entry_days_min'],oos_entry_days=z['entry_days']>=g['oos_entry_days_min'],
                    oos_meaningful_added_notional=economic['net_incremental_notional']>=1e6 and economic['median_added_notional']>=1e6)
            else:checks.update(is_frequency=a['trades']>=old[group]['train']['trades'],oos_frequency=z['trades']>=old[group]['oos']['trades'],is_entry_days=a['entry_days']>=old[group]['train']['entry_days'],oos_entry_days=z['entry_days']>=old[group]['oos']['entry_days'])
            migration['scenario_gates'][group]=checks
    migration['full_objective_qualified']=len(migration['scenario_gates'])==3 and all(all(x.values()) for x in migration['scenario_gates'].values())
    return dict(version='structural-native-results-v6',registration_sha256=sha(experiment.ROOT/'registration.json'),
        candidates=rows,references=refs,completed_cases=done['cases'],oos_evaluated=done['oos_evaluated'],
        training_decision=read(experiment.OUT/'training_decision.json'),
        selection='retain_original_pending_broader_alpha_program',selected_id='c4f49be01ed09b8dc',promotion=False,independent_validation=False)

if __name__=='__main__':
    result=collect();write(experiment.OUT/'results.json',result)
    write(experiment.OUT/'results_collection_receipt.json',dict(utc=utc(),results_sha256=sha(experiment.OUT/'results.json'),collector_sha256=sha(__file__),
        exact_native_sample_audit=True,old_collectors_rerun=False))
    print(dict(cases=result['completed_cases'],oos_evaluated=result['oos_evaluated'],full_objective=result['candidates'][1]['full_objective_qualified']))
