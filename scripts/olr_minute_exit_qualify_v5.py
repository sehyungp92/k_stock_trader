"""Bounded matched-clock experiment; no replacement search or gate relaxation."""
from collections import Counter
from pathlib import Path
import copy,gzip,json,traceback
from scripts import olr_minute_exit_study_v5 as t
from scripts import olr_opening_risk_qualify_v2 as engine
from scripts import olr_targeted_risk_results_v1 as metrics
from scripts import olr_repair_qualify_v4 as repair
n,s,b=t.n,t.s,t.b
WINDOWS=('train','oos')
SCENARIOS=(('baseline',False,'windows'),('combined',False,'stress'),('baseline',True,'capped'))
PRIOR_RESULTS_SHA='be6d3a9f43d8671b556178eedc13dc4ad1124d1c54f2da21f30df18cc78c78d9'
_audit=engine.audit
_gates=engine.gates


def configure():
    t.activate();engine.t=t;engine.audit=audit


def references():
    path=t.SOURCE_OUT/'results.json'
    assert n.sha(path)==PRIOR_RESULTS_SHA
    old=n.read(path);control=old['candidates'][0]
    assert control['id']=='caa25a36d10de3dc4'
    for group in ('windows','stress','capped'):
        for w in WINDOWS:s.forensic.authenticate(Path(control[group][w]['folder']))
    return {**old['references'],'retained':control,'participation_results_sha256':PRIOR_RESULTS_SHA}


def audit(c,w,scenario='baseline',capped=False):
    proof=_audit(c,w,scenario,capped);folder=b.output(c,w,scenario,capped)
    native=n.read(folder/'execution_audit.json');clock=native['minute_clock_audit']
    assert clock['status']=='pass' and not clock['errors'] and clock['unchanged_thresholds']
    assert clock['control_or_treatment']==('treatment' if c['mutations'][t.FLAG] else 'clock_control')
    # Locate the native qualification details without assuming an outer label.
    def find(value):
        if not isinstance(value,dict):return []
        return ([value['minute_sources']] if 'minute_sources' in value else [])+[x for v in value.values() for x in find(v)]
    sources=find(native);assert len(sources)==1
    source=sources[0]
    assert all(source[k] for k in ('passed','origins_reparsed','minute_source_scope_complete','coarse_basis_projection_exact'))
    correction=source['closing_clock_corrections']
    assert correction['clock_corrections_verified'] and correction['corrected_sessions']==(99 if w=='oos' else 0)
    nav=n.read(folder/'minute_nav_evidence.json');result=n.read(folder/'result.json')
    assert abs(nav['final_equity']/1e8*100-100-result['return_pct'])<1e-8
    ready=n.read(t.ENGINES/w/'ready.json');key=scenario+('_capped' if capped else '')
    grid=Path(ready['files'])/ready['study']['minute_original_grids'][key]
    assert nav['original_grid_sha256']==n.sha(grid)
    rows=n.read(grid);assert [r['timestamp'] for r in nav['common_original_grid']]==[r[0] for r in rows]
    measured=metrics.episode([(r['timestamp'],r['equity']) for r in nav['common_original_grid']])
    assert abs(measured['drawdown_pct']-nav['common_original_drawdown_pct'])<1e-8
    assert nav['all_event_drawdown_pct']+1e-8>=nav['common_original_drawdown_pct']
    assert nav['risk_signals']==clock['recorded_signals']
    if not c['mutations'][t.FLAG]:assert nav['risk_signals']==0
    proof.update(minute_clock=clock,minute_source_audit=source,minute_nav_sha256=n.sha(folder/'minute_nav_evidence.json'),
        common_grid_sha256=n.sha(grid),fine_drawdown_pct=nav['all_event_drawdown_pct'],common_drawdown_pct=nav['common_original_drawdown_pct'])
    return proof


def describe(c,w,scenario,capped):
    folder=b.output(c,w,scenario,capped);row=metrics.describe(folder)
    nav=n.read(folder/'minute_nav_evidence.json');trades=n.read(folder/'trades.json')
    row.update(native_minute_close_drawdown_pct=row['drawdown_pct'],drawdown_pct=nav['common_original_drawdown_pct'],
        complete_fine_drawdown_pct=nav['all_event_drawdown_pct'],fine_drawdown_episode=nav['maximum'],
        common_intraday_drawdown=metrics.episode([(r['timestamp'],r['equity']) for r in nav['common_original_grid']]),
        common_grid_policy=nav['common_grid_policy'],fine_nav_points=nav['points'],fine_nav_events=nav['events'],
        entry_notional=sum(x['qty']*x['entry_price'] for x in trades),authentication=audit(c,w,scenario,capped))
    signals=[];events=Counter()
    with gzip.open(folder/'minute_events.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line);events[event['event']]+=1
            if event['event']=='minute_exit_signal_v5':signals.append(event)
    assert len(signals)==nav['risk_signals']
    row['mechanism_activation']=dict(signals=len(signals),events=dict(events),signal_details=signals)
    return row


def comparison(old,new):
    value=metrics.matched(Path(old['folder']),Path(new['folder']))
    key=lambda x:(x['symbol'],x['entry_decision_time'])
    for name in ('added','removed'):value[name].sort(key=key)
    a=n.read(Path(old['folder'])/'trades.json');z={key(x):x for x in n.read(Path(new['folder'])/'trades.json')}
    winners=[x for x in a if x['net_pnl']>0];winning_pnl=sum(x['net_pnl'] for x in winners)
    value.update(whole_old_winners=len(winners),removed_old_winners=sum(key(x) not in z for x in winners),
        whole_old_winner_pnl_retention_pct=100*sum(z.get(key(x),{}).get('net_pnl',0) for x in winners)/winning_pnl if winning_pnl else None,
        frequency_economics=repair.frequency_economics(new,old))
    return value


def gates(a,z,old,fine,refs,group):
    if group=='capped':
        ca,cz=old[group]['train'],old[group]['oos']
        checks=dict(is_selected_return=a['return_pct']>=ca['return_pct']-1.5,
            oos_return=z['return_pct']>=cz['return_pct'],oos_drawdown=z['drawdown_pct']<cz['drawdown_pct'])
    else:
        # Original, guard and old selected returns/DD remain mandatory.
        engine.t=t;checks=_gates(a,z,old,refs,group=='stress')
    if group in ('stress','capped'):
        checks.update(is_frequency=a['trades']>=old[group]['train']['trades'],
            oos_frequency=z['trades']>=old[group]['oos']['trades'],
            is_entry_days=a['entry_days']>=old[group]['train']['entry_days'],
            oos_entry_days=z['entry_days']>=old[group]['oos']['entry_days'])
    else:
        g=n.read(t.ROOT/'registration.json')['gates'];econ=repair.frequency_economics(z,old['windows']['oos'])
        checks.update(is_entry_days=a['entry_days']>=g['is_entry_days_min'],oos_entry_days=z['entry_days']>=g['oos_entry_days_min'],
            oos_meaningful_added_notional=econ['net_incremental_notional']>=1e6 and econ['median_added_notional']>=1e6)
    fa,fz=fine[group]['train'],fine[group]['oos']
    checks.update(matched_is_return=a['return_pct']>=fa['return_pct']-1.5,
        matched_oos_return=z['return_pct']>=fz['return_pct'],
        matched_oos_common_drawdown=z['drawdown_pct']<fz['drawdown_pct'],
        matched_oos_complete_fine_drawdown=z['complete_fine_drawdown_pct']<fz['complete_fine_drawdown_pct'],
        matched_is_frequency=a['trades']>=fa['trades'],matched_oos_frequency=z['trades']>=fz['trades'],
        matched_is_entry_days=a['entry_days']>=fa['entry_days'],matched_oos_entry_days=z['entry_days']>=fz['entry_days'])
    return checks


def risk_gates(checks):
    excluded={'is_frequency','oos_frequency','is_entry_days','oos_entry_days','oos_meaningful_added_notional',
        'matched_is_frequency','matched_oos_frequency','matched_is_entry_days','matched_oos_entry_days'}
    return {k:v for k,v in checks.items() if k not in excluded}


def collect():
    configure();refs=references();rows=[]
    for c in n.read(t.ROOT/'candidate_plan.json')['candidates']:
        row={k:c[k] for k in ('id','label','mutations')}
        for scenario,capped,group in SCENARIOS:
            row[group]={}
            for w in WINDOWS:
                folder=b.output(c,w,scenario,capped)
                if (folder/'assessment.json').exists():row[group][w]=describe(c,w,scenario,capped)
        rows.append(row)
    fine=rows[0]
    for row in rows:
        row['comparisons']={};row['scenario_gates']={};row['risk_component_gates']={}
        for _,_,group in SCENARIOS:
            if len(row[group])!=2 or len(fine[group])!=2:continue
            row['comparisons'][group]={w:{'against_retained':comparison(refs['retained'][group][w],row[group][w]),
                'against_fine_control':comparison(fine[group][w],row[group][w])} for w in WINDOWS}
            checks=gates(row[group]['train'],row[group]['oos'],refs['retained'],fine,refs,group)
            row['scenario_gates'][group]=checks;row['risk_component_gates'][group]=risk_gates(checks)
        row['full_objective_qualified']=len(row['scenario_gates'])==3 and all(all(v.values()) for v in row['scenario_gates'].values())
        row['risk_component_qualified']=len(row['risk_component_gates'])==3 and all(all(v.values()) for v in row['risk_component_gates'].values())
    # JSON canonicalization only; deterministic collection has no timestamp and
    # explicitly sorts only legacy set-derived added/removed lifecycle lists.
    return json.loads(json.dumps(dict(registration_sha256=n.sha(t.ROOT/'registration.json'),references=refs,candidates=rows,
        independent_validation=False,promotion=False,drawdown_gate_grid='common_original_grid',
        fine_control_improvement_required_on_both_grids=True),sort_keys=True))


def check_preflight():
    complete=n.read(t.ROOT/'engineering_complete.json');pre=n.read(t.ROOT/'preflight_complete.json')
    assert complete['no_performance_executed'] and complete['preflight_sha256']==n.sha(t.ROOT/'preflight_complete.json')
    assert complete['harness_sha256']==n.sha(t.ROOT/'engineering_controller_harness.json')
    assert pre['registration_sha256']==n.sha(t.ROOT/'registration.json') and pre['worker_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_preflight_v5.py')
    assert pre['source_hashes']==n.read(t.ROOT/'preparation_complete.json')['sources']
    assert len(pre['evidence'])==2
    for row in pre['evidence']:
        assert row['native_control_and_E_artifacts_exact'] and row['retained_entry_features_rank_size_rejections_exact']
        assert n.sha(row['entry_identity_path'])==row['entry_identity_sha256']
        identity=n.read(row['entry_identity_path'])
        assert identity['worker_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_entry_identity_v5.py')
        for name,record in identity['artifacts'].items():
            assert n.sha(name)==record['sha256'] and n.sha(Path(name).with_suffix('.receipt.json'))==record['receipt_sha256']
        assert n.sha(row['snapshot_path'])==row['snapshot_sha256']
        source=t.ROOT/f'native_source_preflight_{row["window"]}.json';assert n.sha(source)==row['source_preflight_sha256']
        assert n.read(source)['all_source_rows_recomputed']


def verify_harness():
    value=n.read(t.OUT/'research_harness.json')
    for name,digest in value['files'].items():assert n.sha(t.REPO/name)==digest,name
    for name,digest in value['artifacts'].items():assert n.sha(name)==digest,name
    tests=n.read(value['tests_receipt']);assert tests['passed']>=65 and tests['exit_code']==tests['failures']==tests['errors']==tests['skipped']==0
    assert n.sha(value['tests_receipt'])==value['tests_sha256']
    return value


def qualify():
    configure();check_preflight();verify_harness()
    assert not (t.ROOT/'qualification_complete.json').exists(),'Do not rerun completed collector'
    catalog=n.read(t.ROOT/'candidate_plan.json')['candidates'];assert len(catalog)==2
    for scenario,capped,_ in SCENARIOS:
        for c in catalog:
            for w in WINDOWS:
                verify_harness();print(dict(candidate=c['id'],window=w,scenario=scenario,capped=capped),flush=True)
                engine.evaluate(c,w,scenario,capped)
    data=collect();eligible=[r for r in data['candidates'][1:] if r['full_objective_qualified']]
    old=n.read(t.previous.OUT/'selected_research_policy.json')
    if eligible:
        chosen=eligible[0];selected=dict(selected_id=chosen['id'],selected_label=chosen['label'],mutations=chosen['mutations'],
            source_generation_hashes=n.read(t.ROOT/'preparation_complete.json')['sources'])
    else:selected={k:copy.deepcopy(old[k]) for k in ('selected_id','selected_label','mutations','source_generation_hashes')}
    n.write(t.OUT/'results.json',data)
    profile={**selected,'status':'qualified_research_successor' if eligible else 'retain_original_selected_no_full_upgrade',
        'results_sha256':n.sha(t.OUT/'results.json'),'registration_sha256':n.sha(t.ROOT/'registration.json'),
        'history_policy':'literal_carry','promotion':False,'independent_validation':False,
        'risk_component_ids':[r['id'] for r in data['candidates'][1:] if r['risk_component_qualified']],
        'retained_parent_profile_sha256':n.sha(t.previous.OUT/'selected_research_policy.json')}
    n.write(t.OUT/'selected_research_policy.json',profile)
    n.write(t.OUT/'qualified_alternatives.json',dict(full_objective=eligible,risk_components=[r for r in data['candidates'][1:] if r['risk_component_qualified']]))
    n.write(t.ROOT/'qualification_complete.json',dict(utc=s.utc(),selected_id=profile['selected_id'],results_sha256=profile['results_sha256'],
        profile_sha256=n.sha(t.OUT/'selected_research_policy.json')))
    print(profile,flush=True)


if __name__=='__main__':
    try:qualify()
    except Exception:
        n.write(t.ROOT/'qualification_failure.json',dict(utc=s.utc(),traceback=traceback.format_exc()));raise
