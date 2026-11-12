"""Finite population-calibration policies; authenticated forecasts, continuous clock."""
from collections import defaultdict
from copy import deepcopy
from datetime import date
import gzip
import json
from pathlib import Path

from backtests.strategies.kalcb import population_calibration_v1 as pc
from backtests.strategies.kalcb import population_policy_primitives_v1 as pp
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, digest, verify, PERIODS, POLICY

BASE = Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010')
ALIGNED = Path('E:/CodexResearch/k_stock_trader/kalcb_aligned_walk_forward_20261010/v3')
ALIGNMENT = Path('E:/CodexResearch/k_stock_trader/kalcb_selected_target_alignment_20261010/v2')
QUAL = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010')
OLD = Path('C:/Users/sehyu/.codex/research/kalcb_full_oos_recovery_20261009/v3')
PARENTS = ('expanding', 'expanding_atr5', 'decay60', 'decay60_atr5')
MODES = ('full', 'additions_only', 'removals_only')
RECIPES = {p+'_population_'+m: {'parent': p, 'family': p.split('_atr')[0],
    'atr': p.endswith('atr5'), 'mode': m, 'config': {}} for m in MODES for p in PARENTS}
FORECASTS = {'expanding': ALIGNED/'fit/forecasts/original_target_full_history',
    'decay60': Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_mechanisms_20261010/v1/fit/forecasts/original_decay60')}


def select_backout(mode, parent, origins, full, full_origins, records, others, proof, plan, source):
    """Do not change an incumbent's quantity/score in component-only tests.

    Removal means a supported nonpositive corrected value for its exact original
    action/origin. Rank replacement is not called a removal. Checkpoint can later
    admit the issuer independently; this is explicitly not a day-long veto.
    """
    if mode == 'full':
        return full, full_origins
    if mode == 'removals_only':
        selected = []
        for row, score in parent:
            key = row['opportunity_id']; origin = origins[key]
            descriptor = pc.descriptor(records[key], others[key], origin)
            corrected = pc.corrected_values(records[key], others[key], origin, proof)
            if not (pc.supported(descriptor, proof) and min(corrected.values()) <= 0):
                selected.append((row, score))
        return selected, {r['opportunity_id']: origins[r['opportunity_id']] for r, _ in selected}
    if mode != 'additions_only':
        raise ValueError('Unregistered population backout')
    incumbent = {r['base_opportunity_id'] for r, _ in parent}
    additions = [(r, v) for r, v in full if r['base_opportunity_id'] not in incumbent
        and r['features']['action_quantity_scale'] <= .5
        and pc.supported(pc.descriptor(records[r['opportunity_id']], others[r['opportunity_id']],
                                     full_origins[r['opportunity_id']]), proof)]
    seed = read(plan['seed'])['mutations']; q3 = read(plan['q3'])['mutations']
    selected, _ = pp.admit_actions(parent+additions,
        lambda row, rank: pp.ev.eligible(row, seed, q3, rank=rank), pp.config_for(plan), source,
        budget=8, priority_keys=[r['opportunity_id'] for r, _ in parent])
    parent_keys = [r['opportunity_id'] for r, _ in parent]
    actual = {r['opportunity_id']: (r, v) for r, v in selected}
    if any(k not in actual or actual[k] != pair for k, pair in zip(parent_keys, parent)):
        raise ValueError('Protected parent action displaced or repriced')
    return selected, {r['opportunity_id']: origins.get(r['opportunity_id'], full_origins.get(r['opportunity_id']))
                      for r, _ in selected}


def normalized(result):
    return {'snapshots': [s.to_json_dict() for s in result['snapshots'].values()],
            'updates': [s.to_json_dict() for ss in result['updates'].values() for s in ss],
            'admissions': result['admissions'], 'timed_admissions': result['timed_admissions']}


def load_policy(folder):
    m, snaps = pp.verify_policy(folder, POLICY)
    return {'snapshots': snaps, 'updates': pp.timed_policy(folder, m, snaps),
            'admissions': pp.rows(folder/(POLICY+'.admission.jsonl')),
            'timed_admissions': pp.rows(folder/'timed.admission.jsonl')}


def build(work):
    """Prelaunch build: fixed finite design, no funded performance selection."""
    work = Path(work); work.mkdir(parents=True, exist_ok=False)
    inputs = []
    def take(path):
        inputs.append(bound(path)); return read(path)
    previous = take(BASE/'v2/build/completion.json')
    verify(previous['policies']+[previous['control_bridge'], previous['calibration'], previous['lineage']])
    lineage = take(BASE/'v2/build/lineage.json'); verify(lineage)
    audit_path = BASE/'v2/build/decision_audit.jsonl.gz'
    observed = take(BASE/'v3/observational_review_20261010_1200/cohort_transfer.json')
    expected = [b for b in observed['inputs'] if Path(b['path']).name == audit_path.name]
    if not expected or digest(audit_path) != expected[0]['sha256']:
        raise ValueError('Unbound decision/label audit')
    inputs.append(bound(audit_path))
    with gzip.open(audit_path, 'rt', encoding='utf-8') as stream:
        audit = [json.loads(line) for line in stream]
    if len(audit) != previous['audit_rows']:
        raise ValueError('Truncated audit')
    lookup = {(r['family'], r['action_id']): r for r in audit}
    if len(lookup) != len(audit): raise ValueError('Duplicate audit action')
    records = [pc.historical_record(r, lookup[('decay60' if r['family']=='expanding' else 'expanding', r['action_id'])], origin)
               for r in audit for origin in pc.ORIGINS]
    calendar = take(ALIGNED/'fit/calendar.json'); sessions = calendar['sessions']
    scored = [s for s in sessions if PERIODS['is'][0] <= s <= PERIODS['oos'][1]]
    if len(scored)!=225 or sessions.index('2026-04-01')!=217:
        raise ValueError('Continuous history/calendar changed')
    original_proofs = take(BASE/'v2/build/calibration_proofs.json')
    fit_dates = [x['valid_from'] for x in original_proofs['expanding']]
    if fit_dates != scored[::5] or fit_dates != [x['valid_from'] for x in original_proofs['decay60']]:
        raise ValueError('Residual clock reset or drift')
    parent_profiles = take(QUAL/'v3/preparation.json')['profiles']
    profiles = {(p['name'],p['period']):p for p in parent_profiles}
    parents = {}; bases = {}
    for p in PARENTS:
        for period in PERIODS:
            item = profiles[(p,period)]; verify([item[k] for k in ('manifest','plan','policy')])
            inputs.extend(item[k] for k in ('manifest','plan','policy'))
            bases[(p,period)] = Path(item['policy']['path']).parent
            parents[(p,period)] = load_policy(bases[(p,period)])
    alignment = take(ALIGNMENT/'protocol.json')
    cells = {(c['period'],c['kind']): c for c in alignment['cells']}
    plans = {}; sources = {}; byday = {k:defaultdict(list) for k in ('opening','timed')}
    for key, cell in cells.items():
        prepared = Path(cell['prepared']); meta = take(prepared/'prepared.json')
        path = prepared/'opportunities.jsonl'; inputs.append(bound(path))
        if digest(path)!=meta['opportunities_sha256']: raise ValueError('Prepared candidate drift')
        plans[key] = take(cell['parent_plan']); sources[key] = digest(path)
        for row in pp.rows(path):
            if row['action']['ready'] and row['guard_at_rank1']:
                for q in pc.QUANTITIES:
                    byday[key[1]][row['trade_date']].append(pp.scaled_row(row,q))
    raw = {}
    for f, directory in FORECASTS.items():
        raw[f] = {}
        for kind in byday:
            path = directory/(kind+'.jsonl'); inputs.append(bound(path))
            # Authenticate forecast bytes against the earlier registered policy build.
            oldbind = next(b for b in read(BASE/'v2/protocol.json')['bindings'] if Path(b['path']).resolve()==path.resolve())
            verify([oldbind]); raw[f][kind] = {r['opportunity_id']:r for r in pp.rows(path)}
    results = {n:{p:pp.empty_result() for p in PERIODS} for n in RECIPES if not RECIPES[n]['atr']}
    controls = {f:{p:pp.empty_result() for p in PERIODS} for f in FORECASTS}
    proofs = {f:[] for f in FORECASTS}; fits = {}; changes = []; boundary = {}
    for index, day in enumerate(scored):
        period = 'is' if day <= PERIODS['is'][1] else 'oos'
        items = byday['opening'][day]; timed = byday['timed'][day]
        if index % 5 == 0:
            for f in FORECASTS:
                fits[f] = pc.fit_population(records, f, day, sessions); proofs[f].append(fits[f])
        for f in FORECASTS:
            other = 'decay60' if f=='expanding' else 'expanding'
            ar = {r['opportunity_id']:lookup[(f,r['opportunity_id'])] for r in items}
            others = {k:lookup[(other,k)] for k in ar}
            now = {k:raw[f]['opening'][k] for k in ar}
            core = {k:{**now[k],**{s+'_value_bps':r['calibrated_core'][s] for s in pc.SCENARIOS}} for k,r in ar.items()}
            support = {k:{**now[k],**{s+'_value_bps':r['calibrated_support'][s] for s in pc.SCENARIOS}} for k,r in ar.items()}
            for row in items:
                k=row['opportunity_id']
                if ar[k]['available_at']!=row['feature_available_at'] or ar[k]['capacity_pass']!=pp.capacity_ok(row):
                    raise ValueError('Decision timing/capacity mismatch')
                if ar[k]['raw'] != {s:now[k][s+'_value_bps'] for s in pc.SCENARIOS}:
                    raise ValueError('Raw forecast audit mismatch')
            parent, origins, _ = pp.choose_day(items,now,core,support,plans[(period,'opening')],sources[(period,'opening')])
            ds=date.fromisoformat(day)
            old_snap=parents[(f,period)]['snapshots'][ds]
            valid_from=old_snap.metadata['forecast']['valid_from'] if 'forecast' in old_snap.metadata else max((v['valid_from'] for v in now.values()),default='2026-04-01')
            # Exact original admission valid_from also covers sessions without actions.
            if not now:
                valid_from=next(first for first,last in calendar['windows'] if first<=day<=last)
            def package(chosen, chosen_origins):
                return pp.package_day(day,chosen,chosen_origins,timed,raw[f]['timed'],plans[(period,'timed')],
                    sources[(period,'opening')],sources[(period,'timed')],valid_from)
            control=package(parent,origins); pp.merge(controls[f][period],control)
            expected={'snapshots':{ds:old_snap},'updates':{ds:parents[(f,period)]['updates'][ds]} if ds in parents[(f,period)]['updates'] else {},
                'admissions':[r for r in parents[(f,period)]['admissions'] if r['trade_date']==day],
                'timed_admissions':[r for r in parents[(f,period)]['timed_admissions'] if r['trade_date']==day]}
            if normalized(control)!=normalized(expected):
                write(work/'bridge_failure.json',{'family':f,'day':day,'actual':normalized(control),'expected':normalized(expected)})
                raise ValueError('Original policy bridge mismatch '+f+' '+day)
            corrected=[]
            for origin, values in zip(pc.ORIGINS,(core,support)):
                corrected.append({k:{**values[k],**{s+'_value_bps':v for s,v in pc.corrected_values(ar[k],others[k],origin,fits[f]).items()}} for k in ar})
            full, fo, _ = pp.choose_day(items,now,*corrected,plans[(period,'opening')],sources[(period,'opening')])
            for mode in MODES:
                chosen, co = select_backout(mode,parent,origins,full,fo,ar,others,fits[f],plans[(period,'opening')],sources[(period,'opening')])
                name=f+'_population_'+mode; result=package(chosen,co); pp.merge(results[name][period],result)
                changes.append({'variant':name,'day':day,'parent':[{ 'action':r['opportunity_id'],'score':v,'origin':origins[r['opportunity_id']]} for r,v in parent],
                    'selected':[{'action':r['opportunity_id'],'score':v,'origin':co[r['opportunity_id']]} for r,v in chosen],
                    'calibration_valid_from':fits[f]['valid_from'],'base_valid_from':valid_from})
            if day=='2026-04-01':
                if valid_from!='2026-03-10' or fits[f]['valid_from']!='2026-03-31' or fits[f]['observed_through']!='2026-03-27':
                    raise ValueError('April learning state reset')
                boundary[f]={'passed':True,'history_sessions':217,'base_valid_from':valid_from,
                    'next_base_fit':'2026-04-07','residual_fit':fits[f],'first_day_candidate_count':len(items)}
        if index%20==0: print(json.dumps({'built_sessions':index+1,'through':day}),flush=True)
    bridges=[]
    for f in FORECASTS:
        for period in PERIODS:
            if normalized(controls[f][period])!=normalized(parents[(f,period)]):raise ValueError('Full control policy drift')
            bridges.append({'parent':f,'period':period,'passed':True,'sessions':PERIODS[period][2]})
    feature_bridge=take(OLD/'boundary_continuity.json')
    if not feature_bridge['passed'] or feature_bridge['errors'] or feature_bridge['feature_comparisons']!=305:
        raise ValueError('Feature continuity qualification incomplete')
    verify(list(feature_bridge['sources'].values())+[feature_bridge['program']])
    inputs.extend(feature_bridge['sources'].values()); inputs.append(feature_bridge['program'])
    history_audit=take(OLD/'history_boundary_audit.json')
    write(work/'input_bindings.json',inputs+lineage)
    write(work/'boundary_continuity.json',{'passed':True,'families':boundary,
        'feature_bridge':feature_bridge,'history_audit':history_audit})
    registration=work/'design.json'
    write(registration,{'recipes':RECIPES,'scope':'Finite prelaunch policy build; no portfolio-performance selection',
        'input_bindings':bound(work/'input_bindings.json'),'primitive':bound(Path(pc.__file__))})
    pp.jsonl(work/'selection_changes.jsonl',changes); write(work/'calibration_proofs.json',proofs)
    evidence={x['path']:x['sha256'] for x in inputs+lineage}
    for name,spec in RECIPES.items():
        f=spec['family']
        for period in PERIODS:
            folder=work/'policies'/name/period
            base=bases[(f,period)]; chronology=read(base/'chronology.json')
            if spec['atr']:
                pp.transform(work/'policies'/(f+'_population_'+spec['mode'])/period,folder,'atr_quantity_5pct')
            else:
                pp.save_period(folder,base,period,results[name][period],chronology['folds'],
                    chronology['calibration']+proofs[f],evidence,spec,registration)
    for f in FORECASTS:
        for period in PERIODS:
            folder=work/'control_atr'/f/period; pp.transform(bases[(f,period)],folder,'atr_quantity_5pct')
            for file in (POLICY+'.snapshots.json',POLICY+'.admission.jsonl','timed.snapshots.jsonl','timed.admission.jsonl'):
                if read_or_rows(folder/file)!=read_or_rows(bases[(f+'_atr5',period)]/file):raise ValueError('ATR control drift')
            bridges.append({'parent':f+'_atr5','period':period,'passed':True})
    write(work/'control_bridges.json',{'passed':True,'cases':bridges,'continuous_base_sessions':450})
    # Policies can share executable bytes while retaining different explanatory lineage.
    aliases={};seen={}
    for name in RECIPES:
        signature=tuple(digest(work/'policies'/name/p/fn) for p in PERIODS for fn in
            (POLICY+'.snapshots.json',POLICY+'.admission.jsonl','timed.snapshots.jsonl','timed.admission.jsonl'))
        if signature in seen:aliases[name]=seen[signature]
        else:seen[signature]=name
    verify(inputs+lineage)
    write(work/'completion.json',{'complete':True,'policies':[bound(work/'policies'/n/p/'policies_manifest.json') for n in RECIPES for p in PERIODS],
        'aliases':aliases,'control_bridge':bound(work/'control_bridges.json'),'boundary':bound(work/'boundary_continuity.json'),
        'inputs':bound(work/'input_bindings.json'),'calibration':bound(work/'calibration_proofs.json'),'decisions':bound(work/'selection_changes.jsonl')})


def read_or_rows(path):
    return pp.rows(path) if str(path).endswith('.jsonl') else read(path)
