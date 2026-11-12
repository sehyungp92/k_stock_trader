"""Finite, causal removal-only calibration neighborhoods using shared primitives."""
from collections import defaultdict
from datetime import date
import gzip
import json
from pathlib import Path

from backtests.strategies.kalcb import population_calibration_v1 as pc
from backtests.strategies.kalcb import population_policy_primitives_v1 as pp
from backtests.strategies.kalcb.population_policy_v1 import (
    select_backout, normalized, load_policy, ALIGNMENT, ALIGNED)
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, digest, verify, PERIODS, POLICY


def build(out, registration):
    out, registration = Path(out), Path(registration)
    protocol = read(registration)
    verify(protocol['bindings']+protocol['programs'])
    old_root = Path(protocol['screening_root'])
    old_protocol = read(old_root/'protocol.json')
    verify(old_protocol['bindings']+old_protocol['programs'])
    prior_build = Path(old_protocol['build'])
    priors = protocol['calibration_priors']
    if priors != [75., 100., 125.]:
        raise ValueError('Undeclared finite prior neighborhood')
    out.mkdir(parents=True, exist_ok=False)
    dependencies = read(prior_build/'input_bindings.json')
    verify(dependencies)
    audit_path = old_root.parent/'v2/build/decision_audit.jsonl.gz'
    expected = [x for x in dependencies if Path(x['path']).resolve() == audit_path.resolve()]
    if not expected: raise ValueError('Unbound action/label audit')
    verify(expected)
    with gzip.open(audit_path, 'rt', encoding='utf-8') as stream:
        audit = [json.loads(line) for line in stream]
    lookup = {(r['family'], r['action_id']): r for r in audit}
    if len(lookup) != len(audit): raise ValueError('Duplicate action audit')
    history = [pc.historical_record(r, lookup['decay60', r['action_id']], origin)
               for r in audit if r['family'] == 'expanding' for origin in pc.ORIGINS]
    calendar = read(ALIGNED/'fit/calendar.json')['sessions']
    old_fits = read(prior_build/'calibration_proofs.json')['expanding']
    scored = [d for d in calendar if PERIODS['is'][0] <= d <= PERIODS['oos'][1]]
    if [x['valid_from'] for x in old_fits] != scored[::5] or calendar.index('2026-04-01') != 217:
        raise ValueError('Continuous fitting clock drift')
    changes = {r['day']: r for r in pp.rows(prior_build/'selection_changes.jsonl')
               if r['variant'] == 'expanding_population_removals_only'}
    if set(changes) != set(scored): raise ValueError('Missing original parent decisions')
    plans, sources, rows = {}, {}, {'opening': defaultdict(dict), 'timed': defaultdict(list)}
    for cell in read(ALIGNMENT/'protocol.json')['cells']:
        period, kind = cell['period'], cell['kind']
        path = Path(cell['prepared'])/'opportunities.jsonl'
        meta = read(Path(cell['prepared'])/'prepared.json')
        if digest(path) != meta['opportunities_sha256']: raise ValueError('Changed candidate source')
        plans[period, kind], sources[period, kind] = read(cell['parent_plan']), digest(path)
        for row in pp.rows(path):
            if row['action']['ready'] and row['guard_at_rank1']:
                for quantity in pc.QUANTITIES:
                    action = pp.scaled_row(row, quantity)
                    if kind == 'opening': rows[kind][row['trade_date']][action['opportunity_id']] = action
                    else: rows[kind][row['trade_date']].append(action)
    forecast_path = ALIGNED/'fit/forecasts/original_target_full_history/timed.jsonl'
    forecast = {r['opportunity_id']: r for r in pp.rows(forecast_path)}
    profiles = {(p['name'],p['period']):p for p in old_protocol['parent_profiles']}
    parents = {p:Path(profiles['expanding',p]['policy']['path']).parent for p in PERIODS}
    original = {p:load_policy(parents[p]) for p in PERIODS}
    outputs = {(prior,p):pp.empty_result() for prior in priors for p in PERIODS}
    rebuilt_parent = {p:pp.empty_result() for p in PERIODS}
    fits, proofs, decisions = {}, {prior:[] for prior in priors}, []
    for index, day in enumerate(scored):
        period = 'is' if day <= PERIODS['is'][1] else 'oos'
        change = changes[day]
        parent = [(rows['opening'][day][x['action']],x['score']) for x in change['parent']]
        origins = {x['action']:x['origin'] for x in change['parent']}
        records = {r['opportunity_id']:lookup['expanding',r['opportunity_id']] for r,_ in parent}
        others = {k:lookup['decay60',k] for k in records}
        def package(selected, origin):
            return pp.package_day(day, selected, origin, rows['timed'][day], forecast,
                plans[period,'timed'], sources[period,'opening'], sources[period,'timed'], change['base_valid_from'])
        pp.merge(rebuilt_parent[period],package(parent,origins))
        if index % 5 == 0:
            for prior in priors:
                fits[prior] = pc.fit_population(history,'expanding',day,calendar,prior=prior)
                proofs[prior].append(fits[prior])
            if fits[100.] != old_fits[index//5]: raise ValueError('Center calibration proof drift')
        for prior in priors:
            selected, origin = select_backout('removals_only',parent,origins,[],{},records,others,
                                              fits[prior],None,None)
            pp.merge(outputs[prior,period],package(selected,origin))
            decisions.append({'prior':prior,'day':day,'parent_actions':[r['opportunity_id'] for r,_ in parent],
                              'retained_actions':[r['opportunity_id'] for r,_ in selected],
                              'calibration_valid_from':fits[prior]['valid_from']})
        if index % 25 == 0: print(json.dumps({'neighborhood_sessions':index+1,'through':day}),flush=True)
    for p in PERIODS:
        if normalized(rebuilt_parent[p]) != normalized(original[p]):
            raise ValueError('Unchanged parent reconstruction drift')
        center = load_policy(prior_build/'policies/expanding_population_removals_only'/p)
        if normalized(outputs[100.,p]) != normalized(center):
            raise ValueError('Center removal policy drift')
    write(out/'proofs.json',proofs)
    pp.jsonl(out/'decisions.jsonl',decisions)
    evidence = {b['path']:b['sha256'] for b in dependencies+protocol['bindings']}
    policies, bridges = [], []
    for prior in priors:
        for p in PERIODS:
            chronology = read(parents[p]/'chronology.json')
            name = 'expanding_removal_prior'+str(int(prior))
            folder = out/'policies'/name/p
            pp.save_period(folder,parents[p],p,outputs[prior,p],chronology['folds'],
                           chronology['calibration']+proofs[prior],evidence,
                           {'parent':'expanding','mode':'removals_only','prior':prior,'config':{}},registration)
            atr = out/'policies'/('expanding_atr5_removal_prior'+str(int(prior)))/p
            pp.transform(folder,atr,'atr_quantity_5pct')
            policies.extend([bound(folder/'policies_manifest.json'),bound(atr/'policies_manifest.json')])
            if prior == 100.:
                center = load_policy(prior_build/'policies/expanding_atr5_population_removals_only'/p)
                if normalized(load_policy(atr)) != normalized(center): raise ValueError('ATR center drift')
                bridges.append({'period':p,'parent_and_both_centers':True,'sessions':len(outputs[prior,p]['snapshots'])})
    # Alias identical executable policies explicitly; retain documentary variants.
    aliases, signatures = {}, {}
    for parent in ('expanding','expanding_atr5'):
        for prior in (100.,75.,125.):
            name = parent+'_removal_prior'+str(int(prior))
            sig = tuple(digest(out/'policies'/name/p/f) for p in PERIODS for f in
                        (POLICY+'.snapshots.json',POLICY+'.admission.jsonl','timed.snapshots.jsonl','timed.admission.jsonl'))
            if sig in signatures: aliases[name]=signatures[sig]
            else: signatures[sig]=name
    verify(protocol['bindings']+protocol['programs']+dependencies)
    write(out/'completion.json',{'complete':True,'policies':policies,'aliases':aliases,'bridges':bridges,
          'proofs':bound(out/'proofs.json'),'decisions':bound(out/'decisions.jsonl'),
          'scope':'Fresh mature-label calibration at priors75/100/125 on original clocks; center policies exactly bridge. No native performance has been used by this builder.'})
