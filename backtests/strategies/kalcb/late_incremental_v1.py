"""Registered target/population and uncertainty components for the SAME q.25 action.

Outcome units are net basis points per planned notional, never rescaled funded
portfolio PnL. Eligibility is known snapshot membership, not subsequent fills.
"""
from collections import defaultdict
from pathlib import Path
import json
import math
import numpy as np
from backtests.strategies.kalcb import action_value as av, executed_value as ev
from backtests.strategies.kalcb.late_signal_v1 import (
    SCENARIOS, COMPACT_COLUMNS, CompactPayoff, aware, affine_fit, affine_predict)
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, digest, link

KINDS = ('aligned48', 'reserve', 'aligned_reserve')
RULE = {
    'target': 'exact q0.25 full-cost net equity bps / decision-time planned notional-to-equity',
    'learner': 'same16inputs_Ridge120_expanding60mature_original20clock_oneSessionEmbargo',
    'calibration': 'known-new-issuer only; identity prior100; slope0..1; 20 issuer-days / 8 sessions',
    'reserve': 'max(0, mean(session mean positive-forecast overprediction) + t(.95,D-1)*sd(session means)/sqrt(D))',
    'reserve_population': 'earlier genuinely forward positive baseline AND combined forecasts; known new issuer; physical/capacity pass; mature exact-quarter outcomes',
    'reserve_support': {'issuer_days': 20, 'sessions': 8},
    'unsupported': 'abstain, including unknown pre-IS parent membership',
    'ranking': 'inherited min baseline/combined expected equity bps; reserve is admission-only',
    'interpretation': 'session-cluster mean-error sensitivity reserve, not a calibrated individual prediction interval or independent confidence claim',
    'unchanged': 'q.25 native action, completed60minutes, full costs, management, physical gates and ONE checkpoint success',
}


def fraction(row):
    value = float(row['features']['action_notional_equity'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid decision-time planned notional')
    return value


def aligned_fit(rows, labels, sessions, first, through):
    if any(not r['opportunity_id'].endswith('/q0.25') for r in rows):
        raise ValueError('Only independently replayed quarter labels are valid')
    ix, weights = av.training_indices(rows, labels, sessions, first, av.ModelSpec('incremental'))
    train = [rows[i] for i in ix]
    if len({r['opportunity_id'] for r in train}) != len(train):
        raise ValueError('Dependent duplicate target')
    y = np.array([float(labels[r['opportunity_id']]['net_value_equity_bps']) / fraction(r) for r in train])
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite full-cost target')
    model = CompactPayoff().fit(train, y, weights)
    valid = [r for r in rows if first <= r['trade_date'] <= through]
    proof = dict(name='incremental_notional_ridge120', window='expanding', valid_from=first,
        valid_through=through, fit_at=first+'T00:00:00+09:00',
        train_from=min(r['trade_date'] for r in train), train_through=max(r['trade_date'] for r in train),
        label_available_through=max(labels[r['opportunity_id']]['label_available_at'] for r in train),
        train_rows=len(train), train_stock_days=len(train), feature_columns=COMPACT_COLUMNS,
        training_keys_hash=ev.signature([r['opportunity_id'] for r in train]),
        training_target_hash=ev.signature(y.tolist()), embargo_sessions=1,
        minimum_training_sessions=60, quantity_scale=.25, target_units='net_bps_per_planned_notional')
    return model, list(zip(valid, map(float, model.predict(valid)))), proof


def mature_points(records, labels, sessions, first, scenario, eligibility, rows, *, positive=False):
    if scenario not in SCENARIOS or sessions != sorted(set(sessions)) or first not in sessions:
        raise ValueError('Invalid scenario or calendar')
    allowed = set(sessions[:max(0, sessions.index(first)-1)])
    cutoff = aware(first+'T00:00:00+09:00')
    seen = set(); points = []
    for r in records:
        key = r['opportunity_id']; day = key[:10]
        if day not in allowed:
            continue
        if key in seen or not key.endswith('/q0.25') or r['valid_from'] > day:
            raise ValueError('Duplicate, wrong quantity or non-forward forecast')
        seen.add(key)
        if eligibility.get(key) is not True or not r.get('calibration_supported', True):
            continue
        if r['failure_probability'] >= .001 or not r['capacity_pass']:
            continue
        if positive and min(r[s] for s in SCENARIOS) <= 0:
            continue
        label = labels[key]
        if not label['label_valid'] or aware(label['label_available_at']) >= cutoff:
            continue
        if aware(label['label_available_at']) < aware(r['feature_available_at']):
            raise ValueError('Outcome precedes decision')
        denom = fraction(rows[key])
        x = float(r[scenario]); y = float(label['net_value_equity_bps']) / denom
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Nonfinite incremental calibration')
        points.append((key, x, y, label['label_available_at']))
    return points


def population_map(records, labels, sessions, first, scenario, eligibility, rows):
    points = mature_points(records, labels, sessions, first, scenario, eligibility, rows)
    # Reuse the exact identity-prior affine estimator, supplying converted units.
    rr = [dict(opportunity_id=k, valid_from=k[:10], feature_available_at=rows[k]['feature_available_at'],
               failure_probability=0., capacity_pass=True, **{scenario:x}) for k,x,y,t in points]
    ll = {k:dict(label_valid=True,label_available_at=t,net_value_equity_bps=y) for k,x,y,t in points}
    proof = affine_fit(rr, ll, sessions, first, scenario)
    proof.update(kind='eligible_new_issuer_notional_affine100', units='net_bps_per_planned_notional',
                 eligible_training_pairs_hash=ev.signature(points))
    return proof


def reserve_fit(records, labels, sessions, first, scenario, eligibility, rows):
    from scipy.stats import t
    points = mature_points(records, labels, sessions, first, scenario, eligibility, rows, positive=True)
    daily = defaultdict(list)
    for key,x,y,_ in points:
        daily[key[:10]].append(x-y)
    values = np.array([np.mean(v) for _,v in sorted(daily.items())])
    n, d = len(points), len(values)
    supported = n >= 20 and d >= 8
    mean = float(values.mean()) if d else None
    se = float(values.std(ddof=1) / math.sqrt(d)) if d >= 2 else None
    reserve = max(0., mean + float(t.ppf(.95,d-1))*se) if supported else None
    return dict(kind='positive_new_issuer_session_mean_error_reserve', valid_from=first, scenario=scenario,
        supported=supported, issuer_days=n, sessions=d, minimum_issuer_days=20, minimum_sessions=8,
        mean_error=mean, standard_error=se, reserve_bps=reserve, units='net_bps_per_planned_notional',
        observed_through=sessions[sessions.index(first)-2] if sessions.index(first)>=2 else None,
        embargo_sessions=1, label_available_through=max((p[3] for p in points), default=None),
        training_keys_hash=ev.signature([p[0] for p in points]), training_pairs_hash=ev.signature(points),
        session_errors_hash=ev.signature(sorted((k,float(np.mean(v))) for k,v in daily.items())))


def eligibility_maps(protocol, rows):
    from scripts.kalcb_executed_value_replay import verify_policy
    from backtests.strategies.kalcb.late_entry_v1 import cumulative_timed_policy
    output = {}; proofs = {}
    for prior in (100,125):
        maps = {}; bindings = []
        for period in ('is','oos'):
            profile = protocol['profiles'][str(prior)][period]
            source = Path(profile['policy']['path']).parent
            m, opening = verify_policy(source,'interaction_cost_margin')
            updates = cumulative_timed_policy(source,m,opening)
            for day,base in opening.items():
                latest = updates.get(day,[])
                maps[str(day)] = (latest[-1] if latest else base)
            bindings.append(profile['policy'])
        eligible = {}
        for r in rows:
            key = r['opportunity_id']; snap = maps.get(r['trade_date'])
            if snap is None:
                continue  # Unknown pre-IS membership is NOT inferred eligible.
            if snap.generated_at > aware(r['feature_available_at']):
                raise ValueError('Future parent membership')
            eligible[key] = r['symbol'] not in snap.by_symbol()
        output[prior] = eligible
        proofs[str(prior)] = dict(parent_policies=bindings, mapping=eligible,
            mapping_hash=ev.signature(eligible), unknown_pre_IS_abstains=True,
            scope='membership known before minute60; not a broker-position/funding/future-fill filter')
    return output, proofs


def produce(root):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    p = read(root/'protocol.json'); verify(p['programs']+p['input_bindings'])
    if p['incremental_rule'] != RULE:
        raise ValueError('Unregistered formula changes')
    old = Path(p['label_root']); sessions = read(old/'prepared/prepared.json')['sessions']
    rows = [scaled_row(r,.25) for r in ev.read_jsonl(old/'prepared/opportunities.jsonl')
            if r['action']['ready'] and r['guard_at_rank1']]
    by_key = {r['opportunity_id']:r for r in rows}
    if len(rows)!=len(by_key): raise ValueError('Duplicate causal actions')
    labels = {s:{r['opportunity_id']:r for f in sorted((old/'labels'/s).glob('*.jsonl'))
        for r in ev.read_jsonl(f)} for s in SCENARIOS}
    inherited = Path(p['signal_root'])/'models'
    verify([{'path':str(inherited/f),'sha256':h} for f,h in read(inherited/'completion.json')['files'].items()])
    existing = ev.read_jsonl(inherited/'compact_affine.jsonl')
    risk = ev.read_jsonl(inherited/'compact.jsonl')
    for source in (existing,risk):
        for r in source:
            row=by_key[r['opportunity_id']]
            r.update(feature_available_at=row['feature_available_at'],capacity_pass=capacity_ok(row))
    # Old forecasts are converted, not retrained; this is the reserve-only backout.
    existing=[{**r,**{s:float(r[s])/fraction(by_key[r['opportunity_id']]) for s in SCENARIOS}} for r in existing]
    eligible, eligibility_proof = eligibility_maps(p,rows)
    target=root/'models';target.mkdir(parents=True,exist_ok=False)
    write(target/'eligibility.json',eligibility_proof)
    for f in inherited.iterdir():
        if f.is_file():link(f,target/(f.name if f.suffix=='.joblib' else 'inherited_'+f.name))
    forecasts={f'{prior}_{kind}':[] for prior in (100,125) for kind in KINDS}
    raw_history=[]; aligned_history={prior:[] for prior in (100,125)}; proofs=[]; skipped=[]
    with threadpool_limits(limits=1):
        for first,through in ev.outer_windows(sessions,min_train=60,block=20):
            current=[r for r in risk if first<=r['opportunity_id'][:10]<=through]
            if not current:
                skipped.append(dict(first=first,through=through,reason='inherited risk fold unavailable'));continue
            fits={s:aligned_fit(rows,labels[s],sessions,first,through) for s in SCENARIOS}
            scores={s:{r['opportunity_id']:v for r,v in fits[s][1]} for s in SCENARIOS}
            for s,(model,_,proof) in fits.items():
                file=target/f'incremental_{first}_{s}.joblib';joblib.dump(model,file)
                proof.update(scenario=s,model_path=file.name,model_sha256=digest(file));proofs.append(proof)
            new=[{**r,**{s:scores[s][r['opportunity_id']] for s in SCENARIOS}} for r in current]
            for prior in (100,125):
                maps={s:population_map(raw_history,labels[s],sessions,first,s,eligible[prior],by_key) for s in SCENARIOS}
                reserves={kind:{s:reserve_fit(history,labels[s],sessions,first,s,eligible[prior],by_key) for s in SCENARIOS}
                    for kind,history in [('reserve',existing),('aligned_reserve',aligned_history[prior])]}
                for proof in [*maps.values(),*(v for q in reserves.values() for v in q.values())]:
                    proof.update(prior=prior,valid_through=through);proofs.append(proof)
                calibrated=[]
                for r in new:
                    supported=all(m['supported'] for m in maps.values())
                    calibrated.append({**r,'calibration_supported':supported,
                        **({s:affine_predict(r[s],maps[s]) for s in SCENARIOS} if supported else {})})
                for kind in KINDS:
                    source=([r for r in existing if first<=r['opportunity_id'][:10]<=through]
                            if kind=='reserve' else calibrated)
                    rs=reserves.get(kind)
                    for r in source:
                        support=r['calibration_supported'] and (rs is None or all(v['supported'] for v in rs.values()))
                        q=fraction(by_key[r['opportunity_id']]);key=r['opportunity_id']
                        forecasts[f'{prior}_{kind}'].append({**r,'calibration_supported':support,
                            'population_eligible':eligible[prior].get(key) is True,
                            'prior':prior,'signal_kind':kind,'notional_forecast':{s:r[s] for s in SCENARIOS},
                            'reserve_bps':{s:48. if rs is None else rs[s]['reserve_bps'] for s in SCENARIOS},
                            **{s:r[s]*q for s in SCENARIOS}})
                aligned_history[prior].extend(calibrated)
            raw_history.extend(new)
            print(json.dumps(dict(stage='incremental_models',first=first,through=through,rows=len(new))),flush=True)
    for name,values in forecasts.items():
        with (target/(name+'.jsonl')).open('x',encoding='utf-8') as f:
            for row in values:f.write(json.dumps(row,allow_nan=False)+'\n')
    chronology=read(inherited/'chronology.json')
    chronology['folds'] += [v for v in proofs if 'model_path' in v]
    chronology.setdefault('calibration',[]).extend({**v,'latest_label_available_at':v['label_available_through']}
        for v in proofs if 'kind' in v)
    chronology.update(skipped_incremental_folds=skipped,incremental_rule=RULE)
    write(target/'chronology.json',chronology)
    write(target/'completion.json',dict(complete=True,files={f.name:digest(f) for f in target.iterdir() if f.is_file()}))
    build_policies(root,rows,forecasts)


def admitted(score):
    return (score['calibration_supported'] and score['population_eligible'] and
        score['failure_probability']<.001 and score['capacity_pass'] and
        all(score['notional_forecast'][s] > score['reserve_bps'][s] for s in SCENARIOS))


def build_policies(root,rows,forecasts):
    from scripts.kalcb_executed_value_replay import verify_policy
    from scripts.kalcb_executed_value_study import config_for
    from backtests.strategies.kalcb.late_policy_v2 import select_executable
    from backtests.strategies.kalcb.late_entry_v1 import cumulative_timed_policy
    p=read(root/'protocol.json');by_key={r['opportunity_id']:r for r in rows}
    models=read(root/'models/completion.json');aliases=defaultdict(list)
    for name,recipe in p['recipes'].items():
        daily=defaultdict(list)
        for score in forecasts.get(f"{recipe['prior']}_{recipe['signal']}",[]):
            if admitted(score):daily[score['opportunity_id'][:10]].append((by_key[score['opportunity_id']],score))
        for period in ('is','oos'):
            old=p['profiles'][str(recipe['prior'])][period];source=Path(old['policy']['path']).parent
            m,opening=verify_policy(source,'interaction_cost_margin');updates=cumulative_timed_policy(source,m,opening)
            target=root/'policies'/name/period;target.mkdir(parents=True,exist_ok=False)
            for f in m['files']:link(source/f,target/f)
            timed=[];admissions=ev.read_jsonl(source/m['timed_admission_file']);added=[]
            cfg=config_for(read(old['plan']['path']))
            for day,base in sorted(opening.items()):
                parents=updates.get(day,[]);timed.extend(s.to_json_dict() for s in parents)
                latest=parents[-1] if parents else base
                choices=[(r,s) for r,s in daily[str(day)] if r['symbol'] not in latest.by_symbol()]
                selected=select_executable(choices,latest,cfg,False) if choices else None
                if selected:
                    row,score,value,snap=selected;timed.append(snap.to_json_dict())
                    added.append(dict(trade_date=str(day),symbol=row['symbol'],available_at=row['feature_available_at'],
                        score=value,action_id=row['opportunity_id'],quantity_scale=.25,valid_from=score['valid_from'],
                        flat_only=False,signal_kind=recipe['signal'],notional_forecast=score['notional_forecast'],
                        reserve_bps=score['reserve_bps']))
            for filename,values in [('later60.snapshots.jsonl',timed),('later60.admission.jsonl',admissions+added),('later60.new.jsonl',added)]:
                with (target/filename).open('x',encoding='utf-8') as f:
                    for row in values:f.write(json.dumps(row,default=str,allow_nan=False)+'\n')
            chronology=read(source/'chronology.json');new=read(root/'models/chronology.json')
            chronology['folds']+=new['folds'];chronology.setdefault('calibration',[]).extend(new['calibration'])
            (target/'chronology.json').unlink();write(target/'chronology.json',chronology)
            for f in models['files']:link(root/'models'/f,target/(f if f.endswith('.joblib') else 'later_model_'+f))
            write(target/'later_lineage.json',dict(registration=bound(root/'protocol.json'),preparation=bound(root/'prepared/prepared.json'),
                models=bound(root/'models/completion.json'),label_manifests=[bound(Path(p['label_root'])/'labels'/s/'completion.json') for s in SCENARIOS],
                parent_policy=old['policy'],parent_snapshots_preserved=True,recipe=recipe,incremental_rule=RULE,
                action_unchanged='same native completed60-minute quarter action, management, costs and source generation'))
            m.update(version='kalcb-first-hour-incremental-v1',observation_minutes=60,timed_snapshot_file='later60.snapshots.jsonl',
                timed_admission_file='later60.admission.jsonl',contract={**m['contract'],'timed_budget':13,'new_budget':1},
                later_admission_rule=RULE,files={f.name:digest(f) for f in target.iterdir() if f.is_file()})
            write(target/'policies_manifest.json',m);write(target/'completion.json',dict(complete=True))
            actual,ss=verify_policy(target,'interaction_cost_margin');cumulative_timed_policy(target,actual,ss)
            signature=ev.signature(dict(opening=[s.to_json_dict() for s in opening.values()],timed=timed))
            aliases[(period,signature)].append(name)
            print(json.dumps(dict(stage='incremental_policy',name=name,period=period,new_admissions=len(added))),flush=True)
    write(root/'policies/completion.json',dict(complete=True,independent_validation=False,
        effective_aliases=[dict(period=p,signature=s,names=v) for (p,s),v in aliases.items() if len(v)>1]))
