"""Finite first-hour signal ablation; no execution or optimizer lifecycle.

Exact quarter-quantity targets are reused because the native action is unchanged.
Calibration sees only earlier, genuinely forward forecasts and mature labels.
"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import numpy as np

from backtests.strategies.kalcb import action_value as av, continuation_value as cv, executed_value as ev
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, digest, link

SCENARIOS = ('baseline', 'combined')
KINDS = ('raw_affine', 'compact', 'compact_affine')
COMPACT_COLUMNS = ('daily_return_5d', 'daily_return_20d', 'daily_atr_pct',
    'first30_rel_volume_log', 'prior_afternoon_return', 'action_risk_fraction',
    'action_notional_equity', 'checkpoint_return', 'checkpoint_above_or',
    'checkpoint_vwap_distance', 'checkpoint_volume_ratio', 'checkpoint_efficiency',
    'checkpoint_sector_return', 'checkpoint_sector_breadth', 'later_vwap_pullback',
    'later_confirmation_margin')


def aware(text):
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError('Naive information clock')
    return value


def affine_fit(records, labels, sessions, first, scenario, *, prior=100., minimum=20, min_sessions=8):
    """Identity-prior affine reliability map, nonnegative slope capped at one.

    Fit x-centered intercept and slope with a fixed100-observation identity prior.
    No tail clipping, sign reversal, cross-quantity pooling or selected-trade filter.
    Insufficient support is explicit abstention, not an uncalibrated admission.
    """
    if scenario not in SCENARIOS or sessions != sorted(set(sessions)) or first not in sessions:
        raise ValueError('Invalid scenario or calendar')
    if prior <= 0 or not math.isfinite(prior):
        raise ValueError('Invalid calibration prior')
    allowed = set(sessions[:max(0, sessions.index(first)-1)])
    cutoff = aware(first+'T00:00:00+09:00')
    points = []
    seen = set()
    for r in records:
        key = r['opportunity_id']
        day = key[:10]
        if day not in allowed:
            continue  # Never inspect future/embargoed target values.
        if not key.endswith('/q0.25') or r['valid_from'] > day:
            raise ValueError('Wrong quantity or non-forward calibration forecast')
        if key in seen:
            raise ValueError('Duplicate dependent issuer-day')
        seen.add(key)
        if r['failure_probability'] >= .001 or not r['capacity_pass']:
            continue
        label = labels[key]
        if aware(label['label_available_at']) >= cutoff or not label['label_valid']:
            continue
        if aware(label['label_available_at']) < aware(r['feature_available_at']):
            raise ValueError('Outcome precedes decision')
        x, y = float(r[scenario]), float(label['net_value_equity_bps'])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('Nonfinite calibration observation')
        points.append((key, x, y, label['label_available_at']))
    n = len(points)
    days = len({p[0][:10] for p in points})
    supported = n >= minimum and days >= min_sessions
    center = offset = 0.
    slope = 1.
    if supported:
        x = np.array([p[1] for p in points]); y = np.array([p[2] for p in points])
        center = float(x.mean()); dx = x-center
        offset = float((y-center).sum()/(n+prior))
        variance = float(np.mean(dx*dx))
        if variance > 1e-12:
            slope = float(np.clip((np.dot(dx, y-center)+prior*variance)/
                                  (np.dot(dx, dx)+prior*variance), 0., 1.))
    return {'kind':'prequential_affine_identity100', 'scenario':scenario, 'valid_from':first,
        'supported':supported, 'prior':prior, 'minimum_issuer_days':minimum,
        'minimum_sessions':min_sessions, 'issuer_days':n, 'sessions':days,
        'center':center, 'offset':offset, 'slope':slope,
        'observed_through':max(allowed) if allowed else None, 'embargo_sessions':1,
        'label_available_through':max((p[3] for p in points), default=None),
        'training_keys_hash':ev.signature([p[0] for p in points]),
        'training_pairs_hash':ev.signature(points)}


def affine_predict(value, proof):
    if not proof['supported']:
        raise ValueError('Unsupported calibration must abstain')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite forecast')
    return proof['center']+proof['offset']+proof['slope']*(value-proof['center'])


class CompactPayoff:
    """Sixteen declared decision-time inputs; no splines/interactions/issuer IDs."""
    def fit(self, rows, y, weights):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        x = cv.matrix(rows, COMPACT_COLUMNS)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        self.scaler = StandardScaler().fit(self.imputer.transform(x))
        self.ridge = Ridge(alpha=120).fit(self.transform(rows), y, sample_weight=weights)
        return self

    def transform(self, rows):
        return np.clip(self.scaler.transform(self.imputer.transform(cv.matrix(rows, COMPACT_COLUMNS))), -5, 5)

    def predict(self, rows):
        return self.ridge.predict(self.transform(rows))


def compact_fit(rows, labels, sessions, first, through):
    if any(not r['opportunity_id'].endswith('/q0.25') for r in rows):
        raise ValueError('Compact target must be actual quarter-quantity outcome')
    ix, weights = av.training_indices(rows, labels, sessions, first, av.ModelSpec('compact'))
    train = [rows[i] for i in ix]
    valid = [r for r in rows if first <= r['trade_date'] <= through]
    if len({r['opportunity_id'] for r in train}) != len(train):
        raise ValueError('Duplicate training action')
    y = np.array([labels[r['opportunity_id']]['net_value_equity_bps'] for r in train])
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite native target')
    model = CompactPayoff().fit(train, y, weights)
    proof = {'name':'compact_ridge120_exact_quarter', 'window':'expanding',
        'valid_from':first, 'valid_through':through, 'fit_at':first+'T00:00:00+09:00',
        'train_from':min(r['trade_date'] for r in train), 'train_through':max(r['trade_date'] for r in train),
        'label_available_through':max(labels[r['opportunity_id']]['label_available_at'] for r in train),
        'train_rows':len(train), 'train_stock_days':len(train), 'feature_columns':COMPACT_COLUMNS,
        'training_keys_hash':ev.signature([r['opportunity_id'] for r in train]),
        'training_target_hash':ev.signature(y.tolist()), 'embargo_sessions':1,
        'minimum_training_sessions':60, 'quantity_scale':.25}
    return model, list(zip(valid, map(float, model.predict(valid)))), proof


def produce(root):
    """Frozen-source model/policy data producer; shared PhaseRunner runs accounts."""
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    p = read(root/'protocol.json'); verify(p['programs']+p['input_bindings'])
    old = Path(p['label_root']); sessions = read(old/'prepared/prepared.json')['sessions']
    rows = [scaled_row(r,.25) for r in ev.read_jsonl(old/'prepared/opportunities.jsonl')
            if r['action']['ready'] and r['guard_at_rank1']]
    by_key = {r['opportunity_id']:r for r in rows}
    labels = {s:{r['opportunity_id']:r for f in sorted((old/'labels'/s).glob('*.jsonl'))
                 for r in ev.read_jsonl(f)} for s in SCENARIOS}
    raw = [r for r in ev.read_jsonl(old/'models/forecasts.jsonl') if r['opportunity_id'].endswith('/q0.25')]
    for r in raw:
        source = by_key[r['opportunity_id']]
        r.update(feature_available_at=source['feature_available_at'], capacity_pass=capacity_ok(source))
    histories = {'raw':raw, 'compact':[]}; forecasts = {k:[] for k in KINDS}; proofs = []; skipped=[]
    target=root/'models'; target.mkdir(parents=True,exist_ok=False)
    inherited = read(old/'models/chronology.json')
    for f in (old/'models').iterdir():
        if f.is_file():
            link(f, target/(f.name if f.suffix=='.joblib' else 'inherited_'+f.name))
    with threadpool_limits(limits=1):
        for first,through in ev.outer_windows(sessions,min_train=60,block=20):
            current=[r for r in raw if first<=r['opportunity_id'][:10]<=through]
            if not current:
                skipped.append({'first':first,'through':through,'reason':'inherited risk/value fold unavailable'})
                continue
            fits={s:compact_fit(rows,labels[s],sessions,first,through) for s in SCENARIOS}
            scores={s:{r['opportunity_id']:value for r,value in fits[s][1]} for s in SCENARIOS}
            for s,(model,_,proof) in fits.items():
                file=target/f'compact_{first}_{s}.joblib';joblib.dump(model,file)
                proof.update(scenario=s,model_path=file.name,model_sha256=digest(file));proofs.append(proof)
            new=[{**r, **{s:scores[s][r['opportunity_id']] for s in SCENARIOS}} for r in current]
            # Append the current fold only after fitting calibration. The helper also
            # excludes current/future records before inspecting their outcomes.
            for kind,history in [('raw_affine',histories['raw']),('compact_affine',histories['compact'])]:
                maps={s:affine_fit(history,labels[s],sessions,first,s) for s in SCENARIOS}
                for s,proof in maps.items():
                    proof.update(forecast_family=kind,valid_through=through);proofs.append(proof)
                source=current if kind=='raw_affine' else new
                for r in source:
                    supported=all(m['supported'] for m in maps.values())
                    forecasts[kind].append({**r,'calibration_supported':supported,
                        **({s:affine_predict(r[s],maps[s]) for s in SCENARIOS} if supported else {}),
                        'signal_kind':kind})
            forecasts['compact'].extend({**r,'calibration_supported':True,'signal_kind':'compact'} for r in new)
            histories['compact'].extend(new)
            print(json.dumps({'stage':'signal_models','first':first,'through':through,
                'predictions':len(current),'compact_train_issuer_days':fits['baseline'][2]['train_stock_days']}),flush=True)
    for kind,values in forecasts.items():
        with (target/(kind+'.jsonl')).open('x',encoding='utf-8') as f:
            for r in values:f.write(json.dumps(r,allow_nan=False)+'\n')
    write(target/'chronology.json',{'folds':inherited['folds']+[v for v in proofs if 'model_path' in v],
        'calibration':[{**v,'latest_label_available_at':v['label_available_through']} for v in proofs if 'kind' in v],
        'skipped_folds':skipped,
        'inherited_value_risk_chronology':bound(old/'models/chronology.json'),
        'scope':'continuous expanding original20-session clock; exact q0.25 targets; one-session embargo; prequential calibration'})
    write(target/'completion.json',{'complete':True,'files':{f.name:digest(f) for f in target.iterdir() if f.is_file()}})
    build_policies(root,rows,forecasts)


def build_policies(root,rows,forecasts):
    from scripts.kalcb_executed_value_replay import verify_policy
    from scripts.kalcb_executed_value_study import config_for
    from backtests.strategies.kalcb.late_policy_v2 import select_executable
    from backtests.strategies.kalcb.late_entry_v1 import cumulative_timed_policy
    p=read(root/'protocol.json'); by_key={r['opportunity_id']:r for r in rows}
    models=read(root/'models/completion.json'); aliases=defaultdict(list)
    for name,recipe in p['recipes'].items():
        daily=defaultdict(list)
        for score in forecasts.get(recipe['signal'],[]):
            row=by_key[score['opportunity_id']]
            if (score['calibration_supported'] and score['failure_probability']<.001 and score['capacity_pass']
                    and min(score[s] for s in SCENARIOS)>48*row['features']['action_notional_equity']):
                daily[row['trade_date']].append((row,score))
        for period in ('is','oos'):
            old=p['profiles'][str(recipe['prior'])][period];source=Path(old['policy']['path']).parent
            m,opening=verify_policy(source,'interaction_cost_margin'); updates=cumulative_timed_policy(source,m,opening)
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
                    added.append({'trade_date':str(day),'symbol':row['symbol'],'available_at':row['feature_available_at'],
                        'score':value,'action_id':row['opportunity_id'],'quantity_scale':.25,
                        'valid_from':score['valid_from'],'flat_only':False,'signal_kind':recipe['signal']})
            for filename,values in [('later60.snapshots.jsonl',timed),('later60.admission.jsonl',admissions+added),('later60.new.jsonl',added)]:
                with (target/filename).open('x',encoding='utf-8') as f:
                    for row in values:f.write(json.dumps(row,default=str,allow_nan=False)+'\n')
            chronology=read(source/'chronology.json');model_proof=read(root/'models/chronology.json')
            chronology['folds']+=model_proof['folds']
            chronology.setdefault('calibration',[]).extend(model_proof['calibration'])
            (target/'chronology.json').unlink();write(target/'chronology.json',chronology)
            for f in models['files']:link(root/'models'/f,target/(f if f.endswith('.joblib') else 'later_model_'+f))
            write(target/'later_lineage.json',{'registration':bound(root/'protocol.json'),
                'preparation':bound(root/'prepared/prepared.json'),'models':bound(root/'models/completion.json'),
                'label_manifests':[bound(Path(p['label_root'])/'labels'/s/'completion.json') for s in SCENARIOS],
                'parent_policy':old['policy'],'parent_snapshots_preserved':True,'recipe':recipe,
                'target_unchanged':'same native completed60-minute quarter action, management, costs and source generation'})
            m.update(version='kalcb-first-hour-signal-v1',observation_minutes=60,timed_snapshot_file='later60.snapshots.jsonl',
                timed_admission_file='later60.admission.jsonl',contract={**m['contract'],'timed_budget':13,'new_budget':1},
                files={f.name:digest(f) for f in target.iterdir() if f.is_file()})
            write(target/'policies_manifest.json',m);write(target/'completion.json',{'complete':True})
            actual,ss=verify_policy(target,'interaction_cost_margin');cumulative_timed_policy(target,actual,ss)
            signature=ev.signature({'opening':[s.to_json_dict() for s in opening.values()],'timed':timed})
            aliases[(period,signature)].append(name)
            print(json.dumps({'stage':'signal_policy','name':name,'period':period,'new_admissions':len(added)}),flush=True)
    write(root/'policies/completion.json',{'complete':True,'independent_validation':False,
        'effective_aliases':[{'period':p,'signature':s,'names':v} for (p,s),v in aliases.items() if len(v)>1]})
