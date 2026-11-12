"""One fixed causal selector for typed, executable whole-account contrasts.

Ensemble agreement is a support heuristic, not a confidence certificate. IS
out-of-time action discrimination gates expensive sequential qualification;
neither local contrasts nor that gate establish strategy profitability.
"""
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import numpy as np
from .holding_action_v1 import FEATURES,ValueContract
from .retention_research_io_v1 import read,write,bound,verify

SCENARIOS=('baseline','participation','cost','combined')
MODEL_SPEC={'algorithm':'HistGradientBoostingRegressor','max_iter':64,'max_leaf_nodes':7,
    'max_depth':3,'min_samples_leaf':30,'l2_regularization':1.,'learning_rate':.05,
    'early_stopping':False,'random_state':1301,'loss':'squared_error'}


def fit(rows,days):
    from sklearn.ensemble import HistGradientBoostingRegressor
    if len(days)<60 or not rows:raise ValueError('Insufficient mature training sessions')
    if any(r['observation']['day'] not in days for r in rows):raise ValueError('Immature training row')
    X=np.array([[r['observation']['features'][f] for f in FEATURES] for r in rows])
    day_array=np.array([r['observation']['day'] for r in rows])
    counts=Counter(day_array);w=np.array([1/counts[d] for d in day_array]);w/=w.mean()
    center=np.median(X,axis=0);scale=np.maximum(np.percentile(X,75,axis=0)-np.percentile(X,25,axis=0),1e-6)
    from sklearn.neighbors import NearestNeighbors
    support=NearestNeighbors(n_neighbors=6).fit((X-center)/scale)
    distances=support.kneighbors(return_distance=True)[0][:,-1]
    result={'contract':asdict(ValueContract()),'features':FEATURES,'training_days':days,
        'training_rows':len(rows),'low':X.min(axis=0),'high':X.max(axis=0),'center':center,'scale':scale,
        'support':support,'support_radius':float(np.quantile(distances,.95)), 'models':{}}
    for arm in ('half','exit'):
        result['models'][arm]={}
        for target in ('value_bps','dd_change_bps'):
            y=np.array([r['contrasts'][arm]['target'][target] for r in rows])
            models=[]
            for block in np.array_split(np.array(days),3):
                mask=~np.isin(day_array,block)
                if mask.sum()<60:raise ValueError('Insufficient grouped action coverage')
                m=HistGradientBoostingRegressor(**{k:v for k,v in MODEL_SPEC.items() if k!='algorithm'})
                m.fit(X[mask],y[mask],sample_weight=w[mask]);models.append(m)
            result['models'][arm][target]=models
    return result


def choose(model,row):
    ValueContract().validate(model['contract'])
    if row['day']<=model['training_days'][-1]:raise ValueError('Unmatured action model')
    x=np.array([[row['features'][f] for f in FEATURES]])
    if not np.isfinite(x).all():raise ValueError('Nonfinite inference feature')
    if np.any(x[0]<model['low']) or np.any(x[0]>model['high']):
        return {'arm':'hold','reason':'outside_observed_training_range'}
    distance=float(model['support'].kneighbors((x-model['center'])/model['scale'],return_distance=True)[0][0,-1])
    if distance>model['support_radius']:return {'arm':'hold','reason':'sparse_joint_state_support','distance':distance}
    choices=[];predictions={}
    for arm,targets in model['models'].items():
        p={t:[float(m.predict(x)[0]) for m in models] for t,models in targets.items()}
        predictions[arm]=p
        # No arbitrary weighted wealth/risk objective: require positive wealth
        # and nonworsening peak-path risk under every grouped fitted model.
        if min(p['value_bps'])>0 and max(p['dd_change_bps'])<=0:
            choices.append((min(p['value_bps']),arm))
    if not choices:return {'arm':'hold','reason':'no_agreed_positive_value_and_risk','predictions':predictions}
    value,arm=max(choices)
    return {'arm':arm,'reason':'agreed_supported_action','predicted_value_bps':value,
        'predictions':predictions,'distance':distance,'training_through':model['training_days'][-1]}


def selected_contrasts(rows,decisions):
    # Mutually exclusive choice per clock; no repeated intervention per lifetime.
    by_clock=defaultdict(list);seen=set();out=[]
    for r,d in zip(rows,decisions):
        if d['arm']!='hold':by_clock[r['observation']['available_at']].append((r,d))
    for clock,values in sorted(by_clock.items()):
        eligible=[v for v in values if (v[0]['observation']['day'],v[0]['observation']['entry_order_id']) not in seen]
        if not eligible:continue
        r,d=max(eligible,key=lambda v:(v[1]['predicted_value_bps'],v[0]['observation']['symbol']))
        obs=r['observation'];seen.add((obs['day'],obs['entry_order_id']))
        out.append({'day':obs['day'],'symbol':obs['symbol'],'entry_order_id':obs['entry_order_id'],
            'available_at':obs['available_at'],'qty':obs['qty'],'arm':d['arm'],
            'target':r['contrasts'][d['arm']]['target'],'half':r['contrasts']['half']['target'],
            'exit':r['contrasts']['exit']['target'],'decision':d})
    return out


def readiness(selected):
    grouped=defaultdict(list)
    for r in selected:grouped[r['day']].append(r)
    means=np.array([[np.mean([r['target'][k] for r in rows]) for k in ('value_bps','dd_change_bps')]
        for _,rows in sorted(grouped.items())])
    if len(means)<10 or len(selected)<20:
        return {'passed':False,'reason':'insufficient_changed_material_support','days':len(means),'actions':len(selected)}
    # Calendar-ordered five-active-session blocks preserve nearby shared shocks.
    # These are action-evidence sensitivity intervals, never strategy returns.
    rng=np.random.default_rng(1301);samples=[]
    for _ in range(2000):
        starts=rng.integers(0,len(means),size=(len(means)+4)//5)
        idx=np.concatenate([(np.arange(5)+s)%len(means) for s in starts])[:len(means)]
        samples.append(means[idx].mean(axis=0))
    samples=np.array(samples);lower=float(np.quantile(samples[:,0],.05));upper=float(np.quantile(samples[:,1],.95))
    return {'passed':lower>0 and upper<=0,'days':len(means),'actions':len(selected),
        'mean_session_action_value_bps':float(means[:,0].mean()),'mean_session_action_dd_change_bps':float(means[:,1].mean()),
        'block5_lower_action_value_bps':lower,'block5_upper_action_dd_change_bps':upper,
        'worst_action_value_bps':min(r['target']['value_bps'] for r in selected),
        'interpretation':'Local action discrimination only; overlapping suffixes are not additive returns'}


def build(root):
    root=Path(root);dest=root/'models';dest.mkdir(exist_ok=False)
    days=[r['day'] for r in read(root/'cases/control/is/baseline/daily.json')]
    manifest={'contract':asdict(ValueContract()),'model_spec':MODEL_SPEC,'features':FEATURES,
        'fit_min_sessions':60,'embargo_completed_sessions':1,'refit_sessions':20,'snapshots':{},'readiness':{},'inputs':[]}
    for scenario in SCENARIOS:
        case=root/'cases/control/is'/scenario
        verify([read(case/'completion.json')['holding']])
        if not read(case/'identity_bridge.json')['passed']:raise ValueError('Unbridged training account')
        rows=[json.loads(s) for s in (case/'labels.jsonl').read_text(encoding='utf-8').splitlines()]
        manifest['inputs']+=[bound(case/k) for k in ('labels.jsonl','completion.json','hold_identity.json')]
        identities={r['day']:r['exact_hold_eod_state'] for r in read(case/'hold_identity.json')}
        if any(identities.get(r['observation']['day'])!=r['hold_eod_state'] for r in rows):raise ValueError('Unauthenticated hold label')
        snapshots=[];validation=[];predictions=[]
        starts=list(range(61,len(days),20))+[len(days)]
        for n,j in enumerate(starts):
            train_days=days[:j-1];train=[r for r in rows if r['observation']['day'] in train_days]
            model=fit(train,train_days)
            path=dest/(scenario+'_'+str(j)+'.pkl')
            with path.open('xb') as f:pickle.dump(model,f,protocol=5)
            through=days[starts[n+1]-1] if n+1<len(starts) else '9999-12-31'
            effective=days[j] if j<len(days) else '2026-04-01'
            snapshots.append({'effective_from':effective,'through':through,'model':bound(path),
                'training_through':train_days[-1],'training_sessions':len(train_days),'training_rows':len(train)})
            held=[r for r in rows if effective<=r['observation']['day']<=through]
            validation.extend(held);predictions.extend(choose(model,r['observation']) for r in held)
        selected=selected_contrasts(validation,predictions)
        # Preserve all predictions, harmful selected actions and unconditional
        # alternatives. OOS labels cannot enter fitting or this readiness gate.
        write(dest/(scenario+'_out_of_time.json'),{'predictions':[{'observation':r['observation'],'decision':d}
            for r,d in zip(validation,predictions)],'selected':selected,
            'unconditional_local_means':{a:{k:float(np.mean([r['contrasts'][a]['target'][k] for r in validation]))
                for k in ('value_bps','dd_change_bps')} for a in ('half','exit')}})
        manifest['snapshots'][scenario]=snapshots;manifest['readiness'][scenario]=readiness(selected)
    manifest['economic_readiness_pass']=all(r['passed'] for r in manifest['readiness'].values())
    manifest['strategy_objective_achieved']=False
    write(dest/'manifest.json',manifest);return manifest


class Policy:
    def __init__(self,folder,scenario,period):
        self.folder=Path(folder);self.manifest=read(self.folder/'manifest.json');self.scenario=scenario;self.period=period
        ValueContract().validate(self.manifest['contract'])
        if not self.manifest['economic_readiness_pass']:raise ValueError('Unqualified action selector')
        verify(self.manifest['inputs']+[s['model'] for s in self.manifest['snapshots'][scenario]])
        self.cache={}

    def __call__(self,row):
        snapshots=self.manifest['snapshots'][self.scenario]
        candidates=[s for s in snapshots if s['effective_from']<=row['day']<=s['through']]
        if len(candidates)>1:raise ValueError('Overlapping chronological model intervals')
        if not candidates:return {'arm':'hold','reason':'causal_warmup_no_mature_model'}
        spec=candidates[0];key=spec['model']['sha256']
        if key not in self.cache:
            verify([spec['model']])
            with Path(spec['model']['path']).open('rb') as f:self.cache[key]=pickle.load(f)
        return choose(self.cache[key],row)
