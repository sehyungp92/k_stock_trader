"""Finite, prequential reclaim payoff/risk research; no portfolio runner.

Auxiliary same-order labels inform only physical risk. They never enter the
payoff learner or eligibility calibration. No threshold search is performed.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import math
import numpy as np
from backtests.strategies.kalcb import action_value as av, executed_value as ev
from backtests.strategies.kalcb.late_signal_v1 import affine_fit, affine_predict
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, verify, bound, digest

SCENARIOS=('baseline','combined')
PHYSICAL=('action_notional_equity','action_quantity_last_volume',
    'action_quantity_opening_volume','action_low_volume_exit_slices',
    'action_risk_fraction','daily_adv20_krw_log','daily_atr_pct','checkpoint_volume_ratio')
PAYOFF=('daily_return_5d','daily_return_20d','daily_atr_pct','first30_rel_volume_log',
    'prior_afternoon_return','action_risk_fraction','action_notional_equity',
    'checkpoint_return','checkpoint_above_or','checkpoint_vwap_distance',
    'checkpoint_volume_ratio','checkpoint_efficiency','checkpoint_sector_return',
    'checkpoint_sector_breadth','reclaim_pullback_depth','reclaim_confirmation_margin')
RULE={
    'action':'Frozen completed90minute reclaim; exact integer quarter; unchanged management/cost/funding/core.',
    'payoff':'Geometry and native-quarter-ready only; full-cost net per planned notional; sixteen explicit inputs; train-only median/scaling, clip standardized values to[-5,5], Ridge120.',
    'risk':'Same-time/quantity/management physical population, geometry+auxiliary, one issuer-day outcome unsafe if either scenario fails. Inherited FeasibilityModel1/(N+100) or logisticC.1 unchanged. Eight declared execution-context inputs.',
    'domain':'All eight physical features finite and inside the min/max of mature prior pooled training rows. No future envelope or exception.',
    'clock':'Expanding original20session folds;60mature sessions required separately for payoff and physical risk; one-session embargo and label availability before fit.',
    'calibration':'Prior-specific known-new-snapshot geometry only; earlier genuine forward raw forecasts passing physical/domain/capacity; full-cost same-action labels; identity prior100 affine slope0..1, minimum20issuer-days/8sessions. Unknown membership and unsupported map abstain.',
    'recipes':['raw48_component_backout','eligible_affine48'],
    'admission':'New issuer, physical probability<.001, prior-domain and original capacity pass, BOTH scenarios strictly above48planned-notionalbps. No threshold tuning.',
    'ranking':'Inherited min baseline/combined expected equitybps after exact notional conversion; final native combined-rank/account check still required.',
    'limits':'Research estimator is not a certified0.1% true failure bound. Same-session/issuer observations dependent. Auxiliary returns forbidden as payoff targets. No native policy or portfolio performance is produced here.'}

def aware(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None: raise ValueError('Naive information clock')
    return result

def unique(values):
    out={}
    for v in values:
        k=v['opportunity_id']
        if k in out: raise ValueError('Duplicate dependent action')
        out[k]=v
    return out

def features(rows,columns,finite=False):
    # Missing names indicate a schema error, not an imputable observation.
    for row in rows:
        if any(k not in row['features'] for k in columns): raise ValueError('Undeclared missing feature')
    x=np.array([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in columns] for r in rows],dtype=float)
    if not rows:x=np.empty((0,len(columns)))
    if np.isinf(x).any() or (finite and not np.isfinite(x).all()): raise ValueError('Unsupported feature values')
    return x

def mature_rows(rows,labels,sessions,first,kind):
    allowed=set(sessions[:max(0,sessions.index(first)-1)]);cutoff=aware(first+'T00:00:00+09:00')
    selected=[];field='risk_available_at' if kind=='risk' else 'label_available_at'
    for row in rows:
        if row['trade_date'] not in allowed: continue
        label=labels[row['opportunity_id']]
        if kind=='payoff' and (label.get('auxiliary_physical_only') or label.get('return_model_eligible') is False):
            raise ValueError('Auxiliary payoff leakage')
        if kind=='payoff' and not label['label_valid']:continue
        at=label.get(field)
        if not at or aware(at)>=cutoff:continue
        if aware(at)<aware(row['feature_available_at']):raise ValueError('Outcome precedes action')
        selected.append(row)
    return selected

def physical_outcomes(grid,rows):
    if set(grid['baseline'])!=set(grid['combined']):raise ValueError('Unpaired risk scenarios')
    out={}
    for row in rows:
        if not row['quarter_preview']['ready'] or row['action']['quantity']<=0:raise ValueError('No-order risk-count inflation')
        key=row['opportunity_id'];pair=[grid[s][key] for s in SCENARIOS]
        if not all(v.get('risk_available_at') for v in pair):raise ValueError('Missing physical outcome clock')
        out[key]={'risk_available_at':max(v['risk_available_at'] for v in pair),
            'unsafe':any(v['physical_fill_failures'] or v['deadline_violation'] or v['unresolved'] or
                not v['independent_episode_execution_pass'] or v['status']=='execution_invalid' for v in pair)}
    return out

def domain(train,current):
    x=features(train,PHYSICAL,finite=True);z=features(current,PHYSICAL,finite=True)
    if not len(x):raise ValueError('Empty physical domain')
    lo,hi=x.min(axis=0),x.max(axis=0)
    return ((z>=lo)&(z<=hi)).all(axis=1),{'columns':PHYSICAL,'min':lo.tolist(),'max':hi.tolist()}

class Payoff:
    def fit(self,rows,labels):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        if any(not r['guard_at_rank1'] or labels[r['opportunity_id']].get('auxiliary_physical_only') for r in rows):
            raise ValueError('Payoff population changed')
        x=features(rows,PAYOFF)
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        self.scaler=StandardScaler().fit(self.imputer.transform(x))
        y=np.array([labels[r['opportunity_id']]['net_value_planned_notional_bps'] for r in rows],float)
        if not np.isfinite(y).all():raise ValueError('Invalid full-cost payoff target')
        self.ridge=Ridge(alpha=120).fit(self.transform(rows),y)
        return self
    def transform(self,rows):return np.clip(self.scaler.transform(self.imputer.transform(features(rows,PAYOFF))),-5,5)
    def predict(self,rows):return self.ridge.predict(self.transform(rows))

def calibration(records,labels,rows,sessions,first,scenario,prior):
    allowed=set(sessions[:max(0,sessions.index(first)-1)]);cutoff=aware(first+'T00:00:00+09:00')
    eligible=[];converted={};seen=set()
    for value in records:
        key=value['opportunity_id'];day=key[:10]
        if day not in allowed:continue
        if key in seen:raise ValueError('Duplicate calibration record')
        seen.add(key);row=rows[key]
        if value['valid_from']>day:raise ValueError('Non-forward calibration forecast')
        if row['parent_snapshot_eligibility'].get(str(prior)) is not True:continue
        if not value['domain_pass'] or not value['capacity_pass'] or value['failure_probability']>=.001:continue
        label=labels[key]
        if label.get('auxiliary_physical_only'):raise ValueError('Auxiliary calibration leakage')
        if not label['label_valid'] or aware(label['label_available_at'])>=cutoff:continue
        eligible.append({**value,scenario:value['notional_forecast'][scenario]})
        converted[key]={**label,'net_value_equity_bps':label['net_value_planned_notional_bps']}
    proof=affine_fit(eligible,converted,sessions,first,scenario)
    proof.update(kind='reclaim_new_snapshot_notional_affine100',parent_prior=prior,
        units='full_cost_net_bps_per_planned_notional',input_forecasts_hash=ev.signature(eligible))
    return proof

def admitted(value):
    return bool(value['population_eligible'] and value['calibration_supported'] and value['domain_pass'] and
        value['capacity_pass'] and value['failure_probability']<.001 and
        all(value['notional_forecast'][s]>48 for s in SCENARIOS))

def produce(root):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    p=read(root/'protocol.json');verify(p['programs']+p['input_bindings'])
    if p['rule']!=RULE:raise ValueError('Registered formula differs')
    geometry_root=Path(p['geometry_label_root']);aux_root=Path(p['auxiliary_label_root'])
    meta=read(geometry_root/'prepared/prepared.json');sessions=meta['sessions']
    raw=list(ev.read_jsonl(geometry_root/'prepared/opportunities.jsonl'))
    if digest(geometry_root/'prepared/opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed preparation')
    ready=[scaled_row(r,.25) for r in raw if r['quarter_preview']['ready']]
    target=[r for r in ready if r['guard_at_rank1']];by_key=unique(target)
    geometry={};grid={}
    for s in SCENARIOS:
        geometry[s]=unique(v for f in sorted((geometry_root/'labels'/s).glob('*.jsonl')) for v in ev.read_jsonl(f))
        auxiliary=unique(v for f in sorted((aux_root/'labels'/s).glob('*.jsonl')) for v in ev.read_jsonl(f))
        if set(geometry[s])&set(auxiliary):raise ValueError('Overlapping physical populations')
        if any(not v['auxiliary_physical_only'] or v['return_model_eligible'] or v['reclaim_eligible'] for v in auxiliary.values()):raise ValueError('Lost auxiliary role')
        grid[s]={**geometry[s],**auxiliary}
        for r in ready:
            label=grid[s][r['opportunity_id']];notional=r['features']['action_notional_equity']*100_000_000
            if label['planned_quantity']!=r['action']['quantity'] or not math.isclose(notional,label['planned_notional_krw'],abs_tol=1e-6):raise ValueError('Quarter denominator changed')
    risk_labels=physical_outcomes(grid,ready)
    output=root/'models';output.mkdir(exist_ok=False);proofs=[];raw_forecasts=[];scores={f'{prior}_{kind}':[] for prior in (100,125) for kind in ('raw48','affine48')}
    with threadpool_limits(limits=1):
        for first,through in ev.outer_windows(sessions,min_train=60,block=20):
            tr=mature_rows(ready,risk_labels,sessions,first,'risk')
            tp={s:mature_rows(target,geometry[s],sessions,first,'payoff') for s in SCENARIOS}
            counts={'risk':len({r['trade_date'] for r in tr}),**{s:len({r['trade_date'] for r in tp[s]}) for s in SCENARIOS}}
            current=[r for r in target if first<=r['trade_date']<=through]
            if min(counts.values())<60 or not current:
                proofs.append({'valid_from':first,'valid_through':through,'supported':False,'mature_sessions':counts,'reason':'Original60mature-session requirement or empty geometry fold; no backfill'})
                continue
            y=np.array([risk_labels[r['opportunity_id']]['unsafe'] for r in tr],bool)
            risk=av.FeasibilityModel().fit(features(tr,PHYSICAL,finite=True),y)
            prob=risk.predict(features(current,PHYSICAL,finite=True));inside,envelope=domain(tr,current)
            file=output/f'risk_{first}.joblib';joblib.dump(risk,file)
            proofs.append({'name':'reclaim_same_order_physical','valid_from':first,'valid_through':through,'mature_sessions':counts['risk'],
                'train_rows':len(tr),'unsafe_rows':int(y.sum()),'observed_through':sessions[sessions.index(first)-2],
                'label_available_through':max(risk_labels[r['opportunity_id']]['risk_available_at'] for r in tr),
                'training_keys_hash':ev.signature([r['opportunity_id'] for r in tr]),'target_hash':ev.signature(y.tolist()),
                'prior_domain':envelope,'model':bound(file),'probability_not_certified':True})
            predictions={}
            for s in SCENARIOS:
                model=Payoff().fit(tp[s],geometry[s]);predictions[s]=model.predict(current)
                file=output/f'payoff_{first}_{s}.joblib';joblib.dump(model,file)
                proofs.append({'name':'reclaim_notional_Ridge120','scenario':s,'valid_from':first,'valid_through':through,
                    'feature_columns':PAYOFF,'train_rows':len(tp[s]),'train_sessions':counts[s],
                    'train_from':min(r['trade_date'] for r in tp[s]),'train_through':max(r['trade_date'] for r in tp[s]),
                    'label_available_through':max(geometry[s][r['opportunity_id']]['label_available_at'] for r in tp[s]),
                    'training_keys_hash':ev.signature([r['opportunity_id'] for r in tp[s]]),
                    'target_hash':ev.signature([geometry[s][r['opportunity_id']]['net_value_planned_notional_bps'] for r in tp[s]]),
                    'auxiliary_payoff_rows':0,'model':bound(file)})
            current_scores=[]
            for i,row in enumerate(current):
                current_scores.append({'opportunity_id':row['opportunity_id'],'feature_available_at':row['feature_available_at'],
                    'valid_from':first,'valid_through':through,'failure_probability':float(prob[i]),'domain_pass':bool(inside[i]),
                    'capacity_pass':bool(capacity_ok(row)),'notional_forecast':{s:float(predictions[s][i]) for s in SCENARIOS}})
            for prior in (100,125):
                maps={s:calibration(raw_forecasts,geometry[s],by_key,sessions,first,s,prior) for s in SCENARIOS}
                proofs.extend({**m,'valid_through':through} for m in maps.values())
                for v in current_scores:
                    row=by_key[v['opportunity_id']];fraction=row['features']['action_notional_equity']
                    for kind in ('raw48','affine48'):
                        supported=kind=='raw48' or all(m['supported'] for m in maps.values())
                        values=(v['notional_forecast'] if kind=='raw48' else
                            {s:affine_predict(v['notional_forecast'][s],maps[s]) for s in SCENARIOS} if supported else None)
                        record={**v,'prior':prior,'recipe':kind,'calibration_supported':supported,
                            'population_eligible':row['parent_snapshot_eligibility'].get(str(prior)) is True,
                            'notional_forecast':values,'equity_forecast':{s:values[s]*fraction for s in SCENARIOS} if values else None,
                            'planned_notional_fraction':fraction,'admitted':False}
                        record['admitted']=admitted(record);scores[f'{prior}_{kind}'].append(record)
            raw_forecasts.extend(current_scores)
            print(json.dumps({'stage':'reclaim_models','first':first,'through':through,'risk_n':len(tr),
                'payoff_n':len(tp['baseline']),'targets':len(current),'raw_new_issuer_proposals':sum(v['admitted'] for v in scores['100_raw48'])}),flush=True)
    def emit(name,values):
        with (output/name).open('x',encoding='utf-8') as f:
            for v in values:f.write(json.dumps(v,allow_nan=False)+'\n')
    emit('raw_forward.jsonl',raw_forecasts)
    for name,values in scores.items():emit(name+'.jsonl',values)
    write(output/'chronology.json',{'rule':RULE,'proofs':proofs,'source_sessions':sessions,'first_oos_prior_sessions':217,
        'geometry_rows':len(target),'physical_rows':len(ready),'study_complete':False})
    summary={name:{part:{'forecast_rows':sum(v['opportunity_id'][:10]>='2025-08-07' and (v['opportunity_id'][:10]<'2026-04-01')==(part=='is') for v in vals),
        'eligible_proposals':sum(v['admitted'] and (v['opportunity_id'][:10]<'2026-04-01')==(part=='is') for v in vals)} for part in ('is','oos')} for name,vals in scores.items()}
    write(output/'proposal_summary.json',{'recipes':summary,'scope':'Forecast proposals only, not final rank/account availability, submitted orders or fills. No portfolio PnL.',
        'native_policies_produced':0,'native_portfolio_cases':0,'study_complete':False})
    verify(p['programs']+p['input_bindings'])
    write(output/'completion.json',{'complete':True,'files':{f.name:digest(f) for f in output.iterdir() if f.is_file()},'summary':summary,
        'study_complete':False,'portfolio_cases':0,'next':'Review forward support/calibration and proposals. No native score until strict90minute policy/source integration and exact unchanged parent bridges.'})
