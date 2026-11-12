"""Fixed chronological C18 information diagnostics, never a trading selector."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import groupby
from datetime import datetime,timedelta,timezone
import argparse,json,gzip,hashlib,sqlite3,time,math,csv
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def verify(b):assert bound(b['path'])['sha256']==b['sha256']
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for n,l in enumerate(f,1):yield n,json.loads(l)
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def emit(f,x):f.write(json.dumps(x,default=str,allow_nan=False)+'\n')
def csvsave(p,x):
    if not x:return
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)

def holding_pair(base,extended,base_label,extended_label,prefix_bars,participation):
    """Return a verified local contrast; no account omission branch is run."""
    if not all(x['label_valid'] and x['filled_entry_orders']==1 for x in [base_label,extended_label]):return None,'invalid_or_multiple_lifetime'
    buys=lambda x:[(f['timestamp'],f['qty'],f['price']) for f in x['fills'] if f['side']=='BUY']
    if buys(base)!=buys(extended):return None,'different_entry_fills'
    candidates=[(i,r) for i,r in enumerate(base['complete_native_journal']) if r['event']=='submit'
        and r['order']['side']=='SELL' and r['order']['metadata'].get('order_role')=='EXIT'
        and r['order']['metadata'].get('activation_phase')=='decision']
    if not candidates:return None,'no_discretionary_exit'
    i,row=candidates[0];order=row['order'];clock=datetime.fromisoformat(order['metadata']['decision_available_at'])
    deadline=datetime.fromisoformat(base_label['first_fill'])+timedelta(minutes=120)
    if clock>=deadline:return None,'implemented_exit_not_before120'
    prior=base['complete_native_journal'][:i];other=extended['complete_native_journal']
    if prior!=other[:i]:return None,'economic_prefix_not_identical'
    if len(other)>i and other[i]==row:return None,'same_native_action_no_hold_intervention'
    cash=1e8;inventory=Counter();marks={};orders={}
    for event in prior:
        kind=event['event'];o=event.get('order',{});oid=o.get('order_id')
        if kind in ('submit','amend'):orders[oid]=o
        elif kind in ('cancel','expire','reject'):orders.pop(oid,None)
        elif kind in ('opening_marks','closing_marks'):marks.update(event['prices'])
        elif kind=='fill':
            f=event['fill'];sign=1 if f['side']=='BUY' else -1;cost=event['costs']
            fee=f['qty']*f['price']*(cost['commission_bps']+(cost['tax_bps_on_sell'] if sign<0 else 0))/1e4
            assert abs(cash-event['cash_before'])<.02
            cash-=sign*f['qty']*f['price']+fee;inventory[f['symbol']]+=sign*f['qty'];orders.pop(oid,None)
            assert inventory[f['symbol']]>=0 and abs(cash-event['cash_after'])<.02
    symbol=base_label['symbol'];quantity=inventory[symbol]
    if quantity<=0 or order['qty']!=quantity:return None,'not_full_owned_exit'
    observed=[b for b in prefix_bars if b[0]+timedelta(minutes=5)<=clock]
    if not observed:return None,'missing_completed_holding_prefix'
    meta=order['metadata'];risk=meta.get('management_risk_per_share',meta.get('risk_per_share'))
    if not risk or risk<=0:return None,'missing_native_risk_geometry'
    close=observed[-1][3];volume=sum(b[4] for b in observed)
    vwap=sum((b[1]+b[2]+b[3])/3*b[4] for b in observed)/volume if volume else None
    cash_delta=extended_label['net_cash_krw']-base_label['net_cash_krw']
    features=[base_label['raw_score'],meta.get('hold_bars'),meta.get('mfe_r'),meta.get('mae_r'),
        (close-meta['entry_price'])/risk,close/vwap-1 if vwap else None,
        quantity/(observed[-1][4]*participation) if observed[-1][4] else None]
    result=dict(target=cash_delta/(quantity*close)*1e4,target_cash=cash_delta,features=features,
        feature_available_at=clock.isoformat(),label_mature_at=max(base_label['label_mature_at'],extended_label['label_mature_at']),
        native_quantity=quantity,decision_notional=quantity*close,decision_reason=order['reason'],
        prefix_journal_rows=i,common_cash=cash,common_inventory=dict(inventory),common_pending=len(orders),
        current_exit_cash=base_label['net_cash_krw'],continued_cash=extended_label['net_cash_krw'])
    return result,'verified'

def fit(x,y,weights):
    med=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in x.T])
    missing=~np.isfinite(x);clean=np.where(missing,med,x);mean=np.average(clean,axis=0,weights=weights)
    scale=np.sqrt(np.average((clean-mean)**2,axis=0,weights=weights));scale=np.where(scale>1e-12,scale,1.)
    z=np.column_stack([(clean-mean)/scale,missing.astype(float)])
    average=float(np.average(y,weights=weights));sw=np.sqrt(weights);a=z*sw[:,None]
    coef=np.linalg.solve(a.T@a+10*np.eye(a.shape[1]),a.T@((y-average)*sw))
    raw=clean[:,0];rmean=float(np.average(raw,weights=weights));den=float(np.sum(weights*(raw-rmean)**2))
    slope=float(np.sum(weights*(raw-rmean)*(y-average))/den) if den>1e-12 else 0.
    return dict(median=med,mean=mean,scale=scale,coef=coef,average=average,raw_mean=rmean,affine_slope=slope)
def predict(model,x):
    missing=~np.isfinite(x);clean=np.where(missing,model['median'],x)
    z=np.column_stack([(clean-model['mean'])/model['scale'],missing.astype(float)])
    return dict(mean=np.full(len(x),model['average']),affine=model['average']+model['affine_slope']*(clean[:,0]-model['raw_mean']),
        ridge=model['average']+z@model['coef'],raw=clean[:,0])

def chronological(data,calendar):
    results=[];models=[];coverage=Counter();last=None
    # 60 mature sessions, then one complete embargo session, then fit.
    for offset in range(61,len(calendar),20):
        fit_day=calendar[offset];train_through=calendar[offset-2];valid=set(calendar[offset:offset+20])
        train=[r for r in data if r['day']<=train_through and datetime.fromisoformat(r['label_mature_at'])<datetime.fromisoformat(fit_day+'T00:00:00+09:00')]
        test=[r for r in data if r['day'] in valid]
        if len({r['day'] for r in train})<60:
            coverage['insufficient60mature_session_test_rows']+=len(test);continue
        if not test:continue
        counts=Counter(r['day']+'/'+r['symbol'] for r in train)
        weights=np.array([1/counts[r['day']+'/'+r['symbol']] for r in train]);weights*=len(weights)/weights.sum()
        x=np.array([r['features'] for r in train],dtype=float);y=np.array([r['target'] for r in train]);model=fit(x,y,weights)
        train_predictions=predict(model,x);test_predictions=predict(model,np.array([r['features'] for r in test],dtype=float))
        proof=dict(fit_day=fit_day,train_through=train_through,latest_label_maturity=max(r['label_mature_at'] for r in train),
            train_rows=len(train),train_sessions=len({r['day'] for r in train}),test_rows=len(test),valid_days=sorted(valid),
            model={k:v.tolist() if hasattr(v,'tolist') else v for k,v in model.items()},alpha=10,
            weight_basis='inverse issuer-day episode count, normalized to row count; fixed before fit')
        models.append(proof)
        for i,r in enumerate(test):
            result={**r,'fit_day':fit_day,'prediction':{k:float(v[i]) for k,v in test_predictions.items()},'deciles':{}}
            for name,values in train_predictions.items():
                boundaries=np.quantile(values,np.arange(.1,1,.1));result['deciles'][name]=int(np.searchsorted(boundaries,result['prediction'][name],side='right')+1)
            results.append(result)
    coverage.update(predicted_rows=len(results),eligible_rows=len(data),warmup_or_unpredicted_rows=len(data)-len(results))
    return results,models,dict(coverage)

def weighted_auc(y,p,w):
    positive=y>0;order=np.argsort(p,kind='stable');p=p[order];positive=positive[order];w=w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(p)!=0)+1]
    pos=np.add.reduceat(w*positive,starts);neg=np.add.reduceat(w*(~positive),starts)
    if not pos.sum() or not neg.sum():return None
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))

def bootstrap_auc(y,p,w,date_ix,multiplicities):
    order=np.argsort(p,kind='stable');score=p[order];positive=(y[order]>0).astype(float)
    start=np.r_[0,np.flatnonzero(np.diff(score)!=0)+1];result=[]
    for lo in range(0,len(multiplicities),32):
        weight=multiplicities[lo:lo+32,date_ix[order]]*w[order]
        pos=np.add.reduceat(weight*positive,start,axis=1);neg=np.add.reduceat(weight*(1-positive),start,axis=1)
        denominator=pos.sum(axis=1)*neg.sum(axis=1)
        value=np.divide((pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1),denominator,
            out=np.full(len(weight),np.nan),where=denominator>0)
        result.extend(value)
    return np.array(result)

def statistics(data,calendar):
    if not data:return [],[],[]
    y=np.array([r['target'] for r in data]);observed_days=sorted({r['day'] for r in data})
    days=[d for d in calendar if observed_days[0]<=d<=observed_days[-1]];index={d:i for i,d in enumerate(days)}
    date_ix=np.array([index[r['day']] for r in data]);counts=Counter(r['day']+'/'+r['symbol'] for r in data)
    w=np.array([1/counts[r['day']+'/'+r['symbol']] for r in data]);summary=[];deciles=[];uncertainty=[]
    values={name:np.array([r['prediction'][name] for r in data]) for name in ['mean','affine','ridge','raw']}
    for name,pred in values.items():
        auc=weighted_auc(y,pred,w)
        # Raw equity-bp score has ranking value; it is not a notional-bp forecast.
        metrics=dict(method=name,episodes=len(data),issuer_days=len(counts),observed_sessions=len(observed_days),calendar_sessions=len(days),auc=auc,
            mae=None,rmse=None,bias=None,calibration_slope=None,mean_regret_bps=None)
        if name!='raw':
            pm=np.average(pred,weights=w);ym=np.average(y,weights=w);variance=np.sum(w*(pred-pm)**2)
            metrics.update(mae=float(np.average(abs(pred-y),weights=w)),rmse=float(np.sqrt(np.average((pred-y)**2,weights=w))),
                bias=float(np.average(pred-y,weights=w)),calibration_slope=float(np.sum(w*(pred-pm)*(y-ym))/variance) if variance>1e-12 else None,
                mean_regret_bps=float(np.average(np.maximum(y,0)-(pred>0)*y,weights=w)))
        summary.append(metrics)
        for d in range(1,11):
            mask=np.array([r['deciles'][name]==d for r in data])
            deciles.append(dict(method=name,decile=d,episodes=int(mask.sum()),expectancy_bps=float(np.average(y[mask],weights=w[mask])) if mask.any() else None,
                sessions=len({r['day'] for r,ok in zip(data,mask) if ok}),boundaries='training predictions only'))
    denominator=np.bincount(date_ix,weights=w,minlength=len(days))
    for length in [5,10,20]:
        rng=np.random.default_rng(43018+length+len(days));starts=rng.integers(0,len(days),(4000,math.ceil(len(days)/length)))
        ix=((starts[:,:,None]+np.arange(length))%len(days)).reshape(4000,-1)[:,:len(days)]
        multiplicities=np.zeros((4000,len(days)),dtype=np.int16)
        np.add.at(multiplicities,(np.arange(4000)[:,None],ix),1)
        auc_samples={name:bootstrap_auc(y,p,w,date_ix,multiplicities) for name,p in values.items()}
        for name,pred in values.items():
            for reference in [None,'raw'] if name!='raw' else [None]:
                sample=auc_samples[name]-(auc_samples[reference] if reference else 0)
                sample=sample[np.isfinite(sample)];point=weighted_auc(y,pred,w)
                if point is None or not len(sample):continue
                if reference:
                    other=weighted_auc(y,values[reference],w)
                    if other is None:continue
                    point-=other
                lo,hi=np.quantile(sample,[.025,.975]);se=float(sample.std(ddof=1))
                uncertainty.append(dict(candidate=name,reference=reference or 'absolute',metric='AUC',block_sessions=length,
                    delta=point,lower95=float(lo),upper95=float(hi),SE=se,conditional_MDE=2.8016*se,
                    effective_independent_sample_size='unidentified',search_adjusted=False))
            for decile in range(1,11):
                mask=np.array([r['deciles'][name]==decile for r in data]);dw=w*mask
                den=np.bincount(date_ix,weights=dw,minlength=len(days));num=np.bincount(date_ix,weights=dw*y,minlength=len(days))
                denominator_sample=den[ix].sum(axis=1)
                sample=np.divide(num[ix].sum(axis=1),denominator_sample,out=np.full(4000,np.nan),where=denominator_sample>0)
                finite=sample[np.isfinite(sample)]
                if not len(finite):continue
                lo,hi=np.quantile(finite,[.025,.975]);se=float(finite.std(ddof=1))
                uncertainty.append(dict(candidate=name,reference='decile'+str(decile),metric='expectancy_bps',block_sessions=length,
                    delta=float(num.sum()/den.sum()),lower95=float(lo),upper95=float(hi),SE=se,conditional_MDE=2.8016*se,
                    effective_independent_sample_size='unidentified',search_adjusted=False))
        for candidate in ['affine','ridge']:
            for reference in ['mean','affine']:
                if candidate==reference:continue
                for metric in ['absolute_error','regret']:
                    def loss(p):return abs(p-y) if metric=='absolute_error' else np.maximum(y,0)-(p>0)*y
                    difference=loss(values[candidate])-loss(values[reference]);num=np.bincount(date_ix,weights=w*difference,minlength=len(days))
                    samples=num[ix].sum(axis=1)/denominator[ix].sum(axis=1);samples=samples[np.isfinite(samples)];se=float(samples.std(ddof=1));lo,hi=np.quantile(samples,[.025,.975])
                    uncertainty.append(dict(candidate=candidate,reference=reference,metric=metric,block_sessions=length,
                        delta=float(np.average(difference,weights=w)),lower95=float(lo),upper95=float(hi),SE=se,conditional_MDE=2.8016*se,
                        effective_independent_sample_size='unidentified',search_adjusted=False))
    return summary,deciles,uncertainty

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);a=ap.parse_args();reg=read(a.registration)
    for b in reg['bindings']:verify(b)
    root=Path(a.registration).parent;out=root/'results';out.mkdir(exist_ok=False);start=time.monotonic()
    data=defaultdict(list);status=Counter();holding_coverage=Counter();calendars={};connections={};barcache={}
    labelreg=read(reg['label_registration']['path'])
    for period in ['is','oos']:
        plan=read(labelreg['profiles'][period]['plan']['path']);verify(dict(path=plan['database'],sha256=plan['input_sha256'][plan['database']]))
        connections[period]=sqlite3.connect('file:'+Path(plan['database']).as_posix()+'?mode=ro',uri=True)
        calendars[period]=[x['day'] for x in read(Path(reg['controls'][period]['folder'])/'daily.json')]
    for period in ['is','oos']:
        for scenario in ['baseline','combined']:
            item=reg['labels'][period][scenario]
            for b in item.values():verify(b)
            labels={r['episode_id']+'/'+str(r['horizon_minutes']):(n,r) for n,r in rows(item['labels']['path'])}
            for episode,group in groupby(rows(item['details']['path']),lambda nr:nr[1]['episode_id']):
                details={r['horizon_minutes']:(n,r) for n,r in group};numbers={h:labels[episode+'/'+str(h)] for h in [0,30,60,120]}
                for h,(_,lab) in numbers.items():status[(period,scenario,str(h),lab['status'])]+=1
                labelrow,lab=numbers[0];features=lab['source_features']
                common=dict(episode_id=episode,day=lab['day'],symbol=lab['symbol'],cohort=lab['cohort'],scenario=scenario,period=period,
                    source_label=item['labels']['path'],source_label_row=labelrow,source_detail=item['details']['path'],source_detail_row=details[0][0])
                if lab['label_valid'] and lab['filled_entry_orders']==1:
                    known=lab['original_clock_quantity_feature_available']
                    notional=lab['native_requested_quantity']*lab['native_entry_reference'] if known and lab['native_entry_reference'] else None
                    data['entry',scenario].append(dict(**common,target=lab['net_value_notional_bps'],target_cash=lab['net_cash_krw'],
                        features=[lab['raw_score'],features.get('first30_ret'),features.get('first30_vwap_ret'),features.get('first30_range_atr'),
                            features.get('first30_rel_volume_log'),features.get('daily_return_20d'),lab['native_quantity_observed_capacity']],
                        feature_available_at=lab['feature_available_at'],label_mature_at=lab['label_mature_at'],decision_notional=notional))
                key=(period,lab['day'],lab['symbol'])
                if key not in barcache:
                    q=connections[period].execute('SELECT ts,h,l,c,v FROM bars WHERE day=? AND symbol=? ORDER BY ts',(lab['day'],lab['symbol']))
                    barcache[key]=[(datetime.fromisoformat(ts),h,l,c,v) for ts,h,l,c,v in q]
                contrast,reason=holding_pair(details[0][1],details[120][1],lab,numbers[120][1],barcache[key],reg['participation'][scenario])
                holding_coverage[(period,scenario,reason)]+=1
                if contrast:data['holding',scenario].append(dict(**common,**contrast,extended_source_detail_row=details[120][0]))
    for conn in connections.values():conn.close()
    calendar=sorted(set(calendars['is']+calendars['oos']));summaries=[];deciles=[];cis=[];coverage=[];tails=[];recipient=[]
    with gzip.open(out/'diagnostic_rows.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as raw,gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as predictions:
        for (target,scenario),dataset in data.items():
            for r in dataset:emit(raw,dict(target_name=target,**r))
            predicted,models,cover=chronological(dataset,calendar)
            save(out/(target+'_'+scenario+'_models.json'),models)
            for r in predicted:emit(predictions,dict(target_name=target,**r))
            strata={'all':predicted}
            for dimension in ['period','cohort']:
                for value in sorted({r[dimension] for r in predicted}):strata[dimension+':'+value]=[r for r in predicted if r[dimension]==value]
            for stratum,observations in strata.items():
                s,d,c=statistics(observations,calendar)
                summaries.extend(dict(target=target,scenario=scenario,stratum=stratum,**r) for r in s)
                deciles.extend(dict(target=target,scenario=scenario,stratum=stratum,**r) for r in d)
                cis.extend(dict(target=target,scenario=scenario,stratum=stratum,**r) for r in c)
            coverage.append(dict(target=target,scenario=scenario,**cover))
            for method in ['mean','affine','ridge']:
                for direction in ['sacrificed_positive','avoided_negative']:
                    selected=[r for r in predicted if r['prediction'][method]<=0 and ((r['target_cash']>0)==(direction=='sacrificed_positive'))]
                    seen=set()
                    for r in sorted(selected,key=lambda r:-abs(r['target_cash'])):
                        key=(r['day'],r['symbol'])
                        if key in seen:continue
                        seen.add(key)
                        tails.append(dict(target=target,scenario=scenario,method=method,direction=direction,episode=r['episode_id'],day=r['day'],symbol=r['symbol'],
                            target_cash=r['target_cash'],target_bps=r['target'],predicted_bps=r['prediction'][method],source_detail=r['source_detail'],source_detail_row=r['source_detail_row']))
                        if len(seen)>=10:break
            if target=='entry':
                groups=defaultdict(list)
                for r in predicted:
                    if r['decision_notional']:groups[r['day'],r['cohort'],r['feature_available_at']].append(r)
                for key,group in groups.items():
                    if len(group)<2:continue
                    best=max(group,key=lambda r:r['target_cash'])
                    for method in ['mean','affine','ridge','raw']:
                        predicted_cash=lambda r:r['prediction'][method]*(1e8 if method=='raw' else r['decision_notional'])/1e4
                        choice=max(group,key=predicted_cash);chosen=choice if predicted_cash(choice)>0 else None
                        realised=chosen['target_cash'] if chosen else 0.
                        recipient.append(dict(scenario=scenario,day=key[0],cohort=key[1],available_at=key[2],method=method,supported_candidates=len(group),
                            chosen_episode=chosen['episode_id'] if chosen else '',hindsight_episode=best['episode_id'] if best['target_cash']>0 else '',
                            chosen_cash=realised,local_regret_cash=max(0,best['target_cash'])-realised,
                            scope='same-clock standalone recipient diagnostic; no funded displacement or attainable portfolio sum'))
    csvsave(out/'summary.csv',summaries);csvsave(out/'deciles.csv',deciles);csvsave(out/'paired_uncertainty.csv',cis)
    csvsave(out/'tail_cases.csv',tails);csvsave(out/'recipient_regret.csv',recipient)
    save(out/'coverage.json',dict(labels=[dict(period=p,scenario=s,horizon=h,status=k,count=v) for (p,s,h,k),v in status.items()],
        holding=[dict(period=p,scenario=s,reason=k,count=v) for (p,s,k),v in holding_coverage.items()],prediction=coverage))
    save(out/'completion.json',dict(passed=True,seconds=time.monotonic()-start,financial_evaluations=0,eligible_for_scoring=False,
        external_validation=False,holdout_accesses=0,files={p.name:bound(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':main()
