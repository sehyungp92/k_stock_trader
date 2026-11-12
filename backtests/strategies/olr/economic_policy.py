"""Causal, regularized research policy with native funded execution.

Outcome rows are dated by final label availability, not signal date. Cross-
sectional residuals share historical dates across the whole account. Predictions
are estimates; the native replay is the economic and execution test.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from math import log1p
import hashlib
import json
import numpy as np

from strategy_common.actions import CancelOrders, SubmitEntry, SubmitExit, SubmitPartialExit
from strategy_common.events import DecisionEvent
from strategy_olr.core.joint_allocation import signature

OFFSETS=(0.,30.,240.,1440.,2880.,7200.)


def features(candidate,bars,quantity,*,position=None,at=None):
    if not bars:raise ValueError('Completed causal feature prefix required')
    last=bars[-1];get=lambda b,k:float(b[k] if isinstance(b,dict) else getattr(b,k))
    close=get(last,'close');volume=sum(get(b,'volume') for b in bars)
    vwap=sum(get(b,'close')*get(b,'volume') for b in bars)/max(volume,1.)
    first=get(bars[0],'open');high=max(get(b,'high') for b in bars)
    c=candidate if isinstance(candidate,dict) else vars(candidate) if candidate and hasattr(candidate,'__dict__') else asdict(candidate) if candidate else {}
    if position is None:
        return [float(c.get('rank',3))/6,float(c.get('rs_percentile',.5)),
            float(c.get('selection_score',0))/100,float(c.get('daily_atr',0))/close,
            close/max(vwap,1)-1,close/max(first,1)-1,
            (close-get(last,'low'))/max(get(last,'high')-get(last,'low'),1),
            log1p(quantity/max(volume,1)*1000)]
    entry=float(position['entry_price']);peak=float(position['max_favorable_price'])
    age=(at-datetime.fromisoformat(position['entry_time'])).total_seconds()/86400
    return [close/entry-1,close/max(peak,entry)-1,close/max(vwap,1)-1,
        close/max(first,1)-1,float(c.get('daily_atr',position.get('risk_per_share',0)))/close,
        log1p(quantity/max(volume,1)*1000),min(age,5)/5,
        (at.hour*60+at.minute-540)/390]


class DatedResponseModel:
    def __init__(self, rows, *, minimum_dates=18, penalty=20.):
        self.rows=rows;self.minimum_dates=minimum_dates;self.penalty=penalty
        self.cache={};self.fits=[]

    def fit(self,at,kind,scenario):
        key=(at.date().isoformat(),kind,scenario)
        if key in self.cache:return self.cache[key]
        # A daily freeze excludes even earlier same-day outcomes.
        cutoff=at.replace(hour=0,minute=0,second=0,microsecond=0)
        rows=[r for r in self.rows if r['kind']==kind and r['scenario']==scenario
              and datetime.fromisoformat(r['known_at'])<cutoff]
        days=sorted({r['day'] for r in rows})[-80:]
        rows=[r for r in rows if r['day'] in days]
        if len(days)<self.minimum_dates:self.cache[key]=None;return None
        x=np.array([r['x'] for r in rows]);y=np.array([r['y'] for r in rows])
        counts=defaultdict(int)
        for r in rows:counts[r['day']]+=1
        w=np.array([1/counts[r['day']] for r in rows]);w*=len(rows)/w.sum()
        centre=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-centre)**2,axis=0,weights=w));scale=np.maximum(scale,1e-5)
        z=np.column_stack((np.ones(len(x)),np.clip((x-centre)/scale,-4,4)))
        reg=np.eye(z.shape[1])*self.penalty;reg[0,0]=1e-8
        beta=np.linalg.solve(z.T@(w[:,None]*z)+reg,z.T@(w[:,None]*y))
        residual=y-z@beta
        by_day={d:np.mean([residual[i] for i,r in enumerate(rows) if r['day']==d],axis=0) for d in days}
        bound=signature([{'id':r['id'],'known_at':r['known_at'],'x':r['x'],'y':r['y']} for r in rows])
        out=dict(centre=centre,scale=scale,beta=beta,residuals=by_day,days=days,hash=bound,
                 known_at=max(r['known_at'] for r in rows),n=len(rows))
        self.cache[key]=out
        self.fits.append(dict(day=key[0],kind=kind,scenario=scenario,rows=len(rows),dates=len(days),
            label_max_known_at=out['known_at'],source_sha256=bound,penalty=self.penalty,
            coefficients=beta.tolist(),centre=centre.tolist(),scale=scale.tolist()))
        return out

    @staticmethod
    def predict(fit,x,days):
        z=np.r_[1.,np.clip((np.array(x)-fit['centre'])/fit['scale'],-4,4)]
        mean=z@fit['beta']
        return np.array([mean+fit['residuals'].get(d,np.zeros_like(mean)) for d in days])


@dataclass(frozen=True)
class EconomicSettings:
    allocation: bool=True
    management: bool=True
    maximum_tail_drawdown_fraction: float=.08
    minimum_dates: int=18
    mean_uncertainty_fraction: float=.15
    management_margin_bps: float=15.
    # A fixed, modest cost of tying up capital for one more session.
    capital_occupation_bps: float=2.


class EconomicPolicy:
    def __init__(self,rows,config,scenario='baseline',settings=EconomicSettings()):
        self.model=DatedResponseModel(rows,minimum_dates=settings.minimum_dates)
        self.config=config;self.scenario=scenario;self.settings=settings
        self.observed=None;self.trace=[];self.management_quotes={};self._state=None

    def observe_account(self,observed):self.observed=dict(observed)

    def record_allocation(self,receipt):
        self.trace.append(dict(at=receipt['at'],kind='allocation_result',
            context_sha256=receipt['context_sha256'],native_quantities=receipt['native_quantities'],
            selected_quantities=receipt.get('selected_quantities',{}),
            solver_inputs=receipt.get('solver_inputs'),solver_inputs_sha256=receipt.get('solver_inputs_sha256'),
            outcome=receipt['outcome'],fallback=receipt.get('fallback')))

    def size_envelope(self,at):
        return (self.settings.allocation and self.model.fit(at,'entry',self.scenario) is not None
                and self.model.fit(at,'hold',self.scenario) is not None)

    def on_book(self,state,bars,config,snapshot,portfolio,actions):
        self._state=state;self.management_quotes={};events=[];at=bars[0].timestamp+timedelta(minutes=5)
        fit=self.model.fit(at,'hold',self.scenario)
        if fit is None:return actions,events
        days=fit['days'];already={a.symbol for a in actions if isinstance(a,(SubmitExit,SubmitPartialExit))}
        for bar in bars:
            ss=state.symbol_state(bar.symbol);p=ss.position
            if p is None or bar.timestamp.date()<=p.entry_time.date():continue
            if at.time().hour>=15:continue
            px=float(bar.close);data=asdict(p)
            data['entry_time']=p.entry_time.isoformat();data['risk_per_share']=p.metadata.get('risk_per_share',0)
            x=features(ss.candidate,ss.session_bars,p.qty_open,position=data,at=at)
            pred=self.model.predict(fit,x,days)
            self.management_quotes[bar.symbol]=(pred,p.qty_open*px,fit)
            if not self.settings.management or bar.symbol in already:continue
            pending=ss.pending_exit_order_id
            if pending and state.order_roles.get(pending,{}).get('order_type')!='CLOSE_AUCTION':continue
            if ss.pending_entry_order_id:continue
            if bar.symbol in state.meta.get('priority_exit_pending',{}):continue
            # Target is future native net PNL relative to the current marked
            # holding. Compare with a conservative immediate-exit cost proxy;
            # actual liquidation remains capacity limited in the native engine.
            mean=float(pred[:,-1].mean());se=float(pred[:,-1].std()/np.sqrt(len(days)))
            exit_cost=(config.commission_bps+config.tax_bps_on_sell+config.slippage_bps)/10000
            threshold=-exit_cost-self.settings.management_margin_bps/10000
            retire=mean+self.settings.mean_uncertainty_fraction*se<threshold
            record=dict(at=at.isoformat(),kind='management',symbol=bar.symbol,mean=mean,
                standard_error=se,exit_proxy=-exit_cost,retire=retire,source=fit['hash'],dates=len(days))
            self.trace.append(record)
            if not retire:continue
            reason='economic_remaining_value_exit'
            if pending:
                state.meta.setdefault('priority_exit_pending',{})[bar.symbol]=dict(cancel_order_id=pending,reason=reason)
                actions.append(CancelOrders('OLR',bar.symbol,reason,{'order_id':pending}))
            else:
                from strategy_olr.core.logic import _managed_market_exit_at,_exit_plan,_position_risk
                result=_managed_market_exit_at(state,ss,bar.timestamp,p,_exit_plan(config),reason,_position_risk(p))
                actions.extend(result.actions);events.extend(result.decisions)
        return actions,events

    def entry_context(self,request):
        if not self.settings.allocation:return None
        at=datetime.fromisoformat(request['at']);fit=self.model.fit(at,'entry',self.scenario)
        if fit is None:return None
        state=request['state'];account=request['account'];equity=account['equity'];days=fit['days']
        held=np.zeros((len(days),len(OFFSETS)))
        # Existing inventory and pending commitments consume risk even though
        # their cancellation or sale cannot fund the current cohort.
        for symbol,ss in state['symbols'].items():
            if ss.get('position'):
                pos=ss['position'];bars=ss['session_bars']
                if bars:
                    hfit=self.model.fit(at,'hold',self.scenario)
                    if hfit is None:return None
                    x=features(ss.get('candidate'),bars,pos['qty_open'],position=pos,at=at)
                    held[:,1:]+=self.model.predict(hfit,x,days)*pos['qty_open']*bars[-1]['close']
            if ss.get('pending_entry_order_id'):
                # Conservative residual reservation follows the pooled entry
                # distribution. It does not masquerade as an independent name.
                from strategy_olr.core.logic import pending_entry_reservation
                m=ss['pending_entry_metadata'];q,_=pending_entry_reservation(self._state.symbols[symbol])
                if q and ss['session_bars']:
                    x=features(ss.get('candidate'),ss['session_bars'],q)
                    held[:,1:]+=self.model.predict(fit,x,days)[:,:5]*q*m['estimated_entry_price']
        groups={};native={d['intent_id']:d for d in request['native_decisions']}
        for intent in request['intents']:
            key=intent['intent_id'];ss=state['symbols'][intent['symbol']];unit=intent['unit_cost']
            cap=intent['desired_qty'];old=native[key]['approved_qty']
            # Three positive supported research doses; integer, native funding
            # and risk checks still bind. Forecasts are not fill promises.
            qset={0,min(cap,max(intent['minimum_qty'],int(equity*.15/unit))),min(cap,max(intent['minimum_qty'],int(equity*.35/unit))),cap}
            if old>0 and old<cap:qset.add(old)
            choices=[]
            for q in sorted(q for q in qset if q==0 or q>=intent['minimum_qty']):
                values=np.zeros_like(held)
                completion=np.zeros(len(days))
                if q:
                    x=features(ss.get('candidate'),ss['session_bars'],q)
                    raw=self.model.predict(fit,x,days)
                    prediction=raw[:,:5]
                    completion=np.clip(raw[:,5],0,1) if q*unit>=equity*.01 else completion
                    uncertainty=prediction[:,-1].std()/np.sqrt(len(days))*self.settings.mean_uncertainty_fraction
                    prediction=prediction-uncertainty-self.settings.capital_occupation_bps/10000
                    values[:,1:]=prediction*q*unit
                choices.append(dict(action_id=key+':'+str(q),quantity=q,reserve_cash=q*unit,
                    net_paths=values.tolist(),known_at=fit['known_at'],source_sha256=fit['hash'],supported=True,
                    completed_entry_probabilities=completion.tolist()))
            groups[key]=choices
        current_dd=self.observed['drawdown_fraction']
        risk=self.settings.maximum_tail_drawdown_fraction
        # Preserve the actual peak. Restoration handles already-breached states.
        context=dict(at=request['at'],account_identity=request['account_identity'],actual_high_water=self.observed['high_water'],
            actual_equity=self.observed['equity'],
            scenario_ids=days,probabilities=[1/len(days)]*len(days),offsets=list(OFFSETS),
            known_at=fit['known_at'],source_sha256=fit['hash'],held_net_paths=held.tolist(),meaningful_notional=equity*.01,
            constraints=dict(maximum_tail_drawdown_fraction=risk,restore_feasibility=True,
                recovery_tail_budget_fraction=risk*.25,maximum_combinations=100000),actions=groups)
        self.trace.append(dict(at=request['at'],kind='entry',dates=len(days),source=fit['hash'],
            current_drawdown=current_dd,groups=len(groups),risk_limit=risk))
        return context
