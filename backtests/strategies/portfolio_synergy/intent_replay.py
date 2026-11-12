"""Opt-in intent allocation around unchanged native signal/exit policies."""
from collections import Counter, defaultdict, deque
from dataclasses import asdict, replace
from datetime import datetime, timedelta, time
import math
from statistics import median

from backtests.auto.shared.cache_keys import stable_signature
from backtests.engine.capacity import closing_auction_event, continuous_interval
from backtests.strategies.kalcb.runner import KALCBReplayAdapter
from backtests.strategies.olr.runner import OLRReplayAdapter
from deployment.olr_kalcb.intent_allocator import (
    AllocationAccount, EntryIntent, IntentAllocationConfig, allocate_entry_intents)
from strategy_common.actions import SubmitEntry, CancelOrders
from strategy_common.events import DecisionEvent
from strategy_common.execution_session import session_cutoff
from strategy_kalcb.core.state import SymbolStage
from .source_replay import NativeRiskSharedBroker


class ReferenceSizingBroker(NativeRiskSharedBroker):
    """Replay standalone risk-sizing alternatives with physical leverage fixed."""
    def __init__(self,*args,reference_sizing=1.,**kwargs):
        if not math.isfinite(reference_sizing) or not .25<=reference_sizing<=2:
            raise ValueError('Unsupported reference sizing')
        super().__init__(*args,**kwargs)
        self.reference_sizing=reference_sizing

    def strategy_sizing_equity(self):
        return super().strategy_sizing_equity()*self.reference_sizing

    def strategy_risk_cash(self,strategy_id):
        return self.strategy_sizing_equity()-self._strategy_open_notional(strategy_id)


class IntentAdapterMixin:
    def _submit_actions(self, actions, broker, submitted_at):
        if self.strategy_id=='KALCB' and broker.allocation_config.preserve_kalcb_submission:
            for action in actions:
                super()._submit_actions([action],broker,submitted_at)
                if isinstance(action,SubmitEntry):
                    available=broker._decision_available_at or submitted_at
                    broker.allocation_events.append({'event':'native_submission','strategy_id':'KALCB',
                        'symbol':action.symbol,'native_qty':action.qty,'native_metadata':dict(action.metadata),
                        'available_at':available.isoformat(),'timestamp':submitted_at.isoformat(),
                        'intent_id':stable_signature(['KALCB',action.symbol,available.isoformat(),action.metadata.get('candidate_hash'),action.reason]),
                        'submitted_order_id':self.state.symbol_state(action.symbol).pending_entry_order_id})
            return
        for action in actions:
            if isinstance(action, SubmitEntry):
                broker.stage_entry_intent(self, action, submitted_at)
                if self.strategy_id == 'KALCB':
                    # Proposals are not working orders and must not consume
                    # the next symbol's native pending-position slots.
                    state=self.state.symbol_state(action.symbol)
                    state.stage=SymbolStage.WATCHING
                    state.pending_entry_metadata.clear()
            else:
                if isinstance(action, CancelOrders):
                    broker.cancel_entry_intent(self.strategy_id, action.symbol, submitted_at, action.reason)
                super()._submit_actions([action], broker, submitted_at)

    def submit_allocated_entry(self, action, broker, submitted_at):
        super()._submit_actions([action], broker, submitted_at)
        if self.strategy_id == 'KALCB':
            self.state.symbol_state(action.symbol).stage=SymbolStage.ENTRY_QUEUED


class IntentKALCBReplayAdapter(IntentAdapterMixin, KALCBReplayAdapter):
    def _portfolio_view(self, broker):
        view=super()._portfolio_view(broker)
        fraction=broker.allocation_config.kalcb_sizing_fraction
        if fraction==1:return view
        equity=broker.strategy_sizing_equity()*fraction
        return replace(view,equity=equity,cash=equity-broker._strategy_open_notional('KALCB'))


class IntentOLRReplayAdapter(IntentAdapterMixin, OLRReplayAdapter):
    def _portfolio_view(self, broker):
        view=super()._portfolio_view(broker)
        equity=view.equity*broker.allocation_config.olr_sizing_fraction
        return replace(view,equity=equity,cash=min(view.cash,equity),external_entry_allocation=True)


def _quantile(values, q):
    ordered=sorted(values);at=(len(ordered)-1)*q;lo=int(at);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)


class IntentSharedBroker(NativeRiskSharedBroker):
    def __init__(self, *args, allocation_config, value_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.portfolio_policy is None:
            raise ValueError('Intent allocation requires explicit portfolio caps')
        self.allocation_config=allocation_config
        self.value_model=value_model
        self.entry_intents={}
        self.allocation_events=[]
        self.auction_history=defaultdict(lambda:deque(maxlen=20))
        self.continuous_history=defaultdict(lambda:deque(maxlen=3))
        self._observed_liquidity=set()

    def stage_entry_intent(self, adapter, action, submitted_at):
        available=self._decision_available_at or submitted_at
        key=(action.strategy_id,action.symbol)
        if key in self.entry_intents:
            # Native re-observation cannot refresh an old signal's TTL.
            return
        expiry=available+timedelta(minutes=self.allocation_config.defer_minutes)
        for name in ('expiry_timestamp','expiry_ts'):
            raw=action.metadata.get(name)
            if raw:
                end=datetime.fromisoformat(str(raw)) if not isinstance(raw,datetime) else raw
                expiry=min(expiry,end)
        entry_end=session_cutoff(available.date(),time(15,20))
        expiry=min(expiry,available.replace(hour=entry_end.hour,minute=entry_end.minute,second=0,microsecond=0))
        if action.strategy_id=='KALCB':
            # Never carry an intraday entry beyond its native flatten clock.
            raw=adapter.config.flatten_time
            cutoff=session_cutoff(available.date(),time.fromisoformat(raw) if isinstance(raw,str) else raw)
            expiry=min(expiry,available.replace(hour=cutoff.hour,minute=cutoff.minute,second=0,microsecond=0))
        raw=dict(action.metadata)
        intent_id=stable_signature([action.strategy_id,action.symbol,available.isoformat(),raw.get('candidate_hash'),action.reason])
        row={'adapter':adapter,'action':action,'origin':available,'expiry':max(available,expiry),'intent_id':intent_id}
        self.entry_intents[key]=row
        self.allocation_events.append({'event':'entry_intent','intent_id':intent_id,'strategy_id':action.strategy_id,
            'symbol':action.symbol,'available_at':available.isoformat(),'expires_at':row['expiry'].isoformat(),
            'native_qty':action.qty,'native_metadata':raw})

    def cancel_entry_intent(self, sid, symbol, at, reason):
        item=self.entry_intents.pop((sid,symbol),None)
        if item:
            self.allocation_events.append({'event':'intent_cancel','intent_id':item['intent_id'],
                'timestamp':at.isoformat(),'reason':reason})

    def _observe_liquidity(self, bars):
        for bar in bars:
            key=(bar.symbol,bar.timestamp)
            if key in self._observed_liquidity:continue
            self._observed_liquidity.add(key)
            if closing_auction_event(bar):
                self.auction_history[bar.symbol].append((bar.timestamp,float(bar.volume)))
            elif continuous_interval(bar):
                self.continuous_history[bar.symbol].append((bar.timestamp,float(bar.volume)))

    def _capacity_forecast(self, sid, symbol, at):
        cfg=self.allocation_config;limits=[];evidence={}
        fraction=self.capacity_policy.participation if self.capacity_policy else 0
        if cfg.entry_capacity_bars>0:
            observations=[(t,v) for t,v in self.continuous_history[symbol]
                if t.date()==at.date() and t+timedelta(minutes=5)<=at and at-t<=timedelta(minutes=20)]
            qty=math.floor(median([v for _,v in observations])*fraction*cfg.entry_capacity_bars) if observations else 0
            committed=sum(int(o.qty or 0) for o in self.orders if o.symbol==symbol and
                (o.side=='BUY' or (o.side=='SELL' and o.order_type=='MARKET')))
            qty=max(0,qty-committed)
            limits.append(qty)
            evidence['entry']={'samples':[(t.isoformat(),v) for t,v in observations],'quantity':qty,'committed_shares':committed}
        if sid=='OLR' and cfg.exit_capacity_quantile>0:
            observations=[(t,v) for t,v in self.auction_history[symbol] if t.date()<at.date()]
            if len(observations)>=cfg.minimum_exit_observations:
                qty=math.floor(_quantile([v for _,v in observations],cfg.exit_capacity_quantile)*fraction)
                limits.append(qty)
                evidence['exit']={'samples':[(t.isoformat(),v) for t,v in observations],'quantity':qty}
            else:
                evidence['exit']={'status':'insufficient_history_native_cap_fallback','sample_count':len(observations)}
        return min(limits) if limits else None,evidence

    def _account(self, adapters):
        equity=max(0,self._portfolio_equity());caps=self.portfolio_policy.effective_notional_caps(equity)
        symbol=defaultdict(float);sector=defaultdict(float);strategy=defaultdict(float);occupied=set()
        slots=Counter();gross=0.;reserved=0.;holdings=[];pending=[]
        def add(sid,sym,amount):
            nonlocal gross
            sec=self.policy_sectors.get(sym,'UNKNOWN');gross+=amount
            symbol[sym]+=amount;sector[sec]+=amount;strategy[sid]+=amount
            if (sid,sym) not in occupied:slots[sid,sec]+=1
            occupied.add((sid,sym))
        for (sid,sym),p in self.positions.items():
            amount=p.qty*self.last_prices.get(sym,p.avg_price);add(sid,sym,amount)
            holdings.append({'strategy_id':sid,'symbol':sym,'qty':p.qty,'notional':amount})
        held=gross
        for order in self.orders:
            if order.side!='BUY':continue
            costs=self.strategy_costs[order.strategy_id]
            unit=max(float(order.metadata.get('allocation_unit_cost',0)),
                self.last_prices.get(order.symbol,0)*(1+(costs.commission_bps+costs.slippage_bps)/10000))
            amount=int(order.qty or 0)*unit;reserved+=amount;add(order.strategy_id,order.symbol,amount)
            pending.append({'order_id':order.order_id,'strategy_id':order.strategy_id,'symbol':order.symbol,
                'remaining_qty':order.qty,'reserved_notional':amount})
        native=self.portfolio_policy.config.funding_mode=='native_buying_power'
        funding=max(0,equity*self.buying_power_leverage-held-reserved if native else self.cash-reserved)
        cfg=self.allocation_config
        limits={'KALCB':min(self.strategy_leverage['KALCB'],cfg.kalcb_budget_fraction)*equity,
                'OLR':min(self.strategy_leverage['OLR'],cfg.olr_budget_fraction)*equity}
        # OLR's overnight_slot_count selects daily candidates; it is not a
        # native hard limit on carryover positions from earlier sessions.
        max_positions={a.strategy_id:a.config.max_positions for a in adapters if a.strategy_id=='KALCB'}
        max_sector={a.strategy_id:a.config.max_per_sector for a in adapters if a.strategy_id=='KALCB'}
        return AllocationAccount(equity,caps['max_gross_notional'],caps['max_symbol_notional'],caps['max_sector_notional'],
            funding,limits,gross,dict(symbol),dict(sector),dict(strategy),frozenset(occupied),dict(slots),
            max_positions,max_sector,self.portfolio_policy.config.same_symbol_policy=='block'),holdings,pending

    def finalize_entry_intents(self, bars, adapters):
        self._observe_liquidity(bars)
        if not self.entry_intents:return []
        stamp=bars[0].timestamp;at=stamp+timedelta(minutes=5)
        by_symbol={b.symbol:b for b in bars};intents=[];sources={};events=[]
        for key,item in list(self.entry_intents.items()):
            action=item['action'];state=item['adapter'].state.symbol_state(action.symbol)
            if at>item['expiry'] or at.date()!=item['origin'].date():
                self.cancel_entry_intent(*key,at,'signal_validity_expired')
                if key[0]=='KALCB' and not state.pending_entry_order_id and state.position is None:
                    state.stage=SymbolStage.WATCHING
                continue
            if action.symbol not in by_symbol:continue
            bar=by_symbol[action.symbol]
            if not continuous_interval(bar):continue
            price=max(float(bar.close),float(action.metadata.get('estimated_entry_price',0)),float(action.metadata.get('entry_price_ref',0)))
            costs=self.strategy_costs[action.strategy_id];unit=price*(1+(costs.commission_bps+costs.slippage_bps)/10000)
            capacity,evidence=self._capacity_forecast(action.strategy_id,action.symbol,at)
            prediction=self.value_model.prediction(action.strategy_id,action.metadata,at) if self.value_model else None
            # Sparse cohorts do not become synthetic high-confidence forecasts.
            if prediction and prediction['cohort_support']<20:prediction=None
            value=prediction['score_bps'] if prediction else None
            intent=EntryIntent(item['intent_id'],action.strategy_id,action.symbol,
                self.policy_sectors.get(action.symbol,'UNKNOWN'),action.qty,unit,at,item['expiry'],
                int(action.metadata.get('candidate_rank',0) or 1_000_000),capacity,value,
                .25 if action.strategy_id=='KALCB' else 1.5,
                datetime.fromisoformat(prediction['latest_label_exit']) if prediction else None,
                prediction['label_hash'] if prediction else '',evidence.get('entry',{}).get('quantity'))
            intents.append(intent);sources[intent.intent_id]=(key,item,evidence,prediction)
        if not intents:return []
        account,holdings,pending=self._account(adapters)
        decisions=allocate_entry_intents(intents,account,at,self.allocation_config)
        cycle={'event':'allocation_cycle','timestamp':at.isoformat(),'ledger_cursor':len(self.journal),'config':asdict(self.allocation_config),
            'account':{**asdict(account),'occupied':sorted(account.occupied),
                'sector_slots':[[*k,v] for k,v in account.sector_slots.items()]},
            'holdings':holdings,'pending_buys':pending,'intents':[asdict(x) for x in intents],
            'decisions':[asdict(x) for x in decisions],
            'forecasts':{x.intent_id:{'capacity':sources[x.intent_id][2],'value':sources[x.intent_id][3]} for x in intents}}
        self.allocation_events.append(cycle)
        lookup={x.intent_id:x for x in intents}
        for decision in decisions:
            key,item,_,_=sources[decision.intent_id];intent=lookup[decision.intent_id]
            action=item['action'];adapter=item['adapter']
            if decision.approved_qty:
                metadata={**dict(action.metadata),'allocation_intent_id':intent.intent_id,
                    'allocation_origin_at':item['origin'].isoformat(),'allocation_at':at.isoformat(),
                    'allocation_unit_cost':intent.unit_cost,'allocation_native_qty':action.qty,
                    'allocation_rank_basis':decision.rank_basis,'reserved_notional':decision.reserved_notional,
                    'risk_reserved_entry_qty':decision.approved_qty}
                approved=replace(action,qty=decision.approved_qty,metadata=metadata)
                self._decision_available_at=at
                adapter.submit_allocated_entry(approved,self,stamp)
                self.entry_intents.pop(key)
                cycle['decisions'][decisions.index(decision)]['submitted_order_id']=adapter.state.symbol_state(action.symbol).pending_entry_order_id
            elif at>=item['expiry'] or decision.reason in ('nonpositive_supported_value','own_position_or_order'):
                self.cancel_entry_intent(*key,at,decision.reason)
            events.append(DecisionEvent(at,action.strategy_id,action.symbol,
                'PORTFOLIO_ALLOCATED' if decision.approved_qty else 'PORTFOLIO_DEFERRED',decision.reason,
                metadata={**asdict(decision),'native_qty':action.qty,'forecast':sources[intent.intent_id][2]}))
        return events

    def _policy_execution_quantity(self, order, price, quantity, timestamp):
        quantity=super()._policy_execution_quantity(order,price,quantity,timestamp)
        if order.side=='BUY' and quantity:
            cfg=self.allocation_config
            fraction=cfg.olr_budget_fraction if order.strategy_id=='OLR' else cfg.kalcb_budget_fraction
            budget=min(fraction,self.strategy_leverage[order.strategy_id])*max(0,self._portfolio_equity())
            remaining=max(0,budget-self._strategy_open_notional(order.strategy_id))
            unit=price*(1+self.strategy_costs[order.strategy_id].commission_bps/10000)
            approved=min(quantity,max(0,math.floor(remaining/unit)))
            self.journal.append({'event':'strategy_budget_admission','order':asdict(order),
                'timestamp':timestamp.isoformat(),'budget':budget,'own_exposure':self._strategy_open_notional(order.strategy_id),
                'unit_cost':unit,'approved_qty':approved})
            self.allocation_events.append({'event':'strategy_budget_execution','timestamp':timestamp.isoformat(),
                'order_id':order.order_id,'strategy_id':order.strategy_id,'budget':budget,
                'own_exposure':self._strategy_open_notional(order.strategy_id),'unit_cost':unit,
                'offered_qty':quantity,'approved_qty':approved})
            return approved
        return quantity


def audit_intent_allocations(broker):
    """Independently check reservation arithmetic, limits, causality and submits."""
    errors=[];cycles=0;submits={}
    needed={(x['symbol'],stamp) for row in broker.allocation_events if row['event']=='allocation_cycle'
        for x in row['intents'] for part in row['forecasts'][x['intent_id']]['capacity'].values()
        for stamp,_ in part.get('samples',[])}
    source_volumes={(bar['symbol'],bar['timestamp']):bar['volume'] for batch in broker.expected_batches
        for bar in batch if (bar['symbol'],bar['timestamp']) in needed}
    for row in broker.journal:
        if row['event']=='submit' and row['order']['side']=='BUY':
            order=row['order'];submits[order['order_id']]=order
    checked=set();native_submits=0
    for row in broker.allocation_events:
        if row['event']=='native_submission':
            oid=row['submitted_order_id'];order=submits.get(oid)
            if (not broker.allocation_config.preserve_kalcb_submission or row['strategy_id']!='KALCB'
                    or not order or order['strategy_id']!='KALCB' or order['qty']!=row['native_qty']
                    or order['symbol']!=row['symbol']
                    or datetime.fromisoformat(str(order['submitted_at'])).isoformat()!=row['timestamp']
                    or order['metadata'].get('candidate_hash')!=row['native_metadata'].get('candidate_hash')):
                errors.append('native_kalcb_submit_mismatch')
            checked.add(oid);native_submits+=1
        if row['event']=='strategy_budget_execution':
            if row['approved_qty']*row['unit_cost']>max(0,row['budget']-row['own_exposure'])+.001:
                errors.append('strategy_budget_execution_exceeded')
        if row['event']!='allocation_cycle':continue
        cycles+=1;a=row['account'];at=datetime.fromisoformat(row['timestamp']);items={x['intent_id']:x for x in row['intents']}
        gross=a['gross_committed'];added=0.;symbols=dict(a['symbol_committed']);sectors=dict(a['sector_committed']);strategies=dict(a['strategy_committed']);shares=Counter()
        for d in row['decisions']:
            x=items[d['intent_id']];q=d['approved_qty'];amount=q*x['unit_cost']
            if q<0 or q>x['desired_qty'] or abs(amount-d['reserved_notional'])>.001:errors.append('allocation_reservation_mismatch')
            if q and not x['available_at']<=at<=x['expires_at']:errors.append('allocation_outside_validity')
            if x['value_asof'] and x['value_asof']>at:errors.append('allocation_future_value')
            if x['capacity_qty'] is not None and q>x['capacity_qty']:errors.append('allocation_forecast_capacity_exceeded')
            for kind,part in row['forecasts'][x['intent_id']]['capacity'].items():
                for stamp,_ in part.get('samples',[]):
                    observed=datetime.fromisoformat(stamp)
                    if (kind=='entry' and observed+timedelta(minutes=5)>at) or (kind=='exit' and observed.date()>=at.date()):
                        errors.append('allocation_future_capacity')
                if any(source_volumes.get((x['symbol'],stamp))!=volume for stamp,volume in part.get('samples',[])):
                    errors.append('allocation_unbound_capacity_observation')
            if not q:continue
            sid=x['strategy_id'];sym=x['symbol'];sec=x['sector'];gross+=amount;added+=amount
            shares[sym]+=q
            if x['shared_entry_capacity_qty'] is not None and shares[sym]>x['shared_entry_capacity_qty']:
                errors.append('allocation_shared_entry_capacity_exceeded')
            symbols[sym]=symbols.get(sym,0)+amount;sectors[sec]=sectors.get(sec,0)+amount;strategies[sid]=strategies.get(sid,0)+amount
            if (gross>a['gross_limit']+.001 or added>a['funding_limit']+.001 or symbols[sym]>a['symbol_limit']+.001
                    or sectors[sec]>a['sector_limit']+.001 or strategies[sid]>a['strategy_limits'][sid]+.001):
                errors.append('allocation_account_limit_exceeded')
            oid=d.get('submitted_order_id');order=submits.get(oid)
            if (not order or order['qty']!=q or order['metadata'].get('allocation_intent_id')!=x['intent_id']
                    or abs(order['metadata'].get('reserved_notional',0)-amount)>.001):errors.append('allocation_submit_mismatch')
            checked.add(oid)
    if checked!=set(submits):errors.append('entry_without_allocation')
    return {'version':'entry-intent-allocation-audit-v2','pass':not errors,'errors':sorted(set(errors)),
            'cycles':cycles,'approved_entries':len(checked),'native_kalcb_submissions':native_submits,
            'terminal_deferred_intents':len(broker.entry_intents)}
