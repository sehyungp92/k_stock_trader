"""Quantity-consistent entry budgets and a separately registered joint stress cap.

No market I/O: consume only the source-authenticated prior-session panel in state.
Pending sells never release cash or risk. Historical loss envelopes are scenarios,
not guaranteed future loss bounds or predicted executable stop prices.
"""
from collections import defaultdict
from dataclasses import replace
from math import isfinite
from statistics import mean
from .liquidation_tail_v3 import KEY, scenario, evidence as legacy_evidence
from .opening_tail import tail_mean
from deployment.olr_kalcb.intent_allocator import allocate_entry_intents,IntentAllocationConfig

def history(state,at):
    h=state.meta.get(KEY)
    if at.tzinfo is None or not h or not h['through'] or h['through']>=at.date().isoformat():
        raise ValueError('Risk repair requires strictly prior completed history')
    if any(r['day']>=at.date().isoformat() for r in h['observations']):raise ValueError('Future risk scenario')
    return h

def quote(state,symbol,sector,at,notional,participation,mode):
    if mode in ('off','quantity'):return legacy_evidence(state,symbol,sector,at,notional,participation)
    h=history(state,at)
    if mode=='portfolio':
        return dict(version='original-atr-with-separate-joint-stress-v4',fraction=0.,through=h['through'],notional_upper=notional,
                    participation=participation,sessions=len(h['sessions']))
    if mode!='separated':raise ValueError('Unknown risk mode')
    own=[r for r in h['observations'] if r['symbol']==symbol]
    if len(own)<20:raise ValueError('Insufficient own liquidation dates')
    pool=[r for r in h['observations'] if r['symbol']!=symbol and r['sector']==sector];pool_name='sector_price_peers'
    if len({r['symbol'] for r in pool})<3 or len({r['day'] for r in pool})<20:
        pool=[r for r in h['observations'] if r['symbol']!=symbol];pool_name='universe_price_peers'
    byday=defaultdict(list);names=set();unsupported=0
    for r in pool:
        price=scenario(r,r['reference_price'],participation)
        if price['unfilled_shares']:unsupported+=1;continue
        byday[r['day']].append(price['fraction']);names.add(r['symbol'])
    if len(names)<3 or len(byday)<20:raise ValueError('Insufficient independent peer support')
    oq=[scenario(r,notional,participation) for r in own]
    a=tail_mean([q['fraction'] for q in oq]);z=tail_mean([mean(v) for v in byday.values()]);weight=len(own)/(len(own)+20)
    return dict(version='own-liquidity-peer-price-es90-v4',fraction=max(a,weight*a+(1-weight)*z),
        through=h['through'],notional_upper=notional,participation=participation,sessions=len(h['sessions']),
        symbol_observations=len(own),peer_observations=sum(map(len,byday.values())),peer_names=len(names),peer_dates=len(byday),
        pool=pool_name,symbol_tail_mean=a,peer_tail_mean=z,symbol_weight=weight,
        own_incomplete_scenarios=sum(x['unfilled_shares']>0 for x in oq),unsupported_peer_price_rows=unsupported,
        aligned_own_loss_fractions={r['day']:q['fraction'] for r,q in zip(own,oq)})

def largest_feasible(upper,feasible):
    low,high=0,int(upper)
    while low<high:
        mid=(low+high+1)//2
        if feasible(mid):low=mid
        else:high=mid-1
    return low

def exposures(state):
    from .logic import pending_entry_reservation
    result={};marks=state.meta.get('risk_repair_marks_v4',{})
    def add(symbol,sector,qty,price):
        if qty<=0:return
        row=result.setdefault(symbol,dict(sector=sector,notional=0.))
        if row['sector']!=sector:raise ValueError('Inconsistent position sector')
        row['notional']+=qty*price
    for symbol,ss in state.symbols.items():
        if ss.position is not None:
            p=ss.position;mark=marks.get(symbol)
            if not mark:raise ValueError('Missing causal mark for held shock exposure')
            add(symbol,p.sector,p.qty_open,mark['price'])
        if ss.pending_entry_order_id:
            qty,_=pending_entry_reservation(ss);m=ss.pending_entry_metadata
            add(symbol,m['sector'],qty,m['estimated_entry_price'])
    return result

def joint_stress(state,at,positions,participation):
    h=history(state,at)
    if not positions:return dict(supported=True,common_dates=0,total_loss=0.,sector_losses={},worst_day=None)
    grouped={s:{} for s in positions}
    for r in h['observations']:
        if r['symbol'] in grouped:grouped[r['symbol']][r['day']]=r
    common=sorted(set.intersection(*(set(v) for v in grouped.values())))
    if len(common)<20:return dict(supported=False,common_dates=len(common),total_loss=None,sector_losses={},worst_day=None)
    totals={};sector_paths=defaultdict(dict)
    costs=positions.get('__costs__',0.) if '__costs__' in positions else 0.
    for day in common:
        loss=defaultdict(float)
        for symbol,item in positions.items():
            frac=scenario(grouped[symbol][day],item['notional'],participation)['fraction']
            loss[item['sector']]+=item['notional']*(frac+item.get('cost_fraction',0.))
        totals[day]=sum(loss.values())
        for sec,value in loss.items():sector_paths[sec][day]=value
    worst=max(totals,key=totals.get)
    return dict(supported=True,common_dates=len(common),total_loss=totals[worst],worst_day=worst,
        sector_losses={sec:max(v.values()) for sec,v in sector_paths.items()},
        through=h['through'],method='maximum aligned prior-date own-liquidation loss plus costs')

def allocate(intents,account,at,config,state,by_id,*,prior=()):
    from .loss_budget import committed,unit_loss
    symbols,sectors=committed(state,config);running=account;result=[]
    prior_qty={d.intent_id:d.approved_qty for d in prior}
    portfolio=exposures(state) if config.capacity_risk_repair_mode=='portfolio' else None
    costs=(2*config.commission_bps+config.tax_bps_on_sell+2*config.slippage_bps)/10000
    if portfolio is not None:
        for item in portfolio.values():item['cost_fraction']=costs
    def add_amount(symbol,sector,amount):
        symbols[symbol]=symbols.get(symbol,0.)+amount;sectors[sector]=sectors.get(sector,0.)+amount
    def add_exposure(action,q):
        if portfolio is None or not q:return
        item=portfolio.setdefault(action.symbol,dict(sector=action.metadata['sector'],notional=0.,cost_fraction=costs))
        item['notional']+=q*action.metadata['estimated_entry_price']
    for d in prior:
        action=by_id[d.intent_id];add_amount(action.symbol,action.metadata['sector'],d.approved_qty*unit_loss(action.metadata,config));add_exposure(action,d.approved_qty)
    for intent in sorted(intents,key=lambda i:(i.native_rank,i.available_at,i.strategy_id,i.symbol,i.intent_id)):
        action=by_id[intent.intent_id];m=dict(action.metadata);sector=m['sector'];price=m['estimated_entry_price']
        old_q=prior_qty.get(intent.intent_id,0);old_charge=old_q*unit_loss(m,config) if old_q else 0.
        funding=allocate_entry_intents([intent],running,at,IntentAllocationConfig())[0]
        cache={}
        def assess(extra):
            if extra in cache:return cache[extra]
            qty=old_q+extra
            if qty:
                ev=quote(state,intent.symbol,sector,at,qty*price,config.capacity_participation,config.capacity_risk_repair_mode)
                total=qty*unit_loss({**m,'liquidation_tail_evidence':ev},config)
            else:ev=m.get('liquidation_tail_evidence',{});total=0.
            delta=total-old_charge;fail=[]
            for label,fraction,used in [('position_loss_budget',config.capacity_loss_position_pct,symbols.get(intent.symbol,0.)),
                ('sector_loss_budget',config.capacity_loss_sector_pct,sectors.get(sector,0.)),
                ('total_loss_budget',config.capacity_loss_total_pct,sum(symbols.values()))]:
                if fraction and used+delta>account.equity*fraction+1e-7:fail.append(label)
            shock=None
            if portfolio is not None:
                proposed={s:dict(v) for s,v in portfolio.items()}
                if extra:
                    item=proposed.setdefault(intent.symbol,dict(sector=sector,notional=0.,cost_fraction=costs));item['notional']+=extra*price
                shock=joint_stress(state,at,proposed,config.capacity_participation)
                if not shock['supported']:fail.append('joint_history_support')
                else:
                    if shock['total_loss']>account.equity*config.capacity_loss_total_pct+1e-7:fail.append('joint_total_loss_budget')
                    if any(v>account.equity*config.capacity_loss_sector_pct+1e-7 for v in shock['sector_losses'].values()):fail.append('joint_sector_loss_budget')
            cache[extra]=(not fail,ev,delta,fail,shock);return cache[extra]
        qty=largest_feasible(funding.approved_qty,lambda q:assess(q)[0])
        if qty<intent.minimum_qty:qty=0
        valid,ev,delta,fail,shock=assess(qty)
        if qty and not valid:raise AssertionError('Infeasible approved risk quantity')
        reason=funding.reason;binding=funding.binding_limit
        if qty<funding.approved_qty:
            next_fail=assess(qty+1)[3]
            binding=next_fail[0] if next_fail else 'minimum_qty';reason='resized' if qty else 'insufficient_'+binding
        updated=replace(funding,approved_qty=qty,reserved_notional=qty*intent.unit_cost,reason=reason,binding_limit=binding)
        m.update(liquidation_tail_evidence=ev,risk_repair_evidence={
            'mode':config.capacity_risk_repair_mode,'quantity_before':old_q,'approved_increment':qty,'total_quote_quantity':old_q+qty,
            'funding_feasible_increment':funding.approved_qty,'newly_reserved_loss':delta if qty else 0.,
            'risk_used_before':{'symbol':symbols.get(intent.symbol,0.),'sector':sectors.get(sector,0.),'portfolio':sum(symbols.values())},
            'equity':account.equity,'joint_stress':shock,'binding_limit':binding})
        by_id[intent.intent_id]=replace(action,metadata=m)
        if qty:
            add_amount(intent.symbol,sector,delta);add_exposure(action,qty)
            cash=qty*intent.unit_cost
            symcash=dict(running.symbol_committed);seccash=dict(running.sector_committed);strcash=dict(running.strategy_committed)
            symcash[intent.symbol]=symcash.get(intent.symbol,0.)+cash;seccash[sector]=seccash.get(sector,0.)+cash;strcash[intent.strategy_id]=strcash.get(intent.strategy_id,0.)+cash
            running=replace(running,funding_limit=max(0.,running.funding_limit-cash),gross_committed=running.gross_committed+cash,
                symbol_committed=symcash,sector_committed=seccash,strategy_committed=strcash,occupied=running.occupied|{(intent.strategy_id,intent.symbol)})
        result.append(updated)
    return result
