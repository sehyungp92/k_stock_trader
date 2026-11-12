"""Thin C18 standalone labelling over the frozen native core and broker.

No portfolio oracle or policy promotion occurs here. All rows are explicitly
standalone, with native100m sizing and optimistic priority rank1.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from dataclasses import asdict
from collections import defaultdict,Counter
from itertools import groupby
import argparse,json,gzip,hashlib,sys,time,importlib.util

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def emit(f,x):f.write(json.dumps(x,default=str,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);ap.add_argument('--period',required=True)
    ap.add_argument('--scenario',required=True);ap.add_argument('--output',required=True);ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    reg=read(a.registration);profile=reg['profiles'][a.period]
    for b in reg['programs']+[reg['cohorts'],reg['decision_scores'],profile['manifest'],profile['plan']]:assert sha(b['path'])==b['sha256']
    code=Path(profile['manifest']['path']).parent/'files';sys.path.insert(0,str(code))
    hs=importlib.util.spec_from_file_location('c18_horizon',reg['horizon_program']['path']);hm=importlib.util.module_from_spec(hs);hs.loader.exec_module(hm)
    from scripts.kalcb_executed_value_study import load_plan,config_for,connect,load_day
    from backtests.strategies.kalcb import executed_value as ev
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.auto.shared.cache_keys import source_generation_hash
    from kis_core.trading_calendar import get_trading_calendar
    start=time.monotonic();plan=load_plan(Path(profile['plan']['path']));cfg=config_for(plan,a.scenario)
    generation=source_generation_hash();assert generation==profile['source_generation_hash']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    write(out/'request.json',dict(registration=dict(path=a.registration,sha256=sha(a.registration)),arguments=vars(a),
        native_generation=generation,additional_sources=reg['programs'],eligible_for_scoring=False))
    conn=connect(plan);source=plan['input_sha256'][plan['database']];calendar=get_trading_calendar()
    participation,slip,name=ev.SCENARIOS[a.scenario];costs=BrokerCosts(cfg.commission_bps,cfg.tax_bps_on_sell,slip)
    selected=defaultdict(list);coverage=Counter();decision_scores={}
    with gzip.open(reg['decision_scores']['path'],'rt') as f:
        for line in f:
            r=json.loads(line);decision_scores[r['action_id']]=r
    with gzip.open(reg['cohorts']['path'],'rt') as f:
        for n,line in enumerate(f,1):
            row=json.loads(line)
            if row['period']!=a.period:continue
            if a.pilot and row['trade_date'] not in reg['pilot_sessions'][a.period]:continue
            coverage['declared_rows']+=1
            # Native predictive-policy risk sets were already defined before
            # outcomes. Forecast absence never becomes an invented zero label.
            if not row['candidate'] or not row['original_forecasts']:
                coverage['not_labelled_missing_context_or_original_risk_set']+=1;continue
            selected[row['trade_date']].append((n,row))
    count=0;labels=Counter();seen=set();max_cash_error=0.;preparation_seconds=time.monotonic()-start
    with gzip.open(out/'labels.jsonl.gz','xt',compresslevel=1) as summaries,gzip.open(out/'native_records.jsonl.gz','xt',compresslevel=1) as details:
        for di,(day,items) in enumerate(sorted(selected.items())):
            allbars=load_day(conn,day,source,{r['symbol'] for _,r in items});by_symbol=defaultdict(list)
            for bar in allbars:by_symbol[bar.symbol].append(bar)
            for source_row,row in items:
                episode=row['episode_id'];assert episode not in seen;seen.add(episode)
                forecasts=[v['record'] for v in row['original_forecasts']]
                admission=row.get('original_admission');scale=admission['record']['quantity_scale'] if admission else .25
                choices=[v for v in forecasts if abs(v['quantity_scale']-scale)<1e-12]
                if len(choices)!=1:raise ValueError('Original quantity forecast identity not unique '+episode)
                forecast=choices[0];score=forecast['baseline_value_bps'];score_origin='original_checkpoint'
                if row['cohort']=='opening':
                    original=decision_scores[forecast['opportunity_id']]
                    score_origin=('calibrated_support' if admission and admission['record']['forecast_origin']=='new_forecast' else 'calibrated_core')
                    score=original[score_origin]['baseline']
                if admission:assert abs(score-admission['record']['score'])<1e-8
                candidate={**row['candidate'],'metadata':{**row['candidate']['metadata'],'action_quantity_scale':scale}}
                native_row=dict(trade_date=day,symbol=row['symbol'],candidate=candidate,feature_available_at=row['feature_available_at'])
                snapshot=ev.snapshot_for(day,[(native_row,score)],source=reg['cohorts']['sha256'])
                bars=by_symbol[row['symbol']];available=datetime.fromisoformat(row['feature_available_at'])
                for horizon in reg['horizons']:
                    broker=SimBroker(1e8,costs,buying_power_leverage=cfg.intraday_leverage,
                        capacity_policy=BarParticipationScenario(participation,name))
                    adapter=KALCBReplayAdapter(cfg,{snapshot.trade_date:snapshot},initial_equity=1e8,costs=costs)
                    hook=hm.DiagnosticHorizon(horizon);hook.install();decisions=[]
                    try:
                        for bar in bars:
                            decisions.extend(e.to_json_dict() for e in drive_timestamp_batch(broker,(bar,),(adapter,)))
                    finally:hook.restore()
                    adapter._sync_new_fills(broker)
                    buys=[f for f in broker.fills if f.side=='BUY'];sells=[f for f in broker.fills if f.side=='SELL']
                    qty=sum(f.qty for f in buys);notional=sum(f.qty*f.price for f in buys)
                    cash=1e8;inventory=0;fees=0.
                    for f in broker.fills:
                        fee=f.qty*f.price*(costs.commission_bps+(costs.tax_bps_on_sell if f.side=='SELL' else 0))/10000
                        cash+=(-1 if f.side=='BUY' else 1)*f.qty*f.price-fee;fees+=fee
                        inventory+=(1 if f.side=='BUY' else -1)*f.qty
                        assert inventory>=0 and f.timestamp>=datetime.fromisoformat(f.metadata['decision_available_at'])
                        if f.side=='BUY':assert f.timestamp>=available
                    error=abs(cash-broker.cash);max_cash_error=max(max_cash_error,error);assert error<.02
                    assert inventory==sum(p.qty for p in broker.positions.values())
                    issues=ev.physical_fill_failures(broker.fills)
                    closing=datetime.combine(snapshot.trade_date,calendar.session_hours(snapshot.trade_date)[1],available.tzinfo)
                    coverage_end=bars[-1].timestamp+timedelta(minutes=5) if bars else available
                    first=min((f.timestamp for f in buys),default=None)
                    deadline=first+timedelta(minutes=horizon) if first and horizon else None
                    censored=bool(inventory or broker.orders or coverage_end<closing-timedelta(minutes=10)
                        or (deadline and deadline>closing-timedelta(minutes=10))
                        or (horizon and any(min(f.timestamp for f in buys if f.order_id==oid)+timedelta(minutes=horizon)>closing-timedelta(minutes=10) for oid in {f.order_id for f in buys})))
                    valid=bool(buys) and not censored and not issues and not broker.same_bar_fill_violations
                    status='physical_invalid' if issues else 'censored' if censored else 'filled' if buys else 'unfilled' if any(r['event']=='submit' and r['order']['side']=='BUY' for r in broker.journal) else 'no_native_order'
                    if valid:assert abs(sum(t.net_pnl for t in broker.trades)-(cash-1e8))<.02
                    final_fill=max((f.timestamp for f in sells),default=None)
                    # Excursions follow the realised position lifetime, including
                    # execution after the scheduled deadline and mandatory exits.
                    path_end=final_fill or coverage_end
                    path_start=first
                    if first and any(f.metadata.get('execution_phase')!='open' for f in buys if f.timestamp==first):
                        path_start=first+timedelta(minutes=5)
                    obs=[b for b in bars if path_start and path_start<=b.timestamp<path_end]
                    entry=notional/qty if qty else None
                    maxbar=max(obs,key=lambda b:b.high) if obs else None
                    native_intents=[r for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY']
                    intent=native_intents[0]['order'] if native_intents else {}
                    intent_meta=intent.get('metadata',{})
                    prefix=[b for b in bars if b.timestamp+timedelta(minutes=5)<=available]
                    last_volume=prefix[-1].volume if prefix else None
                    intent_available=datetime.fromisoformat(intent_meta['decision_available_at']) if intent_meta.get('decision_available_at') else None
                    contemporaneous_intent=bool(intent_available and intent_available<=available)
                    metadata=dict(candidate.get('metadata',{}))
                    filled_orders=len({f.order_id for f in buys})
                    rec=dict(episode_id=episode,cohort_source_row=source_row,day=day,symbol=row['symbol'],cohort=row['cohort'],
                        scenario=a.scenario,horizon_minutes=horizon,status=status,label_valid=valid,
                        feature_available_at=available.isoformat(),first_fill=first,last_exit_fill=final_fill,
                        label_mature_at=coverage_end if valid else None,
                        deadline=deadline,terminal_inventory=inventory,terminal_pending=len(broker.orders),physical_failures=issues,
                        quantity_scale=scale,quantity_basis='native100m_standalone_rank1_original_admitted_scale_else_quarter',
                        filled_entry_orders=filled_orders,
                        same_issuer_day_multiple_lifetime_warning=filled_orders>1,
                        net_cash_krw=cash-1e8,net_value_notional_bps=(cash-1e8)/notional*10000 if valid and notional else None,
                        net_value_equity_bps=(cash-1e8)/1e8*10000 if valid else None,fees=fees,buy_quantity=qty,buy_notional=notional,
                        native_requested_quantity=intent.get('qty'),native_route=intent_meta.get('entry_route'),
                        native_intent_available_at=intent_available,original_clock_quantity_feature_available=contemporaneous_intent,
                        raw_score=score,raw_score_units='existing decision-time forecast equity_bps',score_origin=score_origin,
                        native_entry_reference=intent_meta.get('entry_price_ref'),
                        source_features=row['features'],source_decision_metadata=metadata,
                        native_quantity_observed_capacity=(intent.get('qty',0)/(last_volume*participation)) if last_volume and contemporaneous_intent else None,
                        mfe_price_pct=100*(maxbar.high/entry-1) if maxbar and entry else None,
                        mae_price_pct=100*(min(b.low for b in obs)/entry-1) if obs and entry else None,
                        time_to_mfe_minutes=(maxbar.timestamp-first).total_seconds()/60 if maxbar else None,
                        excursion_scope='post-first-fill retained5m OHLC through final actual exit; aggregate entry VWAP; descriptive, not executable; finer retained paths separately bound',
                        retained5m_observations=len(obs),
                        excursion_first_bar_excluded_for_nonopen_fill=path_start!=first,
                        retained5m_missing_intervals=sum(max(0,round((b.timestamp-a.timestamp).total_seconds()/300)-1) for a,b in zip(obs,obs[1:])),
                        admitted=bool(admission),eligible_for_scoring=False,standalone_not_funded_portfolio=True)
                    emit(summaries,rec)
                    emit(details,dict(episode_id=episode,horizon_minutes=horizon,fills=[asdict(f) for f in broker.fills],
                        trades=[t.to_json_dict() for t in broker.trades],decisions=decisions,
                        orders=[r for r in broker.journal if r['event'] in ('submit','amend','cancel','expire','reject')],
                        complete_native_journal=broker.journal,
                        source_events_hash=broker.source_events.hexdigest(),source_event_count=broker.source_event_count,
                        scheduled_decisions=hook.records))
                    count+=1;labels[status]+=1
                if len(seen)>reg['max_distinct_episodes']:raise RuntimeError('Registered episode budget exceeded')
                if time.monotonic()-start>reg['worker_seconds_guard']:raise RuntimeError('Registered label batch time guard')
            if di%10==0:
                write(out/('progress_'+str(di)+'.json'),dict(sessions=di+1,episodes=len(seen),labels=count,seconds=time.monotonic()-start))
                print(json.dumps(dict(session=di+1,episodes=len(seen),labels=count,seconds=time.monotonic()-start)),flush=True)
    conn.close()
    assert source_generation_hash()==generation
    for b in reg['programs']:assert sha(b['path'])==b['sha256']
    write(out/'completion.json',dict(passed=True,eligible_for_scoring=False,standalone_only=True,
        seconds=time.monotonic()-start,preparation_seconds=preparation_seconds,replay_seconds=time.monotonic()-start-preparation_seconds,distinct_episodes=len(seen),outcomes=count,statuses=labels,coverage=coverage,
        max_independent_cash_error=max_cash_error,native_generation=generation,
        files={n:dict(path=str(out/n),sha256=sha(out/n)) for n in ['request.json','labels.jsonl.gz','native_records.jsonl.gz']}))
    print('C18 label batch complete',flush=True)

if __name__=='__main__':main()
