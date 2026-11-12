"""Independent fine-clock audits; not activated by existing study runners.

Capacity ledger arithmetic derives from the authenticated v4 native audit. Minute
classification and predictors do not call the execution model quote or budget.
"""
from collections import defaultdict
from datetime import datetime, timedelta
import math

from strategy_common.market import MarketBar
from .capacity import closing_auction_event
from strategy_common.clock import ensure_kst
from strategy_common.execution_session import bar_execution_issue, bar_has_periodic_auction, bar_reopens_after_label
from kis_core.trading_calendar import get_trading_calendar


def continuous_interval(bar):
    if bar is None or bar.timeframe!='1m' or bar_execution_issue(bar):return False
    at=ensure_kst(bar.timestamp)
    opening,closing=get_trading_calendar().session_hours(at.date())
    start=at.replace(hour=opening.hour,minute=opening.minute,second=0,microsecond=0)
    end=at.replace(hour=closing.hour,minute=closing.minute,second=0,microsecond=0)-timedelta(minutes=10)
    return (at.second==at.microsecond==0 and start<=at and at+timedelta(minutes=1)<=end
        and not bar_has_periodic_auction(bar) and not bar_reopens_after_label(bar))


def audit_minute_capacity(broker):
    policy = broker.capacity_policy
    if policy is None:
        return {"status": "not_enabled", "errors": []}
    if policy.contract()['version']!='causal-minute-participation-olr-v5':
        raise ValueError('Wrong minute capacity contract')
    attempts = [r for r in broker.journal if r["event"] == "capacity"]
    needed = {(r["order"]["symbol"], r["timestamp"], r["phase"]) for r in attempts}
    needed.update((f.symbol, f.timestamp.isoformat(), f.metadata.get("execution_phase")) for f in broker.fills)
    quotes, volumes, previous = {}, {}, {}
    # Classification is the shared source/session policy; quantity arithmetic is
    # deliberately independent of BarParticipationScenario.quote/budget.
    for batch in getattr(broker, "expected_batches", ()):
        for raw in batch:
            stamp = str(raw["timestamp"])
            symbol = raw["symbol"]
            phases = [phase for phase in ("open", "intrabar", "auction") if (symbol, stamp, phase) in needed]
            bar = MarketBar(**{**raw, "timestamp": datetime.fromisoformat(stamp)})
            prior = previous.get(symbol)
            for phase in phases:
                quantity, source, reason = 0, None, "available"
                if phase == "open":
                    basis = "previous_contiguous_continuous_1m"
                    if not continuous_interval(bar):
                        reason = "outside_continuous_interval"
                    elif (not continuous_interval(prior) or prior.symbol != symbol
                          or prior.timestamp.date() != bar.timestamp.date()
                          or prior.timestamp + timedelta(minutes=1) != bar.timestamp):
                        reason = "missing_prior_continuous_interval"
                    else:
                        source = prior
                elif phase == "auction":
                    basis = "single_price_closing_auction_event"
                    if closing_auction_event(bar):
                        source = bar
                    else:
                        reason = "unsupported_auction_observation"
                else:
                    basis, reason = "unsupported", "unsupported_execution_phase"
                if source is not None:
                    quantity = max(0, math.floor(float(source.volume) * policy.participation))
                quotes[(symbol, stamp, phase)] = {"budget": quantity,
                    "predictor_volume": float(source.volume) if source else 0,
                    "predictor_timestamp": source.timestamp.isoformat() if source else None,
                    "volume_basis": basis, "reason": reason}
                volumes[(symbol, stamp)] = float(bar.volume)
            # Invalid/missing intervals cannot become a valid predictor. Keeping
            # them here also makes the next adjacency check fail closed.
            previous[symbol] = bar
    errors, offers = [], {}
    used, bar_used = defaultdict(int), defaultdict(int)
    groups, attempt_groups = defaultdict(list), defaultdict(lambda: defaultdict(int))
    by_order = {}
    positions, marks = defaultdict(int), {}
    peak_gross, peak_equity = 0.0, float(broker.initial_equity)
    cash = float(broker.initial_equity)
    for row in broker.journal:
        kind = row["event"]
        if kind in {"submit", "amend"} and row.get("order", {}).get("order_type") not in {"MARKET", "CLOSE_AUCTION"}:
            errors.append("unsupported_capacity_order_type")
        if kind in {"opening_marks", "closing_marks"}:
            marks.update(row["prices"])
        if kind == "capacity":
            order = row["order"]
            key = (order["symbol"], row["timestamp"], row["phase"])
            quote = quotes.get(key)
            requested = int(row["requested_qty"])
            allowed = min(requested, max(0, (quote or {}).get("budget", 0) - used[key]))
            if quote is None or row["quote"] != quote or row["used_qty"] != used[key] or row["allowed_qty"] != allowed or requested < 0:
                errors.append("capacity_quote_mismatch")
            offer_key = (order["order_id"], row["timestamp"], row["phase"])
            if offer_key in offers:
                errors.append("duplicate_capacity_offer")
            offers[offer_key] = allowed
            group = attempt_groups[f"{row['phase']}_{order['side'].lower()}"]
            group["attempts"] += 1
            group["requested_attempt_shares"] += requested
            group["offered_attempt_shares"] += allowed
            group["zero_offer_attempts"] += int(allowed == 0)
            group["partial_offer_attempts"] += int(0 < allowed < requested)
            group["missing_predictor_attempts"] += int(row["quote"]["reason"] == "missing_prior_continuous_interval")
            if row["quote"]["reason"].startswith("unsupported_"):
                errors.append(row["quote"]["reason"])
            by_order.setdefault(order["order_id"], {"requested": requested, "filled": 0, "first_attempt": row["timestamp"]})
        elif kind == "fill":
            fill = row["fill"]
            stamp = fill["timestamp"].isoformat() if hasattr(fill["timestamp"], "isoformat") else fill["timestamp"]
            phase = fill["metadata"].get("execution_phase")
            key = (fill["symbol"], stamp, phase)
            offer = offers.pop((fill["order_id"], stamp, phase), None)
            qty = int(fill["qty"])
            if offer is None or qty > offer:
                errors.append("missing_or_exceeded_capacity_offer")
            used[key] += qty
            bar_used[key[:2]] += qty
            if used[key] > quotes.get(key, {}).get("budget", 0):
                errors.append("shared_capacity_exceeded")
            if bar_used[key[:2]] > volumes.get(key[:2], 0):
                errors.append("shared_bar_volume_exceeded")
            if fill["order_id"] in by_order:
                by_order[fill["order_id"]]["filled"] += qty
                by_order[fill["order_id"]].setdefault("first_fill", stamp)
            positions[(fill["strategy_id"], fill["symbol"])] += qty if fill["side"] == "BUY" else -qty
            cash = row["cash_after"]
        if kind in {"fill", "opening_marks", "closing_marks"}:
            gross = sum(qty * marks.get(symbol, 0) for (_, symbol), qty in positions.items())
            peak_gross = max(peak_gross, gross)
            peak_equity = max(peak_equity, cash + gross)
    violations = []
    for key, qty in used.items():
        volume = volumes.get(key[:2], 0)
        ratio = qty / volume if volume > 0 else None
        if ratio is not None:
            groups[key[2]].append(ratio)
        if volume <= 0 or bar_used[key[:2]] > volume:
            violations.append({"symbol": key[0], "timestamp": key[1], "phase": key[2],
                "filled_shares": qty, "bar_total_filled_shares": bar_used[key[:2]], "printed_shares": volume})
    forecast_ratios = [q["predictor_volume"] / volumes[key[:2]] for key, q in quotes.items()
        if key[2] == "open" and q["reason"] == "available" and volumes.get(key[:2], 0) > 0]
    delays = [(datetime.fromisoformat(r["first_fill"]) - datetime.fromisoformat(r["first_attempt"])).total_seconds()
              for r in by_order.values() if "first_fill" in r]
    def distribution(values):
        ordered = sorted(values)
        return {"count": len(ordered), "p50": ordered[(len(ordered)-1)//2] if ordered else None,
            "p95": ordered[int((len(ordered)-1)*.95)] if ordered else None, "max": max(ordered, default=None)}
    return {"status": "failed" if errors else "model_consistent", "errors": sorted(set(errors)),
        "contract": policy.contract(), "phase_liquidity_verified": False,
        "attempts_by_phase_and_side": {k: dict(v) for k, v in sorted(attempt_groups.items())},
        "combined_participation_by_phase": {phase: {"groups": len(values),
            "p50": sorted(values)[(len(values)-1)//2], "p95": sorted(values)[int((len(values)-1)*.95)],
            "max": max(values), "above_1pct": sum(v > .01 for v in values),
            "above_100pct": sum(v > 1 for v in values)} for phase, values in sorted(groups.items())},
        "impossible_volume_events": violations, "attempted_orders": len(by_order),
        "opening_forecast_to_realized_bar_volume": distribution(forecast_ratios),
        "first_eligible_attempt_to_first_fill_seconds": distribution(delays),
        "orders_with_fills": sum(v["filled"] > 0 for v in by_order.values()),
        "unfilled_order_shares": sum(max(v["requested"] - v["filled"], 0) for v in by_order.values()),
        "capital": {"initial_equity": broker.initial_equity, "peak_equity": peak_equity,
            "peak_gross_exposure": peak_gross, "final_equity": broker.equity_curve[-1]},
        "terminal_open_shares": sum(p.qty for p in broker.positions.values()),
        "terminal_pending_shares": sum(int(o.qty or 0) for o in broker.orders)}


def audit_minute_clock(broker):
    """Recompute clocks, high-water marks and every required fast risk signal.

    Uses bound observations and executed fill/order events, not adapter state,
    native exit helpers or supplied signal pass labels. The usual native ledger
    audit separately reconstructs execution prices, account cash and positions.
    """
    from copy import deepcopy
    errors=[];positions={};archived={};active={};expected_signals={};seen_signals=set()
    fine={batch[0]['timestamp']:batch for batch in broker.expected_batches}
    coarse={batch[0]['timestamp']:batch for batch in broker.coarse_expected_batches}
    seen_fine=[];seen_coarse=[];fine_seen=set();coarse_seen=set()
    closing=None;availability=None;context=None;context_label=None
    plan=broker.minute_exit_plan
    fixed={'mode':'managed','hard_stop_enabled':False,'failed_followthrough_bars':1,
        'failed_followthrough_close_r':-1.,'failed_followthrough_mfe_r':4.,
        'mfe_fade_start_r':1.,'mfe_fade_gap_r':1.25,'mfe_fade_floor_r':0.,
        'vwap_fail_bars':0,'no_mfe_bars':0,'max_hold_bars':0}
    if any(plan.get(k)!=v for k,v in fixed.items()):errors.append('minute_plan_mismatch')
    dt=lambda x: x if isinstance(x,datetime) else datetime.fromisoformat(x)
    next_open_at={};following=None
    for i in range(len(broker.journal)-1,-1,-1):
        next_open_at[i]=following
        if broker.journal[i]['event']=='opening_marks':following=dt(broker.journal[i]['timestamp'])
    def number(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9)
    def mark(raw):
        p=positions.get(raw['symbol']);bar=MarketBar(**{**raw,'timestamp':dt(raw['timestamp'])})
        if p and not bar_execution_issue(bar) and (bar.timestamp>p['at'] or
                (bar.timestamp==p['at'] and p['phase']=='open')):
            p['high']=max(p['high'],bar.high)
        return bar,p
    for i,row in enumerate(broker.journal):
        kind=row['event']
        if kind=='opening_marks':
            at=dt(row['timestamp'])
            if availability and at<availability:errors.append('minute_future_decision_before_open')
            availability=at;context='open';context_label=at
        elif kind=='closing_marks':
            closing=row['timestamp']
        elif kind=='minute_observation_v5':
            stamp=row['timestamp'];batch=fine.get(stamp,())
            if not batch or closing!=stamp or stamp in fine_seen:
                errors.append('minute_observation_disposition')
            seen_fine.append(stamp);fine_seen.add(stamp);at=dt(stamp)+timedelta(minutes=1)
            if row['available_at']!=at.isoformat() or row['symbols']!=[b['symbol'] for b in batch]:
                errors.append('minute_observation_clock')
            if availability and at<availability:errors.append('minute_clock_reversal')
            availability=at;context='minute';context_label=dt(stamp)
            for raw in batch:
                bar,p=mark(raw)
                if not p or not broker.minute_fast_exit or bar.timestamp.date()<=p['at'].date() or not continuous_interval(bar):continue
                sells=[o for o in active.values() if o['symbol']==bar.symbol and o['side']=='SELL']
                if len(sells)>1:errors.append('minute_ambiguous_pending_exits')
                pending=sells[0] if sells else None
                if pending and (not broker.minute_exit_priority or not (
                    (pending['order_type']=='CLOSE_AUCTION' and pending['reason']=='next_close_exit') or
                    (pending['order_type']=='MARKET' and pending['reason']=='weak_followthrough_partial'))):continue
                risk=p['risk'] if p['risk']>0 else p['price']-p['stop'] if p['stop']>0 else 0
                risk=max(risk,p['price']*.001,1.)
                close_r=(bar.close-p['price'])/risk;mfe_r=(p['high']-p['price'])/risk
                reason='failed_followthrough' if close_r<=-1 and mfe_r<4 else (
                    'mfe_fade' if mfe_r>=1 and close_r<=max(0,mfe_r-1.25) else '')
                if reason:expected_signals[(stamp,bar.symbol)]=dict(reason=reason,risk=risk,high_water=p['high'],
                    entry_price=p['price'],entry_time=p['at'].isoformat(),quantity=p['qty'],bar=raw,
                    pending_exit=pending['order_id'] if pending else None,available_at=at.isoformat())
        elif kind=='coarse_observation_v5':
            stamp=row['label'];batch=coarse.get(stamp,());at=dt(stamp)+timedelta(minutes=5)
            delivered=dt(row['delivered_by'])
            if not batch or stamp in coarse_seen:errors.append('minute_coarse_disposition')
            seen_coarse.append(stamp);coarse_seen.add(stamp)
            if (row['available_at']!=at.isoformat() or delivered<at or
                    row['symbols']!=[b['symbol'] for b in batch]):errors.append('minute_coarse_clock')
            # Delivery must occur at a just-completed minute, immediately before
            # a later native opening batch, or as the final nonexecutable auction timer.
            next_open=next_open_at[i]
            # Session-final observations are distinct from executable auctions.
            # Recompute the point-print/calendar conditions independently; a
            # periodic-auction calendar must continue to prohibit all capacity.
            def closing_point(raw):
                bar=MarketBar(**{**raw,'timestamp':dt(raw['timestamp'])})
                stamp=ensure_kst(bar.timestamp)
                return (not bar_execution_issue(bar)
                    and stamp.time().replace(tzinfo=None)==get_trading_calendar().closing_match_time(bar.symbol,stamp.date())
                    and bar.open==bar.high==bar.low==bar.close)
            last_timer=(next_open is None or next_open.date()>at.date()) and bool(batch) and all(closing_point(b) for b in batch)
            completed=dt(closing)+timedelta(minutes=1) if closing else None
            if delivered not in (completed,next_open) and not (last_timer and delivered==at):
                errors.append('minute_coarse_early_delivery')
            if availability and at<availability:errors.append('minute_clock_reversal')
            availability=at;context='coarse';context_label=dt(stamp)
            for raw in batch:mark(raw)
        elif kind=='minute_exit_signal_v5':
            key=(row['timestamp'],row['symbol']);expected=expected_signals.get(key)
            if key in seen_signals or expected is None:errors.append('minute_unearned_exit_signal')
            else:
                for field,value in expected.items():
                    if not (number(row[field],value) if field in ('risk','high_water','entry_price') else row[field]==value):
                        errors.append('minute_exit_evidence_'+field)
                if row['plan']!=plan:errors.append('minute_exit_plan_mismatch')
            if context!='minute' or dt(row['available_at'])!=availability:errors.append('minute_signal_clock')
            seen_signals.add(key)
        elif kind in ('submit','amend'):
            order=row['order'];active[order['order_id']]=deepcopy(order)
            if kind=='submit' and context:
                available=dt(order['metadata']['decision_available_at'])
                if available!=availability:errors.append('minute_order_availability')
                if order['side']=='BUY' and context!='coarse':errors.append('minute_entry_alpha_clock')
                if context=='minute' and order['side']=='SELL' and order['order_type']=='MARKET':
                    key=(context_label.isoformat(),order['symbol'])
                    if key not in seen_signals or order['reason']!=expected_signals.get(key,{}).get('reason'):
                        errors.append('minute_market_exit_without_signal')
        elif kind in ('cancel','expire','reject'):
            active.pop(row['order']['order_id'],None)
        elif kind=='fill':
            f=row['fill'];symbol=f['symbol'];meta=f['metadata'];at=dt(f['timestamp']);qty=f['qty']
            context='open' if meta['execution_phase']=='open' else 'auction';availability=at;context_label=at
            if f['side']=='BUY':
                p=positions.get(symbol) or deepcopy(archived.get(f['order_id']))
                if p:
                    p['price']=(p['price']*p['qty']+f['price']*qty)/(p['qty']+qty);p['qty']+=qty
                else:
                    stop=float(meta.get('protective_stop_price',0) or 0)
                    p=dict(qty=qty,price=f['price'],at=at,high=f['price'],phase=meta['execution_phase'],
                        order_id=f['order_id'],risk=float(meta.get('risk_per_share',0) or 0),stop=stop)
                stop=float(meta.get('protective_stop_price',0) or 0)
                if stop>=f['price']:
                    stop=f['price']*(1-max(float(plan['stop_pct']),.003));p['stop']=stop
                    p['risk']=max(f['price']-stop,f['price']*.001,1.)
                positions[symbol]=p
            elif symbol in positions:
                p=positions[symbol];p['qty']-=qty
                if p['qty']==0:archived[p['order_id']]=deepcopy(p);positions.pop(symbol)
            else:errors.append('minute_sale_without_inventory')
            if qty==row['order']['qty']:active.pop(f['order_id'],None)
    if seen_fine!=list(fine):errors.append('minute_missing_observation')
    if seen_coarse!=list(coarse):errors.append('minute_missing_coarse_observation')
    if set(expected_signals)!=seen_signals:errors.append('minute_missing_exit_signal')
    if broker.minute_clock_schedule!=[dict(kind=r['kind'],label=r['label'],available_at=r['available_at'],
            delivered_by=r['delivered_by'],symbols=r['symbols']) for r in broker.journal if r['event']=='coarse_observation_v5']:
        errors.append('minute_schedule_binding')
    return dict(status='failed' if errors else 'pass',errors=sorted(set(errors)),
        minute_observations=len(seen_fine),coarse_observations=len(seen_coarse),
        reconstructed_required_signals=len(expected_signals),recorded_signals=len(seen_signals),
        unchanged_thresholds=True,control_or_treatment='treatment' if broker.minute_fast_exit else 'clock_control')


def minute_nav_evidence(broker,common_grid=None):
    """Keep all fine events; also sample a separately frozen old completion grid.

    The common grid is old five-minute labels +5 minutes, immediately before
    that boundary's new opening phase. Point auctions use that same conservative
    completion convention. Fine closing marks become visible at label+1, exactly
    as the minute driver declares; no corrected 15:32 print can appear at15:30.
    """
    from datetime import datetime,timedelta
    positions=defaultdict(int);marks={};cash=broker.initial_equity
    peak=cash;max_dd=0.;peak_at=None;worst=None;points=0;counter=defaultdict(int)
    grid=[(datetime.fromisoformat(str(r[0]))+timedelta(minutes=5),str(r[0])) for r in (common_grid or [])]
    if grid!=sorted(grid) or len({at for at,_ in grid})!=len(grid):raise ValueError('Common mark grid must be ordered and unique')
    common=[];cursor=0;previous=cash;previous_key=None
    for row in broker.journal:
        kind=row['event']
        if kind in ('opening_marks','closing_marks'):
            marks.update(row['prices']);at=row['timestamp']
        elif kind=='fill':
            fill=row['fill'];q=fill['qty'];p=fill['price'];meta=fill['metadata']
            positions[fill['symbol']]+=q if fill['side']=='BUY' else -q
            cash+=(-q*p if fill['side']=='BUY' else q*p)-meta['fee']
            at=fill['timestamp'].isoformat() if hasattr(fill['timestamp'],'isoformat') else fill['timestamp']
        else:continue
        if any(q and s not in marks for s,q in positions.items()):raise ValueError('Unmarked position in independent minute NAV')
        key=(datetime.fromisoformat(str(at))+timedelta(minutes=1 if kind=='closing_marks' else 0),0 if kind=='closing_marks' else 1)
        if previous_key is not None and key<previous_key:raise ValueError('Noncausal fine NAV chronology')
        while cursor<len(grid) and (grid[cursor][0],0)<key:
            common.append(dict(timestamp=grid[cursor][1],available_at=grid[cursor][0].isoformat(),equity=previous));cursor+=1
        equity=cash+sum(q*marks.get(s,0) for s,q in positions.items());points+=1;counter[kind]+=1
        previous=equity;previous_key=key
        if equity>peak:peak=equity;peak_at=at
        dd=100*(peak-equity)/peak if peak else 0.
        if dd>max_dd:max_dd=dd;worst=dict(peak_at=peak_at,trough_at=at,peak_equity=peak,trough_equity=equity,event=kind)
    while cursor<len(grid):
        common.append(dict(timestamp=grid[cursor][1],available_at=grid[cursor][0].isoformat(),equity=previous));cursor+=1
    cp=broker.initial_equity;common_dd=0.
    for r in common:cp=max(cp,r['equity']);common_dd=max(common_dd,100*(cp-r['equity'])/cp)
    return dict(all_event_drawdown_pct=max_dd,maximum=worst,points=points,events=dict(counter),
        final_equity=cash+sum(q*marks.get(s,0) for s,q in positions.items()),
        coarse_points=len(broker.minute_coarse_nav),
        risk_signals=sum(r['event']=='minute_exit_signal_v5' for r in broker.journal),
        common_original_grid=common,common_original_drawdown_pct=common_dd if common else None,
        common_grid_policy='Frozen retained-control 5m labels +5 minutes; before new opening phase; fine closes available label+1, including point auctions. Full fine series never filtered.',
        scope='Every native minute opening/closing mark and every fill cash debit or credit; no points removed')
