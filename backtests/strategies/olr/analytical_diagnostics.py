"""Descriptive OLR diagnostics over completed evidence, never a trading policy.

Pure accounting/summary functions are shared by native and generic exporters.
Native interval service additionally supplies source, funding and event paths.
Unavailable inputs stay unavailable; hindsight excursions never become orders.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
from datetime import datetime, timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

VERSION = 'olr-analytical-diagnostics-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def equal(a, b, message):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=.02), message)


def rows(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def ratio(a, b):
    return a / b if b else None


def quantile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pos = (len(values) - 1) * p
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p10=quantile(values, .1), median=quantile(values, .5), p90=quantile(values, .9))


def identity(t):
    return t['symbol'] + ':' + str(t['entry_decision_time'])[:10]


def normalize_trades(trades, capital):
    answer = []
    for index, t in enumerate(trades):
        meta = t.get('route_metadata') or {}
        r = float(meta.get('risk_per_share') or 0)
        qty, net, gross = float(t['qty']), float(t['net_pnl']), float(t['gross_pnl'])
        entry = datetime.fromisoformat(str(t['entry_fill_time']))
        exit_at = datetime.fromisoformat(str(t['exit_fill_time']))
        notional = qty * float(t['entry_price'])
        answer.append(dict(id=identity(t), source_row=index, symbol=t['symbol'],
            day=str(t['entry_decision_time'])[:10], entry=entry.isoformat(), exit=exit_at.isoformat(),
            available_at=meta.get('decision_available_at'), qty=qty, entry_price=t['entry_price'],
            exit_price=t['exit_price'], net=net, gross=gross, fees=gross-net,
            notional=notional, meaningful=notional >= .01 * capital,
            net_r=net/(r*qty) if r > 0 and qty else None,
            gross_r=gross/(r*qty) if r > 0 and qty else None,
            mfe_r=float(t['mfe'])/r if r > 0 and t.get('mfe') is not None else None,
            mae_r=float(t['mae'])/r if r > 0 and t.get('mae') is not None else None,
            mfe_value=float(t['mfe'])*qty if t.get('mfe') is not None else None,
            hold_hours=(exit_at-entry).total_seconds()/3600,
            holding_days=(exit_at.date()-entry.date()).days,
            entry_clock=entry.strftime('%H:%M'), rank=str(meta.get('candidate_rank', 'unknown')),
            score=meta.get('candidate_score'), sector=meta.get('sector', 'unknown'),
            score_band=meta.get('afternoon_score_band_rule', 'unknown'),
            binding=meta.get('allocation_binding_limit', 'unknown'), exit_reason=t.get('exit_reason', 'unknown'),
            risk_per_share=r or None, partial_exits=meta.get('partial_exit_count'),
            exit_legs=meta.get('exit_leg_count'), entry_plan=meta.get('trade_entry_plan'),
            exit_plan=meta.get('trade_exit_plan'), scenario_loss_budget=meta.get('scenario_loss_budget')))
    require(len({t['id'] for t in answer}) == len(answer), 'Duplicate opportunity identity; cannot call rows independent')
    return answer


def trade_stats(trades):
    wins = [t for t in trades if t['net'] > 0]
    losses = [t for t in trades if t['net'] < 0]
    gross = sum(t['gross'] for t in trades)
    win = sum(t['net'] for t in wins)
    loss = -sum(t['net'] for t in losses)
    return dict(n=len(trades), wins=len(wins), losses=len(losses), flat=len(trades)-len(wins)-len(losses),
        meaningful=sum(t['meaningful'] for t in trades), net=win-loss, gross=gross,
        fees=sum(t['fees'] for t in trades), winning_wealth=win, losing_wealth=loss,
        profit_factor=ratio(win, loss), win_rate=ratio(len(wins), len(trades)),
        expectancy=ratio(win-loss, len(trades)), net_bps_on_acquired_notional=ratio(10000*(win-loss), sum(t['notional'] for t in trades)),
        avg_win=ratio(win, len(wins)), avg_loss=ratio(loss, len(losses)),
        net_r=distribution([t['net_r'] for t in trades]), mfe_r=distribution([t['mfe_r'] for t in trades]),
        aggregate_net_per_initial_risk=ratio(sum(t['net'] for t in trades if t['risk_per_share']),
            sum(t['risk_per_share']*t['qty'] for t in trades if t['risk_per_share'])),
        initial_risk_krw=distribution([t['risk_per_share']*t['qty'] if t['risk_per_share'] else None for t in trades]),
        acquired_notional_krw=distribution([t['notional'] for t in trades]),
        mae_r=distribution([t['mae_r'] for t in trades]), hold_hours=distribution([t['hold_hours'] for t in trades]),
        gross_capture_of_recorded_mfe=ratio(gross, sum(t['mfe_value'] or 0 for t in trades)),
        excursion_coverage=sum(t['mfe_r'] is not None and t['mae_r'] is not None for t in trades),
        small_sample=len(trades) < 20)


def cohort_analysis(trades):
    dims = {'entry_month': lambda t: t['day'][:7], 'sector': lambda t: t['sector'],
        'symbol': lambda t: t['symbol'], 'rank': lambda t: t['rank'], 'score_band': lambda t: t['score_band'],
        'entry_clock': lambda t: t['entry_clock'], 'exit_reason': lambda t: t['exit_reason'],
        'binding': lambda t: t['binding'], 'holding_calendar_days': lambda t: str(t['holding_days']),
        'outcome': lambda t: 'winner' if t['net'] > 0 else 'loser' if t['net'] < 0 else 'flat',
        'meaningful': lambda t: 'meaningful' if t['meaningful'] else 'below_1pct_capital',
        'excursion': lambda t: 'unavailable' if t['mfe_r'] is None else 'MFE_under_1R' if t['mfe_r'] < 1 else 'MFE_1R_to_2R' if t['mfe_r'] < 2 else 'MFE_at_least_2R'}
    out = {}
    for dim, key in dims.items():
        groups = defaultdict(list)
        for t in trades:
            groups[key(t)].append(t)
        out[dim] = [dict(label=k, **trade_stats(v)) for k, v in sorted(groups.items())]
    return out


def consistency(trades, daily, capital):
    monthly = {}
    previous = capital
    returns = []
    for d in daily:
        month = d['day'][:7]
        r = float(d['equity']) / previous - 1
        delta = float(d['equity']) - previous
        record = monthly.setdefault(month, dict(month=month, start_equity=previous, end_equity=previous,
            net=0., sessions=0, positive_days=0, negative_days=0))
        record['end_equity'] = d['equity']; record['net'] += delta; record['sessions'] += 1
        record['positive_days'] += r > 0; record['negative_days'] += r < 0
        returns.append(dict(day=d['day'], return_fraction=r, net=delta))
        previous = d['equity']
    for record in monthly.values():
        record['return_fraction'] = record['end_equity']/record['start_equity']-1
    ordered = sorted(trades, key=lambda t: (t['exit'], t['id']))
    rolling = [dict(end_id=ordered[i]['id'], end_at=ordered[i]['exit'], **trade_stats(ordered[i-19:i+1])) for i in range(19, len(ordered))]
    streaks, current, length = Counter(), None, 0
    for t in ordered:
        sign = 'wins' if t['net'] > 0 else 'losses' if t['net'] < 0 else 'flat'
        length = length + 1 if sign == current else 1
        current = sign
        streaks[sign] = max(streaks[sign], length)
    winners = sorted((t for t in trades if t['net'] > 0), key=lambda t: t['net'], reverse=True)
    total = sum(t['net'] for t in trades)
    positive = sum(t['net'] for t in winners)
    shares = [t['net']/positive for t in winners] if positive else []
    positive_days = sorted((d for d in returns if d['net'] > 0), key=lambda d: d['net'], reverse=True)
    effective = ratio(1., sum(s*s for s in shares))
    return dict(monthly=list(monthly.values()), daily=returns, rolling_20_closed_trades=rolling,
        rolling_overlap_note='Overlapping closed-trade windows are stability descriptions, not independent tests.',
        longest_streaks=dict(streaks),
        concentration=dict(positive_pnl_effective_winner_count=effective,
            top1_winner_share=shares[0] if shares else None, top5_winner_share=sum(shares[:5]) if shares else None,
            top_winners=[dict(id=t['id'], net=t['net']) for t in winners[:10]],
            without_best_1_trade=total-sum(t['net'] for t in winners[:1]),
            without_best_5_trades=total-sum(t['net'] for t in winners[:5]),
            without_best_daily_MTM=total-(positive_days[0]['net'] if positive_days else 0),
            fixed_path_only='Deletion holds all later trades/quantities fixed; not an executable funded strategy.'),
        positive_months=sum(m['net'] > 0 for m in monthly.values()), months=len(monthly),
        positive_rolling_windows=sum(t['net'] > 0 for t in rolling), rolling_windows=len(rolling))


def event_analysis(events, capital):
    """Stream the native interval-service event clock and exact MTM attribution.

    Interval/closing labels become known at +1m, matching the accepted N0
    reconstructor. Do not apply this clock to a different execution schema.
    """
    cash = peak = nav = capital
    qty, marks, symbol_pnl, gap_pnl = defaultdict(int), {}, defaultdict(float), defaultdict(float)
    component_pnl = defaultdict(float)
    orders, counts, episodes = {}, Counter(), []
    peak_at = last_at = None
    peak_symbols, peak_components = {}, {}
    worst_pct = worst_money = None
    episode = None
    last_mark_day = {}
    point_count = 0
    daily = {}
    for e in events:
        kind = e['event']; counts[kind] += 1
        if kind == 'submit':
            o = e['order']; m = o.get('metadata') or {}
            require(o['order_id'] not in orders, 'Duplicate submitted order id')
            orders[o['order_id']] = dict(id=o['order_id'], symbol=o['symbol'], side=o['side'],
                qty=o['qty'], filled_qty=0, notional=0., fee=0., fragments=0,
                decision_label=o['submitted_at'], available_at=m.get('decision_available_at'),
                reason=o.get('reason'), order_type=o.get('order_type'), first_fill=None, last_fill=None,
                expiry=m.get('expiry_ts'), events=[], metadata=m)
        elif kind in ('expire', 'cancel', 'reject'):
            o = e.get('order', {}); found = orders.get(o.get('order_id'))
            if found is not None:
                found['events'].append(dict(event=kind, at=e.get('timestamp'), remaining=o.get('qty'), reason=e.get('reason')))
        if kind in ('opening_marks', 'interval_marks', 'closing_marks'):
            at = datetime.fromisoformat(e['timestamp']) + timedelta(minutes=int(kind != 'opening_marks'))
            for s, price in e['prices'].items():
                old = marks.get(s, price)
                delta = qty[s] * (price-old)
                symbol_pnl[s] += delta
                overnight = s in last_mark_day and last_mark_day[s] != at.date()
                component_pnl['overnight_mark' if overnight else 'intraday_mark'] += delta
                if overnight:
                    gap_pnl[s] += delta
                marks[s] = price; last_mark_day[s] = at.date()
        elif kind == 'fill':
            f = e['fill']; at = datetime.fromisoformat(f['timestamp']); s = f['symbol']
            require(s in marks, 'Fill lacks a prior observable mark')
            n, price = f['qty'], f['price']; fee = f['metadata']['fee']
            buy = f['side'] == 'BUY'; signed = n if buy else -n
            equal(cash, e['cash_before'], 'Discontinuous analysis cash')
            cash -= signed * price + fee; qty[s] += signed
            equal(cash, e['cash_after'], 'Analysis cash does not reconcile')
            require(qty[s] >= 0 and cash >= -.01, 'Oversold or unfunded account')
            execution = signed * (marks[s]-price)
            symbol_pnl[s] += execution-fee
            component_pnl['execution_vs_observable_mark'] += execution
            component_pnl['explicit_fees'] -= fee
            o = orders[f['order_id']]
            o['filled_qty'] += n; o['notional'] += n*price; o['fee'] += fee; o['fragments'] += 1
            o['first_fill'] = o['first_fill'] or at.isoformat(); o['last_fill'] = at.isoformat()
            if o['available_at']:
                require(at >= datetime.fromisoformat(o['available_at']), 'Fill before decision availability')
        else:
            continue
        require(at.tzinfo is not None and (last_at is None or at >= last_at), 'Reversed/naive evidence clock')
        nav = cash + sum(n*marks[s] for s, n in qty.items() if n)
        equal(nav-capital, sum(symbol_pnl.values()), 'Symbol MTM attribution failed')
        equal(nav-capital, sum(component_pnl.values()), 'Component MTM attribution failed')
        point_count += 1; daily[at.date().isoformat()] = nav
        if nav >= peak:
            if episode is not None:
                episode.update(recovered_at=at.isoformat(), end=at.isoformat(), censored=False,
                    duration_days=(at-datetime.fromisoformat(episode['peak_at'])).total_seconds()/86400)
                episodes.append(episode); episode = None
            peak, peak_at = nav, at
            peak_symbols = dict(symbol_pnl); peak_components = dict(component_pnl)
        else:
            start = peak_at or at
            decline = peak-nav; fraction = decline/peak
            if episode is None:
                episode = dict(peak_at=start.isoformat(), peak=peak, trough_at=at.isoformat(),
                    trough=nav, drawdown_fraction=fraction, decline_krw=decline)
            if nav < episode['trough']:
                episode.update(trough_at=at.isoformat(), trough=nav, drawdown_fraction=fraction, decline_krw=decline)
            def capture():
                attribution = {s: v-peak_symbols.get(s, 0) for s, v in symbol_pnl.items() if abs(v-peak_symbols.get(s, 0)) > 1e-7}
                components = {s: v-peak_components.get(s, 0) for s, v in component_pnl.items()}
                equal(sum(attribution.values()), -decline, 'Peak/trough attribution mismatch')
                return dict(peak_at=start.isoformat(), trough_at=at.isoformat(), peak=peak, trough=nav,
                    drawdown_fraction=fraction, decline_krw=decline, symbol_attribution=attribution,
                    component_attribution=components)
            if worst_pct is None or fraction > worst_pct['drawdown_fraction']:
                worst_pct = capture()
            if worst_money is None or decline > worst_money['decline_krw']:
                worst_money = capture()
        last_at = at
    if episode is not None:
        episode.update(recovered_at=None, end=last_at.isoformat(), censored=True,
            duration_days=(last_at-datetime.fromisoformat(episode['peak_at'])).total_seconds()/86400)
        episodes.append(episode)
    order_rows = list(orders.values())
    for o in order_rows:
        o['fill_fraction'] = ratio(o['filled_qty'], o['qty'])
        o['first_fill_delay_minutes'] = ((datetime.fromisoformat(o['first_fill'])-datetime.fromisoformat(o['available_at'])).total_seconds()/60
            if o['first_fill'] and o['available_at'] else None)
        o['last_fill_delay_minutes'] = ((datetime.fromisoformat(o['last_fill'])-datetime.fromisoformat(o['available_at'])).total_seconds()/60
            if o['last_fill'] and o['available_at'] else None)
    return dict(terminal_nav=nav, terminal_cash=cash, terminal_inventory={s:n for s,n in qty.items() if n},
        points=point_count, event_counts=dict(counts), daily_nav=daily, symbol_pnl=dict(symbol_pnl),
        mtm_components=dict(component_pnl), overnight_mark_by_symbol=dict(gap_pnl),
        worst_percentage=worst_pct, worst_currency=worst_money,
        drawdown_episodes=sorted(episodes, key=lambda x: x['drawdown_fraction'], reverse=True),
        orders=order_rows)


def execution_summary(orders):
    answer = {}
    for side in ('BUY', 'SELL'):
        selected = [o for o in orders if o['side'] == side]
        answer[side] = dict(orders=len(selected), filled_orders=sum(o['filled_qty'] > 0 for o in selected),
            fragments=sum(o['fragments'] for o in selected),
            partial_or_unfilled_orders=sum(o['filled_qty'] < o['qty'] for o in selected),
            filled_notional=sum(o['notional'] for o in selected), fees=sum(o['fee'] for o in selected),
            expires=sum(any(e['event'] == 'expire' for e in o['events']) for o in selected),
            cancels=sum(any(e['event'] == 'cancel' for e in o['events']) for o in selected),
            first_fill_delay_minutes=distribution([o['first_fill_delay_minutes'] for o in selected]),
            completion_delay_minutes=distribution([o['last_fill_delay_minutes'] for o in selected]),
            availability_clock_coverage=sum(o['available_at'] is not None for o in selected),
            reasons=dict(Counter(o['reason'] for o in selected)))
    answer['scope'] = 'Orders retain original submitted quantity; amendments and fill fragments are not additional opportunities. SELL fallback is a new order, not a new acquisition.'
    return answer


def funding_analysis(receipt, trades):
    trade_by_id = {t['id']: t for t in trades}
    decisions = []
    for context in receipt.get('entry_contexts', []):
        intents = {i['intent_id']: i for i in context['intents']}
        for d in context['decisions']:
            p = context['proposals'][d['intent_id']]; i = intents[d['intent_id']]
            key = p['symbol'] + ':' + context['at'][:10]
            t = trade_by_id.get(key)
            decisions.append(dict(id=key, at=context['at'], symbol=p['symbol'], desired_qty=i['desired_qty'],
                approved_qty=d['approved_qty'], reason=d['reason'], binding=d.get('binding_limit'),
                rank=i.get('rank'), ledger_cash=context['ledger_cash'],
                reservations=context['pending_buy_reservations'], deployable=context['actual_deployable_cash'],
                equity=context['sizing_equity'], gross_committed=context['account']['gross_committed'],
                reserved_notional=d.get('reserved_notional'), actual_net=t['net'] if t else None,
                acquired=t is not None, meaningful=t['meaningful'] if t else False,
                account=context['account'], proposal_metadata=p.get('metadata')))
    groups = defaultdict(list)
    attributed = {}
    for d in decisions:
        groups[str(d['binding'])].append(d)
        # Attribute realised lifecycle PnL once, to the first approved proposal
        # (or last rejection when none was approved). Raw decision rows remain.
        prior = attributed.get(d['id'])
        if prior is None or not prior['approved_qty']:
            attributed[d['id']] = d
    lifecycle_groups = defaultdict(list)
    for d in attributed.values():
        lifecycle_groups[str(d['binding'])].append(d)
    return dict(contexts=len(receipt.get('entry_contexts', [])), proposals=len(decisions),
        distinct_opportunities=len({d['id'] for d in decisions}),
        approved=sum(d['approved_qty'] > 0 for d in decisions),
        rejected=sum(d['approved_qty'] == 0 for d in decisions),
        quantity_reduced=sum(0 < d['approved_qty'] < d['desired_qty'] for d in decisions),
        bindings=[dict(label=k, decisions=len(v), rejected=sum(d['approved_qty'] == 0 for d in v),
            acquired=sum(d['acquired'] for d in lifecycle_groups[k]), meaningful=sum(d['meaningful'] for d in lifecycle_groups[k]),
            realised_net_of_acquired=sum(d['actual_net'] or 0 for d in lifecycle_groups[k])) for k,v in sorted(groups.items())],
        decisions=decisions,
        scope='Native hard-eligible proposal population. Decision counts may repeat; each realised lifecycle is attributed only once, to its first approved proposal. Rejected proposals have unknown alternative payoff; observed binding cohorts are selected, not causal effects.')


def source_funnel(snapshots, episodes, trades, funding, window, slots, session_days=None, entry_session_days=None):
    selected = [s for s in snapshots if window['start'] <= s['trade_date'] <= window['end']]
    trades_by_id = {t['id']:t for t in trades}
    proposals = defaultdict(list)
    for d in funding['decisions']:
        proposals[d['id']].append(d)
    opportunity_rows = []
    snapshot_days = {s['trade_date'] for s in selected}
    require(len(snapshot_days) == len(selected), 'Duplicate snapshot day')
    session_days = sorted(session_days or snapshot_days)
    entry_session_days = set(session_days if entry_session_days is None else entry_session_days)
    stages = {k:Counter() for k in ('daily_frontier','afternoon','build')}
    months = {}
    by_day = {s['trade_date']:s for s in selected}
    for day in session_days:
        month = months.setdefault(day[:7],dict(month=day[:7],sessions=0,snapshots=0,
            no_snapshot=0,empty_snapshot=0,entry_eligible=0,exit_only=0,candidate_days=0,acquired=0,meaningful=0,
            upstream_rejection_tags=Counter()))
        month['sessions'] += 1
        month['entry_eligible'] += day in entry_session_days
        month['exit_only'] += day not in entry_session_days
        snapshot = by_day.get(day)
        if snapshot is None:
            month['no_snapshot'] += day in entry_session_days
            continue
        month['snapshots'] += 1
        month['empty_snapshot'] += not snapshot['candidates']
        month['candidate_days'] += len(snapshot['candidates'])
        meta = snapshot.get('metadata') or {}
        for stage,key in [('daily_frontier','rejected_symbols'),('afternoon','afternoon_rejected_symbols'),('build','build_rejected_symbols')]:
            for reasons in (meta.get(key) or {}).values():
                for reason in set(reasons):
                    stages[stage][reason] += 1
                    month['upstream_rejection_tags'][stage+':'+reason] += 1
        month['acquired'] += sum(t['day']==day for t in trades)
        month['meaningful'] += sum(t['day']==day and t['meaningful'] for t in trades)
    for s in selected:
        for number, c in enumerate(s['candidates']):
            key = c['symbol'] + ':' + s['trade_date']; t = trades_by_id.get(key)
            observed = episodes.get(s['trade_date'], {}).get(c['symbol'], {}).get('episodes', [])
            confirmations = [e for e in observed if e.get('kind') == 'native_confirmation']
            ps = proposals.get(key, [])
            state = ('acquired' if t else 'approved_unfilled' if any(d['approved_qty'] > 0 for d in ps)
                else 'funding_rejected' if ps else 'outside_original_slots' if number >= slots
                else 'snapshot_untradable' if not c.get('tradable', True) else 'no_recorded_proposal')
            opportunity_rows.append(dict(id=key, day=s['trade_date'], symbol=c['symbol'],
                rank=c.get('rank'), selection_score=c.get('selection_score'), source_index=number,
                sector=c.get('sector'), tradable=c.get('tradable', True), inside_slots=number < slots,
                native_confirmation_observed=bool(confirmations),
                confirmation_times=[e.get('known_at') for e in confirmations],
                disposition=state, meaningful=t['meaningful'] if t else False, net=t['net'] if t else None))
    ids = {r['id'] for r in opportunity_rows}
    require(set(trades_by_id) <= ids and set(proposals) <= ids, 'Unmatched source opportunity')
    return dict(snapshot_sessions=len(selected), candidate_days=len(opportunity_rows),
        marked_sessions=len(session_days), entry_eligible_sessions=len(entry_session_days),
        reserved_exit_sessions=sorted(set(session_days)-entry_session_days),
        absent_snapshot_sessions=sorted(set(session_days)-snapshot_days),
        missing_snapshot_sessions=sorted(entry_session_days-snapshot_days),
        unexpected_snapshot_sessions=sorted(snapshot_days-entry_session_days),
        empty_snapshot_sessions=[s['trade_date'] for s in selected if not s['candidates']],
        upstream_rejection_tags={k:dict(v.most_common()) for k,v in stages.items()}, monthly=list(months.values()),
        tradable_candidate_days=sum(r['tradable'] for r in opportunity_rows),
        inside_original_slots=sum(r['inside_slots'] for r in opportunity_rows),
        recorded_confirmation_days=sum(r['native_confirmation_observed'] for r in opportunity_rows),
        dispositions=dict(Counter(r['disposition'] for r in opportunity_rows)),
        rows=opportunity_rows,
        scope='Candidate census plus saved upstream rejection tags, not hypothetical returns on the entire security universe. Rejection tags overlap within and across stages and must not be summed as independent opportunities. Missing snapshots differ from empty snapshots and from an economically valid rejection. Observed confirmation episodes carry no admission authority. No recorded proposal does not distinguish held, failed confirmation, consumed attempt or missing support without additional native state evidence. Outside-slot is selection position, not proof of qualified or profitable alpha.')


def source_entry_calendar(source_path, daily_dates):
    """Invoke only the frozen pure date selector, not the strategy/bundle loader.

    Reusing its actual boundary avoids diagnosing reserved recovery sessions as
    missing data. The selected AST and file digest are saved in the report.
    """
    tree=ast.parse(Path(source_path).read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if
        isinstance(n,ast.FunctionDef) and n.name=='_eligible_snapshot_dates' or
        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS' for t in n.targets)]
    require(len(nodes)==2, 'Unrecognised frozen entry-window boundary; do not assume all marked dates admit entries')
    future=ast.parse('from __future__ import annotations').body
    module=ast.Module(body=future+nodes,type_ignores=[])
    namespace={}
    exec(compile(ast.fix_missing_locations(module),str(source_path),'exec'),namespace)
    dates=tuple(datetime.fromisoformat(d).date() for d in daily_dates)
    eligible=[d.isoformat() for d in namespace['_eligible_snapshot_dates'](dates)]
    return dict(path=str(source_path),sha256=sha(source_path),function='_eligible_snapshot_dates',
        source='\n'.join(ast.unparse(n) for n in nodes),eligible_dates=eligible,
        recovery_sessions=namespace['_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS'])


def compare_cases(control, treatment):
    """Exact endpoint identity; the chosen price/quantity decomposition is ordered."""
    a = {t['id']: t for t in control['trades']}; b = {t['id']: t for t in treatment['trades']}
    changes = []
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        delta = (y['net'] if y else 0)-(x['net'] if x else 0)
        quantity = (y['qty']-x['qty'])*x['net']/x['qty'] if x and y else 0.
        unit = y['qty']*(y['net']/y['qty']-x['net']/x['qty']) if x and y else 0.
        state = 'matched' if x and y else 'added' if y else 'removed'
        changes.append(dict(id=key, kind=state, delta=delta, quantity_at_control_payoff=quantity,
            net_unit_payoff_at_treatment_quantity=unit,
            participation_delta=delta if state != 'matched' else 0.,
            control_qty=x['qty'] if x else 0, treatment_qty=y['qty'] if y else 0,
            control_meaningful=x['meaningful'] if x else False, treatment_meaningful=y['meaningful'] if y else False,
            control_net=x['net'] if x else 0., treatment_net=y['net'] if y else 0.,
            control_exit=x['exit'] if x else None, treatment_exit=y['exit'] if y else None))
    delta = sum(t['net'] for t in b.values())-sum(t['net'] for t in a.values())
    parts = {k:sum(c[k] for c in changes) for k in ('quantity_at_control_payoff', 'net_unit_payoff_at_treatment_quantity', 'participation_delta')}
    equal(sum(parts.values()), delta, 'Scenario decomposition does not reconcile')
    return dict(net_difference=delta, decomposition=parts, identities=dict(Counter(c['kind'] for c in changes)),
        meaningful_difference=sum(t['meaningful'] for t in b.values())-sum(t['meaningful'] for t in a.values()),
        changed_meaningful_contribution=sum(c['delta'] for c in changes if c['control_meaningful'] != c['treatment_meaningful']),
        changed_meaningful=[c for c in changes if c['control_meaningful'] != c['treatment_meaningful']],
        changes=sorted(changes, key=lambda c: abs(c['delta']), reverse=True),
        scope='Endpoint accounting, not causal mediation. Quantity at control unit payoff plus unit payoff at treatment quantity is an exact ordered decomposition; quantity, timing, fees, management and funding are endogenous. Repeated scenarios are not independent market episodes.')


def analyze_case(name, case, spec=None, window=None, slots=6, capital=None):
    path = Path(case['path']); native = spec is not None
    inputs = {}
    def load(filename):
        p = path / filename; inputs[str(p)] = sha(p)
        return read(p)
    metrics = case['metrics']
    capital = float(capital or metrics.get('initial_equity') or 100_000_000)
    if (path / 'trades.json').exists():
        raw = load('trades.json')
        trade_filename = 'trades.json'
    else:
        p = path / 'trades.jsonl.gz'; inputs[str(p)] = sha(p); raw = list(rows(p))
        trade_filename = 'trades.jsonl.gz'
    trades = normalize_trades(raw, capital)
    if (path / 'daily.json').exists():
        daily = load('daily.json')
    else:
        p = path/'equity.jsonl.gz'; inputs[str(p)] = sha(p)
        by_day = {r['timestamp'][:10]:r['equity'] for r in rows(p)}
        daily = [dict(day=d, equity=e) for d,e in sorted(by_day.items())]
    result = dict(version=VERSION, name=name, path=str(path), trade_filename=trade_filename, capital=capital, metrics=metrics,
        trades=trades, summary=trade_stats(trades), cohorts=cohort_analysis(trades),
        consistency=consistency(trades, daily, capital), coverage=dict(trades=True, daily=True,
            event_attribution=False, source_candidates=False, native_funding=False))
    if native:
        p = path/'minute_events.jsonl.gz'; inputs[str(p)] = sha(p)
        events = event_analysis(rows(p), capital)
        equal(events['terminal_nav'], capital*(1+metrics['official_mtm_net_return_pct']), 'Changed final NAV')
        require(math.isclose(events['worst_percentage']['drawdown_fraction'], metrics['max_drawdown_pct'], rel_tol=1e-10, abs_tol=1e-12), 'Changed full DD')
        equal(events['worst_currency']['decline_krw'], metrics['maximum_currency_decline'], 'Changed monetary DD')
        require(events['points'] == metrics['full_event_points'], 'Omitted event points')
        for d in daily:
            equal(events['daily_nav'][d['day']], d['equity'], 'Changed daily NAV')
        require(not events['terminal_inventory'], 'Open native inventory requires explicit attribution support')
        equal(sum(t['net'] for t in trades), events['terminal_nav']-capital, 'Trade PnL does not reconcile')
        equal(sum(t['fees'] for t in trades), -events['mtm_components'].get('explicit_fees', 0), 'Fee reconciliation failed')
        trade_symbols = defaultdict(float)
        for t in trades:
            trade_symbols[t['symbol']] += t['net']
        for s in trade_symbols.keys() | events['symbol_pnl'].keys():
            equal(trade_symbols[s], events['symbol_pnl'].get(s, 0), 'Symbol attribution differs from realised trades')
        funding = funding_analysis(load('funded_population_receipt.json'), trades)
        study = read(spec['study']); inputs[spec['study']] = sha(spec['study'])
        snapshots_path = Path(spec['execution_root'])/study['olr_snapshots']
        manifest = read(study['manifest'])
        require(sha(snapshots_path) == manifest['files'][study['olr_snapshots']], 'Unbound candidate snapshot')
        inputs[str(snapshots_path)] = sha(snapshots_path)
        selector_path=Path(spec['execution_root'])/'backtests/strategies/olr/replay_cache.py'
        require(sha(selector_path)==manifest['files']['backtests/strategies/olr/replay_cache.py'], 'Unbound entry-calendar source')
        inputs[str(selector_path)]=sha(selector_path)
        entry_calendar=source_entry_calendar(selector_path,[d['day'] for d in daily])
        funnel = source_funnel(read(snapshots_path), load('signal_episodes.json'), trades, funding, window, slots,
            [d['day'] for d in daily],entry_calendar['eligible_dates'])
        funnel['entry_calendar']=entry_calendar
        decisions = load_decisions(path/'native_decisions.jsonl.gz', inputs)
        result.update(events=events, execution=execution_summary(events['orders']), funding=funding, funnel=funnel,
            decision_counts=dict(Counter(d['decision_code'] for d in decisions)))
        # Declared deterministic sample: extremes, plus the largest losing-name
        # contributor to the worst drawdown. All orders remain in the JSON.
        chosen = {t['id'] for t in sorted(trades, key=lambda t:t['net'])[:3] + sorted(trades, key=lambda t:t['net'], reverse=True)[:3]}
        worst_names = sorted(events['worst_percentage']['symbol_attribution'], key=events['worst_percentage']['symbol_attribution'].get)[:3]
        for s in worst_names:
            chosen.update(t['id'] for t in trades if t['symbol'] == s and t['entry'] <= events['worst_percentage']['trough_at'] and t['exit'] >= events['worst_percentage']['peak_at'])
        traces = []
        for t in trades:
            if t['id'] not in chosen:
                continue
            # Associate each order by its native order_date when supplied, or
            # by availability inside the actual held interval for SELL orders.
            relevant_orders = [o for o in events['orders'] if o['symbol'] == t['symbol'] and (
                o['side'] == 'BUY' and str(o['decision_label'])[:10] == t['day'] or
                o['side'] == 'SELL' and o['available_at'] and t['entry'] <= o['available_at'] <= t['exit'])]
            relevant_decisions = [d for d in decisions if d['symbol'] == t['symbol'] and
                t['day'] <= d['timestamp'][:10] <= t['exit'][:10]]
            traces.append(dict(trade=t, orders=relevant_orders, decisions=relevant_decisions,
                funding=[d for d in funding['decisions'] if d['id'] == t['id']]))
        result['traces'] = traces
        result['coverage'].update(event_attribution=True, source_candidates=True, native_funding=True)
    else:
        result['unavailable'] = ['Native source census, funded population and fine-event attribution are unavailable in this export schema. No zero-valued substitutes.']
    require(all(sha(p) == h for p,h in inputs.items()), 'Analysis source changed while reading')
    result['input_sha256'] = inputs
    result['reconciliation_passed'] = True
    return result


def load_decisions(path, inputs):
    inputs[str(path)] = sha(path)
    return list(rows(path))


def money(v):
    return 'n/a' if v is None else f'{v/1e6:+.3f}m'


def num(v, digits=2):
    return 'n/a' if v is None else f'{v:.{digits}f}'


def pct(v):
    return 'n/a' if v is None else f'{100*v:.2f}%'


def table(headers, values):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |'] + [
        '| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in values]


def case_report(a):
    m, s, c = a['metrics'], a['summary'], a['consistency']
    text = [f'# OLR {a["name"]}: analytical final diagnostics', '',
        f'Accepted source: `{a["path"]}`. All currency amounts below are KRW; m = million.',
        f'[Trade source]({Path(a["path"]).as_posix()}/{a["trade_filename"]}) and [complete analysis](analysis.json).', '',
        '## Payoff, participation and risk', '',
        f'Net {money(s["net"])} across {s["n"]} independent symbol/decision-date opportunities; {s["meaningful"]} acquired at least 1% of initial capital. '
        f'{s["wins"]} winners contribute {money(s["winning_wealth"])}; {s["losses"]} losers consume {money(-s["losing_wealth"])}. '
        f'PF {num(s["profit_factor"])}; win rate {pct(s["win_rate"])}; mean net R {num(s["net_r"]["mean"])} (n={s["net_r"]["n"]}); '
        f'aggregate net / sum of initial position risk {num(s["aggregate_net_per_initial_risk"])}. The latter weights by risk actually allocated; neither is a portfolio-return series.', '',
        f'Gross realised PnL {money(s["gross"])} already includes simulated execution prices/slippage; explicit fees/tax {money(s["fees"])}. '
        f'Net contribution per acquired-notional unit is {num(s["net_bps_on_acquired_notional"])} bps. This turnover ratio is not a portfolio return.', '',
        '## Where the payoff comes from', '',
        'Cohorts attribute whole realised lifecycles to entry state; they do not estimate the value of a changed rule. Small = fewer than 20 trades, a reporting flag rather than a rejection criterion.', '']
    for dim in ('sector','rank','score_band','entry_clock','exit_reason','holding_calendar_days','meaningful','outcome','excursion'):
        text += [f'### {dim.replace("_", " ")}', '']
        text += table(['Cohort','N','Wins','Meaningful','Net KRW','PF','Mean net R','Risk-weighted R','Mean risk KRW','Mean MFE R','Mean MAE R','Median hold h','Small'],
            [[r['label'],r['n'],r['wins'],r['meaningful'],money(r['net']),num(r['profit_factor']),num(r['net_r']['mean']),num(r['aggregate_net_per_initial_risk']),money(r['initial_risk_krw']['mean']),num(r['mfe_r']['mean']),num(r['mae_r']['mean']),num(r['hold_hours']['median']),r['small_sample']] for r in a['cohorts'][dim]])
        text += ['']
    text += ['## Chronological consistency', '',
        'Month rows use daily account MTM, including held inventory and all sessions. They differ from entry-month realised PnL. Independent account windows are not stitched.', '']
    text += table(['Month','Sessions','Start equity','End equity','MTM PnL','Return','Up/down days'],
        [[r['month'],r['sessions'],money(r['start_equity']),money(r['end_equity']),money(r['net']),pct(r['return_fraction']),f'{r["positive_days"]}/{r["negative_days"]}'] for r in c['monthly']])
    text += ['', f'{c["positive_months"]}/{c["months"]} positive months; {c["positive_rolling_windows"]}/{c["rolling_windows"]} overlapping rolling 20-closed-trade windows positive. '
        f'Longest closed-trade win/loss streaks {c["longest_streaks"].get("wins",0)}/{c["longest_streaks"].get("losses",0)}. These windows are dependent and do not multiply validation evidence.', '',
        '## Concentration and fragility', '']
    conc=c['concentration']
    text += [f'Top winner contributes {pct(conc["top1_winner_share"])} and top five {pct(conc["top5_winner_share"])} of positive PnL. '
        f'Positive-PnL effective winner count (inverse squared-share sum) {num(conc["positive_pnl_effective_winner_count"])}. '
        f'Remaining net after deleting the best one/five trades: {money(conc["without_best_1_trade"])}/{money(conc["without_best_5_trades"])}; '
        f'after deleting the best daily MTM gain: {money(conc["without_best_daily_MTM"])}. '
        'These hold later exposure fixed and are sensitivity descriptions, not funded alternatives or independent confidence intervals.', '']
    text += table(['Largest winning opportunity','Net KRW'], [[r['id'],money(r['net'])] for r in conc['top_winners']])
    text += ['', '## Entry excursions and management', '',
        f'Recorded MFE/MAE coverage {s["excursion_coverage"]}/{s["n"]}; mean MFE {num(s["mfe_r"]["mean"])}R, MAE {num(s["mae_r"]["mean"])}R. '
        f'Aggregate gross PnL / recorded quantity-weighted MFE is {pct(s["gross_capture_of_recorded_mfe"])}. '
        'MFE is a hindsight observed price excursion, not a simultaneously fillable exit; quantities may evolve. No partial-profit or tighter-stop recommendation follows from this ratio.', '',
        'Winner/loser excursion and holding cohorts above show both recoveries and sacrifices; threshold labels summarise paths and are not proposed trading predicates.', '']
    if a['coverage']['event_attribution']:
        ev, f, funding = a['events'], a['funnel'], a['funding']
        text += ['## Opportunity-to-participation funnel', '',
            f'{f["snapshot_sessions"]} source snapshot sessions; {f["candidate_days"]} candidate-days, {f["tradable_candidate_days"]} tradable; '
            f'{f["inside_original_slots"]} inside the original slots; {f["recorded_confirmation_days"]} with observer-recorded native confirmation. '
            f'{funding["distinct_opportunities"]} distinct proposed opportunities in {funding["contexts"]} decision contexts; '
            f'{funding["approved"]} approved proposal rows, {funding["rejected"]} rejected, {funding["quantity_reduced"]} reduced.', '']
        text += table(['Final candidate disposition','Candidate-days'], sorted(f['dispositions'].items()))
        text += ['', f['scope'], '', '### Source coverage and inactive periods', '',
            f'Unexpected missing entry-eligible snapshots: {", ".join(f["missing_snapshot_sessions"]) or "none"}. '
            f'The frozen native date selector reserves exit/recovery sessions {", ".join(f["reserved_exit_sessions"])}; no new snapshot is expected on those dates. '
            'A full marked account curve does not establish a full opportunity feed. Missing source sessions are not counted as market rejections and constrain whole-window opportunity/robustness claims.', '']
        text += table(['Month','Marked sessions','Entry-eligible','Exit-only','Snapshot sessions','Unexpected missing','Empty snapshots','Candidate-days','Acquired/meaningful'],
            [[r['month'],r['sessions'],r['entry_eligible'],r['exit_only'],r['snapshots'],r['no_snapshot'],r['empty_snapshot'],r['candidate_days'],f'{r["acquired"]}/{r["meaningful"]}'] for r in f['monthly']])
        text += ['', '### Upstream rejection reasons (overlapping symbol-day tags)', '']
        for stage, reasons in f['upstream_rejection_tags'].items():
            text += [f'{stage}: '+('; '.join(f'{k}: {v}' for k,v in reasons.items()) or 'no tags recorded')+'.', '']
        text += ['## Funding constraints and later access', '', funding['scope'], '']
        text += table(['Binding at proposal','Decisions','Rejected','Acquired','Meaningful','Realised PnL of acquired'],
            [[r['label'],r['decisions'],r['rejected'],r['acquired'],r['meaningful'],money(r['realised_net_of_acquired'])] for r in funding['bindings']])
        rejected = [d for d in funding['decisions'] if not d['approved_qty']]
        text += ['', 'Every rejected proposal (saved cash is not assigned a hypothetical payoff):', '']
        text += table(['At','Symbol','Desired','Reason','Binding','Deployable cash','Gross committed'],
            [[d['at'],d['symbol'],d['desired_qty'],d['reason'],d['binding'],money(d['deployable']),money(d['gross_committed'])] for d in rejected])
        text += ['', '## Executed orders, expiry and fallback', '']
        text += table(['Side','Orders','Filled orders','Fill fragments','Partial/unfilled','Expired','Cancelled','Median first-fill minutes','p90 completion minutes'],
            [[side,e['orders'],e['filled_orders'],e['fragments'],e['partial_or_unfilled_orders'],e['expires'],e['cancels'],num(e['first_fill_delay_minutes']['median']),num(e['completion_delay_minutes']['p90'])] for side,e in a['execution'].items() if side in ('BUY','SELL')])
        text += ['', a['execution']['scope'], 'Delay starts at decision_available_at, not the earlier bar label; calendar minutes include overnight closure. Native service and source audits do not certify physical queue execution.', '',
            '## Full-event drawdown autopsy', '', 'Percentage and currency maxima are found independently. Flat terminal inventory does not imply recovery of the wealth peak.', '']
        for label,key in [('Largest percentage decline','worst_percentage'),('Largest currency decline','worst_currency')]:
            dd=ev[key]
            text += [f'### {label}', '', f'{dd["peak_at"]} to {dd["trough_at"]}: {money(dd["peak"])} to {money(dd["trough"])}; decline {money(dd["decline_krw"])} / {pct(dd["drawdown_fraction"])}.', '']
            text += table(['Symbol','Actual MTM contribution in peak-to-trough interval'], [[k,money(v)] for k,v in sorted(dd['symbol_attribution'].items(),key=lambda kv:kv[1])])
            text += ['', 'Component attribution: '+', '.join(f'{k} {money(v)}' for k,v in dd['component_attribution'].items())+'.', '']
        text += ['Largest drawdown episodes (full list in JSON):', '']
        text += table(['Peak','Trough','Recovered/end','DD','KRW decline','Calendar days','Censored'],
            [[r['peak_at'],r['trough_at'],r['end'],pct(r['drawdown_fraction']),money(r['decline_krw']),num(r['duration_days']),r['censored']] for r in ev['drawdown_episodes'][:10]])
        text += ['', f'Whole-window MTM components: '+', '.join(f'{k} {money(v)}' for k,v in ev['mtm_components'].items())+'.',
            'Overnight mark is the first observed mark for a symbol on a new session; execution-vs-mark includes modeled price effects and fees are separate. These additive components reconcile to NAV but are not separately investable alphas.', '',
            '## Exposure and occupancy', '',
            f'Calendar time-weighted gross/NAV {pct(m.get("calendar_time_weighted_gross_nav"))}; observed same-day time-weighted gross/NAV {pct(m.get("observed_same_day_time_weighted_gross_nav"))}; '
            f'maximum single-name/NAV {pct(m.get("max_single_name_nav_share"))}. Calendar weighting carries the last observed exposure through closures; intraday excludes between-session time. '
            'Neither is event-count weighting, and lower intraday occupancy need not mean less overnight capital use.', '',
            '## Source-linked supporting and contrary traces', '',
            'Declared sample: three largest winners, three largest losers, and overlapping lifecycles of the three worst symbol contributors to maximum percentage DD. All opportunities/orders/funding rows remain in the JSON. Source row numbers below are zero-based.', '']
        for trace in a['traces']:
            t=trace['trade']
            text += [f'### {t["id"]}', '',
                f'`trades.json[{t["source_row"]}]`: rank {t["rank"]}, {t["sector"]}, original risk/share {num(t["risk_per_share"])}, '
                f'{num(t["qty"],0)} shares; {t["entry"]} -> {t["exit"]}; {t["exit_reason"]}; net {money(t["net"])} / {num(t["net_r"])}R; '
                f'MFE/MAE {num(t["mfe_r"])}/{num(t["mae_r"])}R.', '']
            text += table(['Order id','Side/reason','Decision available','Filled/requested','First fill','Last fill','Terminal events'],
                [[o['id'],str(o['side'])+'/'+str(o['reason']),o['available_at'],f'{o["filled_qty"]}/{o["qty"]}',o['first_fill'],o['last_fill'],','.join(e['event'] for e in o['events'])] for o in trace['orders']])
            text += ['']
    text += ['## Evidence and limitations', '',
        'Reconciled analysis is descriptive, after costs, and based on already consumed history. No benchmark-adjusted alpha, untouched validation, queue robustness, counterfactual rejected payoff, or improved strategy is established. '
        'The source paths and SHA-256s, full cohorts, all drawdown episodes and order/funding records are in analysis.json. No trading transition runs in this engine.', '']
    return '\n'.join(text)


def render_overview(analyses, comparisons):
    text = ['# OLR final-round economic analysis', '',
        'This report explains the accepted trading paths. It does not change the retained policy or turn observed path associations into proposed profitable rules.', '',
        '## Findings that change the interpretation', '']
    if 'train_nominal' in analyses and 'oos_nominal' in analyses:
        x,y=analyses['train_nominal'],analyses['oos_nominal']
        text += [f'- The payoff is asymmetric, not a high-hit-rate process: nominal win rates are {pct(x["summary"]["win_rate"])} IS and {pct(y["summary"]["win_rate"])} later. '
            f'Equal-trade mean net R rises {num(x["summary"]["net_r"]["mean"],3)} -> {num(y["summary"]["net_r"]["mean"],3)}, but net / actually allocated initial risk falls '
            f'{num(x["summary"]["aggregate_net_per_initial_risk"],3)} -> {num(y["summary"]["aggregate_net_per_initial_risk"],3)}. '
            'Opportunity mix, sizing and large loss exposure matter; the aggregate deterioration cannot be attributed solely to weaker equal-sized signals.', '']
        for a in (x,y):
            ex={r['label']:r for r in a['cohorts']['exit_reason']}
            fail=ex.get('failed_followthrough');fallback=ex.get('auction_exit_nonfill_market_fallback')
            if fail and fallback:
                text += [f'- {a["name"]}: {fail["n"]} failed-followthrough lifecycles contribute {money(fail["net"])} '
                    f'({pct(-fail["net"]/a["summary"]["losing_wealth"])} of total losing wealth on a net-cohort basis); '
                    f'{fallback["n"]} auction-fallback lifecycles contribute {money(fallback["net"])}. '
                    'The failure-exit label describes the eventual path, not an entry-time discriminator; fallback cohort PnL does not estimate the incremental value of extending the hold.', '']
            if a.get('risk_components'):
                rc=a['risk_components']
                text += [f'- {a["name"]} realised NAV decomposition: overnight first-mark changes {money(rc.get("overnight_mark",0))}, '
                    f'intraday mark changes {money(rc.get("intraday_mark",0))}, execution-versus-mark {money(rc.get("execution_vs_observable_mark",0))}, '
                    f'fees/tax {money(rc.get("explicit_fees",0))}. This tests where continuation payoff appears without claiming separately tradable components.', '']
            missing=a.get('source_missing',[])
            if missing:
                text += [f'- Coverage limit in {a["name"]}: candidate snapshots are absent on {", ".join(missing)} even though account NAV is marked. '
                    'Those sessions cannot establish strategy opportunity coverage. Missing decisions could help or hurt; their return effect is unknown.', '']
            elif a.get('entry_calendar_counts'):
                ec=a['entry_calendar_counts']
                text += [f'- {a["name"]} has {ec["eligible"]} entry-eligible sessions within {ec["marked"]} marked account sessions. '
                    f'The final {ec["reserved"]} sessions are deliberately reserved by the frozen builder for exit/auction recovery. '
                    'All expected entry snapshots are present; the reserve is part of the evaluation contract, not evidence of a data gap or continued entry operation.', '']
    text += ['## Comparative account evidence', '']
    text += table(['Account','Return','Full DD','Currency DD','Sharpe','Trades/meaningful','Win rate','PF','Mean net R','Top-five winner share'],
        [[name,pct(a['metrics']['official_mtm_net_return_pct']),pct(a['metrics']['max_drawdown_pct']),money(a['metrics'].get('maximum_currency_decline')),
          num(a['metrics'].get('sharpe')),f'{a["summary"]["n"]}/{a["summary"]["meaningful"]}',pct(a['summary']['win_rate']),num(a['summary']['profit_factor']),num(a['summary']['net_r']['mean']),pct(a['consistency']['concentration']['top5_winner_share'])] for name,a in analyses.items()])
    text += ['', 'The IS and development OOS accounts each start with KRW100m; their returns are not stitched. Daily Sharpe uses zero risk-free return and is descriptive for these short, dependent samples. Full-event DD includes within-session risk.', '',
        '## Matched execution and capital sensitivities', '',
        'Stress and cap comparisons replay the same market episodes. Stress changes modeled costs/service; capped sizing removes upward compounding while allowing losses to reduce available capital. Neither creates independent observations.', '']
    text += table(['Treatment minus nominal','Net KRW','Added/removed/matched identities','Meaningful delta','Quantity component','Unit-payoff component','Participation component'],
        [[key,money(v['net_difference']),f'{v["identities"].get("added",0)}/{v["identities"].get("removed",0)}/{v["identities"].get("matched",0)}',v['meaningful_difference'],money(v['decomposition']['quantity_at_control_payoff']),money(v['decomposition']['net_unit_payoff_at_treatment_quantity']),money(v['decomposition']['participation_delta'])] for key,v in comparisons.items()])
    text += ['', 'Quantity/unit-payoff terms form an exact ordered accounting identity, not causal mediation. The unit-payoff term combines price, cost and management differences. The full JSON identifies new, removed and threshold-crossing opportunities separately.', '']
    for key,v in comparisons.items():
        text += [f'### Largest absolute contributions: {key}', '']
        text += table(['Opportunity','Change','Net difference','Quantity before/after','Exit before/after'],
            [[r['id'],r['kind'],money(r['delta']),f'{num(r["control_qty"],0)}/{num(r["treatment_qty"],0)}',f'{r["control_exit"]} / {r["treatment_exit"]}'] for r in v['changes'][:8]])
        text += ['']
    if 'train_nominal' in analyses and 'oos_nominal' in analyses:
        train, oos = analyses['train_nominal'], analyses['oos_nominal']
        text += ['## What transferred, and what deteriorated', '',
            f'Nominal net R per closed trade moves from {num(train["summary"]["net_r"]["mean"])} to {num(oos["summary"]["net_r"]["mean"])}; '
            f'win rate {pct(train["summary"]["win_rate"])} -> {pct(oos["summary"]["win_rate"])} and payoff factor {num(train["summary"]["profit_factor"])} -> {num(oos["summary"]["profit_factor"])}. '
            f'Independent acquisitions per 21 sessions {num(21*train["summary"]["n"]/train["metrics"]["sessions"])} -> {num(21*oos["summary"]["n"]/oos["metrics"]["sessions"])}. '
            'Compare both opportunity rate and unit payoff, rather than interpreting smaller raw OOS profit across fewer sessions as the entire deterioration.', '',
            f'Top-five winner concentration changes {pct(train["consistency"]["concentration"]["top5_winner_share"])} -> {pct(oos["consistency"]["concentration"]["top5_winner_share"])}. '
            f'Deleting the best five trades on each fixed path leaves {money(train["consistency"]["concentration"]["without_best_5_trades"])}/{money(oos["consistency"]["concentration"]["without_best_5_trades"])}. '
            'This measures tail dependence, not the PnL of avoiding those trades.', '']
        for dim in ('sector','rank','exit_reason'):
            ta={r['label']:r for r in train['cohorts'][dim]}; ob={r['label']:r for r in oos['cohorts'][dim]}
            text += [f'### Transfer by {dim.replace("_"," ")}', '']
            text += table(['Cohort','IS N/net/mean R','OOS N/net/mean R'],
                [[k, f'{ta[k]["n"]} / {money(ta[k]["net"])} / {num(ta[k]["net_r"]["mean"])}' if k in ta else 'absent',
                  f'{ob[k]["n"]} / {money(ob[k]["net"])} / {num(ob[k]["net_r"]["mean"])}' if k in ob else 'absent'] for k in sorted(ta.keys()|ob.keys())])
            text += ['']
    text += ['## Consequential research questions and predecessor constraints', '',
        '1. Entry quality versus capital selection: use the source/funding funnel and equal-risk payoff alongside money PnL to distinguish weak proposals from quantity and funding effects. A rejected candidate has no realised outcome here. The completed 0037 census and 0039 access test constrain any claim of abundant missed alpha; added contacts alone did not establish useful contribution.', '',
        '2. Overnight risk versus sacrificed recovery: use the independently reconciled peak-to-trough symbol/mark attribution and winner/loser traces. Do not infer an earlier sell rule from an eventual loss or MFE giveback. 0035/0036 and 0041 already document recovery sacrifices and countervailing funding effects; a quicker exit requires genuinely distinguishing decision-time evidence.', '',
        '3. Management versus subsequent access: exact structural H preserved a narrower IS payoff improvement; H plus access in 0040 damaged the endogenous history/funding path. The factorial evidence prevents summing standalone PnL or assuming released capital buys a nominated winner.', '',
        '4. Transfer and execution sensitivity: inspect named scenario differences, quantities and exit times, then the distinct OOS cohort/episode evidence. Same identities or repeated stress runs are not independent confirmation. Stopped 0042 remains stopped; this report does not authorise another experiment.', '',
        'These are evidence operations for a future authorised design, not automatic policy recommendations. The present task changes diagnostic capability only.', '',
        '## Reading the detailed reports', '',
        'Each account report contains payoff/cohort tables, source and funding funnels, all rejected proposals, execution delays and expiry, monthly/rolling consistency, excursion distributions, concentration, independent percentage/currency drawdown autopsies, occupancy, and supporting/contrary source-linked lifecycle traces. JSON carries complete rows without padding the human report.', '']
    return '\n'.join(text)


def build_analysis(cases, output_dir, plan=None, capital=None):
    """Return report references + a concise economic synthesis for the renderer."""
    output_dir = Path(output_dir); analyses = {}; artifacts = {}
    for name, case in cases.items():
        if not case.get('path'):
            continue
        print('Analytical diagnostics: '+name, flush=True)
        spec = plan['cases'][name] if plan else None
        window = plan['windows'][name.split('_')[0]] if plan else None
        slots = plan['effective_policy']['olr_config']['overnight_slot_count'] if plan else 6
        a = analyze_case(name, case, spec, window, slots, capital)
        directory = output_dir/'diagnostics'/name
        directory.mkdir(parents=True, exist_ok=True)
        jp = directory/'analysis.json'; rp=directory/'analysis.md'
        jp.write_text(json.dumps(a, indent=2, ensure_ascii=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
        rp.write_text(case_report(a), encoding='utf-8', newline='\n')
        artifacts[name] = dict(json=str(jp.relative_to(output_dir)), json_sha256=sha(jp),
            report=str(rp.relative_to(output_dir)), report_sha256=sha(rp),
            coverage=a['coverage'], reconciliation_passed=a['reconciliation_passed'])
        # Keep only the small comparative surface after serialising large traces.
        analyses[name] = {k:a[k] for k in ('name','metrics','summary','trades','cohorts','consistency')}
        if a['coverage']['event_attribution']:
            analyses[name]['risk_components'] = a['events']['mtm_components']
            analyses[name]['source_missing'] = a['funnel']['missing_snapshot_sessions']
            analyses[name]['entry_calendar_counts'] = dict(eligible=a['funnel']['entry_eligible_sessions'],
                marked=a['funnel']['marked_sessions'],reserved=len(a['funnel']['reserved_exit_sessions']))
    comparisons = {}
    for window in ('train', 'oos'):
        control=window+'_nominal'
        if control in analyses:
            for scenario in ('stress','capped'):
                treatment=window+'_'+scenario
                if treatment in analyses:
                    comparisons[treatment+' - '+control] = compare_cases(analyses[control], analyses[treatment])
    overview = render_overview(analyses, comparisons)
    compact = {k:{f:v for f,v in a.items() if f not in ('trades','consistency')} for k,a in analyses.items()}
    return dict(version=VERSION, cases=artifacts, comparisons=comparisons, summaries=compact,
        engine_sha256=sha(__file__), implementation=str(Path(__file__).resolve()),
        changed_trading_policy=False, financial_replays=0), overview
