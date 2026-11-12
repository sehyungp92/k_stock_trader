"""Read-only descriptive OLR diagnostics; never substitutes for native replays."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009')
CID = 'ce14d8377b9e67ac6'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ratio(a, b):
    return a / b if b else None


def account_slice(daily, trades, initial_equity, first, last):
    """Slice observed NAV, retaining wealth and positions from preceding sessions."""
    opening = daily[first-1]['equity'] if first else initial_equity
    rows = daily[first:last+1]
    peak = opening
    drawdown = 0.
    for row in rows:
        peak = max(peak, row['equity'])
        drawdown = max(drawdown, 100*(1-row['equity']/peak))
    entries = sum(rows[0]['day'] <= t['entry_fill_time'][:10] <= rows[-1]['day'] for t in trades)
    return {'start':rows[0]['day'],'end':rows[-1]['day'],'sessions':len(rows),
        'opening_equity':opening,'closing_equity':rows[-1]['equity'],
        'net_nav_change':rows[-1]['equity']-opening,
        'return_pct':100*(rows[-1]['equity']/opening-1),'drawdown_pct':drawdown,
        'entries':entries,'entries_per_21_sessions':21*entries/len(rows)}


def calendar_context(root, records):
    """Descriptive overlapping IS windows and calendar attribution, no fresh replay."""
    output = {}
    for window in ('train','oos'):
        folder = Path(records[window]['result_path']).parent
        daily, trades = read(folder/'daily.json'), read(folder/'trades.json')
        initial = read(root/'protocol.json')['reference_capital']
        months = sorted({r['day'][:7] for r in daily})
        output[window] = {'months':{month:account_slice(daily,trades,initial,
            next(i for i,r in enumerate(daily) if r['day'].startswith(month)),
            max(i for i,r in enumerate(daily) if r['day'].startswith(month))) for month in months}}
        if window == 'train':
            length = records['oos']['sessions']
            slices = [account_slice(daily,trades,initial,i,i+length-1) for i in range(len(daily)-length+1)]
            from statistics import median
            matched = {'sessions':length,'window_count':len(slices),'windows':slices,
                'return_pct':{'min':min(r['return_pct'] for r in slices),
                    'median':median(r['return_pct'] for r in slices),'max':max(r['return_pct'] for r in slices)},
                'entries':{'min':min(r['entries'] for r in slices),
                    'median':median(r['entries'] for r in slices),'max':max(r['entries'] for r in slices)},
                'note':'Overlapping descriptive slices of the continuous IS account, not independent trials or separately reset replays. Entry counts may differ from exits at the boundaries.'}
            (root/'matched_length_is_windows.json').write_text(json.dumps(matched,indent=2)+'\n',encoding='utf-8')
        else:
            cuts = [i for i,r in enumerate(daily) if r['day'] < '2026-06-01']
            if cuts and max(cuts)+1 < len(daily):
                split = max(cuts)+1
                output[window]['april_may'] = account_slice(daily,trades,initial,0,split-1)
                output[window]['june_july'] = account_slice(daily,trades,initial,split,len(daily)-1)
    output['note'] = 'All calendar slices preserve the preceding account wealth and carried positions. June-July here is not the old separately initialized June-only experiment.'
    (root/'calendar_subperiods.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')


def compact_trade(t):
    m = t.get('route_metadata') or {}
    return {**{k: t.get(k) for k in ('symbol', 'qty', 'entry_fill_time', 'exit_fill_time',
        'entry_price', 'exit_price', 'gross_pnl', 'commission', 'net_pnl', 'exit_reason', 'mfe', 'mae')},
        'entry_notional': t['qty'] * t['entry_price'],
        'net_return_on_entry_pct': 100 * t['net_pnl'] / (t['qty'] * t['entry_price']),
        **{k: m.get(k) for k in ('candidate_rank', 'candidate_score', 'sector', 'afternoon_score_band_rule',
           'allocation_weight', 'target_notional', 'allocation_redistributed_qty', 'capacity_recycling',
           'hard_stop_enabled', 'allocation_binding_limit')}}


def analyse(window, folder, info):
    result = read(folder/'result.json')
    trades = read(folder/'trades.json')
    daily = read(folder/'daily.json')
    window_days = {r['day'] for r in daily}
    snapshots = [s for s in read(Path(info['files'])/info['study']['olr_snapshots']) if s['trade_date'] in window_days]
    groups = {name: defaultdict(list) for name in ('month', 'rank', 'sector', 'band', 'regime', 'exit')}
    snapshot_by_day = {s['trade_date']: s for s in snapshots}
    for t in trades:
        m = t.get('route_metadata') or {}
        day = t['entry_fill_time'][:10]
        regime = snapshot_by_day.get(day, {}).get('metadata', {}).get('market', {}).get('regime')
        labels = {'month': day[:7], 'rank': m.get('candidate_rank'), 'sector': m.get('sector'),
                  'band': m.get('afternoon_score_band_rule'), 'regime': regime, 'exit': t['exit_reason']}
        for key, label in labels.items():
            groups[key][str(label)].append(t)
    cohorts = {dimension: {key: {'trades': len(rows), 'wins': sum(t['net_pnl'] > 0 for t in rows),
        'net_pnl': sum(t['net_pnl'] for t in rows),
        'equal_weight_net_return_pct': mean(100*t['net_pnl']/(t['qty']*t['entry_price']) for t in rows)}
        for key, rows in values.items()} for dimension, values in groups.items()}
    funnel = Counter()
    rejections = defaultdict(Counter)
    sole_rejections = defaultdict(Counter)
    regimes = Counter()
    for s in snapshots:
        m = s['metadata']
        regimes[str(m.get('market', {}).get('regime'))] += 1
        for key in ('requested_universe_count', 'complete_universe_count', 'candidate_pool_count',
                    'afternoon_candidate_pool_count', 'selected_symbol_count'):
            funnel[key] += m.get(key, 0)
        funnel['final_candidate_rows'] += len(s['candidates'])
        funnel['days_with_candidates'] += bool(s['candidates'])
        funnel['feature_prefix_excluded_symbol_days'] += len(m.get('feature_prefix_exclusions') or [])
        for stage in ('build_rejected_symbols', 'rejected_symbols', 'afternoon_rejected_symbols'):
            for reasons in (m.get(stage) or {}).values():
                reasons = reasons if isinstance(reasons, list) else [reasons]
                rejections[stage].update(reasons)
                if len(reasons) == 1:
                    sole_rejections[stage].update(reasons)
    net = sum(t['net_pnl'] for t in trades)
    wins = sorted([t['net_pnl'] for t in trades if t['net_pnl'] > 0], reverse=True)
    losses = sorted([t['net_pnl'] for t in trades if t['net_pnl'] < 0])
    return {'window': window, 'result_path': str(folder/'result.json'),
        'audit_status': result['audit_status'], 'return_pct': result['return_pct'],
        'sessions': result['sessions'], 'entry_snapshot_sessions': len(snapshots),
        'trades': len(trades), 'active_days': result['active_days'],
        'trades_per_21_sessions': result['trades_per_21_sessions'], 'win_rate': len(wins)/len(trades),
        'net_pnl': net, 'gross_pnl': sum(t['gross_pnl'] for t in trades),
        'explicit_costs': sum(t['commission'] for t in trades),
        'cost_note': 'Commission field includes native explicit charges; fill prices already include slippage.',
        'average_win': mean(wins), 'average_loss': mean(losses),
        'payoff_ratio': -mean(wins)/mean(losses),
        'break_even_win_rate': -mean(losses)/(mean(wins)-mean(losses)),
        'arithmetic_excluding_two_worst': net-sum(losses[:2]),
        'arithmetic_excluding_two_best': net-sum(wins[:2]),
        'two_worst_share_of_gross_losses': sum(losses[:2])/sum(losses),
        'concentration_warning': 'Removing realized trades is descriptive, not an executable counterfactual.',
        'cohorts': cohorts, 'funnel_counts': dict(funnel),
        'funnel_per_entry_session': {k: v/len(snapshots) for k, v in funnel.items()},
        'regime_sessions': dict(regimes), 'rejection_reasons_overlapping': dict(rejections),
        'sole_rejection_reasons': dict(sole_rejections),
        'rejection_note': 'Reasons overlap. Sole failures isolate one recorded gate but do not prove a trade would execute.',
        'worst_trades': [compact_trade(t) for t in sorted(trades, key=lambda t: t['net_pnl'])[:10]],
        'best_trades': [compact_trade(t) for t in sorted(trades, key=lambda t: -t['net_pnl'])[:10]]}


def loss_paths(trades, info):
    import pandas as pd
    contract = info['manifest']['evidence']['shared_market_inputs']
    answer = []
    for trade in sorted(trades, key=lambda t: t['net_pnl'])[:2]:
        entry = datetime.fromisoformat(trade['entry_fill_time']).astimezone(ZoneInfo('Asia/Seoul'))
        exit_time = datetime.fromisoformat(trade['exit_fill_time']).astimezone(ZoneInfo('Asia/Seoul'))
        pieces = []
        for name in contract['intraday_files'][trade['symbol']]:
            frame = pd.read_parquet(Path(info['files'])/name)
            stamps = pd.to_datetime(frame['timestamp'])
            stamps = stamps.dt.tz_localize('Asia/Seoul') if stamps.dt.tz is None else stamps.dt.tz_convert('Asia/Seoul')
            frame = frame.assign(timestamp=stamps)
            pieces.append(frame[(stamps.dt.date >= entry.date()) & (stamps.dt.date <= exit_time.date())])
        frame = pd.concat(pieces).sort_values('timestamp')
        sessions = []
        for day, rows in frame.groupby(frame.timestamp.dt.date):
            sessions.append({'date': day.isoformat(), 'first_bar': rows.iloc[0].timestamp.isoformat(),
                'open': float(rows.iloc[0].open), 'last_bar': rows.iloc[-1].timestamp.isoformat(),
                'close': float(rows.iloc[-1].close), 'high': float(rows.high.max()), 'low': float(rows.low.min())})
        first_close = sessions[0]['close']
        following_open = sessions[1]['open'] if len(sessions)>1 else None
        answer.append({'trade': compact_trade(trade), 'sessions': sessions,
            'entry_to_first_close_pct': 100*(first_close/trade['entry_price']-1),
            'first_overnight_gap_pct': 100*(following_open/first_close-1) if following_open else None,
            'following_open_to_exit_pct': 100*(trade['exit_price']/following_open-1) if following_open else None,
            'note': 'Observed path decomposition, no assumption that a stop could execute at its threshold.'})
    return answer


def main(root=ROOT):
    records = {}
    protocol = read(root/'protocol.json')
    for window in ('oos', 'train'):
        generation = protocol['generation_directories'][window]
        info_path = root/generation/window/'ready.json'
        if not info_path.exists():
            info_path = root/'generations_v3'/window/'ready.json'
        info = read(info_path)
        folder = root/'cases'/CID/window/'100000000_baseline'
        if not (folder/'result.json').exists() and window == 'train':
            folder = root/'revisions/v3_parser_certificate_failure/training_cases/100000000_baseline'
        records[window] = analyse(window, folder, info)
        if window == 'oos':
            records[window]['two_worst_observed_price_paths'] = loss_paths(read(folder/'trades.json'), info)
    records['windows'] = protocol['windows']
    calendar_context(root, records)
    records['limitations'] = ['Descriptive baseline analysis, not ablation results.',
        'Unverified IS audit status remains explicit until the repaired native replay passes.',
        'Adaptive formerly-OOS search requires later independent validation.']
    destination = root/'mechanism_diagnostics.json'
    destination.write_text(json.dumps(records, indent=2, sort_keys=True, default=str)+'\n', encoding='utf-8')
    print(json.dumps({w: {k: records[w][k] for k in ('audit_status','return_pct','trades','trades_per_21_sessions',
        'payoff_ratio','explicit_costs','arithmetic_excluding_two_worst','funnel_per_entry_session','regime_sessions')}
        for w in ('train','oos')}, indent=2))


if __name__ == '__main__':
    main()
