"""Read-only economic attribution for the next KALCB research assessment.

No policy selection, native replay, new forecasts, date filters or live changes.
Matched-trade decompositions describe observed paths, not counterfactual profit.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import hashlib
import json
import sqlite3
import statistics as st

R = Path('C:/Users/sehyu/.codex/research/kalcb_full_oos_recovery_20261009/v3')
Q = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v4')
P = Q.parent/'v3'
OUT = R/'next_steps_analysis_20261010_v1'
NAMES = ('expanding', 'expanding_atr5', 'decay60', 'decay60_atr5')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def bound(p):
    with Path(p).open('rb') as f:
        return {'path': str(p), 'sha256': hashlib.file_digest(f, 'sha256').hexdigest()}


def stats(trades):
    if not trades:
        return {'trades': 0, 'net_pnl': 0}
    pnl = [t['net_pnl'] for t in trades]
    wins = [p for p in pnl if p > 0]; losses = [p for p in pnl if p < 0]
    total_notional = sum(t['qty']*t['entry_price'] for t in trades)
    bps = [t['net_pnl']/(t['qty']*t['entry_price'])*10000 for t in trades]
    keys = ('first30_ret', 'first30_gap', 'first30_rel_volume')
    return {'trades': len(trades), 'net_pnl': sum(pnl), 'win_pct': 100*len(wins)/len(pnl),
            'mean_net_pnl': st.mean(pnl), 'median_net_pnl': st.median(pnl),
            'mean_winner': st.mean(wins) if wins else None, 'mean_loser': st.mean(losses) if losses else None,
            'unweighted_mean_trade_net_bps': st.mean(bps), 'median_trade_net_bps': st.median(bps),
            'entry_notional_weighted_net_bps': sum(pnl)/total_notional*10000,
            'commission_tax_per_entry_notional_bps': sum(t['commission'] for t in trades)/total_notional*10000,
            'total_entry_notional_krw': total_notional,
            'pnl_without_top5_trades': sum(pnl)-sum(sorted(pnl, reverse=True)[:5]),
            'top5_share_positive_pnl': sum(sorted(wins, reverse=True)[:5])/sum(wins) if wins else None,
            'mean_hold_minutes': st.mean((datetime.fromisoformat(t['exit_fill_time'])-datetime.fromisoformat(t['entry_fill_time'])).total_seconds()/60 for t in trades),
            'entry_routes': dict(Counter(t['cohort_metadata'].get('entry_route', 'unknown') for t in trades)),
            'mean_entry_context': {key: st.mean(t['cohort_metadata'].get(key, t['route_metadata'].get(key)) for t in trades
                                              if t['cohort_metadata'].get(key, t['route_metadata'].get(key)) is not None) for key in keys}}


def grouped(trades):
    result = defaultdict(list)
    for t in trades:
        result[(t['symbol'], t['entry_decision_time'])].append(t)
    return result


def contrast(a, b):
    ga, gb = grouped(a), grouped(b)
    shared = ga.keys() & gb.keys(); removed = ga.keys()-gb.keys(); added = gb.keys()-ga.keys()
    pnl = lambda g, keys: sum(t['net_pnl'] for key in keys for t in g[key])
    quantity_effect = outcome_effect = 0.
    for key in shared:
        qa, qb = [sum(t['qty'] for t in g[key]) for g in (ga, gb)]
        ra, rb = pnl(ga, [key])/qa, pnl(gb, [key])/qb
        quantity_effect += (qb-qa)*(ra+rb)/2
        outcome_effect += (rb-ra)*(qa+qb)/2
    common_change = pnl(gb, shared)-pnl(ga, shared)
    delta = pnl(gb, gb)-pnl(ga, ga)
    assert abs(common_change-quantity_effect-outcome_effect) < 1e-6
    assert abs(delta-(pnl(gb, added)-pnl(ga, removed)+common_change)) < 1e-6
    return {'shared_entry_events': len(shared), 'removed_entry_events': len(removed), 'added_entry_events': len(added),
            'removed_A_net_pnl': pnl(ga, removed), 'added_B_net_pnl': pnl(gb, added),
            'common_trade_net_pnl_change': common_change, 'common_quantity_component': quantity_effect,
            'common_per_share_outcome_component': outcome_effect, 'total_net_pnl_change': delta,
            'scope': 'Exact symmetric accounting decomposition of observed accounts. Sizing includes compounding/cash feedback. Per-share outcome includes entry/exit price, timing and costs. Not causal intervention or hypothetical profit.'}


def market_window(period, days):
    """Fixed-window diagnostic only, with explicit missing/nonstandard coverage."""
    import yaml
    plan = read(P/'profiles/expanding'/period/'plan.json')
    symbols = {str(x).zfill(6) for x in yaml.safe_load(Path(plan['universe']).read_text())['symbols']}
    c = sqlite3.connect('file:'+Path(plan['database']).as_posix()+'?mode=ro', uri=True)
    by_key = defaultdict(dict)
    for day, symbol, ts, price, volume in c.execute("SELECT day,symbol,ts,o,v FROM bars WHERE day>=? AND day<=? AND substr(ts,12,5) IN ('09:00','09:30','14:45')", (min(days), max(days))):
        if symbol in symbols and day in days:
            by_key[(day, symbol)][ts[11:16]] = (price, volume)
    c.close()
    values = defaultdict(list); missing = Counter(); covered = defaultdict(int)
    for day in days:
        for symbol in symbols:
            row = by_key[(day, symbol)]
            if not all(key in row for key in ('09:00', '09:30', '14:45')):
                missing[day] += 1; continue
            if not all(row[key][0] > 0 and row[key][1] > 0 for key in row):
                missing[day] += 1; continue
            a, b, z = [row[key][0] for key in ('09:00', '09:30', '14:45')]
            values['opening_30m_bps'].append((b/a-1)*10000)
            values['after_opening_to_1445_bps'].append((z/b-1)*10000)
            covered[day] += 1
    return {'session_count': len(days), 'universe': len(symbols), 'covered_symbol_sessions': sum(covered.values()),
            'missing_or_zero_volume_symbol_sessions': sum(missing.values()), 'missing_by_day': dict(missing),
            'returns': {k: {'mean_bps': st.mean(v), 'median_bps': st.median(v), 'positive_pct': 100*sum(x>0 for x in v)/len(v)} for k,v in values.items()},
            'scope': 'Descriptive fixed-clock price change for observed 103-universe symbol-sessions with positive volume at all three bars. Missing and nonstandard-session observations remain reported. Cost-free, no capacity/position constraints, no funded portfolio or broad-market total-return benchmark. Never an execution alternative.'}


def main():
    assert not OUT.exists(), 'Preserve prior evidence'
    OUT.mkdir()
    result = {'scope': 'Read-only secondary analysis of completed research; no new policy results or independent validation.',
              'source': bound(Q/'completion.json'), 'program': bound(Path(__file__)), 'trade_statistics': {}, 'contrasts': {}, 'market_windows': {}, 'gates': {}, 'artifacts': []}
    population = {}
    for name in NAMES:
        for period in ('is', 'oos'):
            for scenario in ('baseline', 'combined'):
                key = name+'/'+period+'/'+scenario;folder = Q/'profiles'/name/period/scenario
                t = read(folder/'trades.json'); native = read(folder/'result.json')
                assert len(t) == native['trades']
                assert abs(sum(x['net_pnl'] for x in t) - native['return_pct']*1e6) < .01
                population[key] = t; result['trade_statistics'][key] = stats(t)
                result['artifacts'].append(bound(folder/'trades.json'))
    pairs = [('expanding', 'decay60'), ('expanding', 'expanding_atr5'), ('decay60', 'decay60_atr5'), ('expanding_atr5', 'decay60_atr5')]
    for a, b in pairs:
        for period in ('is', 'oos'):
            for scenario in ('baseline', 'combined'):
                result['contrasts'][a+'->'+b+'/'+period+'/'+scenario] = contrast(population[a+'/'+period+'/'+scenario], population[b+'/'+period+'/'+scenario])
    for period in ('is', 'oos'):
        days = {r['day'] for r in read(Q/'profiles/expanding'/period/'baseline/daily.json')}
        result['market_windows'][period] = market_window(period, days)
    for name, base in [('expanding', Path('E:/CodexResearch/k_stock_trader/kalcb_aligned_walk_forward_20261010/v3/fit/forecasts/original_target_full_history')),
                       ('decay60', Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_mechanisms_20261010/v1/fit/forecasts/original_decay60'))]:
        for kind in ('opening', 'timed'):
            with (base/(kind+'.jsonl')).open() as f:
                forecasts = [json.loads(line) for line in f]
            for period, first, last in [('is', '2025-08-07', '2026-03-31'), ('oos', '2026-04-01', '2026-07-10')]:
                rows = [r for r in forecasts if first<=r['trade_date']<=last]
                feasible = [r for r in rows if r['capacity_ok'] and r['failure_probability']<.001]
                positive = [r for r in feasible if min(r['baseline_value_bps'], r['combined_value_bps']) > 0]
                result['gates'][name+'/'+period+'/'+kind] = {'quantity_actions': len(rows), 'issuer_days':len({r['opportunity_id'].rsplit('/q',1)[0] for r in rows}),
                    'capacity_ok':sum(r['capacity_ok'] for r in rows), 'risk_and_capacity_ok':len(feasible), 'raw_positive_both_scenarios':len(positive),
                    'positive_issuer_days':len({r['opportunity_id'].rsplit('/q',1)[0] for r in positive}),
                    'scope':'Quantity alternatives are dependent. Raw-score positivity precedes calibration and native rank/admission; not final trades.'}
    for ref in result['artifacts']:
        assert bound(ref['path'])['sha256'] == ref['sha256']
    with (OUT/'diagnosis.json').open('x',encoding='utf-8') as f:
        json.dump(result, f, indent=2);f.write('\n')
    print(json.dumps({'saved':str(OUT/'diagnosis.json'),'cases':len(population),'contrasts':len(result['contrasts']),'market_windows':result['market_windows']}))


if __name__ == '__main__':
    main()
