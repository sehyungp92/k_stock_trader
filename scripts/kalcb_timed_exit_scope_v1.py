"""C8-D2 descriptive post-exit prices; never executable policy profit."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_lifecycle_readiness_v1 import ROOT, CODE, SCENARIOS, read, write, bound, verify


def lifetimes(fills):
    active = {}; done = []
    for f in fills:
        symbol = f['symbol']
        if f['side'] == 'BUY':
            p = active.setdefault(symbol, {'entry_order_id': f['order_id'], 'symbol': symbol,
                 'day': f['timestamp'][:10], 'qty': 0, 'fills': [], 'cash': 0.})
            if p['entry_order_id'] != f['order_id']:
                raise ValueError('Ambiguous merged lifetime')
            p['qty'] += f['qty']; p['cash'] -= f['qty'] * f['price'] + f['metadata']['fee']
        else:
            p = active[symbol]; p['qty'] -= f['qty']
            p['cash'] += f['qty'] * f['price'] - f['metadata']['fee']
            if p['qty'] < 0:
                raise ValueError('Invalid inventory')
        p['fills'].append(f)
        if p['qty'] == 0:
            p['exit_reason'] = f['reason']; p['exit_at'] = f['timestamp']
            done.append(p); del active[symbol]
    if active:
        raise ValueError('Nonflat source account')
    return done


def summarise(rows):
    result = {'lifetimes': len(rows), 'days': len({r['day'] for r in rows}),
              'original_net_krw': sum(r['original_net_krw'] for r in rows),
              'original_winners': sum(r['original_net_krw'] > 0 for r in rows), 'horizons': {}}
    for h in ('15', '30', '60', 'flatten'):
        eligible = [r for r in rows if r.get('marks', {}).get(h) is not None]
        v = [r['marks'][h]['price_return_bps'] for r in eligible]
        result['horizons'][h] = {'supported': len(v), 'positive': sum(x > 0 for x in v),
            'mean_price_bps': sum(v)/len(v) if v else None,
            'min_price_bps': min(v) if v else None, 'max_price_bps': max(v) if v else None}
    return result


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    receipt = read(ROOT/'completion_assessment_v1/receipt.json')
    known = {str(Path(b['path']).resolve()): b for b in receipt['inputs'] + receipt['outputs']}
    inputs = []; cases = []
    for period in ('is', 'oos'):
        for s in SCENARIOS:
            case = ROOT/'study_v2/cases/control'/period/s
            for name in ('fills.json', 'labels.jsonl'):
                b = known[str((case/name).resolve())]; verify(b); inputs.append(b)
            cases.append((period, s, case))
    registration = {'at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
        'program': bound(__file__), 'helper': bound(CODE/'scripts/kalcb_lifecycle_readiness_v1.py'),
        'design': bound(CODE/'docs/optimisation/kalcb/experiments/kalcb-exp-0032-continuation-scope.md'),
        'scope': 'C8-D2 fixed post-exit horizons; source marks are not alternative fills; no native run or fit.'}
    write(out/'registration.json', registration)
    reports = {}; source_bindings = {}
    for period, s, case in cases:
        sources = {}
        for line in (case/'labels.jsonl').open(encoding='utf-8'):
            row = json.loads(line); day = row['observation']['day']; b = row['source']
            if day in sources and sources[day] != b:
                raise ValueError('Inconsistent same-day source binding')
            sources[day] = b
        selected = [p for p in lifetimes(read(case/'fills.json'))
            if p['exit_reason'] in ('quick_exit', 'failed_followthrough')
            and p['fills'][0]['metadata'].get('entry_route') == 'first30_open_anchor']
        grouped = defaultdict(list)
        for p in selected: grouped[p['day']].append(p)
        observations = []
        for day, members in sorted(grouped.items()):
            b = sources.get(day)
            stream = None
            if b:
                verify(b); source_bindings[b['path']] = b
                with gzip.open(b['path'], 'rt', encoding='utf-8') as f:stream = json.load(f)
            for p in members:
                last = p['fills'][-1]; at = datetime.fromisoformat(p['exit_at']); base = last['metadata']['base_price']
                row = {'day': day, 'symbol': p['symbol'], 'entry_order_id': p['entry_order_id'],
                       'exit_reason': p['exit_reason'], 'exit_at': p['exit_at'], 'original_net_krw': p['cash'],
                       'reference_base_quote': base, 'marks': {}, 'source_available': bool(b)}
                by_available = {datetime.fromisoformat(bar['timestamp'])+timedelta(minutes=5): bar
                     for batch in (stream or []) for bar in batch if bar['symbol'] == p['symbol']}
                for h in ('15', '30', '60', 'flatten'):
                    t = at+timedelta(minutes=int(h)) if h != 'flatten' else at.replace(hour=14, minute=45)
                    bar = by_available.get(t) if t > at else None
                    row['marks'][h] = None if bar is None else {'available_at': t.isoformat(),
                        'price_return_bps': 10000*(bar['close']/base-1)}
                future = [bar for t,bar in by_available.items() if at < t <= at.replace(hour=14,minute=45)]
                row['future_high_bps'] = 10000*(max(bar['high'] for bar in future)/base-1) if future else None
                row['future_low_bps'] = 10000*(min(bar['low'] for bar in future)/base-1) if future else None
                observations.append(row)
        report = {'summary': summarise(observations), 'by_reason': {reason: summarise([r for r in observations if r['exit_reason']==reason])
             for reason in ('quick_exit','failed_followthrough')},
             'months': {m: summarise([r for r in observations if r['day'].startswith(m)]) for m in sorted({r['day'][:7] for r in observations})},
             'observations': observations}
        reports[period+'/'+s] = report; write(out/(period+'_'+s+'.json'), report)
        print(json.dumps({'case': period+'/'+s, **report['summary']}), flush=True)
    for b in inputs+[registration['program'],registration['helper'],registration['design']]: verify(b)
    write(out/'source_bindings.json',list(source_bindings.values()))
    write(out/'assessment.json',{'reports':reports,'native_policy_test':False,'economic_objective_achieved':False,
        'limitations':'Post-exit prices only: no fill, capital, counterfactual management or policy-return claim; all reused history.'})
    write(out/'completion.json',{'completed':True,'at_utc':datetime.now(timezone.utc).isoformat(),
         'assessment':bound(out/'assessment.json'),'registration':bound(out/'registration.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.output)
    except Exception as exc:
        if a.output.exists() and not (a.output/'failure.json').exists():
            write(a.output/'failure.json',{'error':repr(exc),'at_utc':datetime.now(timezone.utc).isoformat()})
        raise
