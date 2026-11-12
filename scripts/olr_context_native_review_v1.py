"""Review 0034 against saved N0 quantities; no alternate fills or replay."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

BASE = Path('E:/CodexResearch/k_stock_trader')
SOURCE = BASE/'olr_continuing_research_20261016/context_population_v1'
CASE = BASE/'olr_structural_execution_20261013/generation_v12/cases/train/native_control'
OUT = BASE/'olr_continuing_research_20261016/context_native_review_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def run():
    OUT.mkdir(exist_ok=False)
    inputs=[Path(__file__),SOURCE/'native_coverage.json',SOURCE/'holding_rows.json',
            CASE/'trades.json',CASE/'marked_book_observations.json',CASE/'completion.json']
    bindings={str(p):sha(p) for p in inputs}
    completion=read(CASE/'completion.json')
    for name in ['trades.json','marked_book_observations.json']:
        assert bindings[str(CASE/name)]==completion['artifacts'][name]
    save(OUT/'protocol.json',dict(at=datetime.now(timezone.utc).isoformat(),inputs=bindings,
        estimand='All 09:30 native held states; earliest eligible state per acquired lifecycle for nonduplicated wealth. Separate pending exits and buys. No timing or threshold search.',
        continuation='Remaining executed sale proceeds net of 2bp commission and 20bp tax minus current marked notional. Descriptive 27bp immediate sale proxy only.',
        next='Inspect supporting and contrary cases, reachable quantities and later funded opportunities before native diagnostic admission.'))
    trades=read(CASE/'trades.json');books=read(CASE/'marked_book_observations.json')
    book_by_time={b['observed_at']:b for b in books}
    rows=[];unsupported=[]
    for n in read(SOURCE/'native_coverage.json'):
        if n['clock']!='09:30':continue
        at=n['decision_at'];s=n['symbol']
        matches=[(i,t) for i,t in enumerate(trades) if t['symbol']==s and t['entry_fill_time']<=at<t['exit_fill_time']]
        assert len(matches)==1,(s,at,len(matches))
        i,t=matches[0];book=book_by_time[at]
        pending_buy=any(p['symbol']==s and p['side']=='BUY' for p in book['pending'])
        all_legs=t['route_metadata'].get('exit_legs')
        if all_legs is None:
            # Single lifecycle exit exports may retain only the aggregate fill.
            # It is usable here only while the entire acquired quantity remains.
            if n['quantity']!=t['qty']:
                unsupported.append(dict(symbol=s,at=at,reason='aggregate_exit_after_partial_quantity'));continue
            all_legs=[dict(qty=t['qty'],exit_price=t['exit_price'],exit_time=t['exit_fill_time'])]
        assert sum(x['qty'] for x in all_legs)==t['qty']
        full_proceeds=sum(x['qty']*x['exit_price']*(1-.0022) for x in all_legs)
        assert math.isclose(full_proceeds-t['qty']*t['entry_price']*(1+.0002),t['net_pnl'],abs_tol=.02)
        legs=[x for x in all_legs if x['exit_time']>at]
        remaining=sum(x['qty'] for x in legs)
        if remaining!=n['quantity'] or pending_buy:
            unsupported.append(dict(symbol=s,at=at,quantity=n['quantity'],remaining=remaining,pending_buy=pending_buy));continue
        p=next(x for x in book['positions'] if x['symbol']==s)
        proceeds=sum(x['qty']*x['exit_price']*(1-.0022) for x in legs)
        continuation=proceeds-n['notional']
        future=[b for b in books if b['observed_at']>at]
        next_entry=next((b for b in future if any(q['side']=='BUY' for q in b['pending'])),None)
        rows.append(dict(**n,lifecycle=i,entry_at=t['entry_fill_time'],exit_at=t['exit_fill_time'],
            remaining_exit_quantity=remaining,original_entry_price=t['entry_price'],
            entry_r=t['route_metadata']['risk_per_share'],mark=p['mark'],mfe_r=p['mfe_r'],
            continuation_pnl=continuation,hold_minus_marked_exit_proxy=continuation+n['notional']*.0027,
            hold_minus_marked_exit_proxy_bps=(continuation/n['notional']+.0027)*10000,
            next_observed_buy_at=next_entry['observed_at'] if next_entry else None,
            next_observed_buy_cash=next_entry['cash'] if next_entry else None,
            next_observed_buy_symbols=sorted({q['symbol'] for q in next_entry['pending'] if q['side']=='BUY'}) if next_entry else []))
    save(OUT/'rows.json',rows)
    summaries={}
    for scope in ['sector','market']:
        groups=defaultdict(list)
        for r in rows:
            state=r['context'].get(scope+'_state','unsupported')
            groups[(state,'preempted' if r['pending_exit'] else 'no_pending_exit')].append(r)
        for (state,preempted),values in groups.items():
            first={}
            for r in sorted(values,key=lambda x:x['decision_at']):first.setdefault(r['lifecycle'],r)
            values=list(first.values());daily=defaultdict(list)
            for r in values:daily[r['day']].append(r['hold_minus_marked_exit_proxy_bps'])
            wealth=sum(r['hold_minus_marked_exit_proxy'] for r in values)
            dates={d:mean(v) for d,v in daily.items()}
            sums={d:sum(r['hold_minus_marked_exit_proxy'] for r in values if r['day']==d) for d in dates}
            biggest=max(sums,key=lambda d:abs(sums[d]))
            summaries['|'.join([scope,state,preempted])]=dict(lifecycles=len(values),dates=len(dates),
                continuation_pnl=sum(r['continuation_pnl'] for r in values),notional=sum(r['notional'] for r in values),
                hold_minus_marked_exit_proxy=wealth,date_equal_proxy_bps=mean(dates.values()),
                positive_cases=sum(r['hold_minus_marked_exit_proxy']>0 for r in values),
                sacrificed_positive_continuation=sum(max(r['hold_minus_marked_exit_proxy'],0) for r in values),
                avoided_negative_continuation=-sum(min(r['hold_minus_marked_exit_proxy'],0) for r in values),
                largest_absolute_wealth_date=biggest,wealth_without_largest_absolute_date=wealth-sums[biggest],
                cases=[dict(symbol=r['symbol'],at=r['decision_at'],lifecycle=r['lifecycle'],value=r['hold_minus_marked_exit_proxy']) for r in values])
    assert all(sha(Path(p))==h for p,h in bindings.items())
    save(OUT/'assessment.json',dict(at=datetime.now(timezone.utc).isoformat(),summaries=summaries,
        unsupported=unsupported,rows=len(rows),new_replays=0,economic_objective_achieved=False,
        limits=['Actual holdings on a fixed parent policy, not causal account returns. Hypothetical instant sale remains a cost proxy, not an executable fill.',
                'Earliest lifecycle state deduplicates within each partition only; partitions are alternative descriptions and cannot be added.',
                'Original entry average is descriptive; no action is selected using later labels or future average costs.']))
    print(json.dumps(dict(path=str(OUT),rows=len(rows),unsupported=len(unsupported))))


if __name__=='__main__':
    run()
