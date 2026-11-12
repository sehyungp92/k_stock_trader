"""Source-timed context/population diagnostic; no policy selection or replay."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean, median
import traceback
import yaml

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader')
PARENT = BASE/'olr_structural_execution_20261013/generation_v12'
LANES = BASE/'olr_joint_decision_20261013'
OUT = BASE/'olr_continuing_research_20261016/context_population_v1'
CLOCKS = ('09:05', '09:30', '10:00')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(p, data, exclusive=True):
    with Path(p).open('x' if exclusive else 'w', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def status(root, phase, **kwargs):
    value = dict(at=now(), phase=phase, economic_objective_achieved=False, **kwargs)
    save(root/'status.json', value, False)
    print(json.dumps(value), flush=True)


def block(day):
    return 'aug_oct' if day <= '2025-10-31' else 'nov_dec' if day <= '2025-12-31' else 'jan_mar'


def partition(own, peers):
    return ('own_down' if own < 0 else 'own_nonnegative') + '/' + ('peers_down' if peers < 0 else 'peers_nonnegative')


def summarise(rows, value):
    daily = defaultdict(list)
    for row in rows:
        daily[row['day']].append(float(row[value]))
    values = {d: mean(v) for d,v in sorted(daily.items())}
    raw = [float(row[value]) for row in rows]
    if not raw:
        return dict(rows=0, dates=0)
    biggest = max(values, key=lambda d: abs(values[d]))
    return dict(rows=len(rows), issuer_days=len({(r['day'],r['symbol']) for r in rows}), dates=len(values),
        date_equal_mean=mean(values.values()), date_equal_median=median(values.values()),
        positive_dates=sum(v>0 for v in values.values()), negative_dates=sum(v<0 for v in values.values()),
        raw_mean=mean(raw), raw_median=median(raw), positive_rows=sum(v>0 for v in raw),
        raw_positive_sum=sum(max(v,0) for v in raw), raw_negative_sum=sum(min(v,0) for v in raw),
        largest_absolute_date=biggest, largest_absolute_date_value=values[biggest],
        mean_without_largest_absolute_date=mean(v for d,v in values.items() if d!=biggest) if len(values)>1 else None,
        daily=values,
        blocks={b:dict(rows=sum(block(r['day'])==b for r in rows),
                       dates=sum(block(d)==b for d in values),
                       date_equal_mean=mean(v for d,v in values.items() if block(d)==b) if any(block(d)==b for d in values) else None)
                for b in ('aug_oct','nov_dec','jan_mar')})


def fixtures():
    # No future interval may enter a decision prefix; issuer count cannot
    # silently give one date extra weight.
    labels = [datetime.fromisoformat('2025-09-01T09:00:00+09:00'),datetime.fromisoformat('2025-09-01T09:05:00+09:00')]
    at = datetime.fromisoformat('2025-09-01T09:05:00+09:00')
    assert [x for x in labels if x+timedelta(minutes=5)<=at] == labels[:1]
    rows=[dict(day='2025-09-01',symbol='a',v=10),dict(day='2025-09-01',symbol='b',v=0),dict(day='2025-09-02',symbol='c',v=-1)]
    assert summarise(rows,'v')['date_equal_mean']==2
    assert partition(-.01,.01)=='own_down/peers_nonnegative'
    assert partition(-.01,-.01)=='own_down/peers_down'


def run(root):
    fixtures()
    root.mkdir(parents=True,exist_ok=False)
    status(root,'authenticating_inputs')
    study_path=PARENT/'runtime/train/native_control/study.json'
    study=read(study_path); files=Path(study['code']); manifest=read(Path(study['manifest']))['files']
    case=PARENT/'cases/train/native_control'; completion=read(case/'completion.json')
    census_path=BASE/'olr_structural_execution_20261013/generation_v11/existing_opening_book_evidence_v1.json'
    census=read(census_path)
    funnel_path=REPO/'data/backtests/analysis/olr_structural_20261012/opportunity_v1/incumbent_funnel/funnel.json'
    deps={}
    def bind(p, expected=None):
        p=Path(p); digest=sha(p)
        if expected is not None:assert digest==expected,str(p)
        deps[str(p)]=digest
        return p
    for p in (Path(__file__), REPO/'docs/optimisation/olr/experiments/olr-exp-0034.md',
              study_path,Path(study['manifest']),case/'completion.json',census_path,
              LANES/'protocol.json',LANES/'verified_response/summary.json',funnel_path):bind(p)
    diagnostics=read(LANES/'response_diagnostics.json')
    bind(LANES/'verified_response/records.json',diagnostics['records_sha256'])
    bind(LANES/'verified_response/paths.jsonl.gz',read(BASE/'olr_economic_policy_20261013/labels_receipt.json')['source'])
    rels=[study['database'],study['olr_snapshots'],'config/olr/sector_map.yaml','config/universe_103.yaml',
          'data/backtests/integrity/corrective-selection-20261005-v20/source/daily_references.json']
    for rel in rels:bind(files/rel,manifest[rel])
    for name in ('marked_book_observations.json','trades.json','minute_nav_evidence.json'):
        bind(case/name,completion['artifacts'][name])
    save(root/'protocol.json',dict(at=now(),experiment='olr-exp-0034',inputs=deps,
        primary_clock='09:30',support_clocks=CLOCKS,partition='natural zero own session return x median peer session return',
        native_replays=0,conditional_returns_previously_grouped_here=False,
        existing_data='Adaptively reused development, not untouched validation',
        verification='Bound source/response bytes and lane receipt/result pairs; inherited source audits not rerun'))
    records=read(LANES/'verified_response/records.json')
    lookup={(r['day'],r['symbol'],r['scenario'],r['budget']):r for r in records}
    protocol=read(LANES/'protocol.json')
    assert len(protocol['cohort'])==231 and len(lookup)==len(records)==1386
    candidates={(s['trade_date'],c['symbol']):c for s in read(files/study['olr_snapshots']) for c in s['candidates']}
    universe=[str(s).zfill(6) for s in yaml.safe_load((files/'config/universe_103.yaml').read_text(encoding='utf8'))['symbols']]
    sectors={str(k).zfill(6):v.upper().replace(' ','_') for k,v in yaml.safe_load((files/'config/olr/sector_map.yaml').read_text(encoding='utf8'))['sector_map'].items()}
    assert len(universe)==103 and all(s in sectors for s in universe)
    opens={s['day']:datetime.fromisoformat(s['open']['available_at']) for s in census['sessions']}
    targets=defaultdict(set)
    for day in opens:
        for clock in CLOCKS:targets[day].add(day+'T'+clock+':00+09:00')
    for r in protocol['cohort']:
        at=datetime.fromisoformat(r['native_signal_time']).isoformat()
        targets[r['day']].add(at)
    daily=defaultdict(list)
    for r in read(files/rels[-1]):daily[r['symbol']].append(r)
    for values in daily.values():values.sort(key=lambda r:r['date'])
    dates={s:[r['date'] for r in values] for s,values in daily.items()}
    con=sqlite3.connect((files/study['database']).as_uri()+'?mode=ro',uri=True)
    con.execute('PRAGMA query_only=ON')
    own={};coverage=Counter()
    lo=min(targets);hi=(datetime.fromisoformat(max(targets))+timedelta(days=1)).date().isoformat()
    status(root,'building_completed_prefix_context',symbols=len(universe),days=len(targets))
    for n,symbol in enumerate(universe):
        by_day=defaultdict(list)
        for t,o,h,l,c,v in con.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts',(symbol,lo,hi)):
            by_day[t[:10]].append((datetime.fromisoformat(t),o,h,l,c,v))
        for day,times in targets.items():
            bars=by_day.get(day,[]); available=[b[0]+timedelta(minutes=5) for b in bars]
            prior_i=bisect_left(dates.get(symbol,[]),day)-1
            prior=daily[symbol][prior_i] if prior_i>=0 else None
            for text in times:
                at=datetime.fromisoformat(text);cut=bisect_right(available,at);prefix=bars[:cut]
                if (not prefix or prefix[-1][0]+timedelta(minutes=5)!=at or prefix[0][0]!=opens.get(day)
                        or any(b[0]-a[0]!=timedelta(minutes=5) for a,b in zip(prefix,prefix[1:]))):
                    coverage['missing_or_incomplete_prefix']+=1;continue
                volume=sum(b[5] for b in prefix)
                if volume<=0 or prefix[0][1]<=0 or prefix[-1][4]<=0:
                    coverage['nonpositive_price_volume']+=1;continue
                close=prefix[-1][4];opening=prefix[0][1]
                own[symbol,text]=dict(close=close,opening=opening,own_return=close/opening-1,
                    gap=opening/prior['close']-1 if prior and prior['close']>0 else None,
                    prior_daily_date=prior['date'] if prior else None,
                    vwap=sum((b[2]+b[3]+b[4])/3*b[5] for b in prefix)/volume,
                    observed_volume=volume,bar_count=len(prefix),source_last_label=prefix[-1][0].isoformat())
                coverage['supported_own_prefix']+=1
        if (n+1)%20==0:status(root,'building_completed_prefix_context',processed_symbols=n+1,total_symbols=len(universe))
    con.close()
    context_cache={}
    def context(symbol,text):
        key=(symbol,text)
        if key in context_cache:return context_cache[key]
        x=own.get(key)
        if x is None:return dict(available=False,reason='missing_complete_current_prefix')
        peers=[own[s,text]['own_return'] for s in universe if s!=symbol and (s,text) in own and sectors[s]==sectors[symbol]]
        market=[own[s,text]['own_return'] for s in universe if s!=symbol and (s,text) in own]
        total=sum(s!=symbol and sectors[s]==sectors[symbol] for s in universe)
        result=dict(available=True,**x,sector=sectors[symbol],sector_peers=len(peers),sector_total_peers=total,
                    market_peers=len(market),sector_peer_median=median(peers) if len(peers)>=2 else None,
                    market_peer_median=median(market) if len(market)>=2 else None,
                    sector_positive_fraction=mean(v>=0 for v in peers) if len(peers)>=2 else None)
        for name,values in [('sector',peers),('market',market)]:
            if len(values)>=2:
                result[name+'_state']=partition(x['own_return'],median(values))
                result[name+'_relative_return']=x['own_return']-median(values)
        context_cache[key]=result
        return result
    entry=[];hold=[];missing=Counter();receipt_checks=0;hash_checks=[]
    status(root,'joining_authenticated_lane_outcomes',lanes=1386)
    with gzip.open(LANES/'verified_response/paths.jsonl.gz','rt',encoding='utf8') as stream:
        for index,line in enumerate(stream,1):
            r=json.loads(line);key=(r['day'],r['symbol'],r['scenario'],r['budget']);saved=lookup[key]
            assert saved['receipt_sha256']==r['receipt_sha256']
            lane=Path(r['lane']);receipt=read(lane/'receipt.json')
            assert sha(lane/'receipt.json')==r['receipt_sha256']
            assert sha(lane/'result.json')==receipt['result_sha256']
            receipt_checks+=1
            hash_checks.append(dict(lane=str(lane),receipt=r['receipt_sha256'],result=receipt['result_sha256']))
            if not r['execution_valid'] or r['status'] not in ('matured','unfilled'):
                missing[r['status']]+=1;continue
            points=sorted(r['path'],key=lambda p:(p[6],p[1]))
            assert points[-1][5]==0 and math.isclose(points[-1][3],r['policy_cash_pnl'],abs_tol=.01)
            at=datetime.fromisoformat(r['native_signal_time'])
            common={k:r[k] for k in ('day','symbol','scenario','budget','incumbent_disposition','rank','actual_acquired','filled_quantity')}
            cx=context(r['symbol'],at.isoformat())
            entry.append(dict(**common,decision_at=at.isoformat(),context=cx,
                pnl=r['policy_cash_pnl'],request_bps=r['policy_cash_pnl']/r['budget']*10000,
                meaningful=r['actual_acquired']>=1e6,status=r['status']))
            if r['filled_quantity']==0:continue
            future_days=sorted({p[6][:10] for p in points if p[6][:10]>r['day']})
            if not future_days:missing['no_next_session']+=1;continue
            day=future_days[0]
            buy_times=[];prev=0
            for p in points:
                if p[2]=='fill' and p[5]>prev:buy_times.append(p[6])
                prev=p[5]
            for clock in CLOCKS:
                text=day+'T'+clock+':00+09:00'
                # Match the original label boundary: completed closes at this
                # availability precede the new interval's opening/fill actions.
                visible=[p for p in points if p[6]<text or p[6]==text and p[2]=='closing_marks']
                if not visible:missing['no_visible_position']+=1;continue
                p=visible[-1]
                if p[5]<=0:missing['already_flat']+=1;continue
                if not buy_times or max(buy_times)>=text:
                    missing['acquisition_not_complete_at_boundary']+=1;continue
                cx=context(r['symbol'],text)
                if not cx['available']:
                    missing['holding_context_unavailable']+=1;continue
                notional=p[3]-p[4]
                assert notional>0
                continuation=r['policy_cash_pnl']-p[3]
                cost_bps=27 if r['scenario']=='baseline' else 37
                h=dict(**common,decision_at=text,clock=clock,context=cx,quantity=p[5],notional=notional,
                    mark=notional/p[5],coarse_close=cx['close'],entry_price=r['actual_acquired']/r['filled_quantity'],
                    continuation_pnl=continuation,continuation_bps=continuation/notional*10000,
                    hold_minus_marked_exit_proxy_bps=continuation/notional*10000+cost_bps,
                    original_r=candidates.get((r['day'],r['symbol']),{}).get('daily_atr'),
                    latest_acquisition_at=max(buy_times),path_mark_available_at=p[6],
                    path_mark_diff_bps=(notional/p[5]/cx['close']-1)*10000)
                hold.append(h)
            if index%200==0:status(root,'joining_authenticated_lane_outcomes',processed_lanes=index,total_lanes=1386)
    assert receipt_checks==1386
    save(root/'entry_rows.json',entry);save(root/'holding_rows.json',hold);save(root/'lane_receipt_checks.json',hash_checks)
    books=read(case/'marked_book_observations.json');native=[]
    for book in books:
        text=book['observed_at'];clock=text[11:16]
        if clock not in CLOCKS:continue
        for p in book['positions']:
            native.append(dict(day=text[:10],symbol=p['symbol'],clock=clock,decision_at=text,
                quantity=p['quantity'],notional=p['current_notional'],pending_exit=p['pending_exit'],
                state=p['state'],context=context(p['symbol'],text)))
    save(root/'native_coverage.json',native)
    summaries={}
    for kind,rows,value in [('entry',entry,'request_bps'),('hold',hold,'hold_minus_marked_exit_proxy_bps')]:
        groups=defaultdict(list)
        for row in rows:
            base=(row['scenario'],str(row['budget']),row.get('clock','entry'))
            groups[base+('all',)].append(row)
            groups[base+('disposition',row['incumbent_disposition'])].append(row)
            for scope in ('sector','market'):
                state=row['context'].get(scope+'_state','unsupported')
                groups[base+(scope,state)].append(row)
                groups[base+(scope,state,row['incumbent_disposition'])].append(row)
        summaries[kind]={'|'.join(key):summarise(values,value) for key,values in sorted(groups.items())}
    coverage_native=Counter((r['clock'],r['context'].get('sector_state','unsupported'),'preempted' if r['pending_exit'] else 'no_pending_exit') for r in native)
    assessment=dict(at=now(),experiment='olr-exp-0034',entry_rows=len(entry),holding_rows=len(hold),
        unique_opportunities=len({(r['day'],r['symbol']) for r in entry}),receipt_checks=receipt_checks,
        original_funnel_counts=read(funnel_path)['counts'],source_coverage=dict(coverage),missing=dict(missing),
        native_coverage={'|'.join(k):v for k,v in sorted(coverage_native.items())},summaries=summaries,
        mark_mismatch_rows=sum(abs(r['path_mark_diff_bps'])>1e-6 for r in hold),
        economic_objective_achieved=False,new_financial_replays=0,
        limits=['Local native continuation descriptions, not causal alternate fills or funded returns.',
                'Original isolated lane service differs from N0; native coverage is a separate support census.',
                'Sector peers with unavailable current prefixes are omitted with coverage retained, not treated as neutral.',
                'Confirmed but untraded opportunities do not cover no-confirmation or outside-slot populations.',
                'All history development; scenarios/budgets/clocks not independent samples.'])
    save(root/'assessment.json',assessment)
    status(root,'awaiting_context_and_population_review',entry_rows=len(entry),holding_rows=len(hold),
        next='Review primary 09:30 states, opposing cases and untraded populations; then execute named source-observer confound inspection in preparation plan. This is not assignment closure.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=OUT);parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        fixtures();print('Timing, date weighting and state-partition fixtures passed.')
    else:
        try:run(args.root)
        except BaseException as error:
            if args.root.exists():save(args.root/'failure.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc()))
            raise
