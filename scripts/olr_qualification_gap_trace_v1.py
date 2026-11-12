"""Complete the registered OOS reach and loss-timing postmortem from accepted journals."""
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import argparse,gzip,json,math
from pathlib import Path
from olr_qualification_disposition_v1 import read,write,sha


def run(root):
    out=root/'qualification_disposition_v1'/'gap_and_coverage_v1';out.mkdir(exist_ok=False)
    a=read(root/'assessment.json');summary=a['summaries']['oos_nominal_control'];case=Path(summary['case'])
    ready=read(root/'native_engines/oos/ready.json');feature_path=Path(ready['files'])/ready['study']['relative_context_features'];features=read(feature_path)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(out/'protocol.json',dict(at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
        parent_review_sha256=sha(root/'qualification_disposition_v1/assessment.json'),
        inputs={n:sha(case/n) for n in ['minute_events.jsonl.gz','native_decisions.jsonl.gz','trades.json','completion.json']},
        features_sha256=sha(feature_path),scope='Complete fixed morning census where sparse book observations are absent; exact monetary-drawdown contribution and existing opening/first-five-minute loss timing. Zero new replay.'))
    trades=read(case/'trades.json');clock_points=sorted(datetime.fromisoformat(day+'T'+clock+':00+09:00') for day in features['evaluation_days'] for clock in ['09:00','09:05','09:30'])
    cash=high=1e8;qty=defaultdict(int);marks={};cash_by=defaultdict(float);peak_wealth={};peak_at=None;worst=None;index=0;snaps={}
    def wealth():return {s:cash_by[s]+qty[s]*marks.get(s,0) for s in cash_by.keys()|qty.keys()}
    def snapshot():return dict(cash=cash,quantity={s:q for s,q in qty.items() if q},marks={s:marks[s] for s,q in qty.items() if q},wealth=wealth())
    with gzip.open(case/'minute_events.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line);kind=row['event']
            if kind in ['opening_marks','interval_marks','closing_marks']:
                at=datetime.fromisoformat(row['timestamp'])+timedelta(minutes=kind!='opening_marks')
            elif kind=='fill':at=datetime.fromisoformat(row['fill']['timestamp'])
            else:continue
            while index<len(clock_points) and clock_points[index]<at:
                snaps[clock_points[index].isoformat()]=snapshot();index+=1
            if kind=='fill':
                z=row['fill'];sign=1 if z['side']=='BUY' else -1;flow=-sign*z['qty']*z['price']-z['metadata']['fee']
                assert math.isclose(cash,row['cash_before'],abs_tol=1e-5)
                cash+=flow;cash_by[z['symbol']]+=flow;qty[z['symbol']]+=sign*z['qty']
                assert qty[z['symbol']]>=0 and math.isclose(cash,row['cash_after'],abs_tol=1e-5)
            else:marks.update(row['prices'])
            w=wealth();nav=1e8+sum(w.values());assert math.isclose(nav,cash+sum(q*marks[s] for s,q in qty.items()),abs_tol=1e-5)
            if nav>=high:high=nav;peak_at=at.isoformat();peak_wealth=w.copy()
            loss=high-nav
            if worst is None or loss>worst['loss']:
                worst=dict(loss=loss,peak_at=peak_at,trough_at=at.isoformat(),peak_equity=high,trough_equity=nav,
                    symbol_changes={s:w.get(s,0)-peak_wealth.get(s,0) for s in w.keys()|peak_wealth.keys()})
    while index<len(clock_points):snaps[clock_points[index].isoformat()]=snapshot();index+=1
    assert not any(qty.values()) and math.isclose(cash,1e8+summary['return_pct']*1e6,abs_tol=.01)
    assert math.isclose(worst['loss'],summary['maximum_currency_decline'],abs_tol=.01)
    assert math.isclose(sum(worst['symbol_changes'].values()),-worst['loss'],abs_tol=.01)
    exits=[]
    with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            d=json.loads(line)
            if d.get('decision_code')=='EXIT_SUBMITTED':
                exits.append(dict(symbol=d['symbol'],available_at=(datetime.fromisoformat(d['timestamp'])+timedelta(minutes=5)).isoformat(),
                    reason=d['reason'],quantity=d['metadata'].get('qty')))
    holding_rows=[];opening_rows=[]
    for t in trades:
        dates=[day for day in features['evaluation_days'] if t['entry_fill_time'][:10]<day<=t['exit_fill_time'][:10]]
        for day in dates:
            at=day+'T09:30:00+09:00';snap=snaps[at];q=snap['quantity'].get(t['symbol'],0)
            if q:
                candidates=[u for u in trades if u['symbol']==t['symbol'] and u['entry_fill_time']<at<u['exit_fill_time']]
                assert len(candidates)==1 and candidates[0]==t
                f=features['rows'].get(at,{}).get(t['symbol']);prior=[e for e in exits if e['symbol']==t['symbol'] and t['entry_fill_time']<e['available_at']<=at]
                if prior:reason='native_exit_already_submitted'
                elif f is None:reason='unsupported_prefix'
                elif f['own_return']>=0:reason='own_nondecline'
                elif f['market_median']<0:reason='shared_decline'
                else:reason='relative_trigger'
                holding_rows.append(dict(at=at,symbol=t['symbol'],entry_day=t['entry_decision_time'][:10],quantity=q,feature=f,
                    reason=reason,native_exit_submissions=prior,final_net_pnl=t['net_pnl']))
            if day==dates[0]:
                rows={}
                # Wealth includes prior same-symbol realised PNL, removed below using each lifecycle's entry cost.
                for clock in ['09:00','09:05','09:30']:
                    st=day+'T'+clock+':00+09:00';z=snaps[st];current=z['quantity'].get(t['symbol'],0)
                    mark=z['marks'].get(t['symbol'])
                    rows[clock]=dict(quantity=current,mark=mark,
                        gross_entry_to_mark=(mark-t['entry_price'])*current if current else None,
                        mark_minus_entry_R=(mark-t['entry_price'])/t['route_metadata']['risk_per_share'] if current else None,
                        available_at=st)
                opening_rows.append(dict(symbol=t['symbol'],entry_day=t['entry_decision_time'][:10],first_next_session=day,
                    entry_rank=t['route_metadata']['candidate_rank'],entry_score=t['route_metadata']['candidate_score'],
                    entry_price=t['entry_price'],qty=t['qty'],original_R=t['route_metadata']['risk_per_share'],
                    final_pnl=t['net_pnl'],exit_at=t['exit_fill_time'],exit_reason=t['exit_reason'],snapshots=rows,
                    native_exits=[e for e in exits if e['symbol']==t['symbol'] and t['entry_fill_time']<e['available_at']<=t['exit_fill_time']]))
    write(out/'assessment.json',dict(at=datetime.now(timezone.utc).isoformat(),exact_currency_episode=worst,
        journal_morning_rows=holding_rows,first_next_session_states=opening_rows,
        limits=['Opening marks are observed valuation, not guaranteed executable opening sales.',
                'Gross entry-to-mark excludes fee and any partially sold quantity; do not equate it with full eventual loss.',
                'Native five-minute submitted decisions become available five minutes after their label; physical fills may be much later.',
                'No candidate rule or new financial run is produced.']))
    print(json.dumps(dict(output=str(out),held_boundaries=len(holding_rows),episode=worst,
        largest_losses=sorted(opening_rows,key=lambda x:x['final_pnl'])[:7])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();run(args.root)
