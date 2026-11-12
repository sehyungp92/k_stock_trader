"""0041: source-bound first-session timing/service diagnosis, zero account replay."""
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json
import math
import sqlite3
import subprocess
import sys
from olr_selective_half_continuation_v1 import read, write, sha, now
from olr_opportunity_continuity_v1 import ledger

PARENT=Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016/relative_qualification_v2')
REPO=Path(__file__).resolve().parents[1]


def window(out, name):
    dest=out/name;dest.mkdir(exist_ok=False)
    ready=read(PARENT/'native_engines'/name/'ready.json');files=Path(ready['files']);study=ready['study']
    sys.path.insert(0,str(files))
    from strategy_common.market import MarketBar
    from strategy_olr.execution import OLRExitPlan, _completed_bar_exit_reason
    from backtests.engine.olr_interval_service import IntervalParticipationScenario
    from backtests.engine.olr_minute_v5 import continuous_minute
    from kis_core.trading_calendar import get_trading_calendar
    summary=read(PARENT/'assessment.json')['summaries'][name+'_nominal_control'];case=Path(summary['case'])
    assert sha(case/'completion.json')==summary['completion_sha256']
    checks={n:sha(case/n)==h for n,h in read(case/'completion.json')['artifacts'].items()};assert all(checks.values())
    definition=files/study['minute_source_definition'];definition_json=read(definition)
    fine=files/definition_json['minute_database'];coarse=files/definition_json['canonical_database']
    assert fine != coarse and sha(fine)==ready['manifest']['files'][definition_json['minute_database']]
    assert sha(coarse)==ready['manifest']['files'][definition_json['canonical_database']]
    write(dest/'input_receipt.json',dict(at=now(),case=str(case),completion_sha256=sha(case/'completion.json'),
        completion_checks=checks,minute_database=str(fine),minute_sha256=sha(fine),
        coarse_database=str(coarse),coarse_sha256=sha(coarse),definition_sha256=sha(definition),
        native_rule_sha256=sha(files/'strategy_olr/execution.py'),
        calendar_sha256=sha(files/'kis_core/data/krx_sessions.yaml')))
    calendar=get_trading_calendar();trades=read(case/'trades.json');population=[];points=set()
    for t in trades:
        entry=datetime.fromisoformat(t['entry_fill_time']);end=datetime.fromisoformat(t['exit_fill_time'])
        nextday=calendar.next_trading_day(entry.date());opening,_=calendar.session_hours(nextday)
        start=datetime.combine(nextday,opening,entry.tzinfo)
        if end<start:continue
        population.append((t,entry,end,start))
        points.update(start+timedelta(minutes=i) for i in range(6))
    print(json.dumps(dict(at=now(),window=name,phase='ledger',population=len(population))),flush=True)
    book=ledger(case,sorted(points));assert math.isclose(book['terminal_cash'],1e8+summary['return_pct']*1e6,abs_tol=.01)
    db=sqlite3.connect(fine.as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
    service=IntervalParticipationScenario(name='nominal',participation=.01);rows=[]
    for t,entry,end,start in population:
        source=[MarketBar(symbol=t['symbol'],timestamp=datetime.fromisoformat(ts),open=o,high=h,low=l,close=c,volume=v,timeframe='1m',is_completed=True)
            for ts,o,h,l,c,v in db.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day>=? AND day<=? ORDER BY ts',
                (t['symbol'],entry.date().isoformat(),start.date().isoformat()))]
        opening_bars=[b for b in source if start<=b.timestamp<start+timedelta(minutes=5)]
        expected={start+timedelta(minutes=i) for i in range(5)}
        complete={b.timestamp for b in opening_bars}==expected
        risk=t['route_metadata']['risk_per_share'];plan=OLRExitPlan(**t['route_metadata']['trade_exit_plan'])
        assert plan.failed_followthrough_bars==1 and plan.failed_followthrough_close_r==-1 and plan.failed_followthrough_mfe_r==4
        lifecycle_orders=[o for o in book['orders'].values() if o['symbol']==t['symbol'] and
            datetime.fromisoformat(t['entry_decision_time'])<=datetime.fromisoformat(o['submitted_at'])<=end]
        sales=[o for o in lifecycle_orders if o['side']=='SELL']
        first_native=min((datetime.fromisoformat(o['metadata']['decision_available_at']) for o in sales),default=end)
        minute_states=[];trigger=None
        for b in opening_bars:
            at=b.timestamp+timedelta(minutes=1)
            high=max([t['entry_price']]+[x.high for x in source if entry<=x.timestamp and x.timestamp+timedelta(minutes=1)<=at])
            native=_completed_bar_exit_reason((b,),0,t['entry_price'],risk,high,plan,0)['reason']
            independent=(b.close-t['entry_price'])/risk<=-1 and (high-t['entry_price'])/risk<4
            assert (native=='failed_followthrough')==independent
            state=book['snapshots'][at.isoformat()];q=state['quantities'].get(t['symbol'],0)
            pending=[o for o in state['pending'] if o['symbol']==t['symbol']]
            row=dict(label=b.timestamp.isoformat(),available_at=at.isoformat(),close=b.close,high_water=high,
                close_R=(b.close-t['entry_price'])/risk,mfe_R=(high-t['entry_price'])/risk,
                continuous=continuous_minute(b),reason=native,held=q,pending=pending)
            minute_states.append(row)
            # The native minute rule consumes each observed completed minute.
            # A missing later interval cannot retrospectively veto an earlier
            # valid observation; it contributes no executable service volume.
            if trigger is None and continuous_minute(b) and independent and q and at<first_native and not pending:
                trigger=row
        sensitivity=None
        if trigger:
            available=datetime.fromisoformat(trigger['available_at']);horizon=min(first_native,start+timedelta(minutes=5))
            remaining=trigger['held'];fills=[]
            for b in opening_bars:
                if b.timestamp<available or b.timestamp+timedelta(minutes=1)>horizon:continue
                quote=service.quote(b,'interval',None)
                q=min(remaining,quote.budget)
                if not q:continue
                price=b.close*(1-.0005);fee=q*price*.0022
                fills.append(dict(label=b.timestamp.isoformat(),available_at=(b.timestamp+timedelta(minutes=1)).isoformat(),
                    quantity=q,price=price,fee=fee,budget=quote.budget,source_volume=b.volume))
                remaining-=q
            later=[e for o in sales for e in o['events'] if e['kind']=='fill' and datetime.fromisoformat(e['at'])>available]
            assert sum(e['fill_qty'] for e in later)==trigger['held']
            baseline=sum(e['fill_qty']*e['price']-e['fee'] for e in later)/trigger['held']
            sold=trigger['held']-remaining
            sensitivity=dict(first_native_available=first_native.isoformat(),lead_seconds=(first_native-available).total_seconds(),
                served_by_coarse_boundary=sold,unserved=remaining,fills=fills,
                later_actual_net_proceeds_per_share=baseline,
                local_proceeds_difference=sum(f['quantity']*f['price']-f['fee'] for f in fills)-sold*baseline,
                scope='Fixed-parent quantity/proceeds sensitivity, not a funded alternative, physical queue claim, or downstream bound.')
        snap=book['snapshots'][start.isoformat()]
        rows.append(dict(symbol=t['symbol'],entry_day=t['entry_decision_time'][:10],first_next_session=start.date().isoformat(),
            entry_price=t['entry_price'],risk=risk,quantity=t['qty'],final_net_pnl=t['net_pnl'],exit_reason=t['exit_reason'],
            complete_five_minute_prefix=complete,opening_quantity=snap['quantities'].get(t['symbol'],0),
            opening_gross=(opening_bars[0].open-t['entry_price'])*snap['quantities'].get(t['symbol'],0) if opening_bars and opening_bars[0].timestamp==start else None,
            first_native_available=first_native.isoformat(),minute_states=minute_states,first_eligible_trigger=trigger,
            local_sensitivity=sensitivity,actual_sales=sales))
    db.close()
    actionable=[r for r in rows if r['local_sensitivity'] and r['local_sensitivity']['served_by_coarse_boundary']]
    result=dict(at=now(),population=len(rows),full_prefix_count=sum(r['complete_five_minute_prefix'] for r in rows),
        eligible_trigger_count=sum(bool(r['first_eligible_trigger']) for r in rows),
        served_opportunities=len(actionable),served_dates=len({r['first_next_session'] for r in actionable}),
        local_proceeds_difference=sum(r['local_sensitivity']['local_proceeds_difference'] for r in actionable),
        helpful_count=sum(r['local_sensitivity']['local_proceeds_difference']>0 for r in actionable),
        adverse_count=sum(r['local_sensitivity']['local_proceeds_difference']<0 for r in actionable),rows=rows,
        scope='All incumbent first-next-session paths, original failure predicate only, first five minutes. No trading-code change, fitted cutoff or new financial account. Reused development history.')
    write(dest/'assessment.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows']}),flush=True)


def run(out):
    out.mkdir(exist_ok=False)
    design=REPO/'docs/optimisation/olr/experiments/olr-exp-0041.md'
    write(out/'protocol.json',dict(at=now(),script_sha256=sha(__file__),design_sha256=sha(design),
        parent_assessment_sha256=sha(PARENT/'assessment.json'),windows=['train','oos'],fresh_accounts=0,automatic_financial_successor=False))
    (out/'design_at_freeze.md').write_bytes(design.read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for name in ['train','oos']:
        subprocess.run([sys.executable,__file__,str(out),'--window',name],check=True)
    write(out/'assessment.json',dict(at=now(),windows={n:{k:v for k,v in read(out/n/'assessment.json').items() if k!='rows'} for n in ['train','oos']},
        economic_objective_achieved=False,next='Interpret every supporting/contrary timing and service case against exact prior failure-clock accounts before any financial admission.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--window',choices=['train','oos']);args=p.parse_args()
    window(args.out,args.window) if args.window else run(args.out)
