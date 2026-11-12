"""Explain stale-release inactivity from dated development exits and decisions."""
from collections import Counter
from datetime import datetime,timedelta
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path.home()/'.codex/research/portfolio_targeted_validation_20261009'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(Path(p).resolve()),'sha256':sha}


def audit(root):
    study=read(root/'battery/study.json');code=Path(study['code']).resolve();sys.path.insert(0,str(code))
    import kis_core.trading_calendar as calendar_module
    if not Path(calendar_module.__file__).resolve().is_relative_to(code):raise ValueError('Calendar is not from the frozen generation')
    calendar=calendar_module.get_trading_calendar();output={}
    for suffix in ('baseline','cost','participation','combined','price_baseline','price_combined','nonfill_baseline','nonfill_combined'):
        case=root/'battery/qualified'/('k_marginal_'+suffix)
        days=[r['day'] for r in read(case/'daily.json')];index={d:i for i,d in enumerate(days)}
        decisions={}
        with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                if 'EXIT_FALLBACK_SUBMITTED' not in line:continue
                row=json.loads(line)
                if row['strategy_id']=='OLR' and row['decision_code']=='EXIT_FALLBACK_SUBMITTED':
                    key=row['metadata']['expired_order_id']
                    if key in decisions:raise ValueError('Ambiguous fallback decision')
                    decisions[key]=row
        counts=Counter();late=[]
        for trade in read(case/'trades.json'):
            if trade['strategy_id']!='OLR':continue
            entry,exit_at=map(datetime.fromisoformat,(trade['entry_fill_time'],trade['exit_fill_time']))
            held=index[exit_at.date().isoformat()]-index[entry.date().isoformat()]
            counts['olr_trades']+=1;counts['held_'+str(held)+'_sessions']+=1
            if held<2:continue
            earliest=calendar.next_trading_day(calendar.next_trading_day(entry.date()))
            opening,_=calendar.session_hours(earliest)
            threshold=datetime.combine(earliest,opening,entry.tzinfo)+timedelta(minutes=15)
            counts['late_horizon_trades']+=1
            counts['fully_exited_before_stale_eligibility']+=int(exit_at<threshold)
            counts['exit_reason_'+trade['exit_reason']]+=1
            decision=decisions.get(trade['cohort_metadata'].get('expired_order_id'))
            prior=bool(decision and entry<datetime.fromisoformat(decision['timestamp'])<threshold
                       and decision['symbol']==trade['symbol'])
            counts['native_fallback_submitted_before_eligibility']+=int(prior)
            if exit_at>=threshold:
                late.append({'symbol':trade['symbol'],'entry':entry.isoformat(),'exit':exit_at.isoformat(),
                    'stale_eligibility':threshold.isoformat(),'reason':trade['exit_reason'],
                    'fallback_already_submitted':prior,'fallback_at':decision['timestamp'] if decision else None})
        output[case.name]={'counts':dict(counts),'not_fully_exited_before_eligibility':late,
                          'evidence':{n:bound(case/n) for n in ('completion.json','trades.json','daily.json','native_decisions.jsonl.gz')}}
    payload={'scope':'Development realized holding-time and prior-exit diagnostic; not a forecast or selection input.',
        'cases':output,'study':bound(root/'battery/study.json'),
        'calendar':bound(code/'kis_core/data/krx_holidays.yaml'),'program':bound(Path(__file__))}
    (root/'inventory_boundary_diagnostic.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({n:r['counts'] for n,r in output.items()}),flush=True)


if __name__=='__main__':audit(ROOT)
