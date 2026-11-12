"""Bounded existing-account risk materiality study, no fit or counterfactual."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE=BASE/'v29_exit_latency_audit'
HELPER=REPO/'strategy_kalcb/core/risk_contract_v1.py'
LEDGER=REPO/'backtests/strategies/kalcb/exit_latency_audit_v1.py'
TEST=REPO/'tests/test_kalcb_risk_contract_v1.py'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def bound(p):
    with Path(p).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':h}
def verify(bs):
    for b in bs:
        if bound(b['path'])['sha256']!=b['sha256']:raise ValueError('Changed source '+b['path'])
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def register(out,tests):
    if not read(tests)['passed']:raise ValueError('Tests missing')
    receipt=read(SOURCE/'completion.json');verify([receipt['assessment']])
    source=read(SOURCE/'assessment.json');verify(source['artifacts'])
    protocol=read(SOURCE/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    paths=[Path(b['path']) for b in protocol['bindings']]
    paths += [Path(b['path']) for b in source['artifacts']]
    paths += [SOURCE/'completion.json',SOURCE/'assessment.json',SOURCE/'protocol.json',Path(tests)]
    out.mkdir(exist_ok=False)
    write(out/'protocol.json',{'registered_at_utc':now(),'kind':'existing_native_risk_contract_materiality',
       'cases':protocol['cases'],'databases':protocol['databases'],
       'population':'All opening submitted intents and unique filled opening lifetimes in12unchanged audited accounts; prior100 all4scenarios/prior125 base+combined, IS/OOS. No outcome/date/symbol selection.',
       'formulas':'Reference risk and stop from original actual SubmitEntry metadata. Every completed raw close while holding uses native OPEN fills through raw bar timestamp, fees paid once, no next-open fill before decision. First close<=original reference stop strictly BEFORE original complete-parent MARKET EXIT is the sole frozen diagnostic boundary; same-clock parent exits are not independent support. Retain pending/partial cases and eventual winners. This is observational exposure, not a stop policy, native counterfactual or recoverable PnL.',
       'outputs':'Every lifetime max marked loss in filled referenceR, first breach before parent exit, signed subsequent observed movement including future costs; all no-fills; full months/issuers; worst native MTM episode contribution. Source/account identity reused and raw-close MTM independently reconciled again.',
       'decision':'No model/native dispatch. A broad reference-stop rule is unsupported if after-boundary recoveries and IS tails dominate; cannot choose a new threshold from outcomes. Any later intervention requires a separate finite causal mechanism and exact native states/labels.',
       'limits':['Marks are not executable exits; post-boundary movement is not counterfactual PnL.','No actual protective stop proposed or enabled.','Scenarios,priors,issuers/sessions dependent; April-July reused development.','No observed signal claims true financial risk guarantee.'],
       'programs':[bound(p) for p in [Path(__file__).resolve(),HELPER,LEDGER,TEST]],
       'bindings':[bound(p) for p in dict.fromkeys(paths)],'estimate':'12existing accounts, approximately5-10minutes; zero model/native cases.','study_complete':False})

def summarize(rows):
    b=[r for r in rows if r['before_parent_exit_breach']]
    return {'positions':len(rows),'net_krw':sum(r['net_pnl_krw'] for r in rows),
      'reference_stop_breaches':sum(r['reference_stop_breach'] for r in rows),
      'independent_pre_parent_breaches':len(b),'breach_sessions':len({r['entry_day'] for r in b}),
      'breach_eventual_winners':sum(r['net_pnl_krw']>0 for r in b),
      'breach_eventual_winning_pnl_krw':sum(max(0,r['net_pnl_krw']) for r in b),
      'after_breach_observed_net_movement_krw':sum(r['observed_after_breach_net_movement_krw'] for r in b),
      'positive_after_breach_movement_krw':sum(max(0,r['observed_after_breach_net_movement_krw']) for r in b),
      'negative_after_breach_movement_krw':sum(min(0,r['observed_after_breach_net_movement_krw']) for r in b),
      'pending_entry_at_first_breach':sum(r['before_parent_exit_breach']['pending_planned_qty']>0 for r in b),
      'max_marked_net_loss_reference_r':max([r['max_marked_net_loss_reference_r'] for r in rows],default=0),
      'scope':'Observed paths, no recovered-PnL or candidate-return claim.'}

def run(out):
    p=read(out/'protocol.json');verify(p['programs']+p['bindings'])
    import psutil
    me=psutil.Process();write(out/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline(),'at_utc':now()})
    h=module('registered_risk_contract',HELPER);ledger=module('registered_risk_ledger',LEDGER)
    assessments={};outputs=[];summary={}
    for case in p['cases']:
        key=case['key'];folder=Path(case['folder']);ap=case['assessment']
        if ap not in assessments:assessments[ap]=read(ap)
        account=assessments[ap]['accounts'][key];positions=account['positions']
        latency=read(SOURCE/'accounts'/Path(key).with_suffix('.json'))
        exits={r['position_id']:r.get('first_full_exit',{}).get('available_at') for r in latency['positions']}
        fills=read(folder/'fills.json');mapped=ledger.lifetimes(fills,positions)
        need={(r['symbol'],r['entry_day']) for r in positions};bars=defaultdict(list);marks={}
        for db in p['databases']:
            with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as c:
                c.execute('PRAGMA temp_store=MEMORY')
                low=min(d for s,d in need);high=max(d for s,d in need)
                for s,ts,d,price in c.execute('SELECT symbol,ts,day,c FROM bars WHERE day>=? AND day<=? ORDER BY ts,symbol',(low,high)):
                    if (s,d) not in need:continue
                    ts=h.timestamp(ts).isoformat()
                    if (ts,s) in marks:raise ValueError('Duplicate raw source')
                    marks[ts,s]=float(price);bars[s,d].append((ts,float(price)))
        count=ledger.reconcile_curve(fills,read(folder/'equity.json'),marks)
        obs=read(folder/'observer_evidence.json');exposures={e['order_id']:e for e in obs['entry_exposures']}
        rows=[];filled_orders=set()
        for position in positions:
            if position['route']!='first30_open_anchor':continue
            oid=position['entry_order_ids'][0];ref=h.EntryRiskReference.from_exposure(exposures[oid])
            filled_orders.add(oid)
            row=h.analyze_lifetime(ref,position,mapped[position['position_id']],bars[position['symbol'],position['entry_day']],exits[position['position_id']])
            rows.append(row)
        all_open=[e for e in obs['entry_exposures'] if e['action']['metadata'].get('entry_route')=='first30_open_anchor']
        nofills=[{'order_id':e['order_id'],'symbol':e['action']['symbol'],'available_at':e['decision_available_at'],**h.EntryRiskReference.from_exposure(e).evidence()} for e in all_open if e['order_id'] not in filled_orders]
        worst=max(account['mtm']['episodes'],key=lambda x:x['drawdown_pct'])
        ids={r['position_id'] for r in rows if r['before_parent_exit_breach']}
        worst={**worst,'contribution_from_ever_breached_positions_krw':sum(c['mtm_change_krw'] for c in worst['position_contributions'] if c['position_id'] in ids),
               'qualification':'Ever-breached contribution is descriptive; breach can occur after part of episode loss. Not avoidable DD.'}
        result={'account':key,'rows':rows,'no_fill_opening_decisions':nofills,'all_raw_mtm_marks_reconciled':count,'summary':summarize(rows),
            'months':{d:summarize([r for r in rows if r['entry_day'][:7]==d]) for d in sorted({r['entry_day'][:7] for r in rows})},
            'issuers':{s:summarize([r for r in rows if r['symbol']==s]) for s in sorted({r['symbol'] for r in rows})},'worst_mtm_episode':worst}
        path=out/'accounts'/Path(key).with_suffix('.json');write(path,result);outputs.append(bound(path));summary[key]=result['summary']
        print(json.dumps({'account':key,'positions':len(rows),'raw_mtm_marks':count,'complete':True}),flush=True)
        del obs,exposures,marks,bars,fills,mapped
    verify(p['programs']+p['bindings'])
    write(out/'assessment.json',{'accounts':summary,'outputs':outputs,'complete':True,'study_complete':False,'native_cases':0})
    write(out/'completion.json',{'complete':True,'at_utc':now(),'study_complete':False,'assessment':bound(out/'assessment.json'),'outputs':outputs,'native_cases':0})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','run']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tests',type=Path)
    a=parser.parse_args()
    try:register(a.root,a.tests) if a.mode=='register' else run(a.root)
    except Exception as e:
        if a.root.exists():write(a.root/'failure.json',{'error':repr(e),'at_utc':now()})
        raise
