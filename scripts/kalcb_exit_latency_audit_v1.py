"""Finite DATA-ONLY loss-stage audit. Run with -I -B; no application imports."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
HELPER=REPO/'backtests/strategies/kalcb/exit_latency_audit_v1.py'
TEST=REPO/'tests/backtests/strategies/test_kalcb_exit_latency_audit_v1.py'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True),encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def bound(path):
    h=hashlib.sha256();path=Path(path)
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return {'path':str(path),'sha256':h.hexdigest()}
def verify(bindings):
    for b in bindings:
        if bound(b['path'])['sha256']!=b['sha256']:raise ValueError('Changed '+b['path'])
def origins():
    rows=[]
    for name,module in tuple(sys.modules.items()):
        origin=getattr(module,'__file__',None)
        if origin and Path(origin).is_relative_to(REPO):
            p=Path(origin).resolve()
            if p not in {HELPER.resolve(),Path(__file__).resolve()}:raise ValueError('Unregistered repository import '+str(p))
            rows.append({'module':name,**bound(p)})
    return rows


def register(root,tests):
    if (root/'protocol.json').exists():raise ValueError('Already registered')
    receipt=read(tests)
    if not receipt['passed']:raise ValueError('Tests failed')
    verify(receipt['programs'])
    cases=[];sources={}
    for version in [25,28]:
        native=BASE/('v25_pending_funding_screen' if version==25 else 'v28_pending_funding_qualification')
        a=read(native/'assessment_v1/assessment.json')
        verification=native/'full_reports_v1/verification_v1/full_report_verification.json'
        if not read(verification)['passed']:raise ValueError('Missing full reports')
        for p in [native/'assessment_v1/assessment.json',verification,native/'protocol.json',native/'stage_completion.json']:
            sources[str(p)]=bound(p)
        progress=read(native/'progress.json')
        for row in progress:
            name,period,scenario=(row[k] for k in ('name','period','scenario'))
            keep=(version==25 and name in ['control','control_prior125']) or (version==28 and name=='control' and scenario in ['participation','cost'])
            if not keep:continue
            key=f'{name}/{period}/{scenario}';folder=Path(row.get('folder') or native/'cases'/name/period/scenario)
            if a['accounts'][key]['metrics']['audit_status']!='passed':raise ValueError('Unaudited account')
            for n in ['completion.json','result.json','execution_audit.json','independent_account_verification.json','fills.json','equity.json','orders.jsonl.gz','observer_evidence.json','independent_exposure_mtm.json']:
                p=folder/n;sources[str(p)]=bound(p)
            profile=folder.parent/'input_profile.json';sources[str(profile)]=bound(profile)
            cases.append({'key':key,'folder':str(folder),'assessment':str(native/'assessment_v1/assessment.json'),'assessment_key':key})
    if len(cases)!=12 or len({x['key'] for x in cases})!=12:raise ValueError('Expected twelve unique unchanged accounts')
    databases=read(BASE/'v28_pending_funding_qualification/protocol.json')['databases']
    for p in databases:sources[p]=bound(p)
    root.mkdir(parents=True,exist_ok=True)
    protocol={'version':'exit-latency-accounting-v1','registered_at_utc':now(),'cases':cases,'databases':databases,
      'programs':[bound(p) for p in [__file__,HELPER,TEST]],'bindings':list(sources.values()),'tests':bound(tests),
      'question':'Does realized adverse PnL accumulate before the parent requests full liquidation or during execution afterward?',
      'selection':'All physical unique positions in twelve unchanged parent accounts, no outcome/date/issuer/forecast selection. Prior100 four scenarios; prior125 baseline/combined; both IS/OOS.',
      'fixed_formulas':{'first_exit':'First submitted MARKET SELL with order_role EXIT in each physical lifetime; TP/protective intents retained, no relabeling. All sell submissions bound, including nonfilled.',
        'causal_mark':'Raw five-minute bar OPEN timestamp = decision_available_at minus five minutes; require existing completed close. Exclude next-bar fills at equal decision timestamp from pre-decision inventory.',
        'before':'Signed physical cashflows gross before decision plus held inventory at last completed raw close.',
        'after':'Realized terminal gross PnL minus before; contains actual subsequent price changes/execution friction, not achievable saved PnL.',
        'identity':'before + after - ALL actual fees = terminal net. All physical fills/lifetimes and every native raw-close equity mark independently reconcile.',
        'groups':'All, net losers, net winners; full MARKET EXIT reasons; account months and issuers. No policy thresholds or learned bins.'},
      'limitations':['No action labels, predictive fit, forecasts, policies, native portfolio cases or new full diagnostics.','Retrospective accounting and same-parent path description; no causal exit advantage or recoverable PnL.','Scenarios, repeated issuer-days, partials and priors are dependent; do not pool as independent observations.','April-July reused development, not independent validation.'],
      'next_gate':'A native change requires a distinct economically supported causal mechanism and matching fresh full-cost action labels; no hold/threshold/model grid follows automatically.',
      'runtime_estimate':'15-25 minutes including implementation/tests/review; bounded12existing accounts, zero native replays.','native_cases':0,'study_complete':False}
    write(root/'protocol.json',protocol)
    print(json.dumps({'registered':True,'accounts':12,'native_cases':0,'bindings':len(sources)}),flush=True)


def aggregate(rows):
    result={}
    for name,group in [('all',rows),('net_losers',[p for p in rows if p['net_pnl_krw']<0]),('net_winners',[p for p in rows if p['net_pnl_krw']>0])]:
        full=[p for p in group if 'first_full_exit' in p]
        out={'positions':len(group),'full_market_exit_positions':len(full),'without_full_market_exit_positions':len(group)-len(full),
             'net_pnl_krw':sum(p['net_pnl_krw'] for p in group),'fees_krw':sum(p['fees_krw'] for p in group),
             'full_exit_before_gross_krw':sum(p['before_first_full_exit_gross_krw'] for p in full),
             'full_exit_after_gross_krw':sum(p['after_first_full_exit_gross_krw'] for p in full),
             'full_exit_fees_krw':sum(p['fees_krw'] for p in full),
             'full_exit_net_krw':sum(p['net_pnl_krw'] for p in full),
             'known_executed_slippage_krw':sum(p['executed_slippage_krw'] for p in group),
             'exit_reason_counts':dict(Counter(p['first_full_exit']['reason'] for p in full)),
             'full_exit_positive_after_count':sum(p['after_first_full_exit_gross_krw']>0 for p in full),
             'flat_same_clock_count':sum(p['minutes_decision_to_flat']==0 for p in full),
             'max_minutes_decision_to_flat':max([p['minutes_decision_to_flat'] for p in full],default=None),
             'median_minutes_decision_to_flat':sorted(p['minutes_decision_to_flat'] for p in full)[len(full)//2] if full else None,
             'entries_after_exit_decision':sum(p['entry_fills_at_or_after_exit_decision'] for p in full)}
        reasons={}
        for reason in out['exit_reason_counts']:
            rr=[p for p in full if p['first_full_exit']['reason']==reason]
            reasons[reason]={'positions':len(rr),'before_gross_krw':sum(p['before_first_full_exit_gross_krw'] for p in rr),'after_gross_krw':sum(p['after_first_full_exit_gross_krw'] for p in rr),'fees_krw':sum(p['fees_krw'] for p in rr),'net_krw':sum(p['net_pnl_krw'] for p in rr)}
        out['reasons']=reasons;result[name]=out
    return result


def run(root):
    if (root/'supervisor_start.json').exists():raise ValueError('Already started; preserve and use a successor after failure')
    import psutil
    me=psutil.Process()
    write(root/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline(),'at_utc':now()})
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    spec=importlib.util.spec_from_file_location('registered_exit_latency',HELPER)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);sys.modules[spec.name]=h
    before=origins(); assessments={};need=set();positions={}
    for case in p['cases']:
        ap=case['assessment']
        if ap not in assessments:assessments[ap]=read(ap)
        rows=assessments[ap]['accounts'][case['assessment_key']]['positions'];positions[case['key']]=rows
        for row in rows:
            if row['entry_fill_time'][:10]!=row['exit_fill_time'][:10]:raise ValueError('Unexpected overnight position requires explicit source range')
            need.add((row['symbol'],row['entry_fill_time'][:10]))
    symbols=sorted({s for s,d in need});marks={}
    for db in p['databases']:
        with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as conn:
            sql='SELECT symbol,ts,day,c FROM bars WHERE day>=? AND day<=? AND symbol IN ('+','.join('?' for _ in symbols)+')'
            for symbol,ts,day,price in conn.execute(sql,['2025-08-07','2026-07-10',*symbols]):
                if (symbol,day) not in need:continue
                key=(h.stamp(ts),symbol)
                if key in marks:raise ValueError('Duplicate raw close')
                marks[key]=float(price)
        print(json.dumps({'source':db,'consumed_closes':len(marks)}),flush=True)
    write(root/'startup_health.json',{'passed':True,'at_utc':now(),'programs':p['programs'],'inputs_verified':len(p['bindings']),'raw_closes':len(marks),'origins_before':before})
    summaries={};artifacts=[];progress=[]
    for case in p['cases']:
        folder=Path(case['folder']);rows=positions[case['key']];submits=[]
        with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                event=json.loads(line)
                if event['event']=='submit':submits.append(event['order'])
        result=h.account(rows,read(folder/'fills.json'),submits,read(folder/'equity.json'),marks)
        result['groups']=aggregate(result['positions'])
        result['months']={d:aggregate([x for x in result['positions'] if x['entry_fill_time'][:7]==d]) for d in sorted({x['entry_fill_time'][:7] for x in result['positions']})}
        result['issuers']={s:aggregate([x for x in result['positions'] if x['symbol']==s]) for s in sorted({x['symbol'] for x in result['positions']})}
        output=root/'accounts'/Path(case['key']).with_suffix('.json');output.parent.mkdir(parents=True,exist_ok=True);write(output,result)
        artifacts.append(bound(output));summaries[case['key']]={k:result[k] for k in ['groups','all_raw_close_equity_marks_reconciled','all_lifetimes_reconciled','unique_sell_submissions_bound']}
        progress.append({'key':case['key'],'complete':True,'positions':len(rows),'equity_marks':result['all_raw_close_equity_marks_reconciled']});write(root/'progress.json',progress)
        print(json.dumps(progress[-1]),flush=True)
    verify(p['programs']+p['bindings']);after=origins()
    result={'complete':True,'accounts':summaries,'artifacts':artifacts,'origins_before':before,'origins_after':after,'protocol':bound(root/'protocol.json'),'limitations':p['limitations'],'native_cases':0,'study_complete':False}
    write(root/'assessment.json',result);write(root/'completion.json',{'complete':True,'at_utc':now(),'accounts':12,'native_cases':0,'assessment':bound(root/'assessment.json'),'study_complete':False})
    write(root/'status.json',{'stage':'complete_pending_economic_review','at_utc':now(),'complete':True,'accounts':12})
    print(json.dumps({'complete':True,'accounts':12}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','run']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.mode=='register':register(args.root,args.tests)
    else:
        try:run(args.root)
        except Exception as error:
            write(args.root/'failure.json',{'at_utc':now(),'error':repr(error),'complete':False,'study_complete':False})
            raise
