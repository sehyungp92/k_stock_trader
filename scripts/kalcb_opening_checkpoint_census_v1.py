"""Registered data-only minute45 census over authenticated unchanged accounts."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,importlib.util,json,sqlite3,sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE=BASE/'v29_exit_latency_audit'
H=REPO/'backtests/strategies/kalcb/opening_checkpoint_census_v1.py'
L=REPO/'backtests/strategies/kalcb/exit_latency_audit_v1.py'
TEST=REPO/'tests/backtests/strategies/test_kalcb_opening_checkpoint_census_v1.py'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=True),encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def bound(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return {'path':str(p),'sha256':h.hexdigest()}
def verify(rows):
    for b in rows:
        if bound(b['path'])['sha256']!=b['sha256']:raise ValueError('Changed '+b['path'])
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[name]=module;return module
def origins():
    result=[]
    for name,m in tuple(sys.modules.items()):
        f=getattr(m,'__file__',None)
        if f and Path(f).is_relative_to(REPO):
            if Path(f).resolve() not in {H.resolve(),L.resolve(),Path(__file__).resolve()}:raise ValueError('Unexpected application import '+str(f))
            result.append({'module':name,**bound(f)})
    return result

def register(root,tests):
    if (root/'protocol.json').exists():raise ValueError('Already registered')
    receipt=read(tests);assert receipt['passed'];verify(receipt['programs'])
    old=read(SOURCE/'protocol.json');assessment=read(SOURCE/'assessment.json');assert read(SOURCE/'review_v1/completion.json')['complete']
    verify(old['programs']+old['bindings']+assessment['artifacts'])
    protocol={'version':'opening-checkpoint-information-census-v1','registered_at_utc':now(),'cases':old['cases'],'databases':old['databases'],
      'programs':[bound(x) for x in [__file__,H,L,TEST]],'bindings':old['bindings']+assessment['artifacts']+[bound(SOURCE/n) for n in ['protocol.json','assessment.json','review_v1/completion.json']],
      'tests':bound(tests),'selection':'All actual first30_open_anchor lifetimes in twelve unchanged parent accounts; no issuer/date/outcome/forecast filter. Later/nonopening entries and closed-before-checkpoint states retained as exclusions.',
      'clock':'Exactly session minute45 (09:45KST), the existing native checkpoint. Completed09:00..09:40 bars only. This is15minutes after usual09:30 opening fill and before common30/50minute failed-followthrough/quick-exit decisions. No clock grid.',
      'features':'Exactly two natural zero-sign partitions: actual gross marked PnL <0 and completed close < contemporary typical-price sessionVWAP. All four cells reported IS/OOS; no optimized cutpoints or selection of a winner.',
      'source':'Require all nine consecutive positive-volume valid OHLC bars. Native typical-price VWAP formula independently reconstructed. Quotes after decision cannot enter features.',
      'accounting':'Actual executed inventory/cash/fees strictly before checkpoint; same-clock next-bar fills excluded. Remaining actual net = terminal net minus current marked net; fees retained. This is not a fresh action label, hypothetical exit value or feasible saved PnL.',
      'parent_overlap':'A full-market exit decision at/before checkpoint is retained as parent overlap and excluded from still-holding information groups. Parent full-exit identities come from independently reconciled V29. No direct new-exit support claimed.',
      'dependence':'Unique physical lifetimes; counts byissuer-day/session. Scenarios/priors/quantity descendants are dependent. No independence or pooled significance.',
      'next_gate':'Only a credible earlier information distinction could justify fresh paired native hold-versus-liquidate action labels, with chronological support and winner-sacrifice assessment. No fits/policies/native experiments automatically follow.',
      'runtime_estimate':'Under10minutes for12 read-only accounts after registration; zero native replays.','native_cases':0,'study_complete':False}
    root.mkdir(parents=True,exist_ok=True);write(root/'protocol.json',protocol);print(json.dumps({'registered':True,'accounts':12,'native_cases':0}),flush=True)

def run(root):
    if (root/'supervisor_start.json').exists():raise ValueError('Already started')
    import psutil
    me=psutil.Process();write(root/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline(),'at_utc':now()})
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']);h=load('checkpoint_features',H);l=load('lifecycle_accounting',L);before=origins()
    source_assessments={};positions={};need=set()
    for case in p['cases']:
        f=case['assessment']
        if f not in source_assessments:source_assessments[f]=read(f)
        pp=source_assessments[f]['accounts'][case['assessment_key']]['positions'];positions[case['key']]=pp
        need.update((x['symbol'],x['entry_fill_time'][:10]) for x in pp if x['route']=='first30_open_anchor')
    prefixes={};symbols=sorted({s for s,d in need})
    for db in p['databases']:
        with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as conn:
            sql='SELECT symbol,ts,day,o,h,l,c,v FROM bars WHERE day>=? AND day<=? AND symbol IN ('+','.join('?' for _ in symbols)+') ORDER BY symbol,ts'
            for symbol,ts,day,o,hi,lo,c,v in conn.execute(sql,['2025-08-07','2026-07-10',*symbols]):
                if (symbol,day) not in need or datetime.fromisoformat(ts).strftime('%H:%M')>='09:45':continue
                prefixes.setdefault((symbol,day),[]).append({'timestamp':l.stamp(ts),'open':o,'high':hi,'low':lo,'close':c,'volume':v})
    write(root/'startup_health.json',{'passed':True,'programs':p['programs'],'input_hashes_verified':len(p['bindings']),'raw_prefixes':len(prefixes),'origins_before':before})
    summaries={};artifacts=[];progress=[]
    for case in p['cases']:
        pp=positions[case['key']];mapped=l.lifetimes(read(Path(case['folder'])/'fills.json'),pp)
        life={x['position_id']:x for x in read(SOURCE/'accounts'/Path(case['key']).with_suffix('.json'))['positions']}
        rows=[];excluded=[]
        for position in pp:
            pid=position['position_id'];day=position['entry_fill_time'][:10];cutoff=day+'T09:45:00+09:00'
            if position['route']!='first30_open_anchor':excluded.append({'position_id':pid,'reason':'nonopening_route'});continue
            if l.stamp(position['entry_fill_time'])>=cutoff:excluded.append({'position_id':pid,'reason':'entry_not_before_checkpoint'});continue
            qty,cash,fees=h.inventory_before(mapped[pid],cutoff)
            if qty<=0:excluded.append({'position_id':pid,'reason':'already_flat'});continue
            try:features=h.features(day,prefixes.get((position['symbol'],day),[]),qty,cash)
            except ValueError as error:excluded.append({'position_id':pid,'reason':'invalid_prefix','detail':str(error)});continue
            marked=features['gross_marked_pnl_krw']-fees;remaining=position['net_pnl_krw']-marked
            overlap=life[pid].get('first_full_exit',{}).get('available_at','9999')<=cutoff
            rows.append({'position_id':pid,'symbol':position['symbol'],'day':day,'features':features,'held_qty':qty,'entry_equity_krw':position['entry_equity_krw'],
                'marked_net_krw':marked,'actual_remaining_net_krw':remaining,'terminal_net_krw':position['net_pnl_krw'],
                'parent_exit_already_decided':overlap,'entry_fills_at_or_after_checkpoint':sum(f['side']=='BUY' and l.stamp(f['timestamp'])>=cutoff for f in mapped[pid])})
        eligible=[x for x in rows if not x['parent_exit_already_decided']]
        states={state:h.summarize([x for x in eligible if x['features']['state']==state]) for state in ['negative/below_vwap','negative/at_or_above_vwap','nonnegative/below_vwap','nonnegative/at_or_above_vwap']}
        summary={'total_positions':len(pp),'observations':len(rows),'still_holding_without_prior_exit':len(eligible),'parent_exit_overlaps':len(rows)-len(eligible),'exclusions':dict(Counter(x['reason'] for x in excluded)),'states':states,'all_eligible':h.summarize(eligible)}
        result={'summary':summary,'observations':rows,'excluded':excluded,'case':case,'interpretation':'Information census on actual unchanged paths. No new action labels, model, exit return, promotion or independent validation.'}
        path=root/'accounts'/Path(case['key']).with_suffix('.json');path.parent.mkdir(parents=True,exist_ok=True);write(path,result);artifacts.append(bound(path));summaries[case['key']]=summary
        progress.append({'case':case['key'],'complete':True,'observations':len(rows)});write(root/'progress.json',progress);print(json.dumps(progress[-1]),flush=True)
    verify(p['programs']+p['bindings']);after=origins();write(root/'assessment.json',{'complete':True,'accounts':summaries,'artifacts':artifacts,'origins_before':before,'origins_after':after,'protocol':bound(root/'protocol.json'),'native_cases':0,'study_complete':False})
    write(root/'completion.json',{'complete':True,'at_utc':now(),'accounts':12,'native_cases':0,'assessment':bound(root/'assessment.json'),'study_complete':False});print(json.dumps({'complete':True}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','run']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tests',type=Path);args=parser.parse_args()
    if args.mode=='register':register(args.root,args.tests)
    else:
        try:run(args.root)
        except Exception as error:
            write(args.root/'failure.json',{'at_utc':now(),'error':repr(error),'complete':False,'study_complete':False});raise
