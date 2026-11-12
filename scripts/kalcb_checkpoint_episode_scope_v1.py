"""Read-only scope attribution: initial holding, later re-entries, whole episode."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
PARENT=BASE/'v35_checkpoint_hold_bridge'
SOURCE=BASE/'v33_checkpoint_contrast_labels'
HELPER=REPO/'backtests/strategies/kalcb/checkpoint_episode_scope_v1.py'
FILLHELPER=REPO/'backtests/strategies/kalcb/checkpoint_hold_bridge_v1.py'
TEST=REPO/'tests/backtests/strategies/test_kalcb_checkpoint_episode_scope_v1.py'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bound(p):
    with Path(p).open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(Path(p).resolve()),'sha256':s}
def verify(items):
    for x in items:
        if bound(x['path'])['sha256']!=x['sha256']:raise ValueError('Changed source '+x['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def origins():
    rows=[]
    for n,m in tuple(sys.modules.items()):
        f=getattr(m,'__file__',None)
        if f and Path(f).resolve().is_relative_to(REPO):
            if Path(f).resolve() not in (Path(__file__).resolve(),HELPER.resolve(),FILLHELPER.resolve()):raise ValueError('Unexpected import '+n)
            rows.append({'module':n,**bound(f)})
    return rows
def closed():
    import psutil
    r=read(PARENT/'launch_receipt.json')
    try:
        p=psutil.Process(r['pid'])
        if p.cmdline()==r['command'] and abs(p.create_time()-r['process_created'])<.001:raise ValueError('Previous audit still active')
    except psutil.NoSuchProcess:pass
    if not read(PARENT/'completion.json')['complete']:raise ValueError('Previous audit incomplete')
def register(root,tests):
    closed();t=read(tests)
    if t['exit_code']!=0:raise ValueError('Tests failed')
    verify(t['programs']+[t['log']]);p=read(PARENT/'protocol.json');d=read(PARENT/'completion.json')
    bindings=p['bindings']+[d['comparison'],bound(PARENT/'completion.json'),bound(PARENT/'protocol.json'),bound(tests)]
    verify(bindings);root.mkdir(exist_ok=False)
    write(root/'protocol.json',{'at':datetime.now(timezone.utc).isoformat(),'programs':[bound(__file__),bound(HELPER),bound(FILLHELPER),bound(TEST)],'bindings':bindings,'cases':p['cases'],
      'population':'ALL prior V35 same-reference native observations; no return/date/issuer filter. Native same-symbol fills restricted to exactly the source episode timestamp horizon; positions attach once per lifetime.',
      'scope':'Compare first isolated HOLD lifetime with the initial native lifetime and the full isolated HOLD episode with native same-symbol episode. Separately decompose LIQUIDATE-minus-HOLD into first-lifetime and later-reentry cash-flow differences, retaining unresolved inventory.',
      'limits':'Full-path matches/mismatches and descendant outcomes are diagnostics only, never decision-time features or retrospective admission filters. Matching HOLD never proves a changed funded portfolio. Zero terminal inventory required before calling cash flow realized PnL. No summed alternative portfolio PnL.',
      'read_only':True,'fits':0,'native_cases':0,'study_complete':False})
def run(root):
    import psutil
    closed();p=read(root/'protocol.json');verify(p['programs']+p['bindings']);proc=psutil.Process()
    write(root/'supervisor_start.json',{'pid':proc.pid,'process_created':proc.create_time(),'command':proc.cmdline(),'at':datetime.now(timezone.utc).isoformat()})
    h=load('registered_scope',HELPER);f=load('registered_fill_comparison',FILLHELPER);before=origins()
    old=read(PARENT/'comparison.json')['rows'];wanted=defaultdict(list);native={};positions={};assessments={};unique=set()
    for c in p['cases']:
        fills=read(Path(c['folder'])/'fills.json');symbols=defaultdict(list)
        for x in fills:symbols[x['symbol']].append(x)
        native[c['key']]=symbols
        if c['assessment'] not in assessments:assessments[c['assessment']]=read(c['assessment'])
        positions[c['key']]={x['position_id']:x for x in assessments[c['assessment']]['accounts'][c['assessment_key']]['positions']}
        for row in old:
            if row['case']!=c['key']:continue
            key=(row['case'],row['action_id'])
            if key in unique:raise ValueError('Duplicate episode would duplicate descendants')
            unique.add(key);wanted[row['day'],row['case'].split('/')[-1]].append(row)
    refs=read(SOURCE/'labels/completion.json')['artifacts'];byname={Path(x['path']).name:x for x in refs};consumed=[];results=[]
    for (day,scenario),rows in sorted(wanted.items()):
        path=SOURCE/'labels'/f'{day}_{scenario}.events.jsonl.gz';binding=byname[path.name];verify([binding]);consumed.append(binding)
        byaction=defaultdict(list)
        for row in rows:byaction[row['action_id']].append(row)
        tokens=['"opportunity_id":"'+k+'"' for k in byaction];seen=set()
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                if not any(token in line for token in tokens):continue
                v=json.loads(line);action=v['pair']['opportunity_id']
                if action not in byaction:continue
                if action in seen:raise ValueError('Duplicate source episode')
                seen.add(action);hold=v['hold'];liq=v['liquidate'];costs=hold['costs'];decomp=h.decomposition(hold['fills'],liq['fills'],costs)
                hpos=h.lifetimes(hold['fills'],costs);start=f.stamp(hold['timestamps'][0]);end=f.stamp(hold['timestamps'][-1]);clock=v['pair']['checkpoint_at']
                for row in byaction[action]:
                    symbol=row['symbol'];np=positions[row['case']][row['position_id']];nf=native[row['case']][symbol]
                    initial=[x for x in nf if f.stamp(np['entry_fill_time'])<=f.stamp(x['timestamp'])<=f.stamp(np['exit_fill_time'])]
                    first=f.compare(initial,hpos[0]['fills'],symbol,clock,costs,np)
                    episode=[x for x in nf if start<=f.stamp(x['timestamp'])<=end]
                    npos=h.lifetimes(episode,costs);exact=f.fills_canonical(episode,symbol)==f.fills_canonical(hold['fills'],symbol)
                    a=f.account_from_fills(episode,costs);z=f.account_from_fills(hold['fills'],costs)
                    exact=exact and abs(a['fees']-z['fees'])<1e-5 and abs(a['net_cash_flow']-z['net_cash_flow'])<1e-5
                    results.append({k:row[k] for k in ('case','day','symbol','action_id','position_id','quantity_pending_exact','parent_exit_already_decided')} | {
                      'initial_lifetime_bridge':first['hold_lifetime_bridge'],'whole_symbol_episode_bridge':exact,
                      'native_episode_lifetimes':len(npos),'hold_episode_lifetimes':len(hpos),
                      'prior_whole_hold_vs_initial_mismatch':row['exact_fill_prefix'] and not row['hold_lifetime_bridge'],
                      'unresolved_inventory':{'native':a['terminal_inventory'],'hold':z['terminal_inventory']},
                      'action_cash_flow_decomposition':decomp})
        if seen!=set(byaction):raise ValueError('Missing source episodes')
        print(json.dumps({'day':day,'scenario':scenario,'compared':len(results)}),flush=True)
    summaries={}
    for c in p['cases']:
        rows=[r for r in results if r['case']==c['key'] and not r['parent_exit_already_decided']];q=[r for r in rows if r['quantity_pending_exact']]
        summaries[c['key']]={'reference_rows':len(rows),'quantity_pending_exact':len(q),
          'initial_lifetime_bridge_at_exact_quantity':sum(r['initial_lifetime_bridge'] for r in q),
          'whole_symbol_episode_bridge_at_exact_quantity':sum(r['whole_symbol_episode_bridge'] for r in q),
          'episode_reentries_at_exact_quantity':sum(r['hold_episode_lifetimes']>1 for r in q),
          'nonzero_action_descendant_delta_at_exact_quantity':sum(abs(r['action_cash_flow_decomposition']['descendant_net_cash_delta'])>1e-5 for r in q),
          'initial_mismatch_despite_exact_quantity':sum(not r['initial_lifetime_bridge'] for r in q)}
    write(root/'scope.json',{'summaries':summaries,'rows':results,'limits':p['limits']});verify(p['programs']+p['bindings']+consumed)
    write(root/'completion.json',{'complete':True,'at':datetime.now(timezone.utc).isoformat(),'programs':p['programs'],'bindings':p['bindings'],
       'consumed_journals':consumed,'origins_before':before,'origins_after':origins(),'scope':bound(root/'scope.json'),'fits':0,'native_cases':0,'study_complete':False})
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('register','run'));a.add_argument('--root',type=Path,required=True);a.add_argument('--tests',type=Path)
    args=a.parse_args()
    if args.mode=='register':register(args.root,args.tests)
    else:
        try:run(args.root)
        except Exception as e:write(args.root/'failure.json',{'at':datetime.now(timezone.utc).isoformat(),'error':repr(e),'study_complete':False});raise
