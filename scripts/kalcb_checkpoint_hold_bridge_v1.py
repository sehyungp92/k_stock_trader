"""Finite read-only HOLD bridge audit, with no economic selection or replay."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SUPPORT=BASE/'v34_checkpoint_contrast_support'
SOURCE=BASE/'v33_checkpoint_contrast_labels'
HELPER=REPO/'backtests/strategies/kalcb/checkpoint_hold_bridge_v1.py'
TEST=REPO/'tests/backtests/strategies/test_kalcb_checkpoint_hold_bridge_v1.py'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bound(p):
    with Path(p).open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(Path(p).resolve()),'sha256':sha}
def verify(items):
    for x in items:
        if bound(x['path'])['sha256']!=x['sha256']:raise ValueError('Changed input: '+x['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def stamp(s):return datetime.fromisoformat(s)
def closed():
    import psutil
    receipt=read(SUPPORT/'launch_receipt.json')
    try:
        p=psutil.Process(receipt['pid'])
        if p.cmdline()==receipt['command'] and abs(p.create_time()-receipt['process_created'])<.001:
            raise ValueError('Support audit still active')
    except psutil.NoSuchProcess:pass
    if not read(SUPPORT/'completion.json')['complete']:raise ValueError('Support incomplete')
def origins():
    out=[]
    for name,mod in tuple(sys.modules.items()):
        file=getattr(mod,'__file__',None)
        if file and Path(file).resolve().is_relative_to(REPO):
            if Path(file).resolve() not in (Path(__file__).resolve(),HELPER.resolve()):raise ValueError('Unexpected application import: '+name)
            out.append({'module':name,**bound(file)})
    return out


def register(root,tests):
    closed();test=read(tests)
    if test['exit_code']!=0:raise ValueError('Failed tests')
    verify(test['programs']+[test['log']])
    p=read(SUPPORT/'protocol.json');d=read(SUPPORT/'completion.json')
    bindings=p['bindings']+d['artifacts']+[bound(SUPPORT/'protocol.json'),bound(SUPPORT/'completion.json'),bound(tests)]
    bindings=list({x['path']:x for x in bindings}.values());verify(bindings)
    root.mkdir(exist_ok=False)
    write(root/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
          'programs':[bound(__file__),bound(HELPER),bound(TEST)],'bindings':bindings,'tests':bound(tests),'cases':p['cases'],
          'population':'All eight unchanged native prior100/125 IS/OOS baseline/combined V34 observations with a same-source V33 reference. Keep missing references and all mismatches; no future-return filtering.',
          'method':'Compare exact native unique-lifetime fill sequence against isolated HOLD: pre-completed45 prefix, full timestamp/side/integer quantity/price/reason, independently recomputed fees and net cash. Match quantities/pending state from V34 separately.',
          'limits':'Full HOLD identity is a post-outcome diagnostic, never a decision-time feature or retrospective policy filter. Even a matching HOLD does not prove counterfactual liquidation or other funded portfolio paths. No rescaling or summing isolated PnL.',
          'read_only':True,'fits':0,'native_cases':0,'study_complete':False})


def run(root):
    import psutil
    closed();p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    process=psutil.Process();write(root/'supervisor_start.json',{'pid':process.pid,'process_created':process.create_time(),'command':process.cmdline(),'at':datetime.now(timezone.utc).isoformat()})
    spec=importlib.util.spec_from_file_location('registered_hold_bridge',HELPER)
    h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);before=origins()
    comparisons=read(SUPPORT/'selected_parent_comparability.json')['rows'];wanted=defaultdict(list)
    assessments={};native={};positions={}
    for c in p['cases']:
        fills=read(Path(c['folder'])/'fills.json');by_symbol=defaultdict(list)
        for f in fills:by_symbol[f['symbol']].append(f)
        if c['assessment'] not in assessments:assessments[c['assessment']]=read(c['assessment'])
        pos={x['position_id']:x for x in assessments[c['assessment']]['accounts'][c['assessment_key']]['positions']}
        for row in comparisons:
            if row['case']!=c['key'] or not row.get('reference_present'):continue
            key=(c['key'],row['position_id']);v=pos[row['position_id']]
            native[key]=[f for f in by_symbol[row['symbol']] if stamp(v['entry_fill_time'])<=stamp(f['timestamp'])<=stamp(v['exit_fill_time'])]
            positions[key]=v;wanted[(row['day'],c['key'].split('/')[-1])].append(row)
    artifacts=read(SOURCE/'labels/completion.json')['artifacts'];by_name={Path(x['path']).name:x for x in artifacts};consumed=[];results=[]
    for (day,scenario),rows in sorted(wanted.items()):
        path=SOURCE/'labels'/f'{day}_{scenario}.events.jsonl.gz';binding=by_name[path.name];verify([binding]);consumed.append(binding)
        by_action=defaultdict(list)
        for r in rows:by_action[r['action_id']].append(r)
        tokens=['"opportunity_id":"'+k+'"' for k in by_action];seen=set()
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if not any(t in line for t in tokens):continue
                detail=json.loads(line);action=detail['pair']['opportunity_id']
                if action not in by_action:continue
                if action in seen:raise ValueError('Duplicate source pair')
                seen.add(action);hold=detail['hold'];clock=detail['pair']['checkpoint_at']
                for row in by_action[action]:
                    key=(row['case'],row['position_id']);result=h.compare(native[key],hold['fills'],row['symbol'],clock,hold['costs'],positions[key])
                    results.append({k:row[k] for k in ('case','position_id','day','symbol','action_id')} | {
                        'quantity_pending_exact':all(row['state_comparison'].values()),
                        'parent_exit_already_decided':row['parent_exit_already_decided'],
                        'paired_payoff_valid':row['reference']['paired_payoff_valid'],**result})
        if seen!=set(by_action):raise ValueError('Missing reference journals')
        print(json.dumps({'day':day,'scenario':scenario,'compared':len(results)}),flush=True)
    summaries={}
    for c in p['cases']:
        rows=[r for r in results if r['case']==c['key'] and not r['parent_exit_already_decided']]
        qty=[r for r in rows if r['quantity_pending_exact']]
        summaries[c['key']]={'reference_rows':len(rows),'quantity_pending_exact':len(qty),
            'exact_prefix_with_exact_quantity':sum(r['exact_fill_prefix'] for r in qty),
            'full_hold_bridge_with_exact_quantity':sum(r['hold_lifetime_bridge'] for r in qty),
            'full_hold_bridge_all_references':sum(r['hold_lifetime_bridge'] for r in rows),
            'matched_prefix_but_future_mismatch':sum(r['exact_fill_prefix'] and not r['hold_lifetime_bridge'] for r in rows)}
    write(root/'comparison.json',{'summaries':summaries,'rows':results,'limits':p['limits']})
    verify(p['programs']+p['bindings']+consumed)
    write(root/'completion.json',{'complete':True,'at':datetime.now(timezone.utc).isoformat(),'programs':p['programs'],'bindings':p['bindings'],
        'consumed_journals':consumed,'origins_before':before,'origins_after':origins(),'comparison':bound(root/'comparison.json'),
        'native_cases':0,'fits':0,'study_complete':False})


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('register','run'));a.add_argument('--root',type=Path,required=True);a.add_argument('--tests',type=Path)
    args=a.parse_args()
    if args.mode=='register':register(args.root,args.tests)
    else:
        try:run(args.root)
        except Exception as e:write(args.root/'failure.json',{'at':datetime.now(timezone.utc).isoformat(),'error':repr(e),'study_complete':False});raise
