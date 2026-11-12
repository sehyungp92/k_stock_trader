"""Generate separately versioned native targets under selected KALCB management.

No forecast, model fit, selection or canonical policy is changed by this program.
The recorded ready population must bridge under both configurations before replay.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
RECOVERY=Path('C:/Users/sehyu/.codex/research/kalcb_full_oos_recovery_20261009/v3')
DEST=Path('E:/CodexResearch/k_stock_trader/kalcb_selected_target_alignment_20261010/v1')
HISTORY=Path('E:/CodexResearch/k_stock_trader')
FORWARD=Path('C:/Users/sehyu/.codex/research/olr_round1_oos_20261009/forward_features')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    selected=REPO/'config/research/kalcb_intraday_advantage_20261007/selected_config.json'
    config=root/'selected_management.json';write(config,read(selected))
    sources={
      ('is','opening'):(HISTORY/'kalcb_action_value_20261006/quantity_plan_v2.json',HISTORY/'kalcb_continuation_20261006/prepared_v1'),
      ('is','timed'):(HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/plan.json',HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/prepared'),
      ('oos','opening'):(FORWARD/'plan.json',FORWARD/'continuation'),
      ('oos','timed'):(FORWARD/'checkpoint_plan.json',FORWARD/'checkpoint')}
    cells=[];jobs=[];proofs=[bound(__file__),bound(selected),bound(config)]
    for (period,kind),(parent,prepared) in sources.items():
        plan=read(parent);original=read(plan['config']);updated=read(config)
        differences={k:{'old':original.get(k),'selected':updated.get(k)} for k in set(original)|set(updated) if original.get(k)!=updated.get(k)}
        if any(not(k.startswith('exit_') or k in {'entry_checkpoint_enabled','entry_quantity_multiplier'}) for k in differences):
            raise ValueError('Unexpected entry change in target alignment')
        start=1 if period=='oos' else 0;meta=read(prepared/'prepared.json');dates=meta['sessions'][start:]
        expected=('2025-05-12','2026-03-31',217) if period=='is' else ('2026-04-01','2026-07-10',69)
        if (dates[0],dates[-1],len(dates))!=expected:raise ValueError('Wrong target dates')
        old_config=plan['config'];plan['config']=str(config);plan['input_sha256'].pop(old_config);plan['input_sha256'][str(config)]=digest(config)
        plan['target_alignment']={'parent':bound(parent),'selected':bound(selected),'config_changes':differences,
            'scope':'Identical causal ready population, quantity grid and source; native selected management outcomes.'}
        path=root/(period+'_'+kind+'_plan.json');write(path,plan)
        cell={'period':period,'kind':kind,'plan':str(path),'parent_plan':str(parent),'prepared':str(prepared),'start_index':start,'sessions':dates,'config_changes':differences}
        cells.append(cell);proofs.extend([bound(path),bound(parent),bound(prepared/'prepared.json')])
        for scenario in ('baseline','combined'):
            jobs.append({**cell,'name':period+'_'+kind+'_'+scenario,'scenario':scenario,
                         'output':str(root/'labels'/(period+'_'+kind+'_'+scenario))})
    (root/'labels').mkdir()
    write(root/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'code':str(RECOVERY/'inputs/files'),
        'expected_source_generation_hash':read(RECOVERY/'preparation.json')['source_generation_hash'],
        'bound_inputs':proofs,'cells':cells,'jobs':jobs,'quantity_scales':[.25,.5,1.],
        'hypothesis':'Value models should learn outcomes from the same selected management policy they rank. Isolate older-target mismatch before adding features or broad parameter search.',
        'no_outcome_filtering':'All original ready/guard opportunities, losses/no-fills and physical failures retained. Invalid execution has no investable return and remains in risk labels.',
        'maturity':'Label availability and one-session embargo remain mandatory in subsequent fits. Whole-horizon generation is not permission to train on future outcomes.',
        'next_fit_design':'After target verification, rebuild all historical out-of-fold predictions and calibration under the aligned target. Keep original global60-session warm-up and20-session folds across IS/OOS. Do not calibrate new targets using old-target in-sample predictions or reset April1.',
        'control_design':'Original-target continuous learner and selected management form the common control; new aligned-target learner initially keeps architecture, features, sizing, exits, clock and calibration settings fixed.',
        'scope':'Reused development. No model/config promotion. Full target-integrity and source qualification still required before acceptance.'})
    print(json.dumps({'registered':str(root/'protocol.json'),'jobs':len(jobs),'cells':len(cells)}),flush=True)


def verify(root):
    protocol=read(root/'protocol.json')
    for p in protocol['bound_inputs']:
        if digest(p['path'])!=p['sha256']:raise ValueError('Changed input '+p['path'])
    return protocol


def preview(root):
    protocol=verify(root);code=Path(protocol['code']);sys.path.insert(0,str(code))
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_executed_value_study import config_for
    from backtests.strategies.kalcb import executed_value as ev
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_common.market import MarketBar
    if source_generation_hash()!=protocol['expected_source_generation_hash']:raise ValueError('Wrong native source')
    portfolio=KALCBPortfolioView(cash=ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY)
    report=[]
    for cell in protocol['cells']:
        old=read(cell['parent_plan']);new=read(cell['plan']);cfg0=config_for(old);cfg1=config_for(new)
        prepared=Path(cell['prepared']);meta=read(prepared/'prepared.json')
        if digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed prepared observations')
        wanted=set(cell['sessions']);count=Counter()
        with (prepared/'opportunities.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row['trade_date'] not in wanted:continue
                cutoff=datetime.fromisoformat(row['feature_available_at'])
                bars=[MarketBar(row['symbol'],datetime.fromisoformat(b['timestamp']),'5m',
                    *[b[k] for k in ('open','high','low','close','volume')]) for b in row['opening_bars']]
                snap=ev.snapshot_for(row['trade_date'],[(row,0.)],source=new['input_sha256'][new['database']])
                a=preview_entry(bars,snap,cfg0,portfolio,available_at=cutoff)
                b=preview_entry(bars,snap,cfg1,portfolio,available_at=cutoff)
                def signature(p):return (p['ready'],[(v['symbol'],v['quantity'],v['metadata'].get('entry_route')) for v in p['intents']])
                if signature(a)!=signature(b):raise ValueError('Selected management changes causal entry '+row['opportunity_id'])
                if a['ready']!=row['action']['ready'] or sum(v['quantity'] for v in a['intents'])!=row['action']['quantity']:
                    raise ValueError('Original entry preview does not bridge to prepared row '+row['opportunity_id'])
                count['rows']+=1;count['ready_and_guard']+=bool(a['ready'] and row['guard_at_rank1'])
        report.append({'period':cell['period'],'kind':cell['kind'],**count,'passed':True})
        print(json.dumps(report[-1]),flush=True)
    if source_generation_hash()!=protocol['expected_source_generation_hash']:raise ValueError('Native source changed')
    write(root/'preview_bridge.json',{'passed':True,'cells':report,'protocol':bound(root/'protocol.json'),
        'source_generation_hash':protocol['expected_source_generation_hash']})


def run(root):
    import psutil
    import msvcrt
    protocol=verify(root);code=Path(protocol['code'])
    lock=(root/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'completed':[]}
    def status(stage,**kw):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kw);write(root/'status.json',state)
    def wait_for(label,predicate):
        start=time.monotonic()
        while not predicate():
            status('waiting',waiting_for=label)
            if (root/'hold.json').exists():raise ValueError('Explicit hold receipt')
            if time.monotonic()-start>48*3600:raise TimeoutError(label)
            time.sleep(30)
    env=os.environ.copy()
    env.update({k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONUTF8']})
    env.update(PYTHONPATH=str(code),PYTHONIOENCODING='utf-8')
    def execute(args,log):
        verify(root)
        wait_for('at least1GiB available RAM',lambda:psutil.virtual_memory().available>=1024**3)
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-B','-u',*args],cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            status('running',child={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'argv':args,'log':str(log)})
            started=time.monotonic()
            while child.poll() is None:
                status('running')
                if time.monotonic()-started>48*3600:raise TimeoutError('Child preserved after48hours')
                time.sleep(30)
            if child.returncode:raise RuntimeError('Child failed, see '+str(log))
            status('child_completed',child=None)
    try:
        wait_for('completed targeted round diagnostics',lambda:bool(list((RECOVERY/'targeted_round/framework').glob('*/round_1/round_final_full_diagnostics.json'))))
        if not (root/'preview_bridge.json').exists():execute([str(Path(__file__).resolve()),'preview','--root',str(root)],root/'preview_bridge.log')
        if read(root/'preview_bridge.json').get('passed') is not True:raise ValueError('Failed causal entry bridge')
        for job in protocol['jobs']:
            out=Path(job['output'])
            if (out/'completion.json').exists():
                done=read(out/'completion.json')
                if not done.get('complete') or done['source_generation_hash']!=protocol['expected_source_generation_hash']:raise ValueError('Invalid target completion')
                state['completed'].append(job['name']);continue
            if out.exists():raise ValueError('Preserve partial target generation '+str(out))
            execute(['-m','scripts.kalcb_action_quantity_labels','--plan',job['plan'],'--prepared',job['prepared'],'--output',job['output'],
                     '--scenario',job['scenario'],'--scales','.25','.5','1','--start-index',str(job['start_index'])],root/(job['name']+'.log'))
            done=read(out/'completion.json')
            if not done['complete'] or done['source_generation_hash']!=protocol['expected_source_generation_hash']:raise ValueError('Wrong target source')
            state['completed'].append(job['name'])
        write(root/'completion.json',{'complete':True,'protocol':bound(root/'protocol.json'),'preview':bound(root/'preview_bridge.json'),
            'grids':[bound(Path(j['output'])/'grid_manifest.json') for j in protocol['jobs']]})
        status('ready_for_aligned_walk_forward',child=None)
    except Exception as exc:
        status('attention_required',error=repr(exc));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preview','run']);p.add_argument('--root',type=Path,default=DEST);a=p.parse_args()
    {'prepare':prepare,'preview':preview,'run':run}[a.stage](a.root)
