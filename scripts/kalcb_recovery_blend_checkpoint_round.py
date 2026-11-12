"""Finite memory-blend, checkpoint-calibration and sizing contrasts.

All component forecasts were generated out of fold. A blend receives its own
past-only calibration history; averaging finished portfolio returns is forbidden.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT,POLICY,read,write,rows,jsonl,bound,digest,env
from scripts import kalcb_alignment_walk_forward_v3 as aligned
from scripts import kalcb_recovery_mechanism_round as mechanism

DEST=Path('E:/CodexResearch/k_stock_trader/kalcb_blend_checkpoint_20261010/v1')
ROUND='blend_checkpoint_round_v1'
RECIPES={
    'blend_decay25':{'decay_weight':.25,'checkpoint_calibration':False},
    'blend_decay50':{'decay_weight':.5,'checkpoint_calibration':False},
    'blend_decay75':{'decay_weight':.75,'checkpoint_calibration':False},
    'expanding_checkpoint_cal':{'decay_weight':0.,'checkpoint_calibration':True},
    'decay60_checkpoint_cal':{'decay_weight':1.,'checkpoint_calibration':True},
}
SIZING={'decay60_atr4':{'parent':'decay','mode':'atr_quantity_4pct'},
        'decay60_atr5':{'parent':'decay','mode':'atr_quantity_5pct'},
        'blend50_atr5':{'parent':'blend_decay50','mode':'atr_quantity_5pct'}}
CONTROLS={'expanding':aligned.DEST/'fit/policies/original_target_full_history',
          'decay':mechanism.DEST/'fit/policies/original_decay60'}


def blend_values(expanding,decay,weight):
    if weight not in (0.,.25,.5,.75,1.):raise ValueError('Unregistered blend weight')
    if set(expanding)!=set(decay):raise ValueError('Different forecast action populations')
    result={}
    fields=('opportunity_id','trade_date','valid_from','valid_through','quantity_scale','capacity_ok','failure_probability')
    for key,a in expanding.items():
        b=decay[key]
        if any(a[k]!=b[k] for k in fields):raise ValueError('Different decision clocks, quantities or physical forecasts')
        value=deepcopy(a)
        for scenario in ('baseline','combined'):
            field=scenario+'_value_bps'
            if not all(math.isfinite(r[field]) for r in (a,b)):raise ValueError('Nonfinite forecast')
            value[field]=a[field] if weight==0 else b[field] if weight==1 else (1-weight)*a[field]+weight*b[field]
        result[key]=value
    return result


def check(out):
    p=read(out/'protocol.json');aligned.verify_bindings(p['inputs']+p['programs'])
    if p['recipes']!=RECIPES or p['sizing']!=SIZING:raise ValueError('Changed experiment')
    return p


def register(out):
    if out.exists():raise ValueError('Preserve previous registration')
    mechanism.check(mechanism.DEST)
    done=read(mechanism.DEST/'replay_completion.json')
    if not done['complete']:raise ValueError('Previous full round incomplete')
    inputs=[bound(ROOT/'stage4_review.json'),bound(ROOT/'stage3_review.json'),bound(ROOT/'stage2_review.json'),
            bound(mechanism.DEST/'protocol.json'),bound(mechanism.DEST/'fit/completion.json'),
            bound(mechanism.DEST/'replay_completion.json'),done['study'],done['diagnostics'],
            bound(aligned.DEST/'protocol.json'),bound(aligned.DEST/'fit/completion.json'),
            bound(aligned.BOUNDARY/'completion.json'),bound(ROOT/'blend_checkpoint_integrity_tests_20261010.json')]
    tests=read(ROOT/'blend_checkpoint_integrity_tests_20261010.json')
    if not tests['complete'] or tests['exit_code']!=0:
        raise ValueError('Missing integrity tests')
    aligned.verify_bindings([tests['log']]);inputs.append(tests['log'])
    for root,name in [(aligned.DEST,'original_target_full_history'),(mechanism.DEST,'original_decay60')]:
        done=read(root/'fit/completion.json');inputs+=done['models']+done['policies']
        inputs += [bound(root/'fit/forecasts'/name/(k+'.jsonl')) for k in ('opening','timed')]
    programs=set(REPO.glob('scripts/kalcb_*.py'))|set(REPO.glob('backtests/strategies/kalcb/*.py'))
    programs|=set(REPO.glob('strategy_kalcb/**/*.py'))|set(REPO.glob('backtests/auto/shared/*.py'))
    programs.add(REPO/'scripts/portfolio_frozen_inference.py')
    out.mkdir(parents=True)
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'recipes':RECIPES,'sizing':SIZING,
        'inputs':inputs,'programs':[bound(p) for p in sorted(programs)],'periods':aligned.PERIODS,'round':ROUND,
        'mechanisms':{
            'memory_blend':'Fixed25/50/75percent decay60 weights combine expanding and decay60 raw economic forecasts within identical global folds. Earlier OOF forecasts of each blend alone fit its opening calibration. Intended to retain long-history IS information while addressing the observed recency tradeoff. This is an executed policy, not a portfolio-return average.',
            'checkpoint':'Apply the existing60-session/100-prior positive-support residual correction to checkpoint raw forecasts using only older fold observations and mature matched original targets. Preserve half-size cap,48bps hurdle, native admission/timing and physical-risk rules. Controls are expanding and decay60 without checkpoint correction.',
            'sizing':'Existing ATR4/5 on decay60 and ATR5 on the central50percent blend. No score or outcome revaluation after sizing; all real cash/capacity consequences replay natively.'},
        'bridge':'Rebuild both complete225-day expanding/decay controls from bound forecasts/calibration and require exact opening/timed snapshots and admission parity before replay.',
        'invariants':'Same economic core and entry/exit configuration, real fills, costs, frozen market source, continuous global clock and one-session embargo. No outcome-conditioned exclusions, date/symbol filters, live changes or automatic promotion.',
        'support':'Report distinct changed candidate issuer-days and sessions separately from native changed feasible shares. For qualification prefer10 changed issuer-days across8 sessions; downstream compounding cannot supply missing direct intervention support. Keep inactive-equivalent aliases.',
        'selection':'Eight finite variants, all actual new IS and69-session OOS accounts; preserve all failed trials and component/backout evidence. DD8/10 are soft preferences. OOS is reused development. Qualification requires source/config/model/label/account audits, four stress scenarios, concentration and parameter stability.'})
    print({'registered':str(out),'variants':8},flush=True)


def original_targets():
    from scripts.kalcb_recovery_refit import datasets
    data,_,labels,sessions,_,evidence=datasets(ROOT,True)
    done=read(aligned.BOUNDARY/'completion.json')
    aligned.verify_bindings([done['protocol'],done['overlays'],done['prefix_bridge']])
    if not done['complete'] or not read(aligned.BOUNDARY/'prefix_bridge.json')['passed']:raise ValueError('Invalid boundary completion')
    aligned.verify_bindings(read(aligned.BOUNDARY/'protocol.json')['bound_inputs'])
    count=0
    for item in read(aligned.BOUNDARY/'overlays.json'):
        if item['family']!='original':continue
        before,after=item['prior_label'],item['label'];compact=lambda r:{k:r[k] for k in ('label_valid','label_available_at','net_value_equity_bps')}
        old=labels[item['kind']][item['scenario']][after['opportunity_id']]
        if old!=compact(before):raise ValueError('Boundary prefix target changed')
        aligned.economic_label_contract(after)
        labels[item['kind']][item['scenario']][after['opportunity_id']]=compact(after);count+=1
    if count!=4:raise ValueError('Wrong original boundary overlay population')
    evidence.update({str(p):digest(p) for p in (aligned.BOUNDARY/'completion.json',aligned.BOUNDARY/'overlays.json',aligned.BOUNDARY/'prefix_bridge.json')})
    return data,labels,sessions,evidence


def build(out):
    from scripts.kalcb_recovery_refit import assemble
    from scripts.kalcb_quantity_calibration import corrections
    from scripts.portfolio_frozen_inference import calibrated
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    from backtests.strategies.kalcb.executed_value import outer_windows,signature
    check(out);work=out/'build';work.mkdir(exist_ok=False)
    data,labels,sessions,evidence=original_targets()
    windows=outer_windows(sessions,min_train=60,block=20)
    calendar=read(aligned.DEST/'fit/calendar.json')
    if sessions!=calendar['sessions'] or [list(w) for w in windows]!=calendar['windows']:raise ValueError('Changed fit clock')
    cells={(c['period'],c['kind']):c for c in read(aligned.ALIGNMENT/'protocol.json')['cells']}
    plans={k:read(c['parent_plan']) for k,c in cells.items()};sources={k:digest(Path(c['prepared'])/'opportunities.jsonl') for k,c in cells.items()}
    raw={}
    for key,root,name in [('expanding',aligned.DEST,'original_target_full_history'),('decay',mechanism.DEST,'original_decay60')]:
        raw[key]={k:{r['opportunity_id']:r for r in rows(root/'fit/forecasts'/name/(k+'.jsonl'))} for k in data}
    controls={'bridge_expanding':{'decay_weight':0.,'checkpoint_calibration':False},'bridge_decay':{'decay_weight':1.,'checkpoint_calibration':False}}
    bridges=[];support={}
    for name,recipe in {**controls,**RECIPES}.items():
        weight=recipe['decay_weight'];forecasts={k:blend_values(raw['expanding'][k],raw['decay'][k],weight) for k in data}
        histories={k:[] for k in data};calibrations=[];results={p:{'snapshots':{},'updates':{},'admissions':[],'timed_admissions':[]} for p in aligned.PERIODS}
        folds=[]
        for parent in ([CONTROLS['expanding']] if weight==0 else [CONTROLS['decay']] if weight==1 else list(CONTROLS.values())):
            for period in aligned.PERIODS:
                for proof in read(parent/period/'chronology.json')['folds']:
                    if proof not in folds:folds.append(proof)
        for first,last in windows:
            fr={k:[r for r in rs if first<=r['trade_date']<=last] for k,rs in data.items()}
            current={k:{r['opportunity_id']:forecasts[k][r['opportunity_id']] for r in rs} for k,rs in fr.items()}
            cals=aligned.calibrate(histories['opening'],labels['opening'],sessions,first,last)
            calibrations.extend({**c,'kind':'opening'} for c in cals.values())
            timed=current['timed']
            if recipe['checkpoint_calibration']:
                if any(r['trade_date']>=first for r in histories['timed']):raise ValueError('Future checkpoint calibration history')
                _,proof=corrections(histories['timed'],labels['timed'],sessions,first)
                proof.update(mode='offset_all',kind='timed',valid_through=last,history_forecasts_sha256=signature(histories['timed']))
                calibrations.append(proof);timed=calibrated(timed,proof)
            for period,dates in aligned.split_dates(sessions,first,last).items():
                if not dates:continue
                chosen=assemble(*([r for r in fr[k] if r['trade_date'] in dates] for k in ('opening','timed')),
                    current['opening'],timed,cals,plans[(period,'opening')],plans[(period,'timed')],dates,sources[(period,'opening')],sources[(period,'timed')])
                for k in ('snapshots','updates'):results[period][k].update(chosen[k])
                for k in ('admissions','timed_admissions'):results[period][k].extend(chosen[k])
            for k in data:histories[k].extend(current[k].values())
        if name in controls:
            parent=CONTROLS[name.removeprefix('bridge_')]
            for period,result in results.items():
                m,snapshots=verify_policy(parent/period,POLICY)
                expected={'snapshots':snapshots,'updates':timed_policy(parent/period,m,snapshots),
                    'admissions':rows(parent/period/(POLICY+'.admission.jsonl')),'timed_admissions':rows(parent/period/'timed.admission.jsonl')}
                if mechanism.normalized(result)!=mechanism.normalized(expected):raise ValueError('Stored control policy bridge failed '+name+' '+period)
                bridges.append({'control':name,'period':period,'sessions':len(snapshots),'passed':True})
            print({'bridge':name,'passed':True},flush=True)
            continue
        for period,result in results.items():
            aligned.save_period(work/'policies'/name/period,CONTROLS['expanding']/period,period,result,folds,calibrations,evidence,
                {'name':name,**recipe,'models':'component bytes and causal raw forecast blend; no return-series blending'},out/'protocol.json')
            # Raw candidate changes, explicitly not independent feasible-share
            # interventions, are recorded before any native economic result.
            parent=CONTROLS['decay' if weight==1 else 'expanding']/period
            old=rows(parent/(POLICY+'.admission.jsonl'))+rows(parent/'timed.admission.jsonl')
            new=result['admissions']+result['timed_admissions']
            choices=lambda rs:{r['action_id'].rsplit('/q',1)[0]:(r['quantity_scale'],r['score']) for r in rs}
            a,b=choices(old),choices(new);changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
            support[name+'/'+period]={'changed_candidate_issuer_days':len(changed),'changed_sessions':len({k[:10] for k in changed}),
                'added':len(set(b)-set(a)),'removed':len(set(a)-set(b)),'scope':'Candidate score/quantity differences; not broker-effective share evidence or independent samples.'}
        folder=work/'forecasts'/name;folder.mkdir(parents=True)
        for k in data:jsonl(folder/(k+'.jsonl'),forecasts[k].values())
        print({'built':name},flush=True)
    write(work/'control_bridges.json',{'passed':True,'sessions':sum(b['sessions'] for b in bridges),'checks':bridges})
    write(work/'candidate_support.json',support);check(out)
    write(work/'completion.json',{'complete':True,'protocol':bound(out/'protocol.json'),'bridge':bound(work/'control_bridges.json'),
        'support':bound(work/'candidate_support.json'),'policies':[bound(work/'policies'/name/p/'policies_manifest.json') for name in RECIPES for p in aligned.PERIODS]})


def replay(out):
    from scripts.kalcb_recovery_experiment_round import prepare,run
    check(out);done=read(out/'build/completion.json')
    if not done['complete']:raise ValueError('Incomplete build')
    aligned.verify_bindings(done['policies']+[done['protocol'],done['bridge'],done['support']])
    definitions={n:{'config':{},**{p+'_policy':str(out/'build/policies'/n/p) for p in aligned.PERIODS}} for n in RECIPES}
    for name,spec in SIZING.items():
        parent=CONTROLS['decay'] if spec['parent']=='decay' else out/'build/policies'/spec['parent']
        definitions[name]={'config':{},'quantity_rule':spec['mode'],**{p+'_policy':str(parent/p) for p in aligned.PERIODS}}
    write(out/'replay_definitions.json',definitions);prepare(ROOT,ROUND,definitions,1)
    path=ROOT/ROUND/'study.json';study=read(path)
    study['is_interpretation']='Every trial has actual new IS and OOS native replay from the continuous historical policy. No reuse of historical IS for a changed learner.'
    study['purpose']='Fixed memory blends, independent checkpoint residual calibration and registered neighboring ATR controls.'
    study['bound_inputs'] += [bound(out/'protocol.json'),bound(out/'build/completion.json')]
    study['source_programs'] += [bound(__file__)]
    if study['cached_interaction_is']:raise ValueError('Changed IS must replay')
    write(path,study);run(ROOT,ROUND)
    final=list((ROOT/ROUND/'framework').glob('*/round_1/round_final_full_diagnostics.json'))
    if len(final)!=1:raise ValueError('Missing full final diagnostics')
    write(out/'replay_completion.json',{'complete':True,'study':bound(path),'diagnostics':bound(final[0]),
        'next_step':'Review complete stage/components, direct action support and stability; freeze credible finalists for four native stress/source qualification. No automatic promotion.'})


def background(out):
    import psutil,msvcrt
    check(out);lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'child':None}
    def status(stage,**kw):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kw)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    try:
        for stage,receipt in [('build',out/'build/completion.json'),('replay',out/'replay_completion.json')]:
            if receipt.exists():raise ValueError('Preserve prior run')
            check(out)
            while psutil.virtual_memory().available<2*1024**3:status('waiting_for_memory');time.sleep(30)
            if (out/'hold.json').exists():raise ValueError('Explicit hold')
            command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),stage,'--root',str(out)]
            with (out/(stage+'.log')).open('x',encoding='utf-8') as f:
                child=subprocess.Popen(command,cwd=REPO,env=env(REPO),stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                status(stage,child={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command})
                started=time.monotonic()
                while child.poll() is None:
                    if time.monotonic()-started>48*3600:raise TimeoutError('Child preserved after48h')
                    time.sleep(30)
                if child.returncode:raise RuntimeError('Failed '+stage+'; preserve evidence')
            if not read(receipt)['complete']:raise ValueError('Incomplete stage')
            status(stage+'_completed',child=None)
        status('ready_for_blend_checkpoint_review',child=None)
    except Exception as exc:status('attention_required',error=repr(exc));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','build','replay','run']);p.add_argument('--root',type=Path,default=DEST)
    a=p.parse_args();{'register':register,'build':build,'replay':replay,'run':background}[a.stage](a.root)
