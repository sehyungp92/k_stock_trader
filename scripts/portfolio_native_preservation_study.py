"""Controlled follow-up preserving KALCB's native submission lifecycle."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.executed_value import digest,signature,write_json
from scripts.portfolio_synergy_allocation_study import read,run,run_case,verify_case


def prepare(parent_path,output):
    parent=read(parent_path);manifest=read(parent['manifest']);expected=manifest.pop('manifest_hash')
    if signature(manifest)!=expected:raise ValueError('Changed parent generation')
    output.mkdir(parents=True,exist_ok=False);root=output/'inputs/files'
    changed={}
    for name,sha in manifest['files'].items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name in ('deployment/olr_kalcb/intent_allocator.py','backtests/strategies/portfolio_synergy/intent_replay.py'):
            shutil.copy2(REPO/name,target);changed[name]={'before':sha,'after':digest(target)};manifest['files'][name]=digest(target)
        else:os.link(Path(parent['code'])/name,target)
    spec=read(root/parent['spec_path']);base=copy.deepcopy(spec['candidates']['kalcb200_olr20_proportional'])
    new={'joint_control':copy.deepcopy(base)}
    base['allocation']['preserve_kalcb_submission']=True
    new['native_kalcb_olr20']=copy.deepcopy(base)
    base['policy']['max_sector_notional']=2*spec['initial_equity']
    new['native_kalcb_olr20_sector2']=copy.deepcopy(base)
    base['allocation'].update(olr_budget_fraction=.1,olr_sizing_fraction=.1)
    new['native_kalcb_olr10_sector2']=copy.deepcopy(base)
    base=copy.deepcopy(spec['candidates']['kalcb150_olr35_proportional'])
    base['allocation']['preserve_kalcb_submission']=True
    new['native_kalcb_olr35']=base
    spec['candidates'].update(new)
    spec['native_preservation_followup']={
        'trigger':'All 50 previous cases failed the joint acceptance gates. In the KALCB2/OLR20 stressed case, KALCB lost KRW17.756m versus standalone, outweighing OLR profit of KRW8.903m. Changed eligibility, pre-reservation, ranking and execution paths require a direct native-lifecycle control.',
        'design':'Same-configuration joint/native submission pair; sector-cap ablation; smaller OLR sleeve; native KALCB with proportional 35% OLR. Re-run KALCB2 standalone as a source-generation bridge.',
        'unchanged_gates':spec['allocation_selection'],'independent_validation':False,
        'source_scope':'Only allocator contracts/replay integration changed. Concurrent native OLR capacity code is excluded. The shared minimum_qty field is present but remains zero for every intent.'}
    spec_name='research/portfolio_current/native_preservation.json';write_json(root/spec_name,spec)
    manifest['files'][spec_name]=digest(root/spec_name);manifest['evidence']['portfolio_current_study']=spec_name
    program='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    source=subprocess.check_output([sys.executable,'-B','-c',program],cwd=root,text=True).strip()
    manifest['source_generation_hash']=source
    write_json(output/'inputs/manifest.json',{**manifest,'manifest_hash':signature(manifest)})
    write_json(output/'study.json',{**parent,**spec,'code':str(root.resolve()),'source_generation_hash':source,
        'manifest':str((output/'inputs/manifest.json').resolve()),'spec_path':spec_name})
    jobs=[{'name':name+'_'+scenario,'recipe':name,'scenario':scenario,
        'standalone':'KALCB' if name=='kalcb_size_200' else None}
        for name in ['kalcb_size_200',*new] for scenario in ('baseline','combined')]
    write_json(output/'registration.json',{'parent_study':str(parent_path.resolve()),'parent_study_sha256':digest(parent_path),
        'parent_finalist_sha256':digest(parent_path.parent/'finalist.json'),'changed_files':changed,'jobs':jobs,
        'selection':spec['allocation_selection'],'source_generation_hash':source,
        'disclosure':spec['native_preservation_followup'],'portfolios':list(new),
        'frequency_floors':[369,371],
        'bridge_cases':{'joint_control':'kalcb200_olr20_proportional','kalcb_size_200':'kalcb_size_200'}})
    print(json.dumps({'prepared':str(output),'cases':len(jobs),'source':source}),flush=True)


def select(study_path):
    study=read(study_path);reg=read(study_path.parent/'registration.json');parent_path=Path(reg['parent_study'])
    if digest(parent_path)!=reg['parent_study_sha256'] or digest(parent_path.parent/'finalist.json')!=reg['parent_finalist_sha256']:
        raise ValueError('Changed prior study evidence')
    parent=read(parent_path);prior=read(parent_path.parent/'finalist.json');selection=reg['selection']
    results={j['name']:verify_case(study_path.parent/'cases'/j['name'],study) for j in reg['jobs']}
    def pair(name):return [results[name+'_'+scenario] for scenario in ('baseline','combined')]
    # This bridge checks economic identity, not merely equal source filenames.
    bridge_evidence={}
    for new,old in reg['bridge_cases'].items():
        for scenario in ('baseline','combined'):
            a=results[new+'_'+scenario];b=verify_case(parent_path.parent/'cases'/(old+'_'+scenario),parent)
            for key in ('return_pct','drawdown_pct','trades','active_days','strategy_counts'):
                if a[key]!=b[key]:raise ValueError('Source bridge changed economics: '+new+' '+key)
            bridge_evidence[new+'_'+scenario]={
                generation:{'path':str(path),'files':{filename:digest(path/filename) for filename in
                    ('result.json','completion.json','execution_audit.json')}}
                for generation,path in [('current',study_path.parent/'cases'/(new+'_'+scenario)),
                                        ('prior',parent_path.parent/'cases'/(old+'_'+scenario))]}
    reference=pair('kalcb_size_200');floors=[r['return_pct'] for r in reference]
    if floors!=prior['return_floors']:raise ValueError('The repeated reference no longer spans the previous return frontier')
    candidates={};rejected=dict(prior['rejected'])
    for name in reg['portfolios']:
        pair_result=pair(name);why=[]
        if pair_result[0]['drawdown_pct']>selection['baseline_max_dd_pct'] or pair_result[1]['drawdown_pct']>selection['combined_max_dd_pct']:why.append('drawdown_envelope')
        if any(r['return_pct']<=floors[i] for i,r in enumerate(pair_result)):why.append('standalone_return_frontier')
        if any(r['trades']<reg['frequency_floors'][i] for i,r in enumerate(pair_result)):why.append('frequency_floor')
        if why:rejected[name]=why
        else:candidates[name]=pair_result
    def vector(p):return [p[0]['return_pct'],p[1]['return_pct'],min(r['trades'] for r in p),-p[0]['drawdown_pct'],-p[1]['drawdown_pct']]
    frontier={n:p for n,p in candidates.items() if not any(all(a>=b for a,b in zip(vector(o),vector(p))) and any(a>b for a,b in zip(vector(o),vector(p))) for m,o in candidates.items() if m!=n)}
    winner=None
    if frontier:
        top=max(p[1]['return_pct'] for p in frontier.values())
        shortlist={n:p for n,p in frontier.items() if p[1]['return_pct']>=top-selection['near_return_band_pct_points']}
        winner=max(shortlist,key=lambda n:(min(r['trades'] for r in shortlist[n]),-max(r['drawdown_pct'] for r in shortlist[n]),shortlist[n][1]['return_pct']))
    compact={**prior['results'],**{n:[{k:r[k] for k in ('return_pct','drawdown_pct','trades','active_days')} for r in pair(n)] for n in ['kalcb_size_200',*reg['portfolios']]}}
    receipt={'winner':winner,'recipe':study['candidates'].get(winner),'frontier':list(frontier),'rejected':rejected,
        'reference_frontier':prior['reference_frontier'],'return_floors':floors,'selection':selection,
        'frequency_floors':reg['frequency_floors'],
        'source_generation_hash':study['source_generation_hash'],'results':compact,'reference_challenge':None,
        'prior_study':{'path':str(parent_path),'sha256':digest(parent_path),'finalist_sha256':digest(parent_path.parent/'finalist.json')},
        'source_bridge':'KALCB2 standalone and previous joint KALCB2/OLR20 economics exactly repeated in both scenarios.',
        'source_bridge_evidence':bridge_evidence,
        'status':'frozen_research_finalist_pending_qualification' if winner else 'no_candidate_passed_do_not_promote'}
    path=study_path.parent/'finalist.json'
    if path.exists() and read(path)!=receipt:raise ValueError('Finalist already frozen')
    write_json(path,receipt);print(json.dumps({'winner':winner,'frontier':list(frontier),'return_floors':floors,'new_results':{n:compact[n] for n in reg['portfolios']}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','select','qualify'))
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.study,a.output)
    elif a.command=='run':run(a.study,a.workers)
    elif a.command=='select':select(a.study)
    else:
        final=read(a.study.parent/'finalist.json')
        if not final['winner']:raise ValueError('No selected candidate')
        for scenario in ('baseline','combined','cost','participation'):
            r=run_case(a.study,{'name':final['winner']+'_'+scenario,'recipe':final['winner'],'scenario':scenario},True)
            if r['audit_status']!='passed':raise ValueError('Full native qualification failed')
