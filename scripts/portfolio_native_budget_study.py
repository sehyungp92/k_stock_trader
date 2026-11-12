"""Finite, frozen research search for native sleeve budgets; never sends orders.

The default account is KRW25m. Baseline and combined stress share one schedule,
one account and one execution model. All outcomes are reused development data.
"""
from __future__ import annotations

import argparse
import copy
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.executed_value import signature, digest, write_json

SCENARIOS = ('baseline', 'combined')
OWNED = ('deployment/olr_kalcb/native_budget.py',
         'backtests/strategies/portfolio_synergy/native_budget.py',
         'backtests/engine/integrity.py', 'scripts/portfolio_synergy_study.py')
SELECTION = {'baseline_max_dd_pct': 11., 'combined_max_dd_pct': 12.5,
    'frequency_floors': [351, 352], 'stress_return_tie_band_pp': 1.,
    'rule': 'Beat best matched standalone feasible in BOTH risk scenarios in each return; meet native KALCB economic entry frequency; Pareto filter; maximize stressed return; within 1pp prefer minimum scenario frequency, then lower maximum drawdown.',
    'independent_performance_established': False,
    'upward_extension': 'Only if no base-grid candidate passes or winning K=1 lies on risk-feasible boundary with both scenarios retaining native frequency. Then register K=1.25/1.5/1.75; gates unchanged.',
    'risk_challenge': 'On selected budget only, test aggregate planned KALCB stop-risk caps 2%, 2.5%, 3% of current NAV; same selection gates.',
    'robustness': 'Selected account at 20/22.5/25/27.5/30m; no upward compounding; separate cost/participation; OLR auction adverse15bp and nonfill50%; overnight10/20% gap exposure diagnostics. Stress tests do not establish true auction liquidity or guaranteed losses.'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_spec(study):
    """Verify current frozen bytes even when replay outputs are already cached."""
    manifest = read(study['manifest'])
    identity = manifest.pop('manifest_hash')
    if signature(manifest) != identity:
        raise ValueError('Changed parent manifest')
    path = Path(study['code']) / study['spec_path']
    if digest(path) != manifest['files'][study['spec_path']]:
        raise ValueError('Frozen study specification changed: ' + str(path))
    spec = read(path)
    if any(study.get(key) != value for key, value in spec.items()):
        raise ValueError('Study does not match its frozen specification')
    return spec


def candidates(parent):
    rows = copy.deepcopy(parent['candidates']); jobs = []
    for name in rows:
        standalone = 'KALCB' if name == 'kalcb_native' else 'OLR' if name.startswith('olr_') else None
        jobs.extend({'recipe': name, 'scenario': s, 'standalone': standalone, 'group': 'bridge'} for s in SCENARIOS)
    policy = copy.deepcopy(rows['naive_native_combination']['policy'])
    def add(name, k, o, recycle, standalone=None, group='search'):
        r = copy.deepcopy(rows['olr_updated' if recycle else 'olr_median'])
        r['policy'] = copy.deepcopy(policy)
        r['native_budget'] = {'kalcb_sizing_fraction': k, 'olr_fraction': o}
        rows[name] = r
        jobs.extend({'recipe': name, 'scenario': s, 'standalone': standalone, 'group': group} for s in SCENARIOS)
    for k in (.5, .75, 1.):
        for o in (.35, .5, .75, 1.):
            for recycle in (False, True):
                add(f'k{int(k*100):03}_o{int(o*100):03}_r{int(recycle)}', k, o, recycle)
    for recycle in (False, True): add(f'migration_k200_o020_r{int(recycle)}', 2., .2, recycle)
    for k in (.5, .75, 1., 2.): add(f'control_k{int(k*100):03}', k, 1., False, 'KALCB', 'standalone')
    for o in (.35, .5, .75, 1.):
        for recycle in (False, True):
            add(f'control_o{int(o*100):03}_r{int(recycle)}', 1., o, recycle, 'OLR', 'standalone')
    return rows, jobs


def freeze(parent_path, out, recipes, jobs, *, capital=25_000_000, overlay=True, registration=None,
           overlay_files=(), native_source_unchanged=True, data_feed_unchanged=True):
    parent = read(parent_path); m = read(parent['manifest']); identity = m.pop('manifest_hash')
    if signature(m) != identity: raise ValueError('Changed parent manifest')
    spec = verified_spec(parent)
    owned = tuple(dict.fromkeys((*OWNED, *overlay_files)))
    spec_name = 'research/portfolio_native_budget/study.json'
    out.mkdir(parents=True, exist_ok=False); root = out/'inputs/files'; root.mkdir(parents=True)
    qname = m['evidence']['qualification_path']; q = read(Path(parent['code'])/qname)
    declarations = q['capacity']['registered_execution_policies']
    def key(row): return signature({k: row[k] for k in ('initial_equity','buying_power_leverage','policy')})
    seen = {key(r) for r in declarations}
    for r in list(declarations):
        new = {**copy.deepcopy(r), 'initial_equity': capital, 'account': 'native_budget_research'}
        if key(new) not in seen: declarations.append(new); seen.add(key(new))
    for name in list(m['files']):
        # The child specification is rewritten below. Never hardlink it first:
        # writing through that link would mutate every ancestor and sibling.
        if name == spec_name:
            continue
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        if name == qname: write_json(target, q)
        elif overlay and name in owned: shutil.copyfile(REPO/name, target)
        else: os.link(Path(parent['code'])/name, target)
        m['files'][name] = digest(target)
    if overlay:
        for name in owned:
            if name not in m['files']:
                target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/name, target); m['files'][name] = digest(target)
    spec['candidates'] = copy.deepcopy(recipes)
    spec['initial_equity'] = capital; spec['configuration']['initial_equity'] = capital
    for recipe in spec['candidates'].values():
        p = recipe['policy']; ratio = capital/p['notional_cap_reference_equity']
        for key in ('max_gross_notional','max_symbol_notional','max_sector_notional'): p[key] *= ratio
        p['notional_cap_reference_equity'] = capital
    spec['native_budget_registration'] = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent_manifest': identity, 'selection': SELECTION, 'native_strategy_source_unchanged': native_source_unchanged,
        'data_and_feed_unchanged': data_feed_unchanged, 'research_only': True}
    name = spec_name; write_json(root/name, spec); m['files'][name] = digest(root/name)
    m['evidence']['portfolio_current_study'] = name
    program = 'from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    source = subprocess.check_output([sys.executable,'-B','-c',program],cwd=root,text=True).strip()
    m['source_generation_hash'] = source
    write_json(out/'inputs/manifest.json', {**m,'manifest_hash':signature(m)})
    write_json(out/'study.json', {**parent, **spec, 'code':str(root.resolve()), 'spec_path':name,
        'manifest':str((out/'inputs/manifest.json').resolve()), 'source_generation_hash':source})
    write_json(out/'registration.json', {'parent_study':str(Path(parent_path).resolve()),
        'parent_study_sha256':digest(parent_path), 'parent_manifest':identity,
        'source_generation_hash':source, 'changed_files':{n:digest(root/n) for n in owned},
        'capital':capital, 'jobs':jobs, 'selection':SELECTION, **(registration or {})})
    print(json.dumps({'prepared':str(out),'cases':len(jobs),'source':source}),flush=True)


def checked(out, full=False):
    c = read(out/'completion.json'); r = read(out/'result.json'); a = read(out/'execution_audit.json')
    request=read(out/'request.json'); arguments=request['arguments'];study=read(arguments['study'])
    verified_spec(study)
    if (request['study_hash']!=digest(arguments['study'])
            or c['source_generation_hash']!=study['source_generation_hash']
            or r['source_generation_hash']!=study['source_generation_hash']
            or r['recipe']!=arguments['recipe'] or r['scenario']!=arguments['scenario']
            or r['standalone']!=arguments.get('standalone')
            or request['recipe']!=study['candidates'][arguments['recipe']]):
        raise ValueError('Unbound replay request: '+str(out))
    if (not c['complete'] or c['result_sha256'] != digest(out/'result.json')
        or c['audit_sha256'] != digest(out/'execution_audit.json')
        or not r['execution_validity_pass'] or not a['execution_validity_pass']
        or (full and r['audit_status'] != 'passed') or r['terminal_inventory']
        or r['terminal_pending'] or r['intraday_residual_sessions']):
        raise ValueError('Invalid replay: '+str(out))
    for name, sha in c.get('artifacts',{}).items():
        if digest(out/name) != sha: raise ValueError('Changed output: '+str(out/name))
    return r


def one(root, job, full=False, folder='cases'):
    study = read(root/'study.json'); name = job['recipe']+'_'+job['scenario']; out = root/folder/name
    if (out/'completion.json').exists(): return checked(out,full)
    if out.exists(): raise FileExistsError('Preserve incomplete replay: '+str(out))
    log = root/'logs'/(folder+'_'+name+'.log'); log.parent.mkdir(exist_ok=True)
    command = [sys.executable,'-B','-u',str(Path(study['code'])/'scripts/portfolio_synergy_study.py'),
        'replay','--study',str(root/'study.json'),'--recipe',job['recipe'],'--scenario',job['scenario'],'--output',str(out)]
    if full: command += ['--full-audit']
    if job.get('standalone'): command += ['--standalone',job['standalone']]
    if job.get('no_upward_compounding'): command += ['--no-upward-compounding']
    print(json.dumps({'started':name,'folder':folder}),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        done = subprocess.run(command,cwd=study['code'],stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    if done.returncode: raise RuntimeError('Replay failed: '+str(log))
    value = checked(out,full)
    print(json.dumps({'completed':name,**{k:value[k] for k in ('return_pct','drawdown_pct','trades','audit_status')}}),flush=True)
    return value


def bridge(root):
    parent = Path(read(root/'registration.json')['parent_study']).parent
    evidence = {}
    for j in read(root/'registration.json')['jobs']:
        if j.get('group') != 'bridge': continue
        name=j['recipe']+'_'+j['scenario']; a=root/'cases'/name; b=parent/'cases'/name
        checked(a); checked(b)
        hashes={f: (digest(a/f),digest(b/f)) for f in ('trades.json','daily.json','equity.json')}
        if any(x!=y for x,y in hashes.values()): raise ValueError('Native source parity failed: '+name)
        evidence[name]={'current':str(a),'parent':str(b),'path_hashes':{f:x for f,(x,y) in hashes.items()}}
    if not evidence: raise ValueError('No bridge cases')
    write_json(root/'bridge.json',evidence)
    return evidence


def run(root, group, workers, full=False):
    jobs = read(root/'registration.json')['jobs']
    if group == 'smoke': jobs=[j for j in jobs if j['recipe'] in {'control_k100','control_o100_r1'}]
    elif group != 'all': jobs=[j for j in jobs if j.get('group')==group]
    first = {'control_k100', 'control_o100_r1'}
    jobs = sorted(jobs, key=lambda j: 0 if j['recipe'] in first else 1)
    if group in ('all','search','standalone'): bridge(root)
    errors={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,root,j,full):j for j in jobs}
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:
                name=futures[f]['recipe']+'_'+futures[f]['scenario'];errors[name]=str(e)
                print(json.dumps({'error':name,'message':str(e)}),flush=True)
    write_json(root/('run_'+group+'.json'),{'errors':errors,'jobs':jobs})
    if errors: raise RuntimeError('Failed research cases preserved')


def select(root):
    study=read(root/'study.json'); registration=read(root/'registration.json'); results={}; groups={}
    rule=registration['selection']
    if rule!=study['native_budget_registration']['selection']:raise ValueError('Changed selection registration')
    for j in registration['jobs']:
        if j.get('group')=='bridge':continue
        name=j['recipe']; groups[name]=j['group']
        results.setdefault(name,{})[j['scenario']]=checked(root/'cases'/(name+'_'+j['scenario']))
    ordered={n:[r[s] for s in SCENARIOS] for n,r in results.items()}
    inherited = registration.get('standalone_reference_study')
    if inherited:
        parent_path = Path(inherited['path'])
        if digest(parent_path) != inherited['sha256']: raise ValueError('Changed standalone reference study')
        parent = read(parent_path)
        if parent['source_generation_hash'] != study['source_generation_hash']: raise ValueError('Reference source mismatch')
        for j in read(parent_path.parent/'registration.json')['jobs']:
            if j['group'] == 'standalone':
                n=j['recipe']; groups[n]='standalone'
                if n not in ordered: ordered[n]=[None,None]
                ordered[n][SCENARIOS.index(j['scenario'])]=checked(parent_path.parent/'cases'/(n+'_'+j['scenario']))
    def risk(rows):return all(r['drawdown_pct']<=cap for r,cap in zip(rows,(rule['baseline_max_dd_pct'],rule['combined_max_dd_pct'])))
    refs={n:r for n,r in ordered.items() if groups[n]=='standalone' and risk(r)}
    if not refs:raise ValueError('No risk-feasible standalone control')
    floors=[max(r[i]['return_pct'] for r in refs.values()) for i in (0,1)]
    feasible={n:r for n,r in ordered.items() if groups[n]!='standalone' and risk(r)
        and all(x['return_pct']>f and x['trades']>=t for x,f,t in zip(r,floors,rule['frequency_floors']))}
    priority_support=None
    if registration.get('minimum_priority_collision_sessions'):
        parent=registration['retain_parent_on_exact_tie'];days=set()
        for scenario in SCENARIOS:
            case=root/'cases'/(parent+'_'+scenario)
            with gzip.open(case/'allocation_events.jsonl.gz','rt',encoding='utf-8') as stream:
                stamps={e['submitted_order_id']:e['timestamp'] for line in stream if (e:=json.loads(line))['event']=='native_submission'}
            with gzip.open(case/'native_budget_admissions.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    r=json.loads(line);stamp=datetime.fromisoformat(r['order']['submitted_at']).isoformat()
                    group=[a for oid,a in r['ledger']['allocations'].items() if stamps.get(oid)==stamp]
                    if len({a['strategy_id'] for a in group})>1 and any(a['qty']<a['requested_qty'] for a in group):days.add(stamp[:10])
        priority_support={'distinct_potential_conflict_sessions':len(days),'days':sorted(days),
            'minimum':registration['minimum_priority_collision_sessions'],
            'definition':'Both strategies have active buys from the same submission time, and at least one native request is not fully reserved. Sessions deduplicated across stress scenarios. Necessary opportunity support, not proof of predictive value.'}
        if len(days)<priority_support['minimum']:feasible={n:r for n,r in feasible.items() if n==parent}
    def vector(rows):return [rows[0]['return_pct'],rows[1]['return_pct'],min(r['trades'] for r in rows),
                            -rows[0]['drawdown_pct'],-rows[1]['drawdown_pct']]
    def dominates(a,b):return all(x>=y for x,y in zip(a,b)) and any(x>y for x,y in zip(a,b))
    pareto={n:r for n,r in feasible.items() if not any(dominates(vector(v),vector(r)) for m,v in feasible.items() if m!=n)}
    best=max((r[1]['return_pct'] for r in pareto.values()),default=0)
    tied={n:r for n,r in pareto.items() if r[1]['return_pct']>=best-rule['stress_return_tie_band_pp']}
    winner=max(tied,key=lambda n:(min(r['trades'] for r in tied[n]),-max(r['drawdown_pct'] for r in tied[n]),tied[n][1]['return_pct'],n)) if tied else None
    if winner and registration.get('prefer_tighter_risk_cap_on_exact_tie'):
        equivalent = [n for n in tied if all(tied[n][i][k] == tied[winner][i][k]
            for i in (0, 1) for k in ('return_pct', 'drawdown_pct', 'trades'))]
        guarded = [n for n in equivalent if study['candidates'][n]['native_budget'].get('max_kalcb_stop_risk_fraction', 0) > 0]
        if guarded:
            winner = min(guarded, key=lambda n: study['candidates'][n]['native_budget']['max_kalcb_stop_risk_fraction'])
    unchanged=registration.get('retain_parent_on_exact_tie')
    if unchanged in tied and winner and all(
            tied[unchanged][i][k]==tied[winner][i][k] for i in (0,1) for k in ('return_pct','drawdown_pct','trades')):
        winner=unchanged
    value={'winner':winner,'recipe':study['candidates'].get(winner),'source_generation_hash':study['source_generation_hash'],
        'selection':rule,'return_floors':floors,'frequency_floors':rule['frequency_floors'],
        'feasible':list(feasible),'pareto':list(pareto),'standalone_feasible':list(refs),'results':ordered,
        'priority_support':priority_support,
        'source_bridge':read(root/'bridge.json') if (root/'bridge.json').exists() else None}
    write_json(root/'finalist.json',value)
    print(json.dumps({k:value[k] for k in ('winner','return_floors','feasible','pareto','standalone_feasible')}),flush=True)
    return value


def prepare_risk(root, out):
    f=read(root/'finalist.json'); s=read(root/'study.json')
    if not f['winner']:raise ValueError('No feasible base budget')
    recipes={f['winner']:f['recipe']}
    for cap in (.02,.025,.03):
        name=f['winner']+'_risk'+str(int(cap*1000))
        recipes[name]=copy.deepcopy(f['recipe'])
        recipes[name]['native_budget']['max_kalcb_stop_risk_fraction']=cap
    jobs=[{'recipe':n,'scenario':c,'group':'risk'} for n in recipes for c in SCENARIOS]
    freeze(root/'study.json',out,recipes,jobs,overlay=False,registration={
        'standalone_reference_study':{'path':str(root/'study.json'),'sha256':digest(root/'study.json')},
        'parent_finalist_sha256':digest(root/'finalist.json'),
        'prefer_tighter_risk_cap_on_exact_tie': True,
        'risk_tie_registration_utc': datetime.now(timezone.utc).isoformat(),
        'risk_tie_reason': 'Before any cap-challenge replay: when both scenario economics are exactly equal, retain the smallest positive tested risk-unit ceiling. This is a prudent forward admission guard, not evidence of historical risk reduction.',
        'purpose':'Predeclared three-level aggregate planned-stop-risk challenge; same gates and reference controls.'})
    if read(out/'study.json')['source_generation_hash']!=s['source_generation_hash']:
        raise ValueError('Risk challenge changed source')


def prepare_priority(root,out,protocol):
    declaration=read(protocol)
    if declaration['alternative']!=['OLR','KALCB']:raise ValueError('Unsupported priority challenge')
    f=read(root/'finalist.json');registration=read(root/'registration.json')
    if not f['winner']:raise ValueError('No feasible budget to challenge')
    if declaration['source_generation_hash'] != f['source_generation_hash']:
        raise ValueError('Priority protocol belongs to a different source generation')
    reference=registration.get('standalone_reference_study',{'path':str(root/'study.json'),'sha256':digest(root/'study.json')})
    recipes={f['winner']:f['recipe']};name=f['winner']+'_olr_priority'
    recipes[name]=copy.deepcopy(f['recipe']);recipes[name]['policy']['strategy_priority']=declaration['alternative']
    jobs=[{'recipe':n,'scenario':s,'group':'priority'} for n in recipes for s in SCENARIOS]
    freeze(root/'study.json',out,recipes,jobs,overlay=False,registration={
        'standalone_reference_study':reference,'parent_finalist_sha256':digest(root/'finalist.json'),
        'retain_parent_on_exact_tie':f['winner'],
        'minimum_priority_collision_sessions':declaration['minimum_distinct_conflict_sessions'],
        'priority_protocol':{'path':str(protocol.resolve()),'sha256':digest(protocol)},
        'purpose':'Change only cross-strategy precedence for simultaneous native orders. Chronology, signals and budget gates remain fixed; exact economic ties retain existing priority.'})


def perturb(root, out):
    f=read(root/'finalist.json')
    if not f['winner']:raise ValueError('No feasible finalist')
    out.mkdir(parents=True,exist_ok=False)
    registration={'parent_study':str(root/'study.json'),'parent_study_sha256':digest(root/'study.json'),
        'parent_finalist_sha256':digest(root/'finalist.json'),'selected':f['winner'],
        'capital_grid':[20_000_000,22_500_000,27_500_000,30_000_000],
        'auction_stress':{'auction_price15':{'auction_adverse_bps':15},
                          'auction_nonfill50':{'auction_nonfill_rate':.5}},
        'no_upward_compounding':['baseline','combined'],
        'scope':'Predeclared diagnostic robustness; no re-ranking or new search on these outputs.'}
    write_json(out/'registration.json',registration)
    for capital in registration['capital_grid']:
        recipes={f['winner']:f['recipe']};jobs=[{'recipe':f['winner'],'scenario':s,'group':'robustness'} for s in SCENARIOS]
        freeze(root/'study.json',out/('capital_'+str(capital)),recipes,jobs,capital=capital,overlay=False)
    recipes={}
    for label,stress in registration['auction_stress'].items():
        recipes[label]=copy.deepcopy(f['recipe']);recipes[label]['olr_execution_stress']=stress
    jobs=[{'recipe':n,'scenario':s,'group':'robustness'} for n in recipes for s in SCENARIOS]
    freeze(root/'study.json',out/'auction',recipes,jobs,overlay=False)
    jobs=[{'recipe':f['winner'],'scenario':s,'group':'robustness','no_upward_compounding':True} for s in SCENARIOS]
    freeze(root/'study.json',out/'no_upward',{f['winner']:f['recipe']},jobs,overlay=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','bridge','select','qualify','risk','perturb','priority'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--group',choices=('all','bridge','search','standalone','risk','robustness','smoke','priority'),default='all')
    p.add_argument('--workers',type=int,choices=(1,2),default=2);p.add_argument('--full',action='store_true');a=p.parse_args()
    root=a.root.resolve()
    if a.command=='prepare':
        if not a.parent:p.error('prepare requires --parent')
        recipes,jobs=candidates(read(a.parent));freeze(a.parent,root,recipes,jobs)
    elif a.command=='run':run(root,a.group,a.workers,a.full)
    elif a.command=='bridge':print(json.dumps({'verified_cases':len(bridge(root))}))
    elif a.command=='select':select(root)
    elif a.command in ('risk','perturb'):
        if a.output is None:p.error('risk/perturb requires --output')
        (prepare_risk if a.command=='risk' else perturb)(root,a.output.resolve())
    elif a.command=='priority':
        if a.output is None or a.protocol is None:p.error('priority requires --output and --protocol')
        prepare_priority(root,a.output.resolve(),a.protocol)
    else:
        f=read(root/'finalist.json')
        if not f['winner']:raise ValueError('No passing finalist')
        for s in ('baseline','combined','cost','participation'):
            one(root,{'recipe':f['winner'],'scenario':s},True,'qualified')


if __name__=='__main__':main()
