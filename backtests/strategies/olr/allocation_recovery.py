"""Native OLR allocation experiments; PhaseRunner owns selection and acceptance."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import copy
import gzip
import json
import os
import shutil
import subprocess
import sys

from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment, ScoredCandidate, GateCriterion, EndOfRoundArtifacts
from backtests.strategies.kalcb.executed_value import digest, signature, write_json
from backtests.strategies.olr.capacity_qualification import checked_case, native_audit_valid

BASE = {'capacity_joint_allocation': True, 'capacity_exit_quantile': .5}
ROUTED = {**BASE, 'capacity_history_scope': 'orderable'}
SCHEDULED_RELEASE_PROFILES = {
    'median': BASE, 'routed': ROUTED,
    'sparse': {**ROUTED, 'capacity_sparse_exit_mode': 'continuous'},
    'redistribute': {**ROUTED, 'capacity_redistribute': True},
    'release30': {**ROUTED, 'capacity_release_minutes': 30},
    'release60': {**ROUTED, 'capacity_release_minutes': 60},
    'release90': {**ROUTED, 'capacity_release_minutes': 90},
    'release60half': {**ROUTED, 'capacity_release_minutes': 60, 'capacity_release_fraction': .5},
    'release60_sparse': {**ROUTED, 'capacity_release_minutes': 60, 'capacity_sparse_exit_mode': 'continuous'},
    'release60_redistribute': {**ROUTED, 'capacity_release_minutes': 60, 'capacity_redistribute': True},
    'release60_combined': {**ROUTED, 'capacity_release_minutes': 60, 'capacity_sparse_exit_mode': 'continuous', 'capacity_redistribute': True},
    'release90_combined': {**ROUTED, 'capacity_release_minutes': 90, 'capacity_sparse_exit_mode': 'continuous', 'capacity_redistribute': True},
}
ROTATION={**ROUTED,'capacity_rotation':True,'capacity_defer_minutes':10,
    'capacity_defer_min_fraction':1.0}
ROTATION_PROFILES = {
    'median':BASE, 'routed':ROUTED,
    'full_entry':{**ROUTED,'capacity_entry_bars':12},
    'full_entry_redistribute':{**ROUTED,'capacity_entry_bars':12,'capacity_redistribute':True},
    'rotation_half':ROTATION,
    'rotation_full':{**ROTATION,'capacity_rotation_fraction':1.0},
    'rotation_any_half':{**ROTATION,'capacity_rotation_filter':'any'},
    'rotation_rank1':{**ROTATION,'capacity_rotation_max_rank':1},
    'rotation_half_entry':{**ROTATION,'capacity_entry_bars':12},
    'rotation_full_entry':{**ROTATION,'capacity_entry_bars':12,'capacity_rotation_fraction':1.0},
    'rotation_half_entry_redistribute':{**ROTATION,'capacity_entry_bars':12,'capacity_redistribute':True},
}
CAPITALS = (10_000_000, 20_000_000, 50_000_000, 90_000_000, 100_000_000, 110_000_000)
TOPUP={**ROUTED,'capacity_entry_bars':12,'capacity_redistribute':True,
    'capacity_redistribution_evidence':'entry'}
ENTRY_TOPUP_PROFILES={
    'median':BASE,'routed':ROUTED,'full_entry':{**ROUTED,'capacity_entry_bars':12},
    'entry_top2_1p5':{**TOPUP,'capacity_redistribution_max_multiplier':1.5},
    'entry_top2_2':{**TOPUP,'capacity_redistribution_max_multiplier':2},
    'entry_top2_cap':TOPUP,
    'entry_top3_1p5':{**TOPUP,'capacity_redistribution_max_multiplier':1.5,'capacity_redistribute_max_rank':3},
    'entry_top3_2':{**TOPUP,'capacity_redistribution_max_multiplier':2,'capacity_redistribute_max_rank':3},
}
BROAD={**BASE,'capacity_broad_auction_history':True}
VOLUME_PROFILES={
    'median':BASE,'broad':BROAD,
    'broad_entry':{**BROAD,'capacity_entry_bars':12},
    'volume_sqrt':{**BROAD,'capacity_volume_exponent':.5},
    'volume_linear':{**BROAD,'capacity_volume_exponent':1},
    'volume_sqrt_entry':{**BROAD,'capacity_volume_exponent':.5,'capacity_entry_bars':12},
    'volume_linear_entry':{**BROAD,'capacity_volume_exponent':1,'capacity_entry_bars':12},
}
BUDGET_PROFILES={
    'median':BASE, 'broad':BROAD,
    'budget_half':{**BASE,'capacity_session_budget_fraction':.5},
    'budget_65':{**BASE,'capacity_session_budget_fraction':.65},
    'budget_80':{**BASE,'capacity_session_budget_fraction':.8},
    'broad_budget_half':{**BROAD,'capacity_session_budget_fraction':.5},
    'broad_budget_65':{**BROAD,'capacity_session_budget_fraction':.65},
    'broad_budget_80':{**BROAD,'capacity_session_budget_fraction':.8},
}
VALUE_PROFILES={
    'median':BASE,'broad':BROAD,
    'value_rank_half':{**BASE,'capacity_value_mode':'rank'},
    'value_rank_full':{**BASE,'capacity_value_mode':'rank','capacity_value_strength':1},
    'value_band_half':{**BASE,'capacity_value_mode':'band'},
    'value_band_full':{**BASE,'capacity_value_mode':'band','capacity_value_strength':1},
    'broad_value_rank_half':{**BROAD,'capacity_value_mode':'rank'},
    'broad_value_band_half':{**BROAD,'capacity_value_mode':'band'},
}
LEGACY_TOPUP={**BASE,'capacity_entry_bars':12,'capacity_redistribute':True,
    'capacity_redistribution_evidence':'entry'}
ISOLATION_PROFILES={**BUDGET_PROFILES,
    'median_auction_top2':{**BASE,'capacity_redistribute':True},
    'median_entry':{**BASE,'capacity_entry_bars':12},
    'median_entry_top2_1p5':{**LEGACY_TOPUP,'capacity_redistribution_max_multiplier':1.5},
    'median_entry_top2_2':{**LEGACY_TOPUP,'capacity_redistribution_max_multiplier':2},
    'median_entry_top3_1p5':{**LEGACY_TOPUP,'capacity_redistribution_max_multiplier':1.5,'capacity_redistribute_max_rank':3},
}
RECYCLE={**LEGACY_TOPUP,'capacity_redistribution_max_multiplier':1.5,'capacity_recycle_liquidity_only':True}
RECYCLING_PROFILES={
    'median':BASE,'broad':BROAD,
    'recycle_top2':RECYCLE,
    'recycle_top3':{**RECYCLE,'capacity_redistribute_max_rank':3},
    'broad_recycle_top2':{**RECYCLE,'capacity_broad_auction_history':True},
    'broad_recycle_top3':{**RECYCLE,'capacity_broad_auction_history':True,'capacity_redistribute_max_rank':3},
}
CONDITIONED={**BROAD,'capacity_history_scope':'conditional_shrink','capacity_sparse_exit_mode':'continuous'}
SIGNAL_CONDITIONAL_PROFILES={
    'median':BASE,'recycle_top2':RECYCLE,'recycle_top3':{**RECYCLE,'capacity_redistribute_max_rank':3},
    'conditional3':CONDITIONED,
    'conditional10':{**CONDITIONED,'capacity_history_prior_strength':10},
    'conditional3_recycle2':{**RECYCLE,**CONDITIONED},
    'conditional3_recycle3':{**RECYCLE,**CONDITIONED,'capacity_redistribute_max_rank':3},
}
EXIT_CONDITIONED={**CONDITIONED,'capacity_history_scope':'conditional_exit'}
EXIT_CONDITIONAL_PROFILES={
    'median':BASE,
    'conditional_exit3':EXIT_CONDITIONED,
    'conditional_exit10':{**EXIT_CONDITIONED,'capacity_history_prior_strength':10},
    'conditional_exit3_recycle2':{**RECYCLE,**EXIT_CONDITIONED},
}
EXTRA_ONLY={**BASE,'capacity_redistribute':True,
    'capacity_redistribution_evidence':'entry','capacity_redistribution_max_multiplier':1.5}
STRICT_NATIVE_PROFILES={
    'median':BASE,
    'native_entry_top2':EXTRA_ONLY,
    'native_entry_recycle2':{**EXTRA_ONLY,'capacity_recycle_liquidity_only':True},
    'native_auction_top2':{**BASE,'capacity_redistribute':True,'capacity_redistribution_max_multiplier':1.5},
}
AUCTION_RECYCLE={**BASE,'capacity_redistribute':True,
    'capacity_redistribution_max_multiplier':1.5,'capacity_recycle_liquidity_only':True}
PROFILES={
    'median':BASE,
    'native_auction_recycle2':AUCTION_RECYCLE,
    'native_auction_recycle3':{**AUCTION_RECYCLE,'capacity_redistribute_max_rank':3},
    'native_entry_recycle3':{**EXTRA_ONLY,'capacity_recycle_liquidity_only':True,
        'capacity_redistribute_max_rank':3},
}
SCENARIOS = ('baseline', 'combined')
CHANGED = ('strategy_olr/config.py', 'strategy_olr/research.py', 'strategy_olr/core/capacity.py',
    'strategy_olr/core/logic.py', 'strategy_olr/core/allocation.py', 'backtests/strategies/olr/runner.py',
    'strategy_olr/core/capital_rotation.py', 'scripts/portfolio_synergy_study.py',
    'strategy_olr/core/allocation_value.py',
    'backtests/strategies/olr/recovery_evidence.py',
    'scripts/audit_olr_allocation_value.py',
    'backtests/strategies/olr/allocation_recovery.py', 'backtests/strategies/olr/capacity_qualification.py',
    'scripts/olr_allocation_recovery.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(repo, parent_path, root):
    """Copy changed code and inherit immutable inputs; never edit a hardlink."""
    from dataclasses import replace
    from strategy_olr.config import OLRConfig
    parent = read(parent_path)
    request=read(Path(parent['code'])/parent['olr_request'])
    native_config=OLRConfig.from_mapping(request['config'],request['mutations'])
    for settings in PROFILES.values():
        if any(not key.startswith('capacity_') for key in settings):
            raise ValueError('Native replay accepts capacity overrides only')
        replace(native_config, **settings)
    manifest = read(parent['manifest'])
    expected = manifest.pop('manifest_hash')
    if signature(manifest) != expected:
        raise ValueError('Changed parent manifest')
    root.mkdir(parents=True, exist_ok=False)
    code = root/'inputs/files'; code.mkdir(parents=True)
    hashes = {}; changes = {}
    def copy_one(item):
        name,old=item
        dest = code/name; dest.parent.mkdir(parents=True, exist_ok=True)
        source = repo/name if name in CHANGED else Path(parent['code'])/name
        if name in CHANGED:
            shutil.copy2(source, dest);new=digest(dest)
            return name,new,{'parent':old,'new':new}
        if source.drive.lower()==dest.drive.lower():
            os.link(source,dest)
        else:
            # Local-SSD execution may use an external-drive frozen parent.
            # Copy identical bytes across volumes; never mutate the parent.
            shutil.copy2(source,dest)
        return name,old,None
    # Independent immutable hardlinks do not need serialized filesystem waits.
    # Changed source files remain physical copies, never writable hardlinks.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for name,value,change in pool.map(copy_one,manifest['files'].items()):
            hashes[name]=value
            if change is not None:changes[name]=change
    for name in CHANGED:
        if name not in hashes:
            dest=code/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo/name,dest); hashes[name]=digest(dest)
            changes[name]={'parent':None,'new':hashes[name]}
    base = read(Path(parent['code'])/parent['spec_path'])
    parent_reference='control' if 'control' in base['candidates'] else 'median'
    native = copy.deepcopy(base['candidates'][parent_reference])
    base['candidates'] = {name: {**copy.deepcopy(native), 'quality':False,
        'decision_rank':False, 'olr_capacity':settings} for name,settings in PROFILES.items()}
    base['initial_mutations'] = {'recipe':'median'}
    specs = {}
    for capital in CAPITALS:
        spec=copy.deepcopy(base); spec['initial_equity']=capital
        spec['configuration']['initial_equity']=capital
        spec['registration_revision']='olr-allocation-recovery-v1'
        name=f'research/olr_allocation_recovery/{capital}_{signature(spec)[:16]}.json'
        write_json(code/name,spec); hashes[name]=digest(code/name); specs[capital]=(name,spec)
    program='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    source=subprocess.check_output([sys.executable,'-B','-c',program],cwd=code,text=True).strip()
    payload={**manifest,'files':hashes,'source_generation_hash':source,
        'evidence':{**manifest['evidence'],'allocation_recovery_parent':expected}}
    write_json(root/'inputs/manifest.json',{**payload,'manifest_hash':signature(payload)})
    for capital,(name,spec) in specs.items():
        write_json(root/f'study_{capital}.json',{**parent,**spec,'manifest':str(root/'inputs/manifest.json'),
            'code':str(code),'spec_path':name,'source_generation_hash':source})
    write_json(root/'registration.json',{
        'registered_at_utc':datetime.now(timezone.utc).isoformat(), 'parent':str(parent_path),
        'source_generation_hash':source,'changes':changes,'profiles':PROFILES,
        'screening_capitals':[100_000_000], 'qualification_capitals':CAPITALS,'scenarios':SCENARIOS,
        'selection':{'anchor_return_gain_pp':{'baseline':3,'combined':2},
            'per_cell_return_floor':'at least median control',
            'per_cell_drawdown_ceiling':'min(10.75%, median drawdown + 1 percentage point)',
            'per_cell_frequency_floor':'.95 * median completed trades',
            'capital_gap':'10m to 100m return gap must not widen in either scenario',
            'support':'at least 10 distinct issuer-days with direct release orders or changed feasible allocations',
            'tie_break':'highest worst-capital stress return; equal scores retain earlier registered policy',
            'qualification':'all registered capital/stress cells; full source/execution audit for each accepted cell',
            'no_relaxation':True},
        'hypotheses':{'history':'Unrelated symbols cannot alter prior-auction observations',
            'sparse':'Unforecastable exits do not justify infeasible entry reservations',
            'release':'Real prior-session liquidation before entry reduces capital occupation; costs and foregone alpha retained',
            'rotation':'Confirmed rank-one/two funding shortage; prefer incumbent below current session VWAP; cancel acknowledgement then bounded real sale and entry revalidation',
            'entry_horizon':'Bound all requests by completed-volume estimate across actual remaining continuous entry window',
            'entry_supported_topup':'Permit native-preserving top-ups when current entry capacity is observed but auction history is sparse; cap by any available exit forecast, position risk and declared native-size multiple',
            'extra_only_feasibility':'Preserve the selected median native request and funded allocation. Apply the completed-volume remaining-window bound only to the total quantity allowed for optional extra shares; no native order is shrunk by this extra-only forecast. Compare entry evidence, auction evidence and liquidity-displacement recycling with the same fixed 1.5 native multiple.',
            'volume_conditioning':'Median prior-auction quantity multiplied by clipped completed-prefix relative-volume power; full sourced history prevents feed-subscription dependence. Current and future labels never enter the calculation.',
            'session_budget':'Limit cumulative new-session acquisition cost plus working BUY reserves using frozen session equity; old inventory exits normally. The 50% budget matches two overlapping vintages; 65% and 80% test less restrictive neighbors.',
            'causal_value':'Last 60 complete native trade lifecycles closed before the current session; minimum 30 total and 10 group samples; 20-observation shrinkage, 0.25 standard-error penalty, winsorized +/-1500bp labels. Rank or score-band grouping; fixed half/full strength bounded to 25%/50% size shifts, with unchanged 65% name ceiling and median liquidity cap. Missing support preserves native size.',
            'liquidity_recycling':'Top-ups can use only the current session request-funding displacement measured by paired native allocations before/after liquidity sizing. Preserve native allocations and actual cash constraints; do not replace low-capital behavior solely because spare cash exists. No capital-size threshold or future PnL labels.',
            'conditional_history':'Median prior auctions conditional on prior known OLR selections, shrunk by n/(n+3) or n/(n+10) toward the sourced broad median. Conditional observations start empty and accrue through shared live/replay state; broad history is seeded. Missing broad exit history uses observed entry feasibility without inventing an exit forecast.',
            'exit_target_alignment':'Conditional exit volume is observed on the next trading session after sealed OLR selection, matching the planned holding horizon. Never use the selection-day auction as if it were the next-day exit target; retain generation 12 as that separate ablation.',
            'redistribute':'Preserve native quantities and bound extras by measured liquidity and rank-two ceiling'},
        'research_lineage':'Sequential mechanism research; all previous frozen studies retained. Generation 15 shows unconditional auction top-ups fail small-capital return floors, while entry-supported liquidity recycling preserves small-capital results but has only four direct anchor issuer-days. Complete the mechanism interaction: liquidity-displacement recycling with auction evidence, and a previously declared rank-three neighbor for both evidence types. All preserve native funded quantities, actual cash and the fixed 1.5 native multiple. No account-capital cutoff, future outcome label, support gate or economic threshold is added or relaxed. This is reused development evidence, not an independent confirmation.',
        'scope':'reused historical development; independent forward evidence remains separate',
        'live_activated':False,'future_session_completion_gate':None})


def location(root,name,capital,scenario,full=False):
    return root/('qualified' if full else 'cases')/f'{name}_{capital}_{scenario}'


def run_one(root,name,capital,scenario,full=False):
    path=location(root,name,capital,scenario,full)
    if (path/'completion.json').exists():
        return checked_case(path)[0]
    if path.exists():
        raise FileExistsError('Preserving incomplete native case '+str(path))
    study=root/f'study_{capital}.json'; spec=read(study)
    cmd=[sys.executable,'-B','-u',str(Path(spec['code'])/'scripts/olr_capacity_observe.py'),
        '--study',str(study),'--output',str(path),'--recipe',name,'--scenario',scenario]
    if full:cmd.append('--full-audit')
    log=root/'logs'/f'{"qualified_" if full else ""}{path.name}.log';log.parent.mkdir(exist_ok=True)
    print(json.dumps({'started':path.name,'full_audit':full}),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        p=subprocess.run(cmd,cwd=spec['code'],stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    write_json(root/'receipts'/f'{"qualified_" if full else ""}{path.name}.json',{'command':cmd,'returncode':p.returncode})
    if p.returncode:
        raise RuntimeError(str(log))
    result,_=checked_case(path)
    print(json.dumps({'completed':path.name,**{k:result[k] for k in ('return_pct','drawdown_pct','trades')}}),flush=True)
    return result


def support(path):
    kinds={'release':set(),'redistribute':set(),'rotation':set(),'volume_conditioning':set(),'session_budget':set(),'causal_value':set(),'conditional_history':set()}
    with gzip.open(path/'decisions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            e=json.loads(line);m=e.get('metadata') or {};key=(e['timestamp'][:10],e['symbol'])
            if e['reason']=='scheduled_capital_release' and e['decision_code'] in ('EXIT_SUBMITTED','PARTIAL_EXIT_SUBMITTED'):
                kinds['release'].add(key)
            if e['reason']=='opportunity_capital_rotation' and e['decision_code']=='CAPITAL_ROTATION_SUBMITTED':
                kinds['rotation'].add(key)
            # A later fill can retain its original allocation metadata. Its
            # later date is not another direct allocation opportunity.
            if e['decision_code'] not in ('ENTRY_SUBMITTED','ENTRY_SKIPPED'):
                continue
            if m.get('allocation_redistributed_qty',0)>0:kinds['redistribute'].add(key)
            if m.get('capacity_conditioned_qty_delta',0):kinds['volume_conditioning'].add(key)
            if m.get('capacity_budget_qty_delta',0):kinds['session_budget'].add(key)
            if m.get('capacity_value_qty_delta',0):kinds['causal_value'].add(key)
            if m.get('capacity_history_qty_delta',0):kinds['conditional_history'].add(key)
    return {'by_mechanism':{k:len(v) for k,v in kinds.items()},'distinct_issuer_days':len(set.union(*kinds.values()))}


def cell_failures(candidate,control,anchor=False):
    fail=[]
    if candidate['return_pct'] < control['return_pct']-1e-9:fail.append('return')
    if candidate['drawdown_pct'] > min(10.75,control['drawdown_pct']+1)+1e-9:fail.append('drawdown')
    if candidate['trades'] < .95*control['trades']:fail.append('frequency')
    if not candidate['execution_validity_pass'] or candidate['terminal_inventory'] or candidate['terminal_pending'] or candidate.get('terminal_deferred_intents',0) or candidate.get('terminal_rotation_intents',0):
        fail.append('execution')
    if anchor and candidate['return_pct']-control['return_pct'] < (3 if candidate['scenario']=='baseline' else 2):
        fail.append('material_gain')
    return fail


class AllocationRecoveryPlugin:
    name='olr_allocation_recovery'; num_phases=1;ultimate_targets={}
    initial_mutations={'recipe':'median'}

    def __init__(self,root,mode='screen',workers=2):
        self.root=Path(root);self.mode=mode;self.workers=workers
        self.registration=read(self.root/'registration.json')
        self.study=read(self.root/'study_100000000.json')
        self.capitals=(100_000_000,) if mode=='screen' else CAPITALS
        self.names=([n for n in self.registration['profiles'] if n!='median'] if mode=='screen' else
            read(self.root/('qualify_verdict.json' if mode=='accept' else 'screen_verdict.json'))['eligible'])
        self.config={'artifact_promotion_policy':'historical_research_only'}
        self.execution_context={'source_fingerprint':self.study['source_generation_hash'],
            'shared_decision_core':'live_shared_core','capability_level':'real_replay',**self.config}

    def canonicalize_mutations(self,m):
        name=m.get('recipe','median')
        if name not in ['median',*self.names]:raise ValueError('Unregistered finalist')
        return {'recipe':name}

    def batch(self,names):
        jobs=[(n,c,s) for n in dict.fromkeys(names) for c in self.capitals for s in SCENARIOS]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures={pool.submit(run_one,self.root,n,c,s,self.mode=='accept'): (n,c,s) for n,c,s in jobs}
            for future in as_completed(futures):future.result()

    def get_phase_spec(self,phase,state):
        return PhaseSpec('Native allocation mechanisms across capital and stressed execution',
            [Experiment(n,{'recipe':n}) for n in self.names],
            lambda m:[GateCriterion('registered_economic_and_integrity_gates',1,float(m['qualified']),m['qualified'])],
            None,{},PhaseAnalysisPolicy(focus_metrics=['worst_stress_return','worst_baseline_return']),
            max_rounds=1,phase_metric_basis='native_historical_development',
            primary_promotion_metric='worst_stress_return',official_metric_keys=('worst_stress_return',),
            promotion_requires_audit_pass=False)

    def evaluate(self,name):
        records={};failures=[];comparisons=[];audits=[]
        for capital in self.capitals:
            for scenario in SCENARIOS:
                path=location(self.root,name,capital,scenario,self.mode=='accept')
                r,a=checked_case(path);b,_=checked_case(location(self.root,'median',capital,scenario,self.mode=='accept'))
                audits.append(a)
                bad=cell_failures(r,b,capital==100_000_000) if name!='median' else []
                if r['source_generation_hash']!=self.study['source_generation_hash']:bad.append('source')
                if self.mode=='accept':
                    if not native_audit_valid(a,path):bad.append('full_source_execution_audit')
                    if self.registration['profiles'][name].get('capacity_value_mode','off')!='off':
                        from scripts.audit_olr_allocation_value import audit as audit_value
                        if not audit_value(path)['pass']:bad.append('native_matured_value_reconciliation')
                    reference=location(self.root,name,capital,scenario)
                    if any(read(path/f)!=read(reference/f) for f in ('trades.json','equity.json','daily.json')):
                        bad.append('audited_economic_parity')
                failures += [f'{capital}_{scenario}_{f}' for f in bad]
                records[(capital,scenario)]=r
                comparisons.append({'capital':capital,'scenario':scenario,'failures':bad,
                    'result':{k:r[k] for k in ('return_pct','drawdown_pct','trades')},
                    'control':{k:b[k] for k in ('return_pct','drawdown_pct','trades')}})
        evidence=support(location(self.root,name,100_000_000,'baseline',self.mode=='accept'))
        if name!='median' and evidence['distinct_issuer_days']<10:failures.append('direct_support')
        if self.mode!='screen' and name!='median':
            for s in SCENARIOS:
                gap=records[(10_000_000,s)]['return_pct']-records[(100_000_000,s)]['return_pct']
                old=checked_case(location(self.root,'median',10_000_000,s))[0]['return_pct']-checked_case(location(self.root,'median',100_000_000,s))[0]['return_pct']
                if gap>old+1e-9:failures.append(s+'_wider_gap')
        metrics={'qualified':not failures,'research_only':True,'audit_status':'passed' if self.mode=='accept' and not failures else 'unverified',
            'source_fingerprint':self.study['source_generation_hash'],
            'feature_manifest_hash':read(self.study['manifest'])['manifest_hash'],
            'candidate_snapshot_hash':a['audit_receipt']['identity']['qualification_snapshot_hash'],
            'official_metric_basis':'SimBroker.equity_curve_bar_level_mtm',
            'worst_stress_return':min(records[(c,'combined')]['return_pct'] for c in self.capitals),
            'worst_baseline_return':min(records[(c,'baseline')]['return_pct'] for c in self.capitals),
            }
        from backtests.strategies.olr.recovery_evidence import hygiene_metrics
        metrics.update(hygiene_metrics(audits,records.values()))
        record={'recipe':name,'metrics':metrics,'failures':failures,'comparisons':comparisons,
            'support':evidence,'score':100+metrics['worst_stress_return']}
        write_json(self.root/(self.mode+'_comparisons')/f'{name}.json',record)
        return record

    def create_evaluate_batch(self,phase,cumulative_mutations,**kw):
        def evaluate(candidates,current):
            names=[self.canonicalize_mutations({**current,**c.mutations})['recipe'] for c in candidates]
            self.batch(['median',*names])
            out=[]
            for c,name in zip(candidates,names):
                r=self.evaluate(name)
                out.append(ScoredCandidate(c.name,r['score'],bool(r['failures']),','.join(r['failures']),r['metrics']))
            return out
        return evaluate

    def compute_final_metrics(self,mutations):
        name=self.canonicalize_mutations(mutations)['recipe'];self.batch(['median',name])
        return self.evaluate(name)['metrics']

    def run_phase_diagnostics(self,phase,state,metrics,greedy_result):
        return json.dumps(self.evaluate(greedy_result.final_mutations['recipe']),indent=2)

    run_enhanced_diagnostics=run_phase_diagnostics

    def build_end_of_round_artifacts(self,state):
        rows=[read(p) for p in sorted((self.root/(self.mode+'_comparisons')).glob('*.json'))]
        result={'complete':True,'choice':state.cumulative_mutations,
            'eligible':[r['recipe'] for r in rows if r['recipe']!='median' and not r['failures']],
            'trials':rows,'independent_evidence':False,'live_activated':False}
        write_json(self.root/(self.mode+'_verdict.json'),result)
        text=json.dumps(result,indent=2)
        return EndOfRoundArtifacts(text,{'allocation_mechanisms':text},
            'Native development comparison complete; acceptance requires full audited qualification')
