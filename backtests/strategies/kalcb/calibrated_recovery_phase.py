"""Shared-PhaseRunner plugin for native calibrated-selection mechanism studies.

The plugin supplies finite candidate recipes and the existing native evaluator.
It does not implement an optimization lifecycle or an alternative fill engine.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,os,subprocess,sys
from backtests.auto.shared.plugin import PhaseSpec,PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment,ScoredCandidate,GateCriterion,EndOfRoundArtifacts

read=lambda p:json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str)+'\n')


def research_score(m):
    """Explicit soft risk/activity preferences, with no 8%/10% rejection gate."""
    return 100+.5*m['baseline_return']+m['combined_return']-1.5*(m['baseline_dd']+m['combined_dd'])+.02*min(m['baseline_trades'],m['combined_trades'])+.05*min(m['baseline_active'],m['combined_active'])


class KALCBCalibratedRecoveryPlugin:
    name='kalcb_calibrated_recovery'
    num_phases=3
    ultimate_targets={'baseline_dd':8.,'combined_dd':10.}
    initial_mutations={'management':'control','admission':'control','size':1.}

    def __init__(self,study):
        self.study=Path(study);self.spec=read(study);self.root=self.study.parent
        self.code=Path(self.spec['code']);self.cache={};self.config={'artifact_promotion_policy':'research_only_until_full_audit_and_independent_validation'}
        self.execution_context={'source_fingerprint':self.spec['source_generation_hash'],
            'shared_decision_core':'live_shared_core','live_parity_fill_timing':'next_5m_open',
            'primary_promotion_basis':'native_sim_broker_marked_equity','capability_level':'real_replay',
            'artifact_promotion_policy':self.config['artifact_promotion_policy']}
        if self.spec.get('registered_phases'):
            self.num_phases=len(self.spec['registered_phases'])
        self.initial_mutations={**self.initial_mutations,**self.spec.get('initial_mutations',{})}

    def canonicalize_mutations(self,m):return {**self.initial_mutations,**m}

    def get_phase_spec(self,phase,state):
        registered=self.spec.get('registered_phases')
        if registered:
            definition=registered[phase-1]
            candidates=[Experiment(c['name'],c['mutations']) for c in definition['candidates']]
        elif phase==1:
            recipes=[('floor_all',{'management':'floor_all'}),('retention',{'management':'retention'}),
                ('offset_all',{'admission':'offset_all'}),('affine_all',{'admission':'affine_all'})]
            # Each main effect is measured against the original calibrated control.
            candidates=[Experiment(n,{**self.initial_mutations,**m}) for n,m in recipes]
        elif phase==2:
            candidates=[Experiment(m+'__'+a,{'management':m,'admission':a,'size':1.})
                for m in ('floor_all','retention') for a in ('offset_all','affine_all')]
        else:
            candidates=[Experiment('size85',{'size':.85}),Experiment('size70',{'size':.7})]
        return PhaseSpec(focus=definition['focus'] if registered else {1:'Separate profit retention and calibration mechanisms',2:'Native account interactions',3:'Risk allocation and local sensitivity'}[phase],
            candidates=candidates,gate_criteria_fn=lambda m:[GateCriterion('physical_execution',1,float(m['physical_valid']),bool(m['physical_valid']))],
            scoring_weights=None,hard_rejects={},analysis_policy=PhaseAnalysisPolicy(focus_metrics=['baseline_return','combined_return','baseline_dd','combined_dd'],min_effective_score_delta_pct=0),
            max_rounds=1,phase_metric_basis='native_research_screening',primary_promotion_metric='combined_return',
            official_metric_keys=('baseline_return','combined_return'),promotion_requires_audit_pass=False)

    def paths(self,m):
        m=self.canonicalize_mutations(m)
        key=m['management']+'__'+m['admission']+'__'+str(m['size']).replace('.','p')
        folder=self.root/'cases'/key;folder.mkdir(parents=True,exist_ok=True)
        parent=read(self.spec['parent_plan']);config=read(parent['config'])
        changes=self.spec['management'][m['management']];config.update(changes);config['entry_quantity_multiplier']=m['size']
        cp=folder/'config.json';pp=folder/'plan.json'
        if cp.exists() and read(cp)!=config:raise ValueError('Changed immutable recipe')
        if not cp.exists():write(cp,config)
        old=parent['config'];parent['input_sha256'].pop(old);parent['config']=str(cp.resolve());parent['input_sha256'][parent['config']]=digest(cp)
        parent['calibrated_recovery_recipe']=m
        if pp.exists() and read(pp)!=parent:raise ValueError('Changed immutable plan')
        if not pp.exists():write(pp,parent)
        return key,folder,pp,self.spec['admission'][m['admission']]

    def evaluate(self,mutations):
        m=self.canonicalize_mutations(mutations);key,folder,plan,policy=self.paths(m)
        if key in self.cache:return self.cache[key]
        scenarios={};audits={};env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','TEMP':str(self.root/'temp'),'TMP':str(self.root/'temp')}
        (self.root/'temp').mkdir(exist_ok=True)
        for scenario in ('baseline','combined'):
            out=folder/scenario
            if not (out/'completion.json').exists():
                if out.exists():raise FileExistsError('Preserved incomplete native replay: '+str(out))
                command=[sys.executable,'-B','-u',str(self.code/'scripts/kalcb_executed_value_replay.py'),
                    '--plan',str(plan),'--policies',policy['directory'],'--policy',policy['name'],
                    '--scenario',scenario,'--output',str(out)]
                print(json.dumps({'replay_started':key,'scenario':scenario}),flush=True)
                with (folder/(scenario+'.log')).open('x') as log:
                    result=subprocess.run(command,cwd=self.code,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(folder/(scenario+'.receipt.json'),dict(command=command,returncode=result.returncode))
                if result.returncode:raise RuntimeError('Native replay failed: '+str(folder/(scenario+'.log')))
            request=read(out/'request.json');r=read(out/'result.json');done=read(out/'completion.json')
            if len({request['source_generation_hash'],r['source_generation_hash'],done['source_generation_hash'],self.spec['source_generation_hash']})!=1:raise ValueError('Changed replay source')
            if request['plan']!=read(plan) or Path(request['arguments']['policies']).resolve()!=Path(policy['directory']).resolve() or request['arguments']['policy']!=policy['name'] or request['arguments']['scenario']!=scenario:raise ValueError('Replay input mismatch')
            scenarios[scenario]=r;audits[scenario]=read(out/'execution_audit.json')
        valid=all(r['execution_validity_pass'] and not r['residual_sessions'] and not r['terminal_inventory'] and not r['terminal_pending'] for r in scenarios.values())
        metrics={'physical_valid':valid,'audit_status':'unverified','research_only':True,
            'official_metric_basis':'SimBroker.equity_curve_bar_level_mtm','primary_promotion_basis':'SimBroker.equity_curve_bar_level_mtm',
            'source_fingerprint':self.spec['source_generation_hash'],'feature_manifest_hash':self.spec['prepared_sha256'],
            'candidate_snapshot_hash':digest(Path(policy['directory'])/(policy['name']+'.snapshots.json')),
            'end_open_position_count':sum(len(r['terminal_inventory']) for r in scenarios.values())}
        for s,r in scenarios.items():
            metrics.update({s+'_return':r['return_pct'],s+'_dd':r['drawdown_pct'],s+'_trades':r['trades'],s+'_active':r['active_days']})
            if m==self.initial_mutations:
                reference=self.spec.get('control_results',{}).get(s)
                original=read(reference or Path(self.spec['control_accounts'])/('cv_interaction_calibrated_'+s)/'result.json')
                for field in ('return_pct','drawdown_pct','trades','active_days'):
                    if abs(r[field]-original[field])>1e-8:raise ValueError('Disabled-feature control drift: '+field)
        for field in ('same_bar_fill_count','forced_replay_close_count','rejected_order_count','invalid_order_rejection_count'):
            metrics[field]=sum(a[field] for a in audits.values())
        record=dict(key=key,mutations=m,metrics=metrics,score=research_score(metrics),plan=str(plan),policy=policy,
            scenarios={s:str(folder/s/'result.json') for s in scenarios},result_hashes={s:digest(folder/s/'result.json') for s in scenarios})
        write(folder/'case.json',record);self.cache[key]=record
        self.write_ledger()
        print(json.dumps({'replay_complete':key,'metrics':{k:v for k,v in metrics.items() if k.startswith(('baseline','combined'))},'physical_valid':valid}),flush=True)
        return record

    def create_evaluate_batch(self,phase,cumulative_mutations,**kwargs):
        def evaluate(candidates,current):
            result=[]
            for c in candidates:
                r=self.evaluate({**current,**c.mutations});m=r['metrics']
                result.append(ScoredCandidate(c.name,r['score'],not m['physical_valid'],'native_physical_failure' if not m['physical_valid'] else '',m))
            return result
        return evaluate

    def compute_final_metrics(self,mutations):return self.evaluate(mutations)['metrics']

    def write_ledger(self):
        records=[read(p) for p in sorted((self.root/'cases').glob('*/case.json'))]
        def obj(r):
            m=r['metrics'];return [m['baseline_return'],m['combined_return'],-m['baseline_dd'],-m['combined_dd'],m['baseline_trades'],m['combined_trades'],m['baseline_active'],m['combined_active']]
        valid=[r for r in records if r['metrics']['physical_valid']]
        for r in records:
            v=obj(r);r['pareto']=r['metrics']['physical_valid'] and not any(all(a>=b for a,b in zip(obj(o),v)) and any(a>b for a,b in zip(obj(o),v)) for o in valid)
        write(self.root/'trial_ledger.json',{'targets_are_preferences':True,'trials':records})

    def run_phase_diagnostics(self,phase,state,metrics,greedy_result):
        self.write_ledger()
        return json.dumps({'phase':phase,'metrics':metrics,'candidate_evaluations':greedy_result.candidate_evaluations,
            'scope':'Reused chronological development. Physical checks are screening evidence; full source/input audit pending. All Pareto tradeoffs retained.'},indent=2)

    run_enhanced_diagnostics=run_phase_diagnostics

    def build_end_of_round_artifacts(self,state):
        self.write_ledger();record=self.evaluate(state.cumulative_mutations)
        write(self.root/'screening_selection.json',record)
        report=json.dumps({'selected':record,'study':str(self.study),'trial_ledger':str(self.root/'trial_ledger.json'),
            'limits':'Historical research screening; full qualification and attribution required before delivery.'},indent=2)
        return EndOfRoundArtifacts(report,{'native_economics':report},'Research screening complete; no production promotion')
