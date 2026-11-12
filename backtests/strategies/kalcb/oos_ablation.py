"""Finite, non-promoting IS/OOS mechanism diagnosis via the shared PhaseRunner."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager
from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment, ScoredCandidate, GateCriterion, EndOfRoundArtifacts
from strategy_kalcb.models import KALCBDailySnapshot

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')


def candidate_quantity(candidate, mode):
    """Only previously completed daily ATR and prior close enter the scale."""
    old=float(candidate.metadata.get('action_quantity_scale',1))
    if mode=='q1_cap_half':return min(old,.5)
    if mode.startswith('atr_quantity_'):
        cap={'atr_quantity_4pct':.04,'atr_quantity_5pct':.05}[mode]
        if candidate.prior_day_close<=0 or candidate.daily_atr<=0:
            raise ValueError('ATR quantity needs prior, positive price/ATR')
        return old*min(1,cap/(candidate.daily_atr/candidate.prior_day_close))
    return old


def transform_snapshot(raw,mode,excluded=()):
    snapshot=KALCBDailySnapshot.from_json_dict(raw)
    if snapshot.artifact_hash!=raw['artifact_hash']:raise ValueError('Changed parent snapshot')
    candidates=[]
    for c in snapshot.candidates:
        if c.symbol in excluded:continue
        q=candidate_quantity(c,mode)
        old=float(c.metadata.get('action_quantity_scale',1))
        if q!=old:c=replace(c,metadata={**c.metadata,'action_quantity_scale':q,
            'diagnostic_original_quantity':old,'diagnostic_quantity_rule':mode})
        candidates.append(c)
    metadata=deepcopy(snapshot.metadata)
    if excluded:
        metadata['active_symbols']=[s for s in metadata.get('active_symbols',[]) if s not in excluded]
        metadata['active_symbol_count']=len(metadata['active_symbols'])
        for key in ('candidate_pool_count','selection_count','frontier_symbol_count'):
            if key in metadata:metadata[key]=len(candidates)
    return replace(snapshot,candidates=tuple(candidates),metadata=metadata).to_json_dict()


class KALCBOOSAblationPlugin:
    name='kalcb_oos_diagnostic_ablation'
    num_phases=1
    ultimate_targets={}
    initial_mutations={'recipe':'control'}
    requires_full_diagnostics=True

    def __init__(self,study):
        self.study=Path(study).resolve();self.spec=read(self.study);self.root=self.study.parent
        self.study_hash=digest(self.study);self.code=Path(self.spec['code']);self.records={}
        self.config={'initial_equity':100_000_000,'require_full_diagnostics':True,
            'promotion_status':'diagnostic_only_control_retained','artifact_promotion_policy':'No selection or promotion from reused OOS diagnosis',
            'live_parity_fill_timing':'next_5m_open','auction_mode':'non_auction_continuous'}
        self.execution_context={'source_fingerprint':self.spec['source_generation_hash'],
            'capability_level':'real_replay','live_parity_fill_timing':'next_5m_open',
            'replay_mode':'native_is_oos_diagnostic_ablation','study_sha256':self.study_hash,
            'evaluation_role':'reused_development_including_previously_tested_june_july'}
        for item in self.spec['bound_inputs']+self.spec['source_programs']:
            if digest(item['path'])!=item['sha256']:raise ValueError('Changed registered study input')

    def canonicalize_mutations(self,mutations):
        m={**self.initial_mutations,**mutations}
        if set(m)!={'recipe'} or m['recipe'] not in self.spec['recipes']:raise ValueError('Undeclared diagnostic recipe')
        return m

    def get_phase_spec(self,phase,state):
        return PhaseSpec('Paired removal and local perturbation evidence; retain control, do not optimize on the observed OOS',
            [Experiment(name,{'recipe':name}) for name in self.spec['recipes'] if name!='control'],
            lambda m:[GateCriterion('physical_execution',1,int(m['physical_valid']),m['physical_valid'])],
            None,{},PhaseAnalysisPolicy(focus_metrics=['oos_baseline_return','oos_combined_return','is_baseline_return','is_combined_return']),
            max_rounds=1,phase_metric_basis='native_paired_research_screening',primary_promotion_metric='diagnostic_score',
            official_metric_keys=('oos_baseline_return','oos_combined_return'),promotion_requires_audit_pass=False)

    def paths(self,name,period):
        recipe=self.spec['recipes'][name];source=self.spec['periods'][period]
        folder=self.root/'cases'/name/period;folder.mkdir(parents=True,exist_ok=True)
        config=read(source['config']);config.update(recipe['config'])
        cp=folder/'config.json';plan=read(source['plan']);old=plan['config']
        if cp.exists() and read(cp)!=config:raise ValueError('Changed diagnostic configuration')
        if not cp.exists():write(cp,config)
        plan['input_sha256'].pop(old);plan['config']=str(cp.resolve());plan['input_sha256'][plan['config']]=digest(cp)
        plan['diagnostic_recipe']={'name':name,'period':period,'study_sha256':self.study_hash}
        pp=folder/'plan.json'
        if pp.exists() and read(pp)!=plan:raise ValueError('Changed diagnostic plan')
        if not pp.exists():write(pp,plan)
        policies=self.root/'policies'/period/recipe['policy'] if recipe['policy']!='control' else Path(source['policies'])
        return folder,pp,policies

    def run_native(self,name,period,scenario):
        folder,plan,policies=self.paths(name,period);out=folder/scenario
        if not (out/'completion.json').exists():
            if out.exists():raise ValueError('Preserved incomplete native diagnostic: '+str(out))
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'NUMEXPR_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8',
                 'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(self.code)}
            command=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',str(plan),
                '--policies',str(policies),'--policy',self.spec['policy_name'],'--scenario',scenario,'--output',str(out)]
            log=folder/(scenario+'.log')
            print(json.dumps({'started':name,'period':period,'scenario':scenario}),flush=True)
            with log.open('x',encoding='utf-8') as stream:
                code=subprocess.run(command,cwd=self.code,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            write(folder/(scenario+'.receipt.json'),{'command':command,'returncode':code,'log':str(log)})
            if code:raise ValueError('Native diagnostic failed: '+str(log))
        r=read(out/'result.json');request=read(out/'request.json');done=read(out/'completion.json')
        expected=self.spec['source_generation_hash']
        if {r['source_generation_hash'],request['source_generation_hash'],done['source_generation_hash']}!={expected}:
            raise ValueError('Changed source generation in diagnostic cache')
        if request['plan']!=read(plan) or Path(request['arguments']['policies']).resolve()!=policies.resolve():
            raise ValueError('Changed plan or policy in diagnostic cache')
        if request['arguments']['policy']!=self.spec['policy_name'] or request['arguments']['scenario']!=scenario:
            raise ValueError('Changed native replay request')
        if not r['execution_validity_pass'] or r['terminal_inventory'] or r['terminal_pending'] or r['residual_sessions']:
            raise ValueError('Native physical failure: '+str(out))
        trades=read(out/'trades.json');daily=read(out/'daily.json')
        if abs(sum(t['net_pnl'] for t in trades)-(daily[-1]['equity']-100_000_000))>.01:
            raise ValueError('Diagnostic trade/cash reconciliation failed')
        if len(daily)!=self.spec['periods'][period]['sessions']:raise ValueError('Changed session coverage')
        if name=='control':
            original=read(self.spec['periods'][period]['references'][scenario])
            for key in ('return_pct','drawdown_pct','trades','active_days'):
                if abs(r[key]-original[key])>1e-8:raise ValueError('Control bridge drift: '+key)
        print(json.dumps({'completed':name,'period':period,'scenario':scenario,
            'return_pct':r['return_pct'],'drawdown_pct':r['drawdown_pct'],'trades':r['trades']}),flush=True)
        return r

    def record(self,name):
        if name in self.records:return self.records[name]
        metrics={'physical_valid':True,'diagnostic_score':0.0,'research_only':True,'audit_status':'unverified',
                 'promotion_status':'diagnostic_only_control_retained'}
        cases={}
        for period in ('is','oos'):
            for scenario in ('baseline','combined'):
                p=self.root/'cases'/name/period/scenario/'result.json';r=read(p)
                cases[period+'_'+scenario]={'path':str(p),'sha256':digest(p)}
                for a,b in [('return','return_pct'),('dd','drawdown_pct'),('trades','trades'),('active','active_days')]:
                    metrics[period+'_'+scenario+'_'+a]=r[b]
        record={'recipe':name,'changes':self.spec['recipes'][name],'metrics':metrics,'cases':cases}
        write(self.root/'cases'/name/'record.json',record);self.records[name]=record
        return record

    def evaluate_names(self,names):
        for name in names:
            for period in ('is','oos'):self.paths(name,period)
        jobs=[(name,p,s) for name in names for p in ('is','oos') for s in ('baseline','combined')]
        with ThreadPoolExecutor(max_workers=self.spec['workers']) as pool:
            list(pool.map(lambda job:self.run_native(*job),jobs))
        return [self.record(name) for name in names]

    def compute_final_metrics(self,mutations):
        name=self.canonicalize_mutations(mutations)['recipe']
        return (self.records[name] if name in self.records else self.evaluate_names([name])[0])['metrics']

    def create_evaluate_batch(self,phase,cumulative_mutations,**kwargs):
        def evaluate(candidates,current):
            names=[self.canonicalize_mutations({**current,**c.mutations})['recipe'] for c in candidates]
            records=self.evaluate_names(names)
            # Equal diagnostic scores deliberately retain the registered control.
            # All economic vectors are reported; no OOS-tuned winner is promoted.
            return [ScoredCandidate(c.name,0.0,metrics=r['metrics']) for c,r in zip(candidates,records)]
        return evaluate

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Study changed after registration')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        control=next(r for r in records if r['recipe']=='control')['metrics']
        keys=[p+'_'+s+'_'+m for p in ('is','oos') for s in ('baseline','combined') for m in ('return','dd','trades','active')]
        for r in records:
            r['delta_vs_control']={k:r['metrics'][k]-control[k] for k in keys}
            r['pareto_dominates_control_all_16_metrics']=all((r['metrics'][k]<=control[k] if k.endswith('_dd') else r['metrics'][k]>=control[k]) for k in keys) and any(r['metrics'][k]!=control[k] for k in keys)
        payload={'scope':'Reused-history diagnostic evidence. Native execution checks pass; variant full source qualification is not asserted.',
            'no_canonical_promotion':True,'all_scores_zero_control_retained':True,'new_recipe_count':len(records)-1,
            'study':{'path':str(self.study),'sha256':self.study_hash},'trials':records}
        write(self.root/'ablation_ledger.json',payload)
        return payload

    def run_phase_diagnostics(self,*args):return json.dumps(self.diagnostic(),indent=2)
    run_enhanced_diagnostics=run_phase_diagnostics

    def build_end_of_round_artifacts(self,state):
        report=json.dumps(self.diagnostic(),indent=2)
        return EndOfRoundArtifacts(report,{'is_oos_mechanism_effects':report},'Diagnostic round complete; canonical control retained')

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        payload=self.diagnostic();output_dir=Path(output_dir)
        write(output_dir/'round_final_full_diagnostics.json',payload)
        lines=['KALCB IS/OOS NATIVE DIAGNOSTIC ABLATION; NO PROMOTION',
            'All June-July variants are development tests on an already observed period.',
            'No variant has been qualified as an improved independent forward strategy.',
            'Returns/DD in percentage points; B=baseline, S=combined stress.',
            'Recipe | IS B return/DD/trades | IS S return/DD/trades | OOS B return/DD/trades | OOS S return/DD/trades']
        for r in payload['trials']:
            m=r['metrics'];parts=[r['recipe']]
            for p in ('is','oos'):
                for s in ('baseline','combined'):
                    stem=p+'_'+s;parts.append(f"{m[stem+'_return']:.4f}/{m[stem+'_dd']:.4f}/{m[stem+'_trades']}")
            lines.append(' | '.join(parts))
        (output_dir/'round_final_diagnostics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return {'complete':True,'promotion_status':'diagnostic_only_control_retained','trial_count':len(payload['trials']),
            'full_source_qualification_of_variants':False,'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();plugin=KALCBOOSAblationPlugin(args.study)
    manager=RoundManager('stock',plugin.name,base_dir=plugin.root/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Locked IS/OOS diagnostic ablation',
        round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()

if __name__=='__main__':main()
