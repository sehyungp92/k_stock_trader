"""C17 strategy plugin. Shared PhaseRunner owns every search/adoption decision."""
from copy import deepcopy
import json
from pathlib import Path
from backtests.auto.shared.plugin import PhaseSpec,PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment,GateCriterion,ScoredCandidate,EndOfRoundArtifacts,PhaseDecision
from backtests.strategies.common.plugin_base import attach_official_metric_contract
from .round3_contract_v1 import selectable_score
from .round3_native_io_v1 import bundle,read,write,bound,ROOT
from .qualified_artifacts import diagnostic_result
from .round3_reports_v1 import report


class C17RoundPlugin:
    name='kalcb';num_phases=4;ultimate_targets={};capability_level='real_replay';requires_full_diagnostics=True
    max_workers=2

    def __init__(self,service):
        self.service=service;self.initial_mutations=deepcopy(service.arms['v6_h0']['config'])
        self.initial_mutations_override=deepcopy(self.initial_mutations)
        self.source_fingerprint=service.control_receipts['is/baseline']['item']['source_generation_hash']
        self.config=dict(initial_equity=1e8,initial_mutations=self.initial_mutations,
            initial_mutations_path=service.control_receipts['is/baseline']['profile']['configuration'],
            baseline_integrity_required=True,baseline_integrity_tolerance_abs=1e-10,
            # The predecessor used IS return as primary. C16 expressly allows a
            # bounded IS sacrifice for a qualified paired ensemble improvement.
            # This plugin enforces that full rule at evaluation, phase and writer.
            previous_promotion_guard_required=False,promotion_status='C17_research_only',
            date_range=service.control_receipts['is/baseline']['profile']['date_range'],
            live_parity_fill_timing='next_5m_open',auction_mode='non_auction_continuous',
            capability_level='real_replay',risk_basis='full_native_MTM_with_finer_breach_veto',
            required_candidate_policy=service.required_policy,production_enabled=False,
            qualification_manifest=str(service.registration),max_workers=2)
        self.execution_context={**self.config,'source_fingerprint':self.source_fingerprint,
            'feature_manifest_hash':bound(service.registration)['sha256'],
            'candidate_snapshot_hash':service.control_receipts['is/baseline']['item']['policy']['sha256'],
            'expected_universe_size':103,'replay_mode':'C17_frozen_native_ensemble',
            'train_start':'2025-08-07','train_end':'2026-07-10','cost_policy':service.cost_policy}

    def canonicalize_mutations(self,mutations):
        config={**self.initial_mutations,**mutations}
        self.service.name_for_config(config) # Rejects hidden aliases/unregistered final settings.
        return config

    def get_phase_spec(self,phase,state):
        parent=self.canonicalize_mutations(state.cumulative_mutations);candidates=[]
        if phase==1:
            for name,arm in self.service.arms.items():
                if name.startswith('remove_') and self.service.reached(name):
                    candidates.append(Experiment(name,{'__C17_diagnostic__':name}))
        elif phase==2:
            for votes in (5,7):
                config=deepcopy(parent);config['entry_plan_routes'][0]['min_quality_votes']=votes
                name=self.service.name_for_config(config)
                if self.service.reached(name):candidates.append(Experiment(name,{'entry_plan_routes':config['entry_plan_routes']}))
        elif phase==3:
            for hold in (6,8):
                config={**parent,'research_opening_failure_min_hold_bars':hold};name=self.service.name_for_config(config)
                if self.service.reached(name):candidates.append(Experiment(name,{'research_opening_failure_min_hold_bars':hold}))
        elif phase==4:
            for name,arm in self.service.arms.items():
                if arm['selectable'] and self.service.reached(name):
                    candidates.append(Experiment(name,deepcopy(arm['config'])))
        else:raise ValueError('Unregistered phase')
        focus={1:'Existing component re-verification',2:'Pullback vote neighbourhood',
            3:'Opening failure hold neighbourhood',4:'Frozen finite composite and integrated qualification'}[phase]
        policy=PhaseAnalysisPolicy(focus_metrics=['c16_S','c16_R'],
            decide_action_fn=lambda *args:PhaseDecision('advance','Frozen finite family; no scoring redesign, extra arms or retry.'))
        return PhaseSpec(focus,candidates,lambda m:[GateCriterion('complete_C16_eligibility',1,
            int(m['C17_eligible']),bool(m['C17_eligible']))],None,{},policy,max_rounds=1,
            phase_metric_basis='audited_source_replay',primary_promotion_metric='c16_S',
            official_metric_keys=('c16_S','official_mtm_net_return_pct','max_drawdown_pct'),promotion_requires_audit_pass=True)

    def create_evaluate_batch(self,phase,cumulative_mutations,**kwargs):
        plugin=self
        class Evaluator:
            callback=None
            def set_progress_callback(self,callback):self.callback=callback
            def __call__(self,candidates,current):
                results=[]
                if phase==1:plugin.service.baseline_delay(progress=self.callback)
                for candidate in candidates:
                    marker=candidate.mutations.get('__C17_diagnostic__')
                    name=marker or plugin.service.name_for_config({**current,**candidate.mutations})
                    evaluated=plugin.service.evaluate(name,progress=self.callback)
                    if candidate.name=='__baseline__':
                        value=selectable_score(evaluated) # Baseline failure raises; never score zero.
                    elif not evaluated.get('native_valid') or not evaluated.get('qualification',{}).get('promote'):
                        # The shared evaluator accepts only finite scored values.
                        # Withhold invalid/unqualified arms entirely; their complete
                        # rejection/diagnostic receipts remain in the trial registry.
                        if self.callback:self.callback(dict(event='arm_ineligible',arm=name,
                            receipt=evaluated.get('receipt'),reasons=evaluated.get('qualification',{}).get('failures')))
                        continue
                    else:value=selectable_score(evaluated)
                    results.append(ScoredCandidate(candidate.name,value,metrics=dict(c16_S=value,policy=name,qualified=True)))
                return results
        return Evaluator()

    def compute_final_metrics(self,mutations):
        name=self.service.name_for_config(self.canonicalize_mutations(mutations))
        evaluated=self.service.evaluate(name);selectable_score(evaluated)
        receipt=self.service.receipts(name)['is/baseline'];data=bundle(receipt,'baseline');result=diagnostic_result(data)
        result.metrics['invalid_order_rejection_count']=data['cases']['baseline']['audit']['invalid_order_rejection_count']
        from .full_diagnostics import _final_metrics
        metrics=_final_metrics(result.metrics,result.trades,self.config,result.replay_result.equity_curve,timestamps=result.replay_result.timestamps)
        metrics=attach_official_metric_contract(metrics,requires_audit_pass=True,audit_pass=True,
            execution_contract={**self.execution_context,'metric_units':'Canonical return/DD fractions; c16_R/D/S percentage points; KRW cash'})
        metrics.update(c16_R=evaluated['score']['R'],c16_D=evaluated['score']['D'],c16_S=evaluated['score']['S'],
            C17_policy_name=name,C17_eligible=True,archived_native_evidence=True)
        return metrics

    def validate_archived_execution(self,metrics):
        name=metrics['C17_policy_name'];evaluation=self.service.validate_selected(name)
        if not metrics.get('audit_pass') or abs(metrics['c16_S']-selectable_score(evaluation))>1e-10:
            raise ValueError('Stale or unqualified scored result')
        return True

    def phase_acceptance_criteria(self,*,final_metrics,**kwargs):
        valid=self.validate_archived_execution(final_metrics)
        return [GateCriterion('C16_full_qualification',1,int(valid),valid)]

    def run_phase_diagnostics(self,phase,state,metrics,greedy_result):
        return json.dumps(self.service.phase_evidence(phase,metrics['C17_policy_name']),indent=2,default=str)
    run_enhanced_diagnostics=run_phase_diagnostics

    def build_end_of_round_artifacts(self,state):
        name=self.service.name_for_config(self.canonicalize_mutations(state.cumulative_mutations))
        self.service.validate_selected(name)
        self.service.final_freeze(name)
        context=self.service.phase_evidence(4,name)
        text=json.dumps(context,indent=2,default=str)
        return EndOfRoundArtifacts(text,{'C16_qualification_and_coverage':text},
            'Baseline retained' if name=='v6_h0' else 'Qualified research policy; no live activation or untouched holdout claim')

    def write_full_diagnostics(self,state,output_dir,*,round_num=3,round_name=''):
        name=self.service.name_for_config(self.canonicalize_mutations(state.cumulative_mutations));evaluation=self.service.validate_selected(name)
        reports={};summary=None;context=self.service.phase_evidence(4,name)
        for key,receipt in self.service.receipts(name).items():
            p,s=key.split('/');dest=Path(output_dir) if key=='is/baseline' else Path(output_dir)/'scenarios'/p/s
            current=report(receipt,s,dest,self.config,state.cumulative_mutations,round_num=round_num,
                round_name=round_name,round_context={**context,'qualification':evaluation.get('qualification')})
            if key=='is/baseline':summary=current
            reports[key]=bound(dest/'round_final_diagnostics.txt')
        write(Path(output_dir)/'scenario_reports.json',dict(complete=True,reports=reports,individual_scenarios_preserved=True))
        return summary

    def close_pool(self):self.service.close()
