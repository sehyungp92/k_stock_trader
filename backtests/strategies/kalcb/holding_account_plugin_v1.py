"""Finite shared-framework ownership; completion never implies improvement."""
from copy import deepcopy
from pathlib import Path
import json
from backtests.auto.shared.plugin import PhaseSpec,PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment,GateCriterion,EndOfRoundArtifacts
from .opening_account_plugin_v3 import OpeningAccountPlugin,SCENARIOS,objective
from .holding_account_model_v1 import build
from .retention_research_io_v1 import read,write,bound,verify
from .late_artifact_validation_v2 import validate_native


class HoldingAccountPlugin(OpeningAccountPlugin):
    name='kalcb_holding_account_v1'

    def get_phase_spec(self,phase,state):
        return PhaseSpec('Remaining executable account value: chronology, readiness and conditional qualification',
            [Experiment('selective',{'recipe':'selective'})],
            lambda m:[GateCriterion('physical_execution',1,int(m['physical_valid']),m['physical_valid'])],
            None,{},PhaseAnalysisPolicy(focus_metrics=['economic_readiness','economic_qualified','oos_baseline_return','is_baseline_return']),
            max_rounds=1,phase_metric_basis='native_account_readiness_then_conditional_qualification',
            primary_promotion_metric='diagnostic_score',official_metric_keys=('economic_qualified',),promotion_requires_audit_pass=False)

    def evaluate_names(self,names):
        if any(n!='control' for n in names) and 'control' not in self.records:self.evaluate_names(['control'])
        for name in names:
            if name in self.records:continue
            if name=='selective':
                model=build(self.root)
                write(self.root/'economic_readiness.json',model['readiness'])
                if not model['economic_readiness_pass']:
                    record=deepcopy(self.records['control']);record.update(recipe=name,objective=None,
                        native_policy_executed=False,skip_reason='failed_preregistered_IS_action_discrimination',
                        readiness=model['readiness'])
                    record['metrics'].update(recipe_identity=name,economic_readiness=False,economic_qualified=False,
                        diagnostic_score=0.,policy_account_metrics_are_control_reference=True)
                    self.records[name]=record;write(self.root/'cases'/name/'record.json',record);continue
            # Reuse the strict eight-account interface and unchanged frontier.
            super().evaluate_names([name])
            record=self.records[name]
            record.update(native_policy_executed=name!='control')
            record['metrics'].update(economic_readiness=name!='control',
                economic_qualified=bool(record['objective'] and record['objective']['passed']),
                policy_account_metrics_are_control_reference=False)
            write(self.root/'cases'/name/'record.json',record)
        return [self.records[n] for n in names]

    def phase_acceptance_criteria(self,**kwargs):
        name=kwargs['final_metrics']['recipe_identity'];record=self.records[name]
        source_name='control' if not record['native_policy_executed'] else name
        for period in ('is','oos'):
            for scenario in SCENARIOS:
                folder=self.root/'cases'/source_name/period/scenario
                item=read(folder/'input_profile.json')
                verify([record['input_profiles'][f'{period}/{scenario}'],record['cases'][f'{period}/{scenario}']])
                validate_native(folder,item,scenario,save=False)
        return [GateCriterion('all_eight_policy_bound_native_account_receipts',8,8,True)]

    def diagnostic(self):
        result={'trials':list(self.records.values()),'study':bound(self.study),
            'objective_achieved_on_reused_development':any(r.get('objective',{}).get('passed',False)
                for r in self.records.values() if r.get('objective')),
            'mechanical_completion_separate_from_economic_qualification':True,
            'independent_forward_validation':False,'canonical_replaced':False,
            'scope':'Chronological IS action labels; one fixed selector; OOS is reused development. '
                'Local action values are not additive strategy returns. Failed readiness skips policy execution.'}
        write(self.root/'account_frontier.json',result);return result

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        result=self.diagnostic();write(Path(output_dir)/'round_final_full_diagnostics.json',result)
        (Path(output_dir)/'round_final_diagnostics.txt').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return {'complete':True,'full_source_qualification_of_variants':all(r.get('native_policy_executed') for n,r in self.records.items() if n!='control'),
            'promotion_status':'economic_status_is_explicit_not_inferred_from_phase_completion',
            'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}

    def build_end_of_round_artifacts(self,state):
        report=json.dumps(self.diagnostic(),indent=2)
        return EndOfRoundArtifacts(report,{'remaining_action_value':report},
            'Finite evaluation complete. Read economic qualification and remaining objective separately.')
