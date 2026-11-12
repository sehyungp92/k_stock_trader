"""Shared-PhaseRunner adapter for the four-parent selective recovery study."""
import json
from pathlib import Path
from backtests.strategies.kalcb.oos_ablation import KALCBOOSAblationPlugin, read, write, digest


class SelectiveRecoveryPlugin(KALCBOOSAblationPlugin):
    name = 'kalcb_selective_recovery_v1'

    def __init__(self, study, native_case):
        super().__init__(study)
        self.native_case = native_case
        self.config['scope'] = 'Full IS and April1-July10 reused development; all four parents; no promotion'

    def paths(self, name, period):
        folder = self.root/'cases'/name/period
        folder.mkdir(parents=True, exist_ok=True)
        return folder, folder/'plan.json', folder/'inputs/files/research/kalcb_executed_value/policies'

    def run_native(self, name, period, scenario):
        return self.native_case(name, period, scenario)

    def evaluate_names(self, names):
        # Full audit/cash failures remain unscored evidence. An implementation
        # exception fails the phase and is repaired only in a versioned successor.
        for name in names:
            failures = []
            for period in ('is', 'oos'):
                for scenario in ('baseline', 'combined'):
                    result = self.run_native(name, period, scenario)
                    if result.get('research_rejected'):
                        failures.append({'period': period, 'scenario': scenario, 'error':result['error']})
            if failures:
                record={'recipe':name,'changes':self.spec['recipes'][name],'rejected':True,'failures':failures,
                        'metrics':{'physical_valid':False,'diagnostic_score':0.,'research_only':True},'cases':{}}
                self.records[name]=record;write(self.root/'cases'/name/'record.json',record)
            else:
                self.record(name)
        return [self.records[n] for n in names]

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Registered framework study changed')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        control=next(r for r in records if r['recipe']=='control')['metrics']
        keys=[p+'_'+s+'_'+m for p in ('is','oos') for s in ('baseline','combined') for m in ('return','dd','trades','active')]
        for record in records:
            if record.get('rejected'):continue
            record['delta_vs_global_expanding']={k:record['metrics'][k]-control[k] for k in keys}
        payload={'scope':'Source-audited IS and reused April1-July10 development; no independent validation.',
                 'no_canonical_promotion':True,'all_scores_zero_control_retained':True,
                 'new_recipe_count':10,'parent_count':4,'study':{'path':str(self.study),'sha256':digest(self.study)},'trials':records}
        write(self.root/'ablation_ledger.json',payload)
        return payload

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        payload=self.diagnostic();output_dir=Path(output_dir)
        write(output_dir/'round_final_full_diagnostics.json',payload)
        lines=['KALCB SELECTIVE RECOVERY: FOUR PARENTS; FULL IS AND APRIL1-JULY10',
               'Native source audit and independent cash/MTM checks required. Reused development; no promotion.',
               'Recipe | IS base return/DD/trades | IS combined | OOS base | OOS combined']
        for record in payload['trials']:
            if record.get('rejected'):
                lines.append(record['recipe']+' | REJECTED: '+json.dumps(record['failures']));continue
            metrics=record['metrics'];parts=[record['recipe']]
            for period in ('is','oos'):
                for scenario in ('baseline','combined'):
                    stem=period+'_'+scenario
                    parts.append(f"{metrics[stem+'_return']:.4f}/{metrics[stem+'_dd']:.4f}/{metrics[stem+'_trades']}")
            lines.append(' | '.join(parts))
        (output_dir/'round_final_diagnostics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return {'complete':True,'promotion_status':'research_only_control_retained','trial_count':len(payload['trials']),
                'full_source_qualification_of_variants':True,'four_scenario_finalist_qualification_complete':False,
                'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
