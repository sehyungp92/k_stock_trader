"""Account-objective experiment plugin; PhaseRunner owns the whole lifecycle."""
import json
from pathlib import Path

from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment, ScoredCandidate, GateCriterion, EndOfRoundArtifacts
from .oos_ablation import KALCBOOSAblationPlugin
from .retention_research_io_v1 import read, write, bound, verify
from .late_artifact_validation_v2 import validate_native
from backtests.auto.shared.cache_keys import stable_signature

SCENARIOS=('baseline','participation','cost','combined')


def objective(candidate,parent):
    """Declared development frontier; no claim of a unique mathematical optimum."""
    deltas={k:candidate[k]-v for k,v in parent.items() if k.startswith(('is_','oos_'))}
    retention=min(candidate[f'is_{s}_return']/parent[f'is_{s}_return'] for s in SCENARIOS)
    oos_gain=min(deltas[f'oos_{s}_return'] for s in SCENARIOS)
    dd_worsening=max(deltas[f'{p}_{s}_dd'] for p in ('is','oos') for s in SCENARIOS)
    activity=min(candidate[f'{p}_{s}_material']/max(parent[f'{p}_{s}_material'],1)
                 for p in ('is','oos') for s in SCENARIOS)
    passes=retention>=.98 and oos_gain>=1.0 and dd_worsening<=1e-8 and activity>=.95
    return dict(passed=passes,is_return_retention=retention,worst_scenario_oos_return_gain_pp=oos_gain,
        worst_drawdown_change_pp=dd_worsening,material_activity_retention=activity,deltas=deltas,
        score=(oos_gain + min(-deltas[f'oos_{s}_dd'] for s in SCENARIOS)) if passes else 0.)


class OpeningAccountPlugin(KALCBOOSAblationPlugin):
    name='kalcb_opening_account_v2'

    def __init__(self,study,native_case):
        super().__init__(study)
        self.native_case=native_case
        self.config['promotion_status']='development_account_frontier_only'

    def get_phase_spec(self,phase,state):
        return PhaseSpec('Joint opening account risk and marginal value with matched-risk control',
            [Experiment(n,{'recipe':n}) for n in ('joint','uniform')],
            lambda m:[GateCriterion('physical_execution',1,int(m['physical_valid']),m['physical_valid'])],
            None,{},PhaseAnalysisPolicy(focus_metrics=['oos_baseline_return','oos_combined_return','is_baseline_return','is_combined_return']),
            max_rounds=1,phase_metric_basis='native_complete_account_development',
            primary_promotion_metric='diagnostic_score',
            official_metric_keys=('oos_baseline_return','oos_combined_return'),promotion_requires_audit_pass=False)

    def phase_acceptance_criteria(self,**kwargs):
        # The generic gate validates a single CURRENT-process v18 receipt.
        # These eight accounts deliberately execute the frozen parent v16
        # contract. Never forge a current-source receipt or conflate accounts.
        # Revalidate every immutable native source/account receipt instead.
        name=kwargs['final_metrics']['recipe_identity']
        record=self.records[name]
        for period in ('is','oos'):
            item=read(self.root/'runtime'/name/period/'profile.json')
            verify([item[k] for k in ('manifest','policy','plan')])
            for scenario in SCENARIOS:
                verify([record['cases'][f'{period}/{scenario}']])
                validate_native(self.root/'cases'/name/period/scenario,item,scenario,save=False)
        return [GateCriterion('all_eight_frozen_native_source_and_account_receipts',8,8,True)]

    def evaluate_names(self,names):
        if any(n!='control' for n in names) and 'control' not in self.records:
            self.evaluate_names(['control'])
        for name in names:
            if name in self.records:
                continue
            metrics={'physical_valid':True,'diagnostic_score':0.,'research_only':True,
                     'audit_status':'passed','promotion_status':'development_only',
                     'recipe_identity':name,'official_metric_basis':'SimBroker.equity_curve_bar_level_mtm',
                     'audit_gate_owner':'all_eight_frozen_native_source_and_account_receipts',
                     'same_bar_fill_count':0.,'forced_replay_close_count':0.,'rejected_order_count':0.,
                     'invalid_order_rejection_count':0.,'end_open_position_count':0.}
            cases={}
            receipts={}
            for period in ('is','oos'):
                for scenario in SCENARIOS:
                    result=self.native_case(name,period,scenario)
                    for a,b in (('return','return_pct'),('dd','drawdown_pct'),('trades','trades'),
                                ('active','active_days'),('material','material_funded_entries')):
                        metrics[f'{period}_{scenario}_{a}']=result[b]
                    cases[f'{period}/{scenario}']=result['result_binding']
                    audit=read(Path(result['result_binding']['path']).parent/'execution_audit.json')
                    receipts[f'{period}/{scenario}']=audit['audit_receipt']
                    for key in ('same_bar_fill_count','forced_replay_close_count','rejected_order_count','invalid_order_rejection_count'):
                        metrics[key]+=float(audit[key])
            metrics.update(source_fingerprint=stable_signature({k:v['source_generation_hash'] for k,v in receipts.items()}),
                feature_manifest_hash=stable_signature({k:v['identity']['input_manifest']['manifest_hash'] for k,v in receipts.items()}),
                candidate_snapshot_hash=stable_signature({k:v['identity']['qualification_snapshot_hash'] for k,v in receipts.items()}))
            assessment=objective(metrics,self.records['control']['metrics']) if name!='control' else None
            if assessment:
                metrics['diagnostic_score']=assessment['score']
            record={'recipe':name,'metrics':metrics,'cases':cases,'objective':assessment}
            self.records[name]=record
            write(self.root/'cases'/name/'record.json',record)
        return [self.records[n] for n in names]

    def create_evaluate_batch(self,phase,cumulative_mutations,**kwargs):
        def evaluate(candidates,current):
            names=[self.canonicalize_mutations({**current,**c.mutations})['recipe'] for c in candidates]
            rows=self.evaluate_names(names)
            return [ScoredCandidate(c.name,r['metrics']['diagnostic_score'],metrics=r['metrics'])
                    for c,r in zip(candidates,rows)]
        return evaluate

    def diagnostic(self):
        result={'trials':list(self.records.values()),'study':bound(self.study),
            'objective_achieved_on_reused_development':any(r.get('objective',{}).get('passed',False)
                for r in self.records.values() if r.get('objective')),
            'independent_forward_validation':False,'canonical_replaced':False,
            'scope':'Full funded sequential accounts; stress paths use past prices only. '
                    'IS and OOS are reused development. Suffix contrasts are not strategy returns.'}
        write(self.root/'account_frontier.json',result)
        return result

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        result=self.diagnostic()
        write(Path(output_dir)/'round_final_full_diagnostics.json',result)
        lines=['KALCB COMPLETE-ACCOUNT OPENING ALLOCATION',
            'Finite declared joint allocation and matched-risk proportional control.',
            'Native full-source checks required. Reused development; no live deployment.',
            'Recipe | period/scenario | return % | full MTM DD % | funded lifetimes | material lifetimes']
        for row in result['trials']:
            m=row['metrics']
            for p in ('is','oos'):
                for s in SCENARIOS:
                    stem=f'{p}_{s}'
                    lines.append(f"{row['recipe']} | {p}/{s} | {m[stem+'_return']:.6f} | {m[stem+'_dd']:.6f} | {m[stem+'_trades']} | {m[stem+'_material']}")
            if row.get('objective'):
                lines.append('Objective: '+json.dumps(row['objective']))
        (Path(output_dir)/'round_final_diagnostics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return {'complete':True,'full_source_qualification_of_variants':True,
            'promotion_status':'reused_development_only','final_metrics':self.compute_final_metrics(state.cumulative_mutations)}

    def build_end_of_round_artifacts(self,state):
        report=json.dumps(self.diagnostic(),indent=2)
        return EndOfRoundArtifacts(report,{'account_objective':report},
            'Account comparison complete; read objective status separately from process completion.')
