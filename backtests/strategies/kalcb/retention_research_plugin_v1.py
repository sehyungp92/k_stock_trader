"""Shared-framework retention perturbations with direct native exit support."""
from pathlib import Path
from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, digest, bound


class RetentionResearchPlugin(SelectiveRecoveryPlugin):
    name='kalcb_retention_research_v1'

    def record(self,name):
        record=super().record(name)
        for period in ('is','oos'):
            for scenario in ('baseline','combined'):
                if not read(self.root/'cases'/name/period/scenario/'independent_account_verification.json')['passed']:
                    raise ValueError('Unverified native account cannot be scored')
        record['metrics']['audit_status']='passed'
        write(self.root/'cases'/name/'record.json',record)
        return record

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Changed framework study')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        byname={r['recipe']:r for r in records};global_ref=byname['control']['metrics']
        parent_controls={parent:name for name,parent in self.spec['controls'].items()}
        keys=[p+'_'+s+'_'+m for p in ('is','oos') for s in ('baseline','combined') for m in ('return','dd','trades','active')]
        for record in records:
            if record.get('rejected'):continue
            parent=parent_controls[record['changes']['parent']]
            if parent not in byname:continue
            pm=byname[parent]['metrics'];metrics=record['metrics']
            record['local_parent_deltas']={k:metrics[k]-pm[k] for k in keys}
            loss={s:global_ref['is_'+s+'_return']-metrics['is_'+s+'_return'] for s in ('baseline','combined')}
            record['global_IS_sacrifice_pp']=loss
            record['IS_reporting_bands']={str(b):all(x<=b for x in loss.values()) for b in (2,5,10)}
            record['direct_exit_support']={}
            for p in ('is','oos'):
                for s in ('baseline','combined'):
                    evidence=self.root/'cases'/record['recipe']/p/s/'intervention_support.json'
                    if evidence.exists():
                        record['direct_exit_support'][p+'/'+s]={k:v for k,v in read(evidence)['summary'].items() if k!='execution_checks'}
        payload={'scope':'Full IS and April1-July10 reused development. Frozen selectors; management perturbations only. No promotion.',
                 'no_canonical_promotion':True,'all_scores_zero_control_retained':True,
                 'new_recipe_count':len(self.spec['recipes'])-len(self.spec['controls']),
                 'parent_count':4,'study':bound(self.study),'trials':records}
        write(self.root/'ablation_ledger.json',payload)
        return payload

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        payload=self.diagnostic();output_dir=Path(output_dir)
        write(output_dir/'round_final_full_diagnostics.json',payload)
        text=['KALCB RETENTION PERTURBATIONS; FOUR PARENTS; FULL IS/OOS',
              'Native audited research screening; reused development; no promotion.',
              'Recipe | IS baseline return/DD/trades | IS combined | OOS baseline | OOS combined']
        for row in payload['trials']:
            if row.get('rejected'):text.append(row['recipe']+' | REJECTED');continue
            m=row['metrics'];parts=[row['recipe']]
            for p in ('is','oos'):
                for s in ('baseline','combined'):
                    stem=p+'_'+s;parts.append(f"{m[stem+'_return']:.4f}/{m[stem+'_dd']:.4f}/{m[stem+'_trades']}")
            text.append(' | '.join(parts))
        (output_dir/'round_final_diagnostics.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
        return {'complete':True,'promotion_status':'research_only_control_retained','trial_count':len(payload['trials']),
                'full_source_qualification_of_variants':True,'four_scenario_finalist_qualification_complete':False,
                'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
