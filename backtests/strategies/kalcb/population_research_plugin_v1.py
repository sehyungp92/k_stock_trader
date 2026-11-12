"""Population calibration screening through the existing shared phase lifecycle."""
from pathlib import Path
from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest


class PopulationResearchPlugin(SelectiveRecoveryPlugin):
    name='kalcb_population_research_v1'

    def record(self,name):
        record=super().record(name)
        for p in ('is','oos'):
            for s in ('baseline','combined'):
                if not read(self.root/'cases'/name/p/s/'independent_account_verification.json')['passed']:
                    raise ValueError('Unverified account')
        record['metrics']['audit_status']='passed';write(self.root/'cases'/name/'record.json',record)
        return record

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Changed registered study')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        byname={r['recipe']:r for r in records};global_ref=byname['control']['metrics']
        controls={p:n for n,p in self.spec['controls'].items()}
        keys=[p+'_'+s+'_'+m for p in ('is','oos') for s in ('baseline','combined') for m in ('return','dd','trades','active')]
        for r in records:
            if r.get('rejected'):continue
            parent=controls[r['changes']['parent']]
            if parent not in byname:continue
            m=r['metrics'];pm=byname[parent]['metrics']
            r['local_parent_deltas']={k:m[k]-pm[k] for k in keys}
            r['global_IS_sacrifice_pp']={s:global_ref['is_'+s+'_return']-m['is_'+s+'_return'] for s in ('baseline','combined')}
            r['IS_reporting_bands']={str(b):all(v<=b for v in r['global_IS_sacrifice_pp'].values()) for b in (2,5,10)}
            r['native_support']={}
            for p in ('is','oos'):
                for s in ('baseline','combined'):
                    path=self.root/'cases'/r['recipe']/p/s/'intervention_support.json'
                    if path.exists():r['native_support'][p+'/'+s]={k:v for k,v in read(path).items() if k!='changes'}
        result={'scope':'Continuous-history IS and complete69-session OOS, reused development; population selection study, no promotion.',
            'all_scores_zero_control_retained':True,'no_canonical_promotion':True,'parent_count':4,
            'new_recipe_count':len(self.spec['recipes'])-4,'study':bound(self.study),'trials':records}
        write(self.root/'ablation_ledger.json',result);return result

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        result=self.diagnostic();output_dir=Path(output_dir)
        write(output_dir/'round_final_full_diagnostics.json',result)
        lines=['KALCB POPULATION CALIBRATION: FULL IS AND APRIL1-JULY10',
            'Audited native research screening; no independent validation or promotion.',
            'Recipe | IS baseline return/DD/trades | IS combined | OOS baseline | OOS combined']
        for r in result['trials']:
            if r.get('rejected'):lines.append(r['recipe']+' | REJECTED');continue
            m=r['metrics'];parts=[r['recipe']]
            for p in ('is','oos'):
                for s in ('baseline','combined'):
                    k=p+'_'+s;parts.append(f"{m[k+'_return']:.4f}/{m[k+'_dd']:.4f}/{m[k+'_trades']}")
            lines.append(' | '.join(parts))
        (output_dir/'round_final_diagnostics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return {'complete':True,'promotion_status':'research_only_control_retained','trial_count':len(result['trials']),
            'full_source_qualification_of_variants':True,'four_scenario_finalist_qualification_complete':False,
            'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
