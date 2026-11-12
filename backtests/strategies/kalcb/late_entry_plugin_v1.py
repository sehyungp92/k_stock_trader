"""First-hour registered native experiment adapter for shared PhaseRunner."""
from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest


class LateEntryPlugin(SelectiveRecoveryPlugin):
    name='kalcb_late_entry_v1'

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Changed registered study')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        controls={r['changes']['prior']:r for r in records if r['changes']['quantity']==0}
        keys=[p+'_'+s+'_'+m for p in ('is','oos') for s in ('baseline','combined') for m in ('return','dd','trades','active')]
        for r in records:
            if r.get('rejected'):continue
            c=controls[r['changes']['prior']]['metrics'];r['delta_vs_local_parent']={k:r['metrics'][k]-c[k] for k in keys}
            r['IS_sacrifice_pp']={s:max(0,c['is_'+s+'_return']-r['metrics']['is_'+s+'_return']) for s in ('baseline','combined')}
            r['IS_sacrifice_bands']={str(b):{s:v<=b for s,v in r['IS_sacrifice_pp'].items()} for b in (2,5,10)}
        result={'scope':'Reused IS/OOS development; actual funded native comparison; no automatic promotion',
            'study':bound(self.study),'trials':records,'control_count':2,'new_recipe_count':6,'study_complete':False,
            'remaining':'Actual timed-fill and funding attribution, four stress finalist qualification, paired uncertainty, concentration and full existing-engine reports'}
        write(self.root/'screening_ledger.json',result);return result

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        payload=self.diagnostic();write(output_dir/'screening_ledger.json',payload)
        return {'complete':True,'research_screening_only':True,'existing_engine_final_reports_complete':False,
            'promotion_status':'research_only_control_retained','trial_count':len(payload['trials']),
            'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
