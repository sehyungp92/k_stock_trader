"""Shared phase lifecycle for finite population-removal qualification evidence."""
from pathlib import Path
from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest


class PopulationQualificationPlugin(SelectiveRecoveryPlugin):
    name='kalcb_population_qualification_v1'

    def evaluate_names(self,names):
        for name in names:
            if name in self.records:continue
            for period in ('is','oos'):
                for scenario in self.spec['recipes'][name]['scenarios']:
                    self.run_native(name,period,scenario)
            self.record(name)
        return [self.records[n] for n in names]

    def record(self,name):
        if name in self.records:return self.records[name]
        metrics={'physical_valid':True,'diagnostic_score':0.,'research_only':True,'audit_status':'passed',
                 'promotion_status':'qualification_in_progress_no_promotion'}
        cases={}
        for period in ('is','oos'):
            for scenario in self.spec['recipes'][name]['scenarios']:
                folder=self.root/'cases'/name/period/scenario
                if not read(folder/'independent_account_verification.json')['passed']:
                    raise ValueError('Unverified account')
                result=read(folder/'result.json')
                cases[period+'/'+scenario]=bound(folder/'result.json')
                for key,source in (('return','return_pct'),('dd','drawdown_pct'),('trades','trades'),('active','active_days')):
                    metrics[period+'_'+scenario+'_'+key]=result[source]
        record={'recipe':name,'changes':self.spec['recipes'][name],'metrics':metrics,'cases':cases}
        self.records[name]=record;write(self.root/'cases'/name/'record.json',record)
        return record

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Registered study changed')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        byname={r['recipe']:r for r in records}
        for r in records:
            parent='control' if r['changes']['parent']=='expanding' else 'control_expanding_atr5'
            m=r['metrics'];pm=byname[parent]['metrics'];gm=byname['control']['metrics']
            r['local_deltas']={k:m[k]-pm[k] for k in m if k.startswith(('is_','oos_')) and k in pm}
            r['global_IS_sacrifice_pp']={s:gm['is_'+s+'_return']-m['is_'+s+'_return'] for s in ('baseline','combined')}
            r['IS_reporting_bands']={str(b):all(v<=b for v in r['global_IS_sacrifice_pp'].values()) for b in (2,5,10)}
        result={'trials':records,'study':bound(self.study),'all_scores_zero_control_retained':True,
                'scope':'Finite population-removal qualification; reused development, no promotion.',
                'funded_constant_capital_and_final_review_pending':True}
        write(self.root/'qualification_ledger.json',result)
        return result

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        result=self.diagnostic();output_dir=Path(output_dir)
        write(output_dir/'round_final_full_diagnostics.json',result)
        lines=['KALCB POPULATION REMOVAL QUALIFICATION: FINITE PRIOR NEIGHBORHOOD',
               'All four stresses on centers and parents; baseline/combined on neighbors.',
               'Reused development. Further funded constant-capital and final review required.']
        for r in result['trials']:
            lines.append(r['recipe'])
            for p in ('is','oos'):
                for s in r['changes']['scenarios']:
                    m=r['metrics'];k=p+'_'+s
                    lines.append(f"  {k}: return {m[k+'_return']:.6f}%, MTM DD {m[k+'_dd']:.6f}%, trades {m[k+'_trades']}")
        (output_dir/'round_final_diagnostics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return {'complete':True,'promotion_status':'qualification_in_progress_no_promotion',
                'trial_count':len(result['trials']),'study_complete':False,
                'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
