"""Finite four-scenario evidence, including authenticated reuse, via PhaseRunner."""
from pathlib import Path
from .selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from .retention_research_io_v1 import read, write, bound

SCENARIOS=('baseline','participation','cost','combined')


class PendingFundingQualificationPlugin(SelectiveRecoveryPlugin):
    name='kalcb_pending_funding_qualification_v1'

    def evaluate_names(self,names):
        if any(n!='control' for n in names) and 'control' not in self.records:
            self.evaluate_names(['control'])
        for name in names:
            if name in self.records: continue
            for period in ('is','oos'):
                for scenario in SCENARIOS: self.run_native(name,period,scenario)
            self.record(name)
        return [self.records[n] for n in names]

    def record(self,name):
        rows=[r for r in read(self.root/'progress.json') if r['name']==name]
        if len(rows)!=8 or len({(r['period'],r['scenario']) for r in rows})!=8:
            raise ValueError('Missing/duplicate four-scenario cases')
        metrics={'physical_valid':True,'diagnostic_score':0.,'research_only':True,
                 'audit_status':'passed','promotion_status':'pending_funding_qualification_no_promotion'}
        for row in rows:
            if row['metrics']['audit_status']!='passed': raise ValueError('Unaudited qualification case')
            for key,source in (('return','return_pct'),('dd','drawdown_pct'),('trades','trades'),('active','active_days')):
                metrics[row['period']+'_'+row['scenario']+'_'+key]=row['metrics'][source]
        record={'recipe':name,'changes':self.spec['recipes'][name],'metrics':metrics,
                'cases':{r['period']+'/'+r['scenario']:r['result'] for r in rows}}
        self.records[name]=record;write(self.root/'cases'/name/'record.json',record)
        return record

    def diagnostic(self):
        records=[self.records[n] for n in self.spec['recipes']]
        parent=self.records['control']['metrics']
        for row in records:
            row['local_deltas']={k:v-parent[k] for k,v in row['metrics'].items() if k.startswith(('is_','oos_'))}
        result={'trials':records,'study':bound(self.study),'study_complete':False,
            'fresh_native_cases':8,'authenticated_reuse':8,
            'scope':'Prior100 opening pending-funding cap, all four stresses, reused development; unchanged prior125 remains competing reference.',
            'promotion_status':'pending_funding_qualification_no_promotion','existing_engine_full_reports_complete':False}
        write(self.root/'qualification_ledger.json',result);return result

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        write(Path(output_dir)/'qualification_ledger.json',self.diagnostic())
        (Path(output_dir)/'round_final_diagnostics.txt').write_text(
            'INTERIM QUALIFICATION LEDGER. Joint economic assessment and existing-engine reports remain required. No promotion.\n',encoding='utf-8')
        return {'complete':True,'existing_engine_final_reports_complete':False,
            'promotion_status':'pending_funding_qualification_no_promotion','final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
