"""Finite shared PhaseRunner screening; controls gate all variants."""
from pathlib import Path
from .selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from .retention_research_io_v1 import read,write,bound


class PendingFundingPlugin(SelectiveRecoveryPlugin):
    name='kalcb_pending_opening_funding_v1'
    def evaluate_names(self,names):
        if any(self.spec['recipes'][n]['reservation'] for n in names):
            super().evaluate_names([n for n,r in self.spec['recipes'].items() if not r['reservation'] and n not in self.records])
        pending=[n for n in names if n not in self.records]
        if pending:super().evaluate_names(pending)
        return [self.records[n] for n in names]

    def diagnostic(self):
        rows=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        result={'trials':rows,'study_complete':False,'canonical_replaced':False,
            'scope':'Finite opening pending-funding screening; original policies, two priors, baseline/combined IS/OOS. Reused development.',
            'existing_engine_full_reports_complete':False,'promotion_status':'pending_funding_screening_only_no_promotion'}
        write(self.root/'screening_ledger.json',result);return result

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        write(Path(output_dir)/'screening_ledger.json',self.diagnostic())
        (Path(output_dir)/'round_final_diagnostics.txt').write_text('INTERIM SCREENING LEDGER; independent full assessment and existing-engine reports remain required. No promotion.\n',encoding='utf-8')
        return {'complete':True,'existing_engine_final_reports_complete':False,
            'promotion_status':'pending_funding_screening_only_no_promotion','final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
