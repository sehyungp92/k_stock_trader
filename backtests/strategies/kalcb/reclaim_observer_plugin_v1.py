"""Shared PhaseRunner lifecycle for eight unchanged-parent observational cases."""
from pathlib import Path
from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound


class ReclaimObserverPlugin(SelectiveRecoveryPlugin):
    name='kalcb_reclaim_parent_observer_v1'
    def diagnostic(self):
        rows=[]
        for p in sorted((self.root/'cases').glob('*/*/*/observer_evidence.json')):
            if not read(p.parent/'identity_bridge.json')['passed'] or not read(p.parent/'independent_account_verification.json')['passed']:raise ValueError('Unaudited parent observer')
            rows.append({'case':str(p.parent),'summary':read(p)['summary'],'evidence':bound(p)})
        payload={'cases':rows,'study_complete':False,'scope':'Unchanged parent account/exposure and conditional order previews only.',
            'new_action_portfolio_qualified':False,'new_economic_variants':0,'canonical_replaced':False}
        write(self.root/'observer_ledger.json',payload);return payload
    def write_full_diagnostics(self,state,output_dir,**kwargs):
        write(Path(output_dir)/'observer_ledger.json',self.diagnostic())
        (Path(output_dir)/'round_final_diagnostics.txt').write_text('INTERIM OBSERVER LEDGER. Parent identity/source/accounts required; no new strategy or final economic qualification.\n',encoding='utf-8')
        return {'complete':True,'research_observer_only':True,'existing_engine_final_reports_complete':False,
            'promotion_status':'unchanged_parent_observation_only','final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
