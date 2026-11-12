"""Shared PhaseRunner owner for finite opening-loss mechanism screening."""
from pathlib import Path
from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound


class MechanismPlugin(SelectiveRecoveryPlugin):
    name='kalcb_opening_mechanism_v1'

    def diagnostic(self):
        rows=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        controls={r['changes']['prior']:r for r in rows if not r['changes']['hold']}
        keys=[p+'_'+s+'_'+m for p in ('is','oos') for s in ('baseline','combined') for m in ('return','dd','trades','active')]
        for row in rows:
            parent=controls[row['changes']['prior']]['metrics'];m=row['metrics']
            row['local_parent_deltas']={k:m[k]-parent[k] for k in keys}
            row['global_IS_sacrifice_pp']={s:v-m['is_'+s+'_return'] for s,v in [('baseline',73.1303145472),('combined',47.8121465627)]}
            row['IS_reporting_bands']={str(b):{s:v<=b for s,v in row['global_IS_sacrifice_pp'].items()} for b in (2,5,10)}
            row['support']={}
            for period in ('is','oos'):
                for scenario in ('baseline','combined'):
                    folder=self.root/'cases'/row['recipe']/period/scenario
                    if not read(folder/'independent_account_verification.json')['passed']: raise ValueError('Unverified account')
                    summary=read(folder/'mechanism_evidence.json')['summary']
                    row['support'][period+'/'+scenario]={k:v for k,v in summary.items() if k!='executions'}
        payload={'scope':'Exact-state observer and finite management ablation on reused development',
                 'study':bound(self.study),'trials':rows,'study_complete':False,'canonical_replaced':False,
                 'next':'Full economic attribution, incremental selection development, justified interactions and four-stress finalist qualification.'}
        write(self.root/'screening_ledger.json',payload);return payload

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        payload=self.diagnostic();write(Path(output_dir)/'screening_ledger.json',payload)
        (Path(output_dir)/'round_final_diagnostics.txt').write_text(
            'INTERIM OPENING-MECHANISM SCREENING LEDGER; NOT FINAL QUALIFICATION\n'
            'Source/account audits and parent no-op bridges required. Final existing-engine reports pending.\n',encoding='utf-8')
        return {'complete':True,'research_screening_only':True,'existing_engine_final_reports_complete':False,
                'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
