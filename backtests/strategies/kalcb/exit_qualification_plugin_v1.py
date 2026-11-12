"""All native stress/allocation phases remain owned by shared PhaseRunner."""
from pathlib import Path
from backtests.strategies.kalcb.population_qualification_plugin_v1 import PopulationQualificationPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest


class ExitQualificationPlugin(PopulationQualificationPlugin):
    name='kalcb_exit_qualification_v1'

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Changed qualification study')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        byname={r['recipe']:r for r in records};pm=byname['control']['metrics']
        for row in records:
            m=row['metrics'];row['local_deltas']={k:m[k]-pm[k] for k in m if k.startswith(('is_','oos_'))}
            row['global_IS_sacrifice_pp']={s:v-m['is_'+s+'_return'] for s,v in [('baseline',73.1303145472),('combined',47.8121465627)]}
            row['IS_bands']={str(b):{s:v<=b for s,v in row['global_IS_sacrifice_pp'].items()} for b in (2,5,10)}
        payload=dict(study=bound(self.study),trials=records,mode=self.spec['mode'],study_complete=False,
            scope='Native opening-exit qualification on reused development; no promotion or independent validation claim.',
            remaining='Joint stress, funding, intervention, stability, concentration, paired uncertainty and existing-engine full reports.')
        write(self.root/'qualification_ledger.json',payload);return payload

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        payload=self.diagnostic();out=Path(output_dir)
        write(out/'qualification_ledger.json',payload)
        (out/'round_final_diagnostics.txt').write_text('INTERIM EXIT QUALIFICATION LEDGER\nExisting-engine full reports and final economic acceptance remain pending.\n',encoding='utf-8')
        return dict(complete=True,research_screening_only=True,existing_engine_final_reports_complete=False,
            promotion_status='qualification_pending_no_promotion',final_metrics=self.compute_final_metrics(state.cumulative_mutations))
