"""Report-contract repair; the registered action and native cases are unchanged."""
import json
from pathlib import Path
from backtests.strategies.kalcb.late_entry_plugin_v1 import LateEntryPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import write


class LateEntryPluginV2(LateEntryPlugin):
    def __init__(self,study,native_case):
        super().__init__(study,native_case)
        self.config['scope']='Two expanding priors100/125; fresh completed-first-hour additions; continuous IS/OOS reused development; no promotion'

    def get_phase_spec(self,phase,state):
        spec=super().get_phase_spec(phase,state)
        spec.focus='One causal first-hour addition: quarter quantity, flat-only backout and half-flat neighbor on both expanding priors'
        return spec

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        payload=self.diagnostic();output_dir=Path(output_dir)
        write(output_dir/'screening_ledger.json',payload)
        (output_dir/'round_final_diagnostics.txt').write_text(
            'REGISTERED FIRST-HOUR SCREENING LEDGER\n'
            'Source-audited native baseline/combined IS/OOS cases. Reused development.\n'
            'This is an interim screening ledger; four-stress qualification and existing-engine full reports remain pending.\n'
            +json.dumps(payload,indent=2,default=str)+'\n',encoding='utf-8')
        return {'complete':True,'research_screening_only':True,'existing_engine_final_reports_complete':False,
            'promotion_status':'research_only_control_retained','trial_count':len(payload['trials']),
            'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
