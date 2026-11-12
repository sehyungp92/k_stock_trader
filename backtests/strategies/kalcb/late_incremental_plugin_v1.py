"""Shared-framework lifecycle, explicitly interim incremental screening ledger."""
from backtests.strategies.kalcb.late_entry_plugin_v2 import LateEntryPluginV2


class IncrementalPlugin(LateEntryPluginV2):
    name='kalcb_late_incremental_v1'

    def __init__(self,study,native_case):
        super().__init__(study,native_case)
        self.config['scope']='Expanding100/125; exact-quarter target/population alignment, uncertainty-reserve component, single interaction; reused development'

    def get_phase_spec(self,phase,state):
        spec=super().get_phase_spec(phase,state)
        spec.focus='Aligned net/notional target with fixed48 reserve; inherited signal with prequential reserve; their interaction. Native action unchanged.'
        return spec
