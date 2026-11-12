"""Shared PhaseRunner adapter for the registered signal/component ablation."""
from backtests.strategies.kalcb.late_entry_plugin_v2 import LateEntryPluginV2


class LateSignalPlugin(LateEntryPluginV2):
    name='kalcb_late_signal_v1'

    def __init__(self,study,native_case):
        super().__init__(study,native_case)
        self.config['scope']='First-hour signal ablation: expanding priors100/125, exact quarter action, prequential calibration and compact model; reused development'

    def get_phase_spec(self,phase,state):
        spec=super().get_phase_spec(phase,state)
        spec.focus='Raw plus affine calibration; compact16 ridge; compact16 plus affine calibration. Two parents, unchanged execution.'
        return spec
