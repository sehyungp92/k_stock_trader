"""Oracle/label-only scheduled horizon; disabled installation changes nothing.

Native position management, protective children, execution and liquidation stay
in the shared core. Future choice must never be passed to causal features.
"""
from dataclasses import replace
from datetime import timedelta


class DiagnosticHorizon:
    def __init__(self,minutes=0,chooser=None):
        if minutes not in (0,30,60,120):raise ValueError('Unregistered horizon')
        self.minutes=minutes;self.chooser=chooser;self.installed=False;self.config_cache={};self.records=[]

    def install(self):
        if not self.minutes and self.chooser is None:return
        from strategy_kalcb.core import logic
        self.original=logic._exit_reason;owner=self
        def reason(ss,position,bar,cfg,**kwargs):
            horizon=owner.chooser(ss,position,bar) if owner.chooser is not None else owner.minutes
            if horizon not in (0,30,60,120):raise ValueError('Oracle chose an unregistered exit')
            if not horizon:return owner.original(ss,position,bar,cfg,**kwargs)
            # Discretionary triggers are disabled only within the shared exit
            # reason function. Native stop/partial handling runs before it.
            key=id(cfg)
            if key not in owner.config_cache:
                owner.config_cache[key]=replace(cfg,exit_conditional_target_enabled=False,exit_target_r=0,
                    exit_no_mfe_bars=0,exit_failed_followthrough_bars=0,exit_shadow_failed_followthrough_bars=0,
                    exit_vwap_fail_bars=0,exit_profit_retention_enabled=False,exit_path_quality_enabled=False,
                    exit_mfe_giveback_enabled=False,exit_mfe_floor_enabled=False,exit_late_giveback_start_bars=0,
                    exit_time_decay_bars=0,exit_max_hold_bars=0,quick_exit_enabled=False,mfe_conviction_enabled=False,
                    flow_reversal_enabled=False,exit_timed_acceptance_extension_bars=0,research_opening_failure_min_hold_bars=0)
            mandatory=owner.original(ss,position,bar,owner.config_cache[key],**kwargs)
            if mandatory:return mandatory
            available=bar.timestamp+timedelta(minutes=5)
            deadline=position.entry_time+timedelta(minutes=horizon)
            if available>=deadline:
                owner.records.append(dict(symbol=position.symbol,entry_time=position.entry_time.isoformat(),
                    available_at=available.isoformat(),deadline=deadline.isoformat(),horizon=horizon,
                    scope='oracle_or_label_only_not_causal_feature'))
                return 'research_scheduled_horizon'
            return ''
        logic._exit_reason=reason;self.installed=True

    def restore(self):
        if self.installed:
            from strategy_kalcb.core import logic
            logic._exit_reason=self.original;self.installed=False
