"""Private frozen registration; no canonical execution-rule mutation."""
from scripts.olr_minute_exit_patches_v5 import once


def patch_runner(text):
    return once(text,"        from backtests.strategies.olr.funded_action_probe import FundedActionProbeDriver\n        minute_driver=FundedActionProbeDriver(broker,minute_adapter,mode=structural['mode'],\n            output=args.output,interval_service=structural['interval_service'],probe=bound('funded_probe_settings'))", """        from backtests.strategies.olr.funded_population import FundedPopulationDriver
        minute_driver=FundedPopulationDriver(broker,minute_adapter,mode=structural['mode'],
            output=args.output,interval_service=structural['interval_service'],probe=bound('funded_probe_settings'),
            population=bound('funded_population_settings'))""")


def patch_allocation(text):
    return once(text,'    if decision_context_provider is not None and intents:',"""    if state.meta.get('observe_funded_entry_context_v1') and intents:
        from dataclasses import asdict
        # Evidence only: these are actual native hard-eligible proposals. A
        # price episode or pre-cap rejected request is not counted as one.
        state.meta.setdefault('funded_entry_contexts_v1',[]).append(dict(
            at=at.isoformat(),ledger_cash=float(portfolio.cash),pending_buy_reservations=reserve,
            actual_deployable_cash=max(0.,float(portfolio.cash)-reserve),
            sizing_equity=equity,account={**asdict(account),'occupied':[list(k) for k in sorted(account.occupied)]},
            intents=[asdict(i) for i in intents],decisions=[asdict(d) for d in decisions],
            proposals={k:action_to_json_dict(v) for k,v in by_id.items()},
            scope='Existing native signal/lifecycle/capacity-approved proposals; not all raw episodes or guaranteed fills.'))
    if decision_context_provider is not None and intents:""")


def canonicalize_account_evidence(text):
    """Upgrade only the existing V6 observer; native allocation is unchanged."""
    return once(text,'            sizing_equity=equity,account=asdict(account),',
        "            sizing_equity=equity,account={**asdict(account),'occupied':[list(k) for k in sorted(account.occupied)]},")
