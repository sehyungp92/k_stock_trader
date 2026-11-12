"""Reviewable opt-in patches for a separately frozen native generation.

Every anchor is exact; an unexpected parent is rejected rather than silently
patching a different engine. Parent hard links must never be written in place.
"""
from scripts.olr_minute_exit_patches_v5 import once

VERSION='olr-completed-interval-service-v1'


def patch(name,text):
    if name=='backtests/engine/sim_broker.py':
        text=once(text,'        start = len(self.fills)\n',
            "        interval_service = bool(self.capacity_policy and self.capacity_policy.contract().get('version') == '"+VERSION+"')\n        start = len(self.fills)\n")
        text=once(text,'        for phase in ("open", "intrabar", "auction"):', '''        if interval_service:
            # These marks and service are observed at completion, after every
            # eligible order has committed at/before the original interval start.
            from .olr_minute_v5 import continuous_minute
            service_marks={s:float(b.close) for s,b in closing_bars.items() if continuous_minute(b)}
            self.last_prices.update(service_marks)
            if service_marks:self.journal.append(dict(event='interval_marks',timestamp=timestamp.isoformat(),prices=service_marks))
        for phase in (("interval", "auction") if interval_service else ("open", "intrabar", "auction")):''')
        text=once(text,'                if intended_phase != phase:', '''                if interval_service and order.order_type == 'MARKET':intended_phase='interval'
                if interval_service and order.order_type not in ('MARKET','CLOSE_AUCTION'):
                    raise ValueError('Interval service requires native market/auction orders')
                if intended_phase != phase:''')
        text=once(text,'                price = self._fill_price(order, pricing_bar)',
            "                if phase=='interval':pricing_bar=replace(bar,open=bar.close)\n                price = self._fill_price(order, pricing_bar)")
        text=once(text,'                    base = bar.open\n',"                    base = bar.close if phase=='interval' else bar.open\n")
        text=once(text,'                fill = self._apply_fill(replace(order, qty=allowed_qty), price, timestamp)',
            "                fill_at=timestamp+timedelta(minutes=1) if phase=='interval' else timestamp\n                fill = self._apply_fill(replace(order, qty=allowed_qty), price, fill_at)")
        text=once(text,'                    self._decision_available_at = None\n                    on_fills()',
            "                    self._decision_available_at = fill_at if phase=='interval' else None\n                    on_fills()")
        text=once(text,'position.route_metadata.get("entry_execution_phase") == "open"):',
            'position.route_metadata.get("entry_execution_phase") in ("open","interval")):')
        return text
    if name=='backtests/engine/capacity_audit.py':
        return once(text,'    attempts = [r for r in broker.journal',
            "    if policy.contract().get('version')=='"+VERSION+"':\n        from .olr_interval_service import audit_interval_capacity\n        return audit_interval_capacity(broker)\n    attempts = [r for r in broker.journal")
    if name=='backtests/engine/integrity.py':
        text=once(text,'    errors = []\n',"    interval_service=bool(broker.capacity_policy and broker.capacity_policy.contract().get('version')=='"+VERSION+"')\n    errors = []\n")
        text=text.replace("contract().get('version')=='causal-minute-participation-olr-v5'", "contract().get('version') in ('causal-minute-participation-olr-v5','"+VERSION+"')")
        text=once(text,'        if kind in {"opening_marks", "closing_marks"}:',
            '        if kind in {"opening_marks", "closing_marks", "interval_marks"}:')
        text=once(text,'            else:\n                record_deadline(datetime.fromisoformat(row["timestamp"]), completed=True)',
            '            elif kind == "closing_marks":\n                record_deadline(datetime.fromisoformat(row["timestamp"]), completed=True)')
        # Delayed auction marks do not become available at interval service.
        text=once(text,'            if expected is not None and row["prices"] != expected_marks.get((row["timestamp"], field), {}):',
            '''            expected_prices=expected_marks.get((row['timestamp'],field),{})
            if kind=='interval_marks':
                from .olr_minute_audit_v5 import continuous_interval
                expected_prices={s:p for s,p in expected_prices.items() if continuous_interval(
                    MarketBar(**{**expected_bars[(s,row['timestamp'])],'timestamp':datetime.fromisoformat(row['timestamp'])}))}
            if expected is not None and row["prices"] != expected_prices:''')
        text=once(text,'or bar.timestamp != fill["timestamp"]:',
            'or (bar.timestamp + timedelta(minutes=1 if interval_service and meta.get("execution_phase")=="interval" else 0)) != fill["timestamp"]:')
        text=once(text,'{"open": 0, "intrabar": 1, "auction": 2}.get(phase, -1)',
            '{"open": 0, "intrabar": 1, "interval": 1, "auction": 2}.get(phase, -1)')
        text=once(text,'            expected_base = bar.open\n            if phase != "open":',
            '            expected_base = bar.close if interval_service else bar.open\n            if phase != ("interval" if interval_service else "open"):')
        text=once(text,'                    base = float(source["open"])',
            '                    base = float(source["close"] if interval_service else source["open"])')
        return text
    if name=='backtests/engine/qualification_evidence.py':
        text=text.replace("contract().get('version')=='causal-minute-participation-olr-v5'", "contract().get('version') in ('causal-minute-participation-olr-v5','"+VERSION+"')")
        return once(text,'        model_version=MINUTE_CAPACITY_VERSION',
            '        model_version=broker.capacity_policy.contract()["version"]')
    if name=='backtests/engine/olr_minute_audit_v5.py':
        text=once(text,"p['phase']=='open'", "p['phase'] in ('open','interval')")
        # Filter after the original independent predicate, matching the registered
        # mechanism clock; coarse native management still observes both rules.
        anchor="                if reason:expected_signals"
        text=once(text,anchor,"                if reason not in getattr(broker,'minute_exit_reasons',('failed_followthrough','mfe_fade')):reason=''\n"+anchor)
        text=once(text,"if kind in ('opening_marks','closing_marks'):\n            marks.update(row['prices']);at=row['timestamp']", "if kind in ('opening_marks','closing_marks','interval_marks'):\n            marks.update(row['prices']);at=row['timestamp']")
        anchor="        if previous_key is not None and key<previous_key:raise ValueError('Noncausal fine NAV chronology')"
        text=once(text,anchor,"""        if broker.capacity_policy.contract().get('version')=='olr-completed-interval-service-v1':
            # Completed interval marks/fills precede the next opening at the
            # same timestamp. A point-auction fill follows its own opening;
            # assigning every fill to the completion phase reverses that pair.
            new_opening_phase=(kind=='opening_marks' or
                kind=='fill' and row['fill']['metadata']['execution_phase']=='auction')
            key=(datetime.fromisoformat(str(at))+timedelta(minutes=1 if kind in ('closing_marks','interval_marks') else 0),1 if new_opening_phase else 0)
        if previous_key is not None and key<previous_key:raise ValueError('Noncausal fine NAV chronology')""")
        return text
    if name=='backtests/engine/olr_minute_source_v5.py':
        anchor="    if contract!={'source_definition':name,'fast_exit':broker.minute_fast_exit,'version':'olr-minute-clock-v5'}:"
        text=once(text,anchor,"""    structural=identity['configuration'].get('structural_decision_v1')
    if structural:
        settings=json.loads(generation.read(generation.root/structural['settings_path']))
        masks={'control':(),'failure_clock':('failed_followthrough',),'fade_clock':('mfe_fade',),'both_clocks':('failed_followthrough','mfe_fade'),'structural_partial':()}
        if (tuple(broker.minute_exit_reasons)!=masks[settings['mode']] or broker.minute_fast_exit!=bool(masks[settings['mode']])
                or (broker.capacity_policy.contract()['version']=='olr-completed-interval-service-v1')!=settings['interval_service']):
            raise ValueError('Structural mechanism/source registration mismatch')
    if contract!={'source_definition':name,'fast_exit':broker.minute_fast_exit,'version':'olr-minute-clock-v5'}:""")
        return text
    if name=='strategy_olr/core/logic.py':
        text=once(text,'    protective_stop = _initial_protective_stop(',
            "    from .precision_v3 import entry_evidence\n    metadata.update(entry_evidence(symbol_state.session_bars,bar,config))\n    protective_stop = _initial_protective_stop(")
        return once(text,'    pending_id = symbol_state.pending_exit_order_id\n',
            "    if config.capacity_structural_partial:\n        from .precision_v3 import observe_entry_day\n        observe_entry_day(position,bar)\n    pending_id = symbol_state.pending_exit_order_id\n")
    raise ValueError('Unregistered structural patch '+name)
