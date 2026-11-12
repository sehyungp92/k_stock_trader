"""Reviewable isolated native-audit changes; does not touch previous generations."""


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Native patch anchor mismatch: '+old)
    return text.replace(old,new)


def patch(name,text):
    if name=='strategy_olr/config.py':
        text=once(text,'    afternoon_empty_sector_flow_rescue: bool = False\n',
            '    afternoon_empty_sector_flow_rescue: bool = False\n    research_minute_exit: bool = False\n')
        text=once(text,'        import math\n','        import math\n        if type(self.research_minute_exit) is not bool:\n            raise ValueError("Minute exit mode must be boolean")\n')
        return once(text,'    "olr.afternoon.empty_sector_flow_rescue":',
            '    "olr.research.minute_exit": "research_minute_exit",\n    "olr.afternoon.empty_sector_flow_rescue":')
    if name=='strategy_olr/research.py':
        return once(text,'    "afternoon_empty_sector_flow_rescue",\n','    "afternoon_empty_sector_flow_rescue",\n    "research_minute_exit",\n')
    if name=='backtests/engine/capacity_audit.py':
        return once(text,'    attempts = [r for r in broker.journal', '''    if policy.contract().get('version')=='causal-minute-participation-olr-v5':
        from .olr_minute_audit_v5 import audit_minute_capacity
        return audit_minute_capacity(broker)
    attempts = [r for r in broker.journal''')
    if name=='backtests/engine/integrity.py':
        text=once(text,'    errors.extend(capacity_report["errors"])', '''    errors.extend(capacity_report["errors"])
    minute_clock_report=None
    if broker.capacity_policy is not None and broker.capacity_policy.contract().get('version')=='causal-minute-participation-olr-v5':
        from .olr_minute_audit_v5 import audit_minute_clock
        minute_clock_report=audit_minute_clock(broker)
        errors.extend(minute_clock_report['errors'])''')
        text=once(text,'        available = stamp + timedelta(minutes=5) if completed else stamp', '''        # Infer the deadline clock from bound input events, never the strategy's
        # requested risk clock. Five-minute paths retain their original deadline.
        if completed:
            timeframes={bar['timeframe'] for (symbol,at),bar in expected_bars.items() if at==stamp.isoformat()}
            if len(timeframes)!=1 or next(iter(timeframes)) not in ('1m','5m'):
                raise ValueError('Unbound or mixed completion interval')
            minutes=int(next(iter(timeframes))[:-1])
            available=stamp+timedelta(minutes=minutes)
        else:available=stamp''')
        # Build once: a fine replay can contain millions of bars.
        text=once(text,'    expected_marks = {}','    expected_marks = {}\n    expected_timeframes = defaultdict(set)')
        text=once(text,'                expected_bars[key] = bar','                expected_bars[key] = bar\n                expected_timeframes[key[1]].add(bar["timeframe"])')
        text=once(text,"timeframes={bar['timeframe'] for (symbol,at),bar in expected_bars.items() if at==stamp.isoformat()}","timeframes=expected_timeframes.get(stamp.isoformat(),set())")
        text=once(text,'    receipt["continuous_session_end_inventory"] = deadline_positions',
            '    if minute_clock_report is not None:receipt["minute_clock_audit"] = minute_clock_report\n    receipt["continuous_session_end_inventory"] = deadline_positions')
        return text
    if name=='backtests/engine/qualification_evidence.py':
        text=once(text,'        expected = [bar for batch in getattr(broker, "expected_batches", ()) for bar in batch]', '''        if broker.capacity_policy is not None and broker.capacity_policy.contract().get('version')=='causal-minute-participation-olr-v5':
            from .olr_minute_source_v5 import verify_minute_sources
            with generation.activate():
                details['minute_sources']=verify_minute_sources(broker,generation,evidence,identity)
            # This projection is exclusively for existing coarse source-feature
            # obligations. Fine ledger, prices, capacity and marks keep the full
            # independent expected_batches stream; no broker view is substituted.
            source_batches=broker.coarse_expected_batches
        else:source_batches=getattr(broker,'expected_batches',())
        expected = [bar for batch in source_batches for bar in batch]''')
        text=once(text,'    declaration = evidence["capacity"]\n', '''    declaration = evidence["capacity"]
    model_version=CAPACITY_VERSION
    if broker.capacity_policy is not None and broker.capacity_policy.contract().get('version')=='causal-minute-participation-olr-v5':
        from .olr_minute_v5 import MinuteParticipationScenario,MINUTE_CAPACITY_VERSION
        if not isinstance(broker.capacity_policy,MinuteParticipationScenario) or not evidence.get('minute_source_definition'):
            raise ValueError('Unbound minute research model')
        model_version=MINUTE_CAPACITY_VERSION
''')
        text=once(text,'or declaration.get("model_version") != CAPACITY_VERSION','or declaration.get("model_version") != model_version')
        return once(text,'"model_version": CAPACITY_VERSION, "policy": execution_policy(broker),',
            '"model_version": model_version, "policy": execution_policy(broker),')
    if name=='scripts/portfolio_synergy_study.py':
        text=once(text,'    broker = broker_class(study["initial_equity"], strategy_costs=costs,', '''    minute_clock=study.get('minute_source_definition')
    if minute_clock:
        if args.standalone!='OLR' or allocation is not None or native_budget is not None:
            raise ValueError('Minute study requires unchanged standalone native OLR allocation')
        from backtests.engine.olr_minute_v5 import MinuteParticipationScenario,MinuteOLRDriver
        from backtests.engine.olr_minute_source_v5 import load_minute_day
        capacity_model=MinuteParticipationScenario(fraction,scenario_name)
    else:capacity_model=BarParticipationScenario(fraction,scenario_name)
    broker = broker_class(study["initial_equity"], strategy_costs=costs,''')
        text=once(text,'capacity_policy=BarParticipationScenario(fraction, scenario_name), **broker_options)',
            'capacity_policy=capacity_model, **broker_options)')
        text=once(text,'    daily, reasons = [], Counter()', '''    minute_driver=None
    if minute_clock:
        import sqlite3
        minute_definition=bound('minute_source_definition')
        minute_path=generation.root/minute_definition['minute_database']
        with generation.open_verified(minute_path):pass
        minute_connection=sqlite3.connect(minute_path.as_uri()+'?mode=ro',uri=True)
        minute_adapter=next(a for a in adapters if a.strategy_id=='OLR')
        minute_driver=MinuteOLRDriver(broker,minute_adapter,fast_exit=ocfg.research_minute_exit)
    daily, reasons = [], Counter()''')
        old='''        for _, batch in groupby(bars, key=lambda bar: bar.timestamp):
            for event in drive_timestamp_batch(broker, tuple(batch), adapters):
                reasons[event.reason] += 1
                if decision_stream is not None:
                    decision_stream.write(json.dumps(event.to_json_dict(), default=str) + "\\n")'''
        new='''        if minute_driver is not None:
            fine_symbols={c.symbol for c in osnaps[day].candidates} if day in osnaps else set()
            fine_symbols.update(symbol for _,symbol in broker.positions)
            fine_symbols.update(order.symbol for order in broker.orders)
            fine_symbols.add('000100')
            fine=load_minute_day(minute_connection,day,fine_symbols)
            events=minute_driver.run_day(fine,tuple(bars),symbols=fine_symbols)
            for event in events:
                reasons[event.reason]+=1
                if decision_stream is not None:decision_stream.write(json.dumps(event.to_json_dict(),default=str)+'\\n')
        else:
            for _, batch in groupby(bars, key=lambda bar: bar.timestamp):
                for event in drive_timestamp_batch(broker, tuple(batch), adapters):
                    reasons[event.reason] += 1
                    if decision_stream is not None:
                        decision_stream.write(json.dumps(event.to_json_dict(), default=str) + "\\n")'''
        text=once(text,old,new)
        text=once(text,'    connection.close()\n', '''    if minute_driver is not None:
        minute_connection.close()
        from backtests.engine.olr_minute_audit_v5 import minute_nav_evidence
        grid_key=args.scenario+('_capped' if args.no_upward_compounding else '')
        grid_name=study['minute_original_grids'][grid_key]
        original_grid=json.loads(generation.read(generation.root/grid_name))
        nav_proof=minute_nav_evidence(broker,original_grid)
        if abs(nav_proof['final_equity']-broker._portfolio_equity())>1e-5:
            raise ValueError('Independent fine NAV differs from native ledger')
        nav_proof['original_grid_sha256']=generation.digest(generation.root/grid_name)
        write_json(args.output/'minute_nav_evidence.json',nav_proof)
        write_json(args.output/'minute_coarse_equity.json',broker.minute_coarse_nav)
        with gzip.open(args.output/'minute_events.jsonl.gz','wt',encoding='utf-8') as stream:
            for row in broker.journal:
                if row['event'] in ('minute_day_scope_v5','minute_exit_signal_v5','fill','submit','cancel','expire','reject'):
                    stream.write(json.dumps(row,default=str)+'\\n')
    connection.close()
''')
        text=once(text,'    if args.standalone:\n        identity["configuration"]["source_dependency_scopes"]', '''    if minute_driver is not None:
        identity['configuration']['minute_clock']={'version':'olr-minute-clock-v5',
            'source_definition':minute_clock,'fast_exit':bool(ocfg.research_minute_exit)}
    if args.standalone:
        identity["configuration"]["source_dependency_scopes"]''')
        return text
    raise ValueError(name)
