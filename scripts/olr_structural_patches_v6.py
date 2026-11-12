"""Small native integration patch, applied only to a fresh generation."""

CHANGED=('strategy_olr/core/capacity.py','scripts/portfolio_synergy_study.py')
NEW=('strategy_olr/core/liquidity_v6.py','strategy_olr/core/liquidity_runtime_v6.py','backtests/engine/olr_liquidity_audit_v6.py')

def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Structural patch anchor mismatch: '+old)
    return text.replace(old,new)

def patch(name,text):
    if name=='strategy_olr/core/capacity.py':
        return once(text,'    auction = [(datetime.fromisoformat(t), v) for t, v in history.get(\'auction\', [])]',
            "    from .liquidity_runtime_v6 import override_forecast\n    structural = override_forecast(state, bar, config, evidence, limits)\n    if structural is not None:return structural\n    auction = [(datetime.fromisoformat(t), v) for t, v in history.get('auction', [])]")
    if name!='scripts/portfolio_synergy_study.py':raise ValueError(name)
    text=once(text,'    daily, reasons = [], Counter()', '''    structural_panel=None
    if study.get('structural_liquidity_contract'):
        from strategy_olr.core.liquidity_v6 import CausalLiquidityPanel
        from strategy_olr.core.liquidity_runtime_v6 import KEY,day_context,observed_prefix
        from backtests.engine.olr_liquidity_audit_v6 import verify_projection
        structural_definition=bound('structural_liquidity_contract')
        mode=structural_definition['mode']
        if mode not in ('off','broad_median'):raise ValueError('Unregistered liquidity model')
        structural_data=json.loads(generation.read(generation.root/structural_definition['panel']))
        structural_hash=digest(generation.root/structural_definition['panel'])
        projection=verify_projection(connection,structural_data,study['database_hash'])
        structural_panel=CausalLiquidityPanel(structural_data['rows'],structural_data['sessions'],structural_data['universe'])
        structural_adapter=next(a for a in adapters if a.strategy_id=='OLR')
        if study.get('capacity_checkpoint'):
            restored=checkpoint['structural_observed_prefix_v6']
            independent=observed_prefix(structural_panel,checkpoint['through'])
            if restored!=independent or checkpoint['structural_mode_v6']!=mode:
                raise ValueError('Structural observation continuity mismatch')
            projection.update(actual_is_prefix_restored=True,prior_rows=len(restored))
        write_json(args.output/'structural_projection_audit.json',projection)
    daily, reasons = [], Counter()''')
    if 'decision_stream = gzip.open(args.output / "native_decisions.jsonl.gz", "wt", encoding="utf-8")\n' not in text:
        raise ValueError('Native admission ledger must already be unconditional')
    text=once(text,'    for index, day in enumerate(sorted(ks)):', '''    for index, day in enumerate(sorted(ks)):
        if structural_panel is not None and mode!='off':
            structural_adapter.state.meta[KEY]=day_context(structural_panel,str(day),structural_hash)''')
    text=once(text,"            'capacity_history_v1':olr_adapter.state.meta.get('capacity_history_v1',{}),", """            'structural_observed_prefix_v6':observed_prefix(structural_panel,str(max(ks))) if structural_panel is not None else None,
            'structural_mode_v6':mode if structural_panel is not None else None,
            'capacity_history_v1':olr_adapter.state.meta.get('capacity_history_v1',{}),""")
    text=once(text,'    admissions = [row for row in broker.journal if row["event"] == "portfolio_admission"]', '''    if structural_panel is not None:
        from backtests.engine.olr_liquidity_audit_v6 import audit_requests
        structural_audit=audit_requests(args.output/'native_decisions.jsonl.gz',structural_data,mode,
            structural_hash,ocfg.capacity_participation)
        write_json(args.output/'structural_request_audit.json',structural_audit)
    admissions = [row for row in broker.journal if row["event"] == "portfolio_admission"]''')
    text=once(text,'    if args.standalone:\n        identity["configuration"]["source_dependency_scopes"]', '''    if structural_panel is not None:
        identity['configuration']['structural_liquidity_v6']={'contract':study['structural_liquidity_contract'],
            'mode':mode,'panel_sha256':structural_hash,'observation_policy':'fixed_103_universe_completed_sources',
            'future_labels_forbidden':True}
    if args.standalone:
        identity["configuration"]["source_dependency_scopes"]''')
    return text
