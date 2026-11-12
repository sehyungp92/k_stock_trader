"""Exact, narrow registration of the research intervention in native audits."""
from scripts.olr_minute_exit_patches_v5 import once


def patch_runner(text):
    anchor = '    reserve_state = None'
    text = once(text, anchor, """    if study.get('funded_probe_settings'):
        from backtests.strategies.olr.funded_action_probe import FundedActionProbeDriver
        minute_driver=FundedActionProbeDriver(broker,minute_adapter,mode=structural['mode'],
            output=args.output,interval_service=structural['interval_service'],probe=bound('funded_probe_settings'))
    reserve_state = None""")
    text = once(text, "    if structural:\n        if not minute_driver.checkpoint_receipt", "    if study.get('funded_probe_settings'):\n        minute_driver.export_probe()\n    if structural:\n        if not minute_driver.checkpoint_receipt")
    return text


def patch_audit(text):
    text = once(text, '    next_open_at={};following=None', """    probe_contract=getattr(broker,'funded_probe_contract',None)
    probe_boundary=None;probe_orders=[]
    import hashlib,json
    probe_source=hashlib.sha256();probe_source_count=0
    next_open_at={};following=None""")
    text = once(text, '            for raw in batch:\n                bar,p=mark(raw)', """            for raw in batch:
                probe_source.update(json.dumps(raw,sort_keys=True,default=str).encode());probe_source_count+=1
                bar,p=mark(raw)""")
    anchor = "        elif kind=='minute_exit_signal_v5':"
    text = once(text, anchor, """        elif kind=='funded_probe_boundary_v1':
            from backtests.strategies.olr.funded_action_probe import VERSION,ACTIONS
            raw=row['bar'];symbol=row['symbol'];p=positions.get(symbol)
            valid=bool(probe_contract and probe_contract.get('version')==VERSION and
                probe_contract.get('action') in ACTIONS and probe_boundary is None and row['ordinal']==1 and p)
            if valid:
                bar=MarketBar(**{**raw,'timestamp':dt(raw['timestamp'])})
                prior=fine.get(raw['timestamp'],())
                sells=[o for o in active.values() if o['symbol']==symbol]
                risk=max(p['risk'] if p['risk']>0 else p['price']-p['stop'] if p['stop']>0 else 0,p['price']*.001,1.)
                close_r=(bar.close-p['price'])/risk;mfe_r=(p['high']-p['price'])/risk
                valid=bool(raw in prior and continuous_interval(bar) and bar.timestamp.date()>p['at'].date()
                    and dt(row['available_at'])==availability==bar.timestamp+timedelta(minutes=1)
                    and p['qty']>=2 and row['quantity']==p['qty'] and number(row['risk'],risk)
                    and number(row['entry_price'],p['price']) and row['entry_time']==p['at'].isoformat()
                    and number(row['high_water'],p['high']) and mfe_r>=1 and close_r<=max(0,mfe_r-1.25)
                    and not (close_r<=-1 and mfe_r<4)
                    and len(sells)<=1 and all(o['side']=='SELL' and o['order_type']=='CLOSE_AUCTION'
                        and o['reason']=='next_close_exit' for o in sells)
                    and row['pending_exit']==(sells[0]['order_id'] if sells else None)
                    and row['source_prefix']==probe_source.hexdigest() and row['source_events']==probe_source_count)
            if not valid:errors.append('funded_probe_unearned_boundary')
            else:probe_boundary=row;context='funded_probe';context_label=dt(raw['timestamp'])
        elif kind=='minute_exit_signal_v5':""")
    anchor = "                if context=='minute' and order['side']=='SELL' and order['order_type']=='MARKET':"
    text = once(text, anchor, """                funded_order=order['reason']=='funded_inventory_target'
                if funded_order:
                    p=positions.get(order['symbol']);meta=order['metadata']
                    action=probe_contract.get('action') if probe_contract else None
                    target=(probe_boundary['quantity']//2 if action=='half' else 0) if probe_boundary else None
                    if (not probe_boundary or action not in ('half','exit') or probe_orders or not p
                            or order['symbol']!=probe_boundary['symbol'] or order['side']!='SELL' or order['order_type']!='MARKET'
                            or order['qty']!=max(0,p['qty']-target) or order['qty']<=0
                            or meta.get('funded_action_id')!='first_native_fade_1' or meta.get('target_inventory')!=target
                            or meta.get('committed_at')!=probe_boundary['available_at']
                            or any(o['order_id']!=order['order_id'] and o['symbol']==order['symbol'] for o in active.values())):
                        errors.append('funded_probe_invalid_target_submission')
                    probe_orders.append(order['order_id'])
                if context=='funded_probe' and order['side']=='SELL' and not funded_order:
                    errors.append('funded_probe_unregistered_order')
                if context=='minute' and order['side']=='SELL' and order['order_type']=='MARKET' and not funded_order:""")
    text = once(text, "    if seen_fine!=list(fine):errors.append('minute_missing_observation')", """    if probe_contract:
        if probe_boundary is None:errors.append('funded_probe_missing_boundary')
        expected_count=0 if probe_contract['action']=='hold' else 1
        if len(probe_orders)!=expected_count:errors.append('funded_probe_missing_target')
    if seen_fine!=list(fine):errors.append('minute_missing_observation')""")
    text = once(text, '        unchanged_thresholds=True,control_or_treatment=', '        funded_probe_contract=probe_contract,funded_probe_orders=probe_orders,\n        unchanged_thresholds=True,control_or_treatment=')
    return text
