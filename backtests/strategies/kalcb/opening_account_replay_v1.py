"""Research adapter for joint opening allocation through the native engine.

The full account is previewed before a timestamp batch, including fills that
are already eligible before the new decision. The chosen integer allocation
then runs through the ordinary core and broker from the original state.
"""
from dataclasses import asdict
from datetime import timedelta
import numpy as np

from .account_history_v1 import clone_complete_account
from .opening_book_risk_v1 import HistoricalPaths, allocate
from strategy_kalcb.core.account_allocation_v1 import entry_allocation


class OpeningAccountReplay:
    def __init__(self, mode, databases, *, pair_limit=0, pair_store=None):
        if mode not in ('control', 'joint', 'uniform'):
            raise ValueError('Unknown opening account intervention')
        self.mode = mode
        self.paths = HistoricalPaths(databases)
        self.records = []
        self.pairs = []
        self.active_pair = None
        self.pair_limit = pair_limit
        self.pair_store = pair_store

    def install(self):
        from backtests.engine import replay
        self.kernel = replay.drive_timestamp_batch
        replay.drive_timestamp_batch = self.drive

    def restore(self):
        from backtests.engine import replay
        replay.drive_timestamp_batch = self.kernel
        self.paths.close()

    def preview(self, adapter, broker, batch):
        aa, bb = clone_complete_account(adapter, broker)
        before_ids = {o.order_id for o in bb.orders}
        self.kernel(bb, batch, (aa,))
        orders = [o for o in bb.orders if o.order_id not in before_ids and o.side == 'BUY'
                  and o.strategy_id == 'KALCB' and o.metadata.get('entry_route') == 'first30_open_anchor']
        return aa, bb, orders

    def decision(self, adapter, broker, batch):
        aa, bb, orders = self.preview(adapter, broker, batch)
        stamp = batch[0].timestamp
        record = {'day':str(stamp.date()), 'available_at':str(stamp+timedelta(minutes=5)),
                  'mode':self.mode, 'intents':[], 'changed':False}
        if not orders:
            record['reason'] = 'no_opening_intents'
            return record, None
        intents = [dict(symbol=o.symbol, qty=o.qty, price=float(o.metadata['entry_price_ref']),
                        score=float(o.metadata['frontier_selection_score']),
                        reference_risk_per_share=float(o.metadata['risk_per_share']),
                        management_reference=float(o.metadata['entry_price_ref']),
                        hard_stop_enabled=bool(o.metadata.get('hard_stop_enabled')),
                        source_artifact_hash=o.metadata['source_artifact_hash']) for o in orders]
        if len({x['symbol'] for x in intents}) != len(intents):
            raise ValueError('Multiple opening lifetimes for one symbol')
        new_ids = {o.order_id for o in orders}
        fixed_qty = {}
        for (_, symbol), pos in bb.positions.items():
            fixed_qty[symbol] = fixed_qty.get(symbol, 0) + pos.qty
        for order in bb.orders:
            if order.side == 'BUY' and order.order_id not in new_ids:
                fixed_qty[order.symbol] = fixed_qty.get(order.symbol, 0) + order.qty
        # Existing pending exits do not free capital until the broker fills them.
        symbols = [r['symbol'] for r in intents]
        extra = sorted(set(fixed_qty)-set(symbols))
        all_symbols = symbols + extra
        history, support = self.paths.matrix(str(stamp.date()), all_symbols)
        record.update(intents=intents, support=support, existing_pending=[asdict(o) for o in bb.orders if o.order_id not in new_ids],
                      held_quantities={str(k):p.qty for k,p in bb.positions.items()})
        if history is None:
            record['reason'] = 'unsupported_history_parent_preserved'
            return record, None
        equity = bb._portfolio_equity()
        fixed_weights = np.array([fixed_qty.get(s, 0)*bb.last_prices[s]/equity if fixed_qty.get(s, 0) else 0
                                  for s in all_symbols])
        fixed = np.einsum('dts,s->dt', history, fixed_weights)
        pending_notional = sum(o.qty*bb.last_prices[o.symbol] for o in bb.orders
                               if o.side == 'BUY' and o.order_id not in new_ids)
        room = max(0., bb._buying_power_available()-pending_notional)
        budget = adapter.config.risk_per_trade_pct * adapter.config.heat_cap_r
        costs = bb.costs
        cost = (2*costs.commission_bps+costs.tax_bps_on_sell+2*costs.slippage_bps)/10000
        held_notional = sum(pos.qty*bb.last_prices[s] for (_,s),pos in bb.positions.items())
        future_fixed_cost = held_notional*(costs.commission_bps+costs.tax_bps_on_sell+costs.slippage_bps)/10000
        future_fixed_cost += pending_notional*cost
        fixed -= future_fixed_cost/equity
        solution = allocate(intents, history[:, :, :len(intents)], equity=equity,
                            buying_power=room, budget=budget, fixed=fixed, cost_fraction=cost)
        record.update(reason='supported', equity=equity, running_peak=max(bb.equity_curve),
                      budget_fraction=budget, fixed_weights=fixed_weights.tolist(),
                      reserved_pending_notional=pending_notional, available_buying_power=room,
                      cost_fraction=cost, fixed_future_cost_krw=future_fixed_cost,
                      allocation=solution, changed=solution['changed'])
        books = {arm:{r['symbol']:{'parent_qty':r['qty'], 'qty':q, 'available_at':record['available_at'],
                                  'mechanism':'joint_historical_path_budget_v1', 'arm':arm}
                      for r, q in zip(intents, solution[arm])} for arm in ('joint','uniform')}
        return record, books

    def drive(self, broker, bars, adapters, *, accepts=None):
        bars = tuple(bars)
        if len(adapters) != 1 or accepts is not None:
            raise ValueError('Account allocation requires one complete native KALCB adapter')
        adapter = adapters[0]
        stamp = bars[0].timestamp
        if self.active_pair and self.active_pair['day'] != str(stamp.date()):
            self.finish_pair()
        books = None
        if (stamp.hour, stamp.minute) == (9, 25):
            record, books = self.decision(adapter, broker, bars)
            self.records.append(record)
            if books is not None and record['changed'] and len(self.pairs) < self.pair_limit and self.mode == 'control':
                from .account_decision_v1 import capture, fork
                boundary = capture(adapter, broker, stamp+timedelta(minutes=5), self.pair_store)
                self.active_pair = dict(day=str(stamp.date()), boundary=boundary, arms={
                    name:fork(adapter, broker, boundary) for name in ('hold','joint','uniform')},
                    initial_cash=broker.cash, initial_positions={s:p.qty for (_,s),p in broker.positions.items()},
                    initial_marks=dict(broker.last_prices))
        selected = books.get(self.mode) if books else None
        with entry_allocation(selected):
            result = self.kernel(broker, bars, adapters)
        if self.active_pair:
            for name, (aa, bb) in self.active_pair['arms'].items():
                with entry_allocation(books.get(name) if books else None):
                    self.kernel(bb, bars, (aa,))
            self.active_pair['parent'] = adapter, broker
        return result

    def finish_pair(self):
        if self.active_pair is None:
            return
        from .account_decision_v1 import assert_identity, contrast, verify_suffix
        p = self.active_pair
        assert_identity(p['parent'], p['arms']['hold'])
        from backtests.engine.integrity import audit_broker
        audits={name:audit_broker(bb) for name,(aa,bb) in p['arms'].items()}
        if any(not a['execution_validity_pass'] for a in audits.values()):
            raise ValueError('Native fork execution ledger failed independent engine checks')
        checked={name:verify_suffix(p['boundary'],bb,p['initial_cash'],p['initial_positions'],p['initial_marks'])
                 for name,(aa,bb) in p['arms'].items()}
        ledgers={}
        if self.pair_store is not None:
            import gzip,json
            from pathlib import Path
            from .retention_research_io_v1 import bound
            for name,(aa,bb) in p['arms'].items():
                output=Path(self.pair_store).parent/'forks'/p['day']/(name+'.json.gz')
                output.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(output,'xt',encoding='utf-8') as stream:
                    json.dump({'boundary':asdict(p['boundary']), 'initial_cash':p['initial_cash'],
                        'initial_positions':p['initial_positions'],'initial_marks':p['initial_marks'],
                        'fills':[asdict(f) for f in bb.fills[p['boundary'].fills:]],
                        'source_batches':bb.expected_batches[p['boundary'].marks:],
                        'value':contrast(p['boundary'],p['arms']['hold'][1],bb),
                        'engine_checks':audits[name],'independent_suffix':checked[name]},
                        stream,default=str,separators=(',',':'),allow_nan=False)
                ledgers[name]=bound(output)
        self.pairs.append(dict(day=p['day'], boundary=asdict(p['boundary']), exact_disabled_identity=True,
            independent_suffix=checked,ledgers=ledgers,
            execution_checks={name:{'execution_validity_pass':a['execution_validity_pass'],
                'full_source_qualification':a['audit_status']} for name,a in audits.items()},
            contrasts={name:contrast(p['boundary'], p['arms']['hold'][1], p['arms'][name][1])
                       for name in ('joint','uniform')}))
        self.active_pair = None
