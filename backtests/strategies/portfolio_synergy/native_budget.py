"""Native strategy replay with separately sized sleeves and one funding ledger."""
from dataclasses import asdict
import math

from deployment.olr_kalcb.native_budget import NativeBudgetConfig, reservation_ledger, order_key, cash_limited_transfer
from strategy_common.actions import SubmitEntry
from .source_replay import NativeRiskSharedBroker


class NativeBudgetSharedBroker(NativeRiskSharedBroker):
    def __init__(self, *args, native_budget, **kwargs):
        super().__init__(*args, **kwargs)
        if self.portfolio_policy is None or self.quality_model or self.decision_rank:
            raise ValueError('Native budgets require explicit physical caps and native ordering')
        self.native_budget = native_budget
        self.allocation_events = []
        self.sleeve_cash = {'KALCB': self.initial_equity*native_budget.kalcb_capital_fraction,
                           'OLR': self.initial_equity*(1-native_budget.kalcb_capital_fraction)}
        self._last_transfer_day = None

    def _sleeve_equities(self):
        result = dict(self.sleeve_cash)
        for (sid, sym), p in self.positions.items(): result[sid] += p.qty*self.last_prices[sym]
        return result

    def _sleeve_sizing(self, sid):
        if self.native_budget.capital_mode == 'shared': return self._sizing_base()
        value = self._sleeve_equities()[sid]
        fraction = self.native_budget.kalcb_capital_fraction if sid == 'KALCB' else 1-self.native_budget.kalcb_capital_fraction
        return max(0., min(value, self.sizing_equity_cap*fraction) if self.sizing_equity_cap > 0 else value)

    def _apply_fill(self, order, price, timestamp):
        fill = super()._apply_fill(order, price, timestamp)
        if fill is not None:
            costs = self.strategy_costs[fill.strategy_id]
            fee = fill.qty*fill.price*(costs.commission_bps+(costs.tax_bps_on_sell if fill.side == 'SELL' else 0))/10000
            self.sleeve_cash[fill.strategy_id] += (1 if fill.side == 'SELL' else -1)*fill.qty*fill.price-fee
        return fill

    def process_batch(self, bars, *, on_fills=None):
        bars = tuple(bars)
        if bars and self.native_budget.capital_mode == 'daily_rebalanced' and bars[0].timestamp.date() != self._last_transfer_day:
            # Before the day's first opening marks: only already observed marks
            # and realized cash flows are available. This is a zero-sum bookkeeping
            # transfer, never assumed liquidation proceeds or a forecast signal.
            wealth = self._sleeve_equities()
            desired,transfer = cash_limited_transfer(wealth,self.sleeve_cash,self.native_budget.kalcb_capital_fraction)
            self.sleeve_cash['KALCB'] += transfer; self.sleeve_cash['OLR'] -= transfer
            self.journal.append({'event':'native_sleeve_transfer','timestamp':bars[0].timestamp.isoformat(),
                'requested_transfer':desired,'kalcb_transfer':transfer,'prior_wealth':wealth})
            self._last_transfer_day = bars[0].timestamp.date()
        return super().process_batch(bars,on_fills=on_fills)

    def _phase_order_key(self, order):
        key = super()._phase_order_key(order)
        if order.side != 'BUY' or self.native_budget.allocation_mode == 'symbol':
            return key
        return (key[0], *order_key(asdict(order), self.portfolio_policy.config.strategy_priority, self.native_budget))

    def _sizing_base(self):
        return NativeRiskSharedBroker.strategy_sizing_equity(self)

    def strategy_sizing_equity(self):
        # This hook is used by KALCB. OLR uses strategy_account_view instead.
        return self._sleeve_sizing('KALCB') * self.native_budget.kalcb_sizing_fraction

    def strategy_risk_cash(self, strategy_id):
        if strategy_id != 'KALCB':
            raise ValueError('Only KALCB uses native risk cash')
        return self.strategy_sizing_equity() - self._strategy_open_notional(strategy_id)

    def _budget_inputs(self, extra_order=None):
        orders = [asdict(o) for o in self.orders if o.side == 'BUY']
        if extra_order is not None:
            if any(o['order_id'] == extra_order.order_id for o in orders):
                raise ValueError('Execution order double-counted')
            orders.append(asdict(extra_order))
        for order in orders:
            order['filled_qty'] = self.order_filled.get(order['order_id'], 0)
        return dict(cash=self.cash,
            holdings=[{'strategy_id': sid, 'symbol': sym, 'qty': p.qty,
                       'risk_per_share': float(p.route_metadata.get('risk_per_share') or 0.)}
                      for (sid, sym), p in self.positions.items()],
            orders=orders, marks=dict(self.last_prices),
            costs={sid: asdict(c) for sid, c in self.strategy_costs.items()},
            leverage=self.strategy_leverage,
            caps=self.portfolio_policy.effective_notional_caps(max(0., self._portfolio_equity())),
            sectors=self.policy_sectors, priority=self.portfolio_policy.config.strategy_priority,
            config=self.native_budget, sizing_equity_cap=self.sizing_equity_cap,
            sleeve_equities=self._sleeve_equities() if self.native_budget.capital_mode != 'shared' else None)

    def strategy_account_view(self, strategy_id):
        if strategy_id != 'OLR':
            return super().strategy_account_view(strategy_id)
        equity = self._sleeve_sizing('OLR') * self.native_budget.olr_fraction
        foreign = 0.
        if any(o.side == 'BUY' and o.strategy_id != strategy_id for o in self.orders):
            ledger = reservation_ledger(**self._budget_inputs())
            foreign = sum(x['reserved_notional'] for x in ledger['allocations'].values()
                          if x['strategy_id'] != strategy_id)
        physical = min(max(0., self._portfolio_equity()) * self.buying_power_leverage,
                       self.portfolio_policy.effective_notional_caps(max(0., self._portfolio_equity()))['max_gross_notional'])
        # OLR subtracts its own pending reservation internally exactly once.
        funding = max(0., min(equity - self._strategy_open_notional('OLR'),
                             physical - self._open_notional() - foreign))
        return equity, funding

    def submit(self, action, submitted_at):
        oid = super().submit(action, submitted_at)
        if isinstance(action, SubmitEntry):
            ledger = reservation_ledger(**self._budget_inputs())
            self.allocation_events.append({'event': 'native_submission', 'strategy_id': action.strategy_id,
                'symbol': action.symbol, 'timestamp': submitted_at.isoformat(),
                'available_at': (self._decision_available_at or submitted_at).isoformat(),
                'submitted_order_id': oid, 'native_qty': action.qty,
                'native_metadata': dict(action.metadata),
                'funding': ledger['allocations'].get(oid), 'equity': ledger['equity']})
        return oid

    def _policy_execution_quantity(self, order, price, quantity, timestamp):
        allowed = super()._policy_execution_quantity(order, price, quantity, timestamp)
        if order.side != 'BUY':
            return allowed
        unit = price * (1 + self.strategy_costs[order.strategy_id].commission_bps / 10000)
        inputs = self._budget_inputs(order)
        ledger = reservation_ledger(**inputs, execution=(order.order_id, unit))
        reservation = ledger['allocations'][order.order_id]
        approved = min(allowed, reservation['qty'])
        self.journal.append({'event': 'native_budget_admission', 'order': asdict(order),
            'timestamp': timestamp.isoformat(), 'unit_cost': unit, 'offered_qty': allowed,
            'approved_qty': approved, 'config': asdict(self.native_budget),
            'reservation': reservation, 'ledger': ledger})
        return approved


def audit_native_budgets(broker):
    """Reconstruct cash, holdings and working orders from the execution journal.

No copied reservation snapshot is trusted as an account input. Physical source,
marks, fill clocks and costs are independently checked by audit_broker.
"""
    active, positions, marks, approvals, filled = {}, {}, {}, {}, {}
    cash = float(broker.initial_equity)
    sleeve_cash = {'KALCB': cash*broker.native_budget.kalcb_capital_fraction,
                  'OLR': cash*(1-broker.native_budget.kalcb_capital_fraction)}
    transfer_days = set(); transfers = []
    errors = []
    mismatches = []
    checked = 0
    def differences(expected, actual, path='ledger'):
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected) != set(actual): return [path + ':keys']
            return [item for key in expected for item in differences(expected[key], actual[key], path+'.'+str(key))]
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            equal = math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-7)
        else:
            equal = expected == actual
        return [] if equal else [{'field':path,'expected':expected,'actual':actual}]
    for row in broker.journal:
        event = row['event']
        order = row.get('order', {})
        oid = order.get('order_id')
        if event == 'native_sleeve_transfer':
            wealth = dict(sleeve_cash)
            for (sid, sym), p in positions.items(): wealth[sid] += p['qty']*marks[sym]
            desired,transfer = cash_limited_transfer(wealth,sleeve_cash,broker.native_budget.kalcb_capital_fraction)
            day = row['timestamp'][:10]
            if (broker.native_budget.capital_mode != 'daily_rebalanced' or day in transfer_days
                or differences(wealth,row['prior_wealth']) or differences(transfer,row['kalcb_transfer'])
                or differences(desired,row['requested_transfer'])):
                errors.append('invalid_native_sleeve_transfer')
            transfer_days.add(day)
            transfers.append({'timestamp':row['timestamp'],'requested_transfer':desired,
                'kalcb_transfer':transfer,'donor_cash':sleeve_cash['OLR' if desired > 0 else 'KALCB']})
            sleeve_cash['KALCB'] += transfer; sleeve_cash['OLR'] -= transfer
        elif event in ('opening_marks', 'closing_marks'):
            marks.update(row['prices'])
        elif event in ('submit', 'amend'):
            active[oid] = order
        elif event in ('cancel', 'expire', 'reject'):
            active.pop(oid, None)
            approvals.pop(oid, None)
        elif event == 'native_budget_admission':
            checked += 1
            # Match the physical account's marked addition order. Quantities are
            # exact; monetary comparisons tolerate sub-won floating roundoff.
            equity = cash
            for p in positions.values(): equity += p['qty'] * marks[p['symbol']]
            equity = max(0., equity)
            if active.get(oid) != order or row['config'] != asdict(broker.native_budget):
                errors.append('unbound_native_budget_order_or_config')
                continue
            wealth = dict(sleeve_cash)
            for (sid, sym), p in positions.items(): wealth[sid] += p['qty']*marks[sym]
            expected = reservation_ledger(cash=cash, holdings=list(positions.values()),
                orders=[{**o, 'filled_qty': filled.get(oid, 0)} for oid, o in active.items()], marks=marks,
                costs={sid: asdict(c) for sid, c in broker.strategy_costs.items()},
                leverage=broker.strategy_leverage,
                caps=broker.portfolio_policy.effective_notional_caps(equity),
                sectors=broker.policy_sectors, priority=broker.portfolio_policy.config.strategy_priority,
                config=broker.native_budget, execution=(oid, row['unit_cost']),
                sizing_equity_cap=broker.sizing_equity_cap,
                sleeve_equities=wealth if broker.native_budget.capital_mode != 'shared' else None)
            reservation = expected['allocations'][oid]
            diff = differences(expected, row['ledger']) + differences(reservation, row['reservation'], 'reservation')
            if diff:
                errors.append('native_reservation_account_mismatch')
                if len(mismatches) < 3: mismatches.append({'order_id':oid,'differences':diff[:10]})
            maximum = min(row['offered_qty'], reservation['qty'])
            if row['approved_qty'] != maximum:
                errors.append('native_budget_quantity_mismatch')
            approvals[oid] = maximum
        elif event == 'fill':
            fill = row['fill']; sid, sym, qty = fill['strategy_id'], fill['symbol'], fill['qty']
            key = (sid, sym)
            sleeve_cash[sid] += (1 if fill['side'] == 'SELL' else -1)*qty*fill['price']-fill['metadata']['fee']
            if fill['side'] == 'BUY':
                filled[oid] = filled.get(oid, 0) + qty
                if qty > approvals.pop(oid, -1):
                    errors.append('fill_exceeds_native_reservation')
                prior = positions.get(key, {})
                previous = prior.get('qty', 0)
                positions[key] = {'strategy_id': sid, 'symbol': sym, 'qty': previous + qty,
                    'risk_per_share': prior.get('risk_per_share', float(fill['metadata'].get('risk_per_share') or order.get('metadata', {}).get('risk_per_share') or 0.))}
                cash -= qty * fill['price'] + fill['metadata']['fee']
            else:
                positions[key]['qty'] -= qty
                if not positions[key]['qty']:
                    positions.pop(key)
                cash += qty * fill['price'] - fill['metadata']['fee']
            # A partial remainder is explicitly reinstated by the following amend.
            active.pop(oid, None)
    if not math.isclose(cash, broker.cash, abs_tol=1e-5):
        errors.append('native_budget_cash_reconciliation')
    if differences(sleeve_cash,broker.sleeve_cash): errors.append('native_sleeve_cash_reconciliation')
    if {k:p['qty'] for k,p in positions.items()} != {k:p.qty for k,p in broker.positions.items()}:
        errors.append('native_budget_position_reconciliation')
    return {'version': 'native-budget-audit-v1', 'pass': not errors,
            'errors': sorted(set(errors)), 'mismatches': mismatches, 'checked_admissions': checked,
            'terminal_deferred_intents': 0, 'native_submissions': len(broker.allocation_events),
            'terminal_pending_orders': len(broker.orders),
            'capital_mode':broker.native_budget.capital_mode,'terminal_sleeve_cash':sleeve_cash,
            'wealth_transfers':transfers,
            'semantics': 'Native quantities/lifecycles preserved; one feasible pending-funding ledger; no anticipated exits.'}
