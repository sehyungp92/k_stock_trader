"""Outcome-blind, bounded native action coverage and funded entry evidence.

Calendar months and cash relative to the current holding define coverage,
not a trading rule. Every selected half sale must represent at least the
inherited meaningful-notional floor. No observed response selects an action.
"""
import json
from datetime import timedelta
from pathlib import Path

from backtests.engine.olr_minute_v5 import continuous_minute, fixed_exit_plan
from .funded_action_probe import FundedActionProbeDriver

VERSION = 'monthly-native-funded-population-v1'


def candidates(driver):
    from strategy_olr.core.logic import _position_risk, _completed_bar_exit_reason
    c = driver.session_cursor
    for bar in sorted(c.last_batch, key=lambda b: b.symbol):
        ss = driver.adapter.state.symbols.get(bar.symbol)
        p = ss.position if ss else None
        if (p is None or p.qty_open < 2 or ss.pending_entry_order_id or
                bar.timestamp.date() <= p.entry_time.date() or not continuous_minute(bar) or
                c.available_at != bar.timestamp + timedelta(minutes=1) or
                bar.symbol in driver.adapter.state.meta.get('priority_exit_pending', {})):
            continue
        resting = [o for o in driver.broker.orders if o.symbol == bar.symbol]
        if any(o.side != 'SELL' or o.order_id != ss.pending_exit_order_id or
               o.order_type != 'CLOSE_AUCTION' or o.reason != 'next_close_exit' for o in resting):
            continue
        if bool(resting) != bool(ss.pending_exit_order_id):
            continue
        risk = _position_risk(p)
        reason = _completed_bar_exit_reason((bar,), 0, p.entry_price, risk,
            p.max_favorable_price, fixed_exit_plan(driver.adapter.config), 0)['reason']
        if reason == 'mfe_fade':
            yield dict(symbol=bar.symbol, quantity=p.qty_open, risk=risk,
                entry_price=p.entry_price, entry_time=p.entry_time.isoformat(),
                high_water=p.max_favorable_price, bar=bar.to_json_dict(),
                available_at=c.available_at.isoformat(), pending_exit=ss.pending_exit_order_id or None)


def funding_context(driver, candidate):
    from strategy_olr.core.logic import pending_entry_reservation
    b = driver.broker
    reserved = sum(pending_entry_reservation(ss)[1] for ss in driver.adapter.state.symbols.values())
    deployable = max(0., b.cash-reserved)
    marked = candidate['quantity']*candidate['bar']['close']
    state = 'constrained' if deployable < marked else 'cash_rich'
    return dict(cell=candidate['available_at'][:7]+'_'+state,
        ledger_cash=b.cash, pending_buy_reservations=reserved, deployable_cash=deployable,
        current_marked_notional=marked, proposed_half_sale_notional=(candidate['quantity']-candidate['quantity']//2)*candidate['bar']['close'],
        meaningful_notional_floor=.01*b.initial_equity,
        current_equity=b._portfolio_equity(),
        observed_position_giveback=candidate['quantity']*max(0.,candidate['high_water']-candidate['bar']['close']))


def decision_evidence(candidate):
    return {k:v for k,v in candidate.items() if k not in ('source_prefix','source_events','decision_view_hash','ordinal')}


def audit_context(broker):
    """Independent cash/reservation arithmetic at the committed boundary."""
    from math import isclose
    cash = broker.initial_equity
    reserve = 0.
    marks = {}
    quantities = {}
    errors = []
    count = 0
    for row in broker.journal:
        if row['event'] == 'fill':
            f = row['fill']; sign = 1 if f['side']=='BUY' else -1
            cash -= sign*f['qty']*f['price']+f['metadata']['fee']
            quantities[f['symbol']] = quantities.get(f['symbol'],0)+sign*f['qty']
        elif row['event'] in ('opening_marks','interval_marks','closing_marks'):
            marks.update(row['prices'])
        elif row['event'] == 'strategy_state':
            reserve = sum(v['reserved_notional'] for v in row['pending_entries'].values())
        elif row['event'] == 'funded_probe_boundary_v1':
            count += 1
            c = row.get('funding_context')
            if c is None:
                if broker.funded_population_cell is not None: errors.append('missing_population_context')
                continue
            marked = row['quantity']*row['bar']['close']
            half = (row['quantity']-row['quantity']//2)*row['bar']['close']
            deployable = max(0.,cash-reserve)
            cell = row['available_at'][:7]+('_constrained' if deployable < marked else '_cash_rich')
            expected = dict(ledger_cash=cash,pending_buy_reservations=reserve,deployable_cash=deployable,
                current_marked_notional=marked,proposed_half_sale_notional=half,
                meaningful_notional_floor=.01*broker.initial_equity,
                current_equity=cash+sum(q*marks[s] for s,q in quantities.items() if q),
                observed_position_giveback=row['quantity']*max(0.,row['high_water']-row['bar']['close']))
            if c['cell'] != cell or cell != broker.funded_population_cell or half < .01*broker.initial_equity:
                errors.append('unearned_population_cell')
            if any(not isclose(c[k],v,abs_tol=1e-5) for k,v in expected.items()):
                errors.append('population_funding_arithmetic')
    if count != 1: errors.append('population_boundary_count')
    return dict(pass_=not errors,errors=errors,scope='Native execution audit separately verifies prices, marks, quantities and reservations.')


class FundedPopulationDriver(FundedActionProbeDriver):
    def __init__(self,*args,population,**kwargs):
        if set(population) != {'version','cell','catalog_reference'} or population['version'] != VERSION:
            raise ValueError('Unregistered population protocol')
        if population['cell'] is None and kwargs['probe']['action'] != 'hold':
            raise ValueError('Catalog cannot trade')
        if (population['cell'] is None) != (population['catalog_reference'] is None):
            raise ValueError('Every population action needs the sealed catalog reference')
        super().__init__(*args,**kwargs)
        self.population = population
        self.population_cells = {}
        self.broker.funded_population_cell = population['cell']
        self.adapter.state.meta['observe_funded_entry_context_v1'] = True

    def eligible(self):
        if self.population['cell'] is not None and self.probe_selection is not None:
            return None
        selected = None
        for row in candidates(self):
            context = funding_context(self,row)
            cell = context['cell']
            meaningful = context['proposed_half_sale_notional'] >= context['meaningful_notional_floor']
            evidence = dict(row,funding_context=context)
            if meaningful and cell not in self.population_cells:
                self.population_cells[cell] = evidence
            if self.probe_selection is None and selected is None:
                if self.population['cell'] is None:
                    # The catalog retains the exact V5 first-probe hold path.
                    selected = row
                elif meaningful and cell == self.population['cell']:
                    if evidence != self.population['catalog_reference']:
                        raise ValueError('Population selection differs from the sealed native catalog')
                    selected = evidence
        return selected

    def export_probe(self):
        result = super().export_probe()
        audited = audit_context(self.broker)
        if not audited['pass_']: raise ValueError('Population funding context audit failed: '+str(audited))
        value = dict(version=VERSION,cell=self.population['cell'],cells=self.population_cells,
            audit=audited,entry_contexts=self.adapter.state.meta.get('funded_entry_contexts_v1',[]),
            no_policy_authority=True,economic_objective_achieved=False)
        with (Path(self.output)/'funded_population_receipt.json').open('x',encoding='utf8') as f:
            json.dump(value,f,indent=2,default=str)
        return result
