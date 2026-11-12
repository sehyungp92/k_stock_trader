"""Causal scenario-loss budgets for OLR's native ranked entry cohort.

The ATR allowance is a sizing scenario, never a guaranteed stop fill. Pending
buys consume the envelope; pending sells release it only after actual fills.
"""
from dataclasses import replace
from math import floor, isfinite

from deployment.olr_kalcb.intent_allocator import IntentAllocationConfig, allocate_entry_intents


def enabled(config):
    return any((config.capacity_loss_position_pct, config.capacity_loss_sector_pct,
                config.capacity_loss_total_pct))


def unit_loss(metadata, config):
    price = float(metadata.get('estimated_entry_price', metadata.get('entry_price', 0)))
    risk = float(metadata.get('risk_per_share', 0))
    if not all(isfinite(x) and x > 0 for x in (price, risk)):
        raise ValueError('Loss budgeting requires positive sourced entry price and ATR risk')
    costs = (2 * config.commission_bps + config.tax_bps_on_sell + 2 * config.slippage_bps) / 10000
    distance = risk * config.capacity_loss_atr_multiple
    if config.capacity_opening_tail:
        evidence = metadata.get('opening_tail_evidence')
        if not evidence or not isfinite(evidence['fraction']) or evidence['fraction'] < 0:
            raise ValueError('Opening-tail sizing requires sourced finite evidence')
        distance = max(distance, price * evidence['fraction'])
    return distance + price * costs


def committed(state, config):
    from .logic import pending_entry_reservation
    symbols, sectors = {}, {}
    def add(symbol, sector, qty, metadata):
        if qty <= 0:
            return
        amount = qty * unit_loss(metadata, config)
        symbols[symbol] = symbols.get(symbol, 0.) + amount
        sectors[sector] = sectors.get(sector, 0.) + amount
    for symbol, s in state.symbols.items():
        if s.position is not None:
            p = s.position
            add(symbol, p.sector, p.qty_open, {**p.metadata, 'entry_price': p.entry_price})
        if s.pending_entry_order_id:
            qty, _ = pending_entry_reservation(s)
            add(symbol, s.pending_entry_metadata.get('sector', 'UNKNOWN'), qty, s.pending_entry_metadata)
    return symbols, sectors


def allocate(intents, account, at, config, state, by_id, *, prior=()):
    """Apply risk and funding together; released cash reaches lower native ranks.

    `prior` is this cohort's already approved base allocation when allocating
    top-ups. It prevents redistribution from bypassing any scenario-loss cap.
    """
    if not enabled(config):
        return allocate_entry_intents(intents, account, at, IntentAllocationConfig())
    symbols, sectors = committed(state, config)
    def charge(action, qty):
        amount = qty * unit_loss(action.metadata, config)
        sector = action.metadata.get('sector', 'UNKNOWN')
        symbols[action.symbol] = symbols.get(action.symbol, 0.) + amount
        sectors[sector] = sectors.get(sector, 0.) + amount
    for d in prior:
        charge(by_id[d.intent_id], d.approved_qty)
    running = account
    results = []
    for intent in sorted(intents, key=lambda i: (i.native_rank, i.available_at, i.strategy_id, i.symbol, i.intent_id)):
        action = by_id[intent.intent_id]
        loss = unit_loss(action.metadata, config)
        bounds = {}
        for label, fraction, used in (
            ('position_loss_budget', config.capacity_loss_position_pct, symbols.get(intent.symbol, 0.)),
            ('sector_loss_budget', config.capacity_loss_sector_pct, sectors.get(intent.sector, 0.)),
            ('total_loss_budget', config.capacity_loss_total_pct, sum(symbols.values()))):
            if fraction:
                bounds[label] = floor(max(0., account.equity * fraction - used) / loss)
        binding, limit = min(bounds.items(), key=lambda item: item[1])
        cap = min(limit, intent.capacity_qty) if intent.capacity_qty is not None else limit
        d = allocate_entry_intents([replace(intent, capacity_qty=cap)], running, at, IntentAllocationConfig())[0]
        if d.binding_limit == 'forecast_capacity' and cap == limit:
            d = replace(d, binding_limit=binding, reason='resized' if d.approved_qty else 'insufficient_' + binding)
        results.append(d)
        if d.approved_qty:
            charge(action, d.approved_qty)
            amount = d.reserved_notional
            symbol_cash = dict(running.symbol_committed)
            sector_cash = dict(running.sector_committed)
            strategy_cash = dict(running.strategy_committed)
            slots = dict(running.sector_slots)
            symbol_cash[intent.symbol] = symbol_cash.get(intent.symbol, 0.) + amount
            sector_cash[intent.sector] = sector_cash.get(intent.sector, 0.) + amount
            strategy_cash[intent.strategy_id] = strategy_cash.get(intent.strategy_id, 0.) + amount
            slots[intent.strategy_id, intent.sector] = slots.get((intent.strategy_id, intent.sector), 0) + 1
            running = replace(running, funding_limit=max(0., running.funding_limit - amount),
                gross_committed=running.gross_committed + amount, symbol_committed=symbol_cash,
                sector_committed=sector_cash, strategy_committed=strategy_cash, sector_slots=slots,
                occupied=running.occupied | {(intent.strategy_id, intent.symbol)})
    return results
