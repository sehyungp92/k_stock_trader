"""Native fixed-request lanes, isolated imports and explicit missing outcomes."""
from collections import Counter
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, time
from pathlib import Path
import argparse
import gzip
import importlib.util
import json
import math
import os
import sqlite3
import sys
import traceback


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def adapter_class():
    from backtests.strategies.olr.runner import OLRReplayAdapter

    class FixedSignalAdapter(OLRReplayAdapter):
        def __init__(self, config, snapshot, symbol, budget):
            super().__init__(config, {snapshot.trade_date: snapshot})
            if budget <= 0 or budget > 20_000_000:
                raise ValueError('Unregistered fixed request budget')
            if symbol not in {c.symbol for c in snapshot.candidates[:config.overnight_slot_count] if c.tradable}:
                raise ValueError('Only original in-slot incumbents may be labeled')
            self.symbol = symbol
            self.state.meta['fixed_signal_budget_v8'] = float(budget)

        def _should_step_symbol(self, symbol, snapshot):
            return symbol == self.symbol and super()._should_step_symbol(symbol, snapshot)

    return FixedSignalAdapter


def make_lane(config, snapshot, symbol, budget, scenario):
    from backtests.engine.sim_broker import BrokerCosts
    from backtests.engine.olr_minute_v5 import MinuteParticipationScenario, MinuteOLRDriver
    from backtests.strategies.portfolio_synergy.source_replay import NativeRiskSharedBroker
    fraction, slip = {'baseline': (.01, 5), 'combined': (.005, 15)}[scenario]
    cfg = replace(config, slippage_bps=slip, tax_bps_on_sell=20)
    # The inherited capacity exit guard is off. No fabricated historical
    # forecast can become an extra exit in an isolated lane.
    if cfg.capacity_exit_guard or cfg.capacity_value_mode != 'off' or cfg.research_minute_exit:
        raise ValueError('Unexpected additional forecast/exit intervention')
    adapter = adapter_class()(cfg, snapshot, symbol, budget)
    broker = NativeRiskSharedBroker(100_000_000, strategy_costs={'OLR': BrokerCosts(2, 20, slip)},
        strategy_leverage={'OLR': 1}, capacity_policy=MinuteParticipationScenario(fraction, scenario))
    broker.policy_sectors = {c.symbol: c.sector for c in snapshot.candidates}
    driver = MinuteOLRDriver(broker, adapter, fast_exit=False)
    return broker, adapter, driver


def closed(broker):
    return not broker.positions and not broker.orders


def capital_risk(broker, budget):
    """Dollar path, independent of the lane's deliberately idle account cash."""
    holdings, marks = {}, {}
    cash_delta = peak_pnl = max_drawdown = 0.
    for row in broker.journal:
        if row['event'] in ('opening_marks', 'closing_marks'):
            marks.update(row['prices'])
        elif row['event'] == 'fill':
            fill = row['fill']; qty = fill['qty']; price = fill['price']
            buy = fill['side'] == 'BUY'
            holdings[fill['symbol']] = holdings.get(fill['symbol'], 0) + (qty if buy else -qty)
            cash_delta += (-qty*price if buy else qty*price)-fill['metadata']['fee']
        else:
            continue
        assert all(not qty or symbol in marks for symbol, qty in holdings.items())
        pnl = cash_delta+sum(qty*marks.get(symbol, 0) for symbol, qty in holdings.items())
        peak_pnl = max(peak_pnl, pnl)
        max_drawdown = max(max_drawdown, peak_pnl-pnl)
    return dict(max_drawdown_krw=max_drawdown, max_drawdown_over_target_pct=100*max_drawdown/budget,
        basis='Maximum dollar peak-to-trough loss across every native opening/closing/fill-cost event, divided by fixed target; not portfolio maximum drawdown.')


def assess(broker, adapter, budget, *, unsupported=None):
    """Native execution arithmetic plus separately bound source verification.

    The generic portfolio qualifier expects a whole portfolio day's selected
    universe; these isolated lanes deliberately have a different scope. Its
    execution errors MUST be empty, while its unverified portfolio source
    coverage is retained verbatim and never relabeled 'pass'.
    """
    from backtests.engine.integrity import audit_broker
    from backtests.engine.olr_minute_audit_v5 import minute_nav_evidence
    adapter._sync_new_fills(broker)
    audit = audit_broker(broker)
    if not audit['execution_validity_pass']:
        raise ValueError('Native execution audit: ' + repr(audit['audit_receipt']['errors']))
    grid = [(row['timestamp'], 0) for row in broker.minute_coarse_nav]
    nav = minute_nav_evidence(broker, grid)
    assert math.isclose(nav['final_equity'], broker._portfolio_equity(), abs_tol=1e-5)
    buys = [f for f in broker.fills if f.side == 'BUY']
    sells = [f for f in broker.fills if f.side == 'SELL']
    acquired = sum(f.qty*f.price for f in buys)
    cost = sum(f.metadata['fee'] for f in broker.fills)
    pnl = sum(f.qty*f.price for f in sells) - acquired - cost
    entry_orders = [r['order'] for r in broker.journal if r['event'] == 'submit' and r['order']['side'] == 'BUY']
    assert len(entry_orders) <= 1, 'Exactly one original opportunity per lane'
    for order in entry_orders:
        meta = order['metadata']
        assert meta['target_notional'] == budget
        upper = math.floor(budget/(meta['estimated_entry_price']*(1+meta['entry_cost_buffer_pct'])))
        assert 0 < order['qty'] <= upper, 'Native loss allocation cannot exceed fixed request'
    status = ('unsupported_source' if unsupported else 'unfilled' if not buys and closed(broker)
              else 'matured' if buys and closed(broker) else 'right_censored')
    if status == 'matured':
        assert sum(f.qty for f in buys) == sum(f.qty for f in sells)
        assert math.isclose(broker.cash-100_000_000, pnl, abs_tol=1e-5)
    result = dict(status=status, unsupported_reason=unsupported, budget=budget,
        requested_quantity=sum(o['qty'] for o in entry_orders), filled_quantity=sum(f.qty for f in buys),
        filled_notional=acquired, total_fees=cost, residual_quantity=sum(p.qty for p in broker.positions.values()),
        pending_orders=len(broker.orders), net_pnl=pnl if status == 'matured' else None,
        net_bps=10000*pnl/acquired if status == 'matured' else None,
        budget_net_bps=10000*pnl/budget if status == 'matured' else None,
        execution_arithmetic_pass=True, generic_portfolio_audit_pass=audit['audit_pass'],
        independent_source_proof_required=True,
        native_requests=entry_orders, native_exit_reasons=dict(Counter(f.reason for f in sells)),
        fill_count=len(broker.fills), nav=nav, fixed_target_risk=capital_risk(broker, budget))
    return result, audit


def run_labels(c, design, ready, guard, clock):
    from backtests.core.input_manifest import InputManifest
    from backtests.engine.olr_minute_source_v5 import origin_verifier, load_minute_day, same, FIELDS
    from strategy_olr.models import OLRDailySnapshot
    from strategy_olr.config import OLRConfig
    from strategy_olr.core.logic import _entry_plan
    from strategy_olr.core.precision_v3 import entry_plan
    from strategy_olr.execution import find_olr_entry_signal
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff
    from scripts.kalcb_executed_value_study import connect, load_day
    from scripts.update_kis_intraday import aggregate_timeframe
    import pandas as pd

    if (c.ROOT/'labels_complete.json').exists() or (c.ROOT/'lanes').exists():
        raise FileExistsError('No duplicate run or implicit resume of a label generation')
    (c.ROOT/'lanes').mkdir()
    generation = InputManifest(ready['study']['manifest'])
    files = generation.root
    request = c.read(files/ready['study']['olr_request'])
    config = OLRConfig.from_mapping(request['config'], request['mutations'])
    config = replace(config, **ready['study']['candidates']['candidate']['olr_capacity'])
    assert config.overnight_slot_count == 6
    snapshots = {s.trade_date: s for raw in c.read(files/ready['study']['olr_snapshots'])
                 if (s := OLRDailySnapshot.from_json_dict(raw))}
    definition = c.read(files/ready['study']['minute_source_definition'])
    coarse_db = files/definition['canonical_database']
    fine_db = files/definition['minute_database']
    for path in (coarse_db, fine_db):
        assert c.sha(path) == generation.digest(path)
    coarse_connection = connect({'database': coarse_db})
    fine_connection = sqlite3.connect(fine_db.as_uri()+'?mode=ro', uri=True)
    dates = [r[0] for r in coarse_connection.execute('SELECT DISTINCT day FROM bars WHERE day<=? ORDER BY day', ('2026-03-31',))]
    results, source_proofs = [], {}
    corporate = c.read(files/ready['study']['corporate_actions'])['events'] if ready['study'].get('corporate_actions') else []
    with generation.activate(), clock.olr_completeness_clock():
        reparse, clear, source_inputs = origin_verifier(generation, definition)
        cache = {}

        def sourced(symbol, day):
            key = symbol+':'+day
            if key not in cache:
                frame, digest, proof = reparse(symbol, day)
                fine = load_minute_day(fine_connection, date.fromisoformat(day), {symbol})
                coarse = load_day(coarse_connection, day, ready['study']['database_hash'], {symbol})
                # Exact independently reparsed origin -> native loader values,
                # including fingerprint, then native aggregate -> coarse basis.
                from strategy_common.market import MarketBar
                expected = [MarketBar(symbol, r.timestamp.to_pydatetime(), '1m', r.open, r.high, r.low, r.close, r.volume,
                    source='minute_source_plan_v5', source_fingerprint=digest).to_json_dict() for r in frame.itertuples(index=False)]
                assert expected == [b.to_json_dict() for b in fine], key
                canonical = pd.DataFrame(coarse_connection.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',
                    (symbol, day)).fetchall(), columns=FIELDS)
                same(aggregate_timeframe(frame, '5m'), canonical)
                cache[key] = fine, coarse
                source_proofs[key] = dict(origin=proof, fine_rows=len(fine), coarse_rows=len(coarse),
                    fine_hash=c.signature(expected), coarse_hash=c.signature([b.to_json_dict() for b in coarse]),
                    exact_origin_and_aggregation=True)
            return cache[key]

        # Group by symbol to bound native parquet cache memory and avoid
        # reparsing the same origin for each of the four paired lanes.
        cohort = sorted(design['cohort'], key=lambda r: (r['symbol'], r['day']))
        for index, item in enumerate(cohort):
            symbol, day = item['symbol'], item['day']
            snapshot = snapshots[date.fromisoformat(day)]
            candidate = next(x for x in snapshot.candidates if x.symbol == symbol)
            start = dates.index(day)
            horizon = dates[start:start+design['max_sessions_per_lane']]
            fine, coarse = sourced(symbol, day)
            assert fine and coarse, 'Original confirmed signal has no source stream'
            cutoff = datetime.combine(snapshot.trade_date, session_cutoff(snapshot.trade_date, time(14, 30), anchor='close'), KST)
            prefix = [b for b in coarse if b.timestamp+timedelta(minutes=5) <= cutoff]
            _, excluded = clock.close_anchored_feature_bars({(snapshot.trade_date, symbol): prefix}, time(14, 30))
            assert not excluded, 'Confirmed incumbent lacks causal feature prefix'
            plan = entry_plan(_entry_plan(config), config)
            signal = find_olr_entry_signal(tuple(coarse), candidate, plan, require_fill_bar=False)
            assert signal is not None
            signal_time = coarse[signal.signal_index].timestamp+timedelta(minutes=5)
            assert signal_time == datetime.fromisoformat(item['native_signal_time'])
            strict = find_olr_entry_signal(tuple(coarse[:signal.signal_index+1]), candidate, plan, require_fill_bar=False)
            assert strict == signal or (strict.signal_index == signal.signal_index and strict.reason == signal.reason)
            for scenario in c.SCENARIOS:
                for budget in c.BUDGETS:
                    broker, adapter, driver = make_lane(config, snapshot, symbol, budget, scenario)
                    unsupported, consumed = None, []
                    decisions = []
                    for current in horizon:
                        if current != day and closed(broker):
                            break
                        if broker.positions and any(e.get('symbol') == symbol and e['ex_date'] == current for e in corporate):
                            unsupported = 'corporate_action_with_open_inventory'; break
                        fine, coarse = sourced(symbol, current)
                        if not fine or not coarse:
                            unsupported = 'missing_native_session_stream:'+current; break
                        before_fine, before_coarse = len(broker.expected_batches), len(broker.coarse_expected_batches)
                        decisions.extend(driver.run_day(fine, coarse, symbols={symbol}))
                        # Compare both consumed streams against independently
                        # reconstructed source, never replace expected_batches.
                        assert [r for batch in broker.expected_batches[before_fine:] for r in batch] == [b.to_json_dict() for b in fine]
                        assert [r for batch in broker.coarse_expected_batches[before_coarse:] for r in batch] == [b.to_json_dict() for b in coarse]
                        consumed.append(symbol+':'+current)
                    result, audit = assess(broker, adapter, budget, unsupported=unsupported)
                    submissions = [r for r in broker.journal if r['event'] == 'submit' and r['order']['side'] == 'BUY']
                    for event in submissions:
                        # Native submission timestamp is the completed coarse
                        # label; actual availability is bound in order metadata.
                        assert datetime.fromisoformat(event['order']['metadata']['decision_available_at']) == signal_time
                    result.update(day=day, symbol=symbol, rank=item['rank'], scenario=scenario,
                        native_signal_time=signal_time.isoformat(), source_sessions=consumed,
                        available_for_training_after=(consumed[-1].split(':')[1]+'T23:59:59+09:00') if result['status']=='matured' else None,
                        source_scope='single original confirmed incumbent; isolated independent account',
                        source_proof_hash=c.signature({k: source_proofs[k] for k in consumed}),
                        decisions=[e.to_json_dict() for e in decisions])
                    audit['diagnostic_scope'] = dict(
                        warning='Native execution/clock/capacity/ledger arithmetic only; generic whole-portfolio source qualification remains unverified. Isolated source scope is independently bound in source_proofs.json.',
                        unchanged_parent_directory_hash=audit['audit_receipt']['source_generation_hash'],
                        explicit_overlay_binding=design['overlays'],
                        effective_harness_sha256=c.sha(c.ROOT/'frozen_harness.json'))
                    key=f'{day}_{symbol}_{scenario}_{budget}'
                    lane=c.ROOT/'lanes'/key
                    c.write(lane/'result.json', result)
                    c.write(lane/'execution_audit.json', audit)
                    with gzip.open(lane/'journal.jsonl.gz', 'wt', encoding='utf-8') as stream:
                        for row in broker.journal:
                            stream.write(json.dumps(row, sort_keys=True, default=str)+'\n')
                    c.write(lane/'receipt.json', dict(result_sha256=c.sha(lane/'result.json'),
                        audit_sha256=c.sha(lane/'execution_audit.json'), journal_sha256=c.sha(lane/'journal.jsonl.gz'),
                        frozen_harness_sha256=c.sha(c.ROOT/'frozen_harness.json'),
                        source_proof_hash=result['source_proof_hash'], utc=c.utc()))
                    results.append(dict(path=str(lane), result_sha256=c.sha(lane/'result.json'), receipt_sha256=c.sha(lane/'receipt.json')))
            print(json.dumps(dict(phase='fixed_signal_labels', opportunities=index+1, total=len(cohort), completed_lanes=len(results))), flush=True)
            if index+1 == len(cohort) or cohort[index+1]['symbol'] != symbol:
                clear(); cache.clear()
    fine_connection.close(); coarse_connection.close()
    assert len(results) == design['max_lanes']
    c.write(c.ROOT/'source_proofs.json', dict(sessions=source_proofs, source_inputs=source_inputs,
        unchanged_source_manifest_sha256=c.sha(ready['study']['manifest']), origins_independently_reparsed=True,
        original_clock_corrections_unchanged=True, original_eligibility_clock_repair_sha256=c.sha(c.BASE/'opportunity_v1/clock_repair.json')))
    c.write(c.ROOT/'labels_complete.json', dict(utc=c.utc(), lanes=results, complete=True,
        frozen_harness_sha256=c.sha(c.ROOT/'frozen_harness.json'), source_proofs_sha256=c.sha(c.ROOT/'source_proofs.json'),
        imports=c.imports(guard), oos_read=False, new_portfolio_replays=0, promotion=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['tests', 'labels'])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    c = load('fixed_signal_contract', repo/'scripts/olr_fixed_signal_v8.py')
    c.temp()
    design, ready = c.authenticate(args.phase == 'labels', args.phase == 'labels')
    files = Path(ready['files']).resolve()
    clock = load('fixed_signal_clock', repo/'scripts/olr_opportunity_clock_v7.py')
    guard = c.NativeOverlay(files, ready['manifest']['files'], design['overlays'])
    sys.meta_path.insert(0, guard)
    sys.path[:] = [str(files)] + [p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(repo)]
    os.chdir(files)
    sys.modules['fixed_signal_runtime'] = sys.modules[__name__]
    if args.phase == 'tests':
        import pytest
        testfile = repo/'tests/strategy_olr/test_fixed_signal_v8.py'
        rc = pytest.main([str(testfile), '-q', '--junitxml='+str(c.ROOT/'tests.xml')])
        imported = c.imports(guard)
        for key in design['overlays']:
            assert key[:-3].replace('/', '.') in imported
        c.write(c.ROOT/'test_imports.json', dict(exit_code=rc, imports=imported,
            test_file_sha256=c.sha(testfile), overlay_imports_verified=True,
            worker_sha256=c.sha(__file__), contract_sha256=c.sha(repo/'scripts/olr_fixed_signal_v8.py')))
        raise SystemExit(rc)
    try:
        run_labels(c, design, ready, guard, clock)
    except BaseException:
        c.write(c.ROOT/'labels_failure.json', dict(utc=c.utc(), traceback=traceback.format_exc(), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
