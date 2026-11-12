"""Observational bridge from frozen capacity replays to standard OLR reports.

No strategy transitions, selection, re-fitting, or new execution assumptions.
Native receipts are validated in their own schema against the frozen engine.
"""
from collections import defaultdict
from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from statistics import median

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _atomic_write_json
from backtests.auto.shared.types import EndOfRoundArtifacts
from backtests.strategies.common.plugin_base import attach_official_metric_contract, build_execution_contract
from backtests.strategies.olr.capacity_qualification import checked_case, native_audit_valid
from strategy_olr.config import OLRConfig

FILES = ('request.json', 'completion.json', 'result.json', 'execution_audit.json',
         'trades.json', 'equity.json', 'daily.json', 'fills.jsonl.gz',
         'journal.jsonl.gz', 'decisions.jsonl.gz')
STATUS = 'user_selected_research_only_support_unqualified_independent_validation_pending'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-5), message)


def restore_datetimes(value):
    # The observational exporter used default=str for native datetime objects;
    # restore those objects for the engine's canonical ledger hash. Existing
    # ISO strings already containing T are left as their original strings.
    if isinstance(value, dict):
        return {k: restore_datetimes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_datetimes(v) for v in value]
    if isinstance(value, str) and re.match(r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d', value):
        return datetime.fromisoformat(value)
    return value


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(row) for row in stream]


def verify_recycling_orders(journal):
    """Check causal forecast timestamps and extra-quantity bounds on real orders."""
    checked, interventions = 0, set()
    for row in journal:
        if row['event'] != 'submit' or row['order']['side'] != 'BUY':
            continue
        order = row['order']
        m = order['metadata']
        if 'capacity_recycling' not in m:
            continue
        checked += 1
        evidence = m['capacity_forecast']
        available = datetime.fromisoformat(m['decision_available_at'])
        at = datetime.fromisoformat(evidence['asof'])
        require(at <= available, 'Capacity forecast unavailable at order decision')
        close(evidence['participation'], .01, 'Changed strategy forecast participation')
        samples = evidence['entry']['samples']
        require(all(datetime.fromisoformat(t).date() == at.date()
                    and datetime.fromisoformat(t)+timedelta(minutes=5) <= at for t, _ in samples),
                'Uncompleted or future entry-volume forecast sample')
        expected = math.floor(median(v for _,v in samples)*.01)*evidence['entry']['horizon'] if samples else 0
        close(evidence['entry']['quantity'], expected, 'Entry forecast does not match completed samples')
        auctions = evidence['exit']['samples']
        require(all(datetime.fromisoformat(t).date() < at.date() for t,_ in auctions), 'Future auction-volume sample')
        exit_qty = evidence['exit']['quantity']
        if exit_qty is not None:
            close(exit_qty, math.floor(median(v for _,v in auctions)*.01), 'Exit cap does not match prior median')
        before, extra = m['allocation_before_redistribution'], m['allocation_redistributed_qty']
        require(extra >= 0 and order['qty'] == before+extra, 'Recycling changed native funded quantity')
        envelope = m['capacity_recycling']
        require(envelope['day'] == at.date().isoformat() and 0 <= envelope['assigned_extras']
                <= envelope['displaced_request_funding']+1e-6, 'Unfunded recycling envelope')
        if extra:
            require(before > 0 and 1 <= m['candidate_rank'] <= 2 and samples
                    and order['qty'] <= evidence['entry']['quantity']
                    and order['qty'] <= math.floor(1.5*m['allocation_native_qty'])
                    and (exit_qty is None or order['qty'] <= exit_qty), 'Extra shares exceed evidence or native-size limit')
            interventions.add((at.date().isoformat(), order['symbol']))
    return {'checked_native_buy_orders':checked, 'direct_extra_order_issuer_days':len(interventions),
            'issuer_days':sorted(interventions), 'causal_forecasts_and_extra_bounds_pass':True}


def reconcile_exports(path, result, audit, capital, window):
    """Authenticate journal bytes and independently rebuild cash/MTM from fills."""
    path = Path(path)
    journal = read_rows(path/'journal.jsonl.gz')
    receipt = audit['audit_receipt']
    require(stable_signature(restore_datetimes(journal)) == receipt['ledger_hash'], 'Changed native execution journal')
    recycling = verify_recycling_orders(journal)
    fills = read_rows(path/'fills.jsonl.gz')
    require([row['fill'] for row in journal if row['event'] == 'fill'] == fills, 'Fill export differs from audited journal')
    trades, equity, daily = [read(path/name) for name in ('trades.json', 'equity.json', 'daily.json')]
    cash, positions, rebuilt, entry_ids = float(capital), defaultdict(int), [], set()
    prices = {}
    for row in journal:
        if row['event'] == 'opening_marks':
            prices.update(row['prices'])
        elif row['event'] == 'fill':
            fill = row['fill']
            require(fill['strategy_id'] == 'OLR' and fill['side'] in ('BUY', 'SELL'), 'Unexpected fill owner/side')
            buy = fill['side'] == 'BUY'
            qty, price = int(fill['qty']), float(fill['price'])
            require(qty > 0 and qty == fill['qty'] and price > 0, 'Invalid fill quantity/price')
            fee = qty * price * (2 + (0 if buy else 20)) / 10000
            close(fill['metadata']['fee'], fee, 'Incorrect per-fill commission/tax')
            close(row['cash_before'], cash, 'Discontinuous cash before fill')
            cash += (-1 if buy else 1) * qty * price - fee
            close(row['cash_after'], cash, 'Cash after fill does not reconcile')
            require(cash >= -1e-5, 'Unfunded cash account')
            positions[fill['symbol']] += qty if buy else -qty
            require(positions[fill['symbol']] >= 0, 'Oversold inventory')
            if buy:
                entry_ids.add((fill['metadata'].get('order_date'), fill['order_id']))
        elif row['event'] == 'closing_marks':
            prices.update(row['prices'])
            mark = cash + sum(q * prices[s] for s, q in positions.items() if q)
            rebuilt.append((datetime.fromisoformat(row['timestamp']), mark, cash, dict(positions)))
    require(len(rebuilt) == len(equity) and len(trades) == result['trades'], 'Omitted equity marks or incorrect trade denominator')
    closes = {}
    peak, dd = float(capital), 0.
    for (stamp, value), (at, mark, day_cash, inventory) in zip(equity, rebuilt, strict=True):
        require(datetime.fromisoformat(stamp) == at and window['start'] <= stamp[:10] <= window['end'], 'Incorrect equity clock/window')
        close(value, mark, 'Equity differs from independently marked ledger')
        peak = max(peak, mark)
        dd = max(dd, 1 - mark / peak)
        closes[stamp[:10]] = (mark, day_cash, {s: q for s, q in inventory.items() if q})
    require(len(daily) == len(closes) == result['sessions'], 'Missing or extra daily account records')
    require([d['day'] for d in daily] == sorted(closes), 'Unordered or duplicate sessions')
    for day in daily:
        mark, day_cash, inventory = closes[day['day']]
        close(day['equity'], mark, 'Daily MTM mismatch')
        close(day['cash'], day_cash, 'Daily cash mismatch')
        require(day['positions'] == {str(('OLR', s)): q for s, q in inventory.items()}, 'Daily inventory mismatch')
    require(not any(positions.values()) and not result['terminal_inventory'] and not result['terminal_pending']
            and not result.get('terminal_deferred_intents', 0) and not result.get('terminal_rotation_intents', 0), 'Unresolved terminal inventory or intents')
    close(sum(t['net_pnl'] for t in trades), cash-capital, 'Net trades do not reconcile to account equity')
    close(sum(t['gross_pnl']-t['net_pnl'] for t in trades), sum(f['metadata']['fee'] for f in fills), 'Trade costs do not reconcile to fill costs')
    close(result['return_pct'], 100*(cash/capital-1), 'Inflated whole-account return')
    close(result['drawdown_pct'], 100*dd, 'Incorrect MTM drawdown')
    coverage = receipt['daily_equity_coverage']
    require(coverage['status'] == 'complete' and coverage['declared_window'] == window
            and coverage['session_count'] == len(daily), 'Incomplete declared-window mark coverage')
    from backtests.analysis.metrics import compute_trade_metrics
    from strategy_common.events import TradeOutcome
    parsed = []
    for original in trades:
        row = dict(original)
        for key in ('entry_decision_time', 'entry_fill_time', 'exit_fill_time'):
            row[key] = datetime.fromisoformat(row[key]) if row[key] else None
        parsed.append(TradeOutcome(**{f.name: row[f.name] for f in fields(TradeOutcome) if f.name in row}))
    computed = compute_trade_metrics(parsed, [capital, *[v for _, v in equity]], initial_equity=capital,
        timestamps=[datetime.fromisoformat(t) for t, _ in equity])
    for key in ('profit_factor', 'sharpe', 'net_profit', 'max_drawdown_pct', 'net_return_pct', 'total_trades', 'win_rate'):
        close(computed[key], result['metrics'][key], 'Incorrect saved metric: '+key)
    groups = defaultdict(list)
    for t in trades:
        for dimension, label in (('entry_month', t['entry_fill_time'][:7]), ('symbol', t['symbol']),
                ('exit_reason', t['exit_reason']), ('rank', t['route_metadata'].get('candidate_rank', 'unknown'))):
            groups[(dimension, str(label))].append(t)
    cohorts = [{'dimension': d, 'label': label, 'closed_entry_count': len(rows),
        'wins': sum(t['net_pnl'] > 0 for t in rows), 'gross_pnl_krw': sum(t['gross_pnl'] for t in rows),
        'net_pnl_krw': sum(t['net_pnl'] for t in rows),
        'explicit_fees_krw': sum(t['gross_pnl']-t['net_pnl'] for t in rows)} for (d,label),rows in sorted(groups.items())]
    metrics = {**result['metrics'], **audit, 'official_mtm_net_return_pct': result['return_pct']/100,
        'decision_counts': result.get('decision_counts', {}), 'sessions': result['sessions'],
        'active_entry_sessions': result.get('active_days'),
        'trades_per_21_sessions': result.get('trades_per_21_sessions'),
        'entry_lifecycle_count': len(entry_ids), 'daily_equity_coverage': coverage,
        'end_open_position_count': 0, 'source_fingerprint': result['source_generation_hash'],
        'feature_manifest_hash': receipt['identity']['input_manifest']['manifest_hash'],
        'candidate_snapshot_hash': receipt['identity']['qualification_snapshot_hash'],
        'capability_level': 'real_replay', 'native_audit_valid': True,
        'official_metric_basis': 'SimBroker.equity_curve_bar_level_mtm',
        'net_return_pct_basis': 'closed_net_pnl_divided_by_initial_account_equity'}
    return {'metrics': metrics, 'path': str(path), 'cohorts': cohorts,
        'recycling_order_audit': recycling,
        'account_reconciliation': {'passed': True, 'cash_krw': cash, 'fill_count': len(fills),
            'mark_count': len(equity), 'session_count': len(daily), 'ledger_hash': receipt['ledger_hash']}}


def validate_case(expected, source):
    path = Path(expected['path'])
    require(digest(expected['study']) == expected['study_sha256'], 'Changed registered diagnostic study')
    study = read(expected['study'])
    result, audit = checked_case(path)
    request = read(path/'request.json')
    args = request['arguments']
    require(study['source_generation_hash'] == source and study['initial_equity'] == expected['capital'], 'Wrong diagnostic source/capital')
    require(Path(args['study']).resolve() == Path(expected['study']).resolve()
            and args['recipe'] == expected['recipe'] and args['scenario'] == expected['scenario']
            and args['standalone'] == 'OLR' and args['absolute_caps'] is False
            and args['canonical_costs'] is False and args['no_upward_compounding'] == expected['no_upward_compounding']
            and request['recipe'] == study['candidates'][expected['recipe']], 'Diagnostic case identity mismatch')
    require(native_audit_valid(audit, path), 'Missing, changed or failed native source/execution audit')
    return reconcile_exports(path, result, audit, expected['capital'], study['configuration']['date_range'])


class NativeCapacityDiagnosticsPlugin:
    name = 'olr'
    num_phases = 0
    requires_full_diagnostics = True
    capability_level = 'real_replay'

    def __init__(self, config, plan_path, *, output_dir):
        self.config = dict(config)
        self.training_config = dict(config)
        self.plan_path = Path(plan_path)
        self.plan = read(plan_path)
        self.output_dir = Path(output_dir)
        self.initial_mutations = dict(config['initial_mutations'])
        self.source_fingerprint = self.plan['source_generation_hash']
        self.exported_cases = {}
        self.reporting_source = None
        self.require_configuration()
        self.execution_context = {**config, 'source_fingerprint': self.source_fingerprint,
            'date_window': config['date_range'], 'shared_decision_core': 'strategy_olr.core',
            'live_parity_fill_timing': 'completed_5m_signal_then_next_eligible_open; resting_exits_at_actual_closing_match',
            'auction_mode': 'closing_auction_exits_with_native_unfilled_market_fallback',
            'replay_mode': 'native_capacity_study_frozen_subscription_feed',
            'official_promotion_gate': 'user_research_save; support qualification remains failed',
            'cost_policy': {'commission_bps': 2, 'tax_bps_on_sell': 20, 'slippage_bps': 5,
                            'auction_slippage_bps': 0, 'participation': .01}, 'risk_basis': 'mark_to_market'}
        self.diagnostic_scope = config['diagnostic_scope']
        self.diagnostic_limitations = config['diagnostic_limitations']

    def require_configuration(self):
        config, plan = self.config, self.plan
        required = {f'native_entry_recycle2_{c}_{s}' for c in (10_000_000,20_000_000,50_000_000,90_000_000,100_000_000,110_000_000)
                    for s in ('baseline','combined')} | {'participation','cost','no_upward_compounding'}
        require(set(plan['cases']) == required, 'Incomplete final diagnostic scenario/capital grid')
        require(config['native_case_plan_sha256'] == digest(self.plan_path), 'Changed diagnostic plan')
        anchor = plan['cases']['native_entry_recycle2_100000000_baseline']
        study = read(anchor['study'])
        request = read(Path(study['code'])/study['olr_request'])
        expected = replace(OLRConfig.from_mapping(request['config'], request['mutations']),
            slippage_bps=5, tax_bps_on_sell=20, **study['candidates']['native_entry_recycle2']['olr_capacity'])
        require(asdict(OLRConfig.from_mapping(config, self.initial_mutations)) == asdict(expected),
                'Saved configuration differs from the executed native recipe')
        require(config['initial_equity'] == anchor['capital'] and config['date_range'] == study['configuration']['date_range']
                and config['input_manifest_path'] == study['manifest'] and config['promotion_status'] == STATUS,
                'Wrong saved account, window, inputs or qualification status')
        require(config['independent_oos_pass'] is False and config['live_promotion'] is False
                and config['development_qualified'] is False, 'Scientific qualification cannot be inferred from a user-selected save')

    def canonicalize_mutations(self, mutations):
        require(mutations == self.initial_mutations, 'Final diagnostics cannot change the selected recipe')
        return dict(mutations)

    def require_bound_cases(self, mutations):
        self.canonicalize_mutations(mutations)
        if self.exported_cases:
            return
        seals = {}
        for name, expected in self.plan['cases'].items():
            print('Authenticating native case: '+name, flush=True)
            before = {f: digest(Path(expected['path'])/f) for f in FILES}
            case = validate_case(expected, self.source_fingerprint)
            if name.startswith('native_entry_recycle2_'):
                previous = Path(self.plan['g15'])/'cases'/name
                require(all(read(Path(expected['path'])/f) == read(previous/f)
                            for f in ('trades.json','equity.json','daily.json')), 'Unexpected economic drift from frozen selected candidate')
            after = {f: digest(Path(expected['path'])/f) for f in FILES}
            require(before == after, 'Diagnostic reporting changed native replay files')
            self.exported_cases[name] = case
            seals[name] = {'path': expected['path'], 'files': after, 'audit_valid': True,
                           'account_reconciliation': case['account_reconciliation']}
        _atomic_write_json({'plan_sha256': digest(self.plan_path), 'source_generation_hash': self.source_fingerprint,
            'cases': seals, 'all_native_source_execution_audits_pass': True,
            'support_qualified': False, 'independent_oos_pass': False}, self.output_dir/'native_diagnostic_evidence.json')

    def compute_final_metrics(self, mutations):
        self.require_bound_cases(mutations)
        metrics = dict(self.exported_cases['native_entry_recycle2_100000000_baseline']['metrics'])
        metrics.update(promotion_status=STATUS, research_only=True, development_qualified=False,
            independent_oos_pass=False, live_promotion=False, native_receipt_schema='study_recipe_configuration',
            native_receipt_validation_required=True)
        metrics = attach_official_metric_contract(metrics, requires_audit_pass=False,
            execution_contract=build_execution_contract(self, metrics))
        metrics['metric_contract'].update(native_receipt_schema='study_recipe_configuration',
            native_receipt_validation_required=True, native_receipt_validation_pass=True,
            generic_receipt_gate='not_applicable_to_native_study_identity; mandatory native gate enforced before reporting')
        return metrics

    def build_end_of_round_artifacts(self, state):
        self.require_bound_cases(state.cumulative_mutations)
        text = ('User-selected conditional capital recycling research round. All registered native source/execution '
                'receipts and account reconciliations must pass. The original direct-support gate failed and remains failed; '
                'independent validation and production qualification are not established. No optimization phases were run for this save.')
        return EndOfRoundArtifacts(text, {'selection_and_integrity': text}, text)

    def write_full_diagnostics(self, state, output_dir, *, round_num, round_name):
        path = Path(__file__).with_name('final_diagnostics.py')
        spec = importlib.util.spec_from_file_location('backtests.strategies.olr.native_report_formatter', path)
        formatter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formatter)
        self.reporting_source = {'adapter_sha256': digest(__file__), 'formatter_sha256': digest(path),
            'case_plan_sha256': digest(self.plan_path), 'executing_native_source_generation': self.source_fingerprint,
            'mode': 'authenticated_native_capacity_exports'}
        aliases = {'native_entry_recycle2_100000000_baseline':'training',
                   'native_entry_recycle2_100000000_combined':'combined-stress',
                   'participation':'participation-stress', 'cost':'cost-stress'}
        cases = {aliases.get(k,k):v for k,v in self.exported_cases.items()}
        for name, case in cases.items():
            _atomic_write_json(case, Path(output_dir)/'diagnostics'/name/'summary.json')
        primary = cases['training']
        m = primary['metrics']
        lines = [self.config['diagnostic_summary'], '', 'Reference-account native diagnostics:',
            f"Net profit KRW {m['net_profit']:,.2f}; win rate {100*m['win_rate']:.4f}%; net expectancy per closed entry KRW {m['expectancy']:,.2f}.",
            f"Active entry sessions {m['active_entry_sessions']}/{m['sessions']}; closed entries per 21 sessions {m['trades_per_21_sessions']:.4f}.",
            f"Gross MFE capture (hindsight descriptive metric): {100*m['mfe_capture']:.4f}%.",
            'Native decision reasons: '+json.dumps(m['decision_counts'],sort_keys=True),
            'Independent account reconstruction: '+json.dumps(primary['account_reconciliation'],sort_keys=True),
            'Submitted recycling-order evidence: '+json.dumps(primary['recycling_order_audit'],sort_keys=True),
            'Audit coverage: '+json.dumps(m['audit_receipt']['coverage'],sort_keys=True),
            '', 'Executed closed-entry cohorts (each dimension partitions the same executed population):',
            'dimension | cohort | entries | wins | gross PnL KRW | fees KRW | net PnL KRW']
        for row in primary['cohorts']:
            lines.append(f"{row['dimension']} | {row['label']} | {row['closed_entry_count']} | {row['wins']} | "
                f"{row['gross_pnl_krw']:,.2f} | {row['explicit_fees_krw']:,.2f} | {row['net_pnl_krw']:,.2f}")
        lines += ['', 'Unexecuted-candidate funnels are not reconstructed from trades; no accepted/rejected selection-alpha claim is made.',
            'Every case includes its full native audit, capacity diagnostics, executed cohorts and authenticated raw-export paths in the JSON report.']
        return formatter.render_full_diagnostics(self, state, Path(output_dir), round_num=round_num,
            round_name=round_name, cases=cases, training_text='\n'.join(lines))
