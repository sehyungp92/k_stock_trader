"""Package the bounded foundation result without promoting a migration model."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import difflib
import json
import shutil
from scripts import olr_structural_native_v6 as e
from scripts.olr_structural_v6 import REPO, OUT as PROJECT_OUT, read, sha, write, utc

LABELS = {'legacy_clock_control': 'Matched minute control',
          'action_independent_median': 'Action-independent median'}
GROUPS = [('windows', 'Nominal'), ('stress', 'Combined execution stress'), ('capped', 'No upward compounding')]
LIMITS = ('Sparse auction history; a fixed universe and taxonomy rather than point-in-time membership; '
          'inferred archive clocks; and conservative minute-price and lagged-volume execution proxies remain. '
          'The data do not describe actual exchange queues or order-book depth. April–July has been repeatedly '
          'inspected and is development evidence, not independent validation. Source/audit correctness does '
          'not establish future expected returns. The selected policy remains c4f49be01ed09b8dc; no live activation.')


def number(value, places=3):
    return '—' if value is None else f'{value:,.{places}f}'


def money(value):
    return number(value / 1e6, 3)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def para(text):
    return '<p>' + escape(text) + '</p>'


def profile(data):
    path = e.PARENT_OUT / 'selected_research_policy.json'
    assert sha(path) == 'aa2d612718023b915e1b87f65ae9f3a8cf5575edaf30e564b4aadbd9ece3c9a9'
    result = read(path)
    result.update(status='retain_original_pending_broader_alpha_program',
        results_sha256=sha(e.OUT / 'results.json'), registration_sha256=sha(e.ROOT / 'registration.json'),
        retained_parent_profile_sha256=sha(path), promotion=False, independent_validation=False,
        structural_migration_adopted=False, selected_id='c4f49be01ed09b8dc')
    return result


def reference_row(label, window, row):
    folder = Path(row['folder'])
    e.s.forensic.authenticate(folder)
    trades = read(folder / 'trades.json')
    return [label, window, number(row['return_pct']), number(row['drawdown_pct']),
            number(row.get('complete_fine_drawdown_pct')), len(trades), row['entry_days'],
            money(sum(t['qty'] * t['entry_price'] for t in trades))]


def chart(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    windows = ('train', 'oos') if data['oos_evaluated'] else ('train',)
    fig, axes = plt.subplots(len(windows), 3, figsize=(15, 4.3 * len(windows)), squeeze=False, layout='constrained')
    colours = ['#31638d', '#b27331']
    x = np.arange(3)
    for row, window in enumerate(windows):
        for col, (key, title) in enumerate([('return_pct', 'Net return (%)'), ('complete_fine_drawdown_pct', 'Complete fine max DD (%)'), ('unique_entries', 'Actual filled entries')]):
            ax = axes[row, col]
            for index, candidate in enumerate(data['candidates']):
                values = [candidate[group][window][key] for group, _ in GROUPS]
                bars = ax.bar(x + (index - .5) * .36, values, width=.34, color=colours[index], label=LABELS[candidate['label']])
                ax.bar_label(bars, fmt='%.2f' if col < 2 else '%.0f', fontsize=8)
            ax.set_xticks(x, ['Nominal', 'Stress', 'Capped'])
            ax.set_title(('IS' if window == 'train' else 'Inspected OOS') + ' · ' + title)
            ax.axhline(0, color='#777', lw=.6)
            ax.margins(y=.18)
            ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('OLR structural foundation comparison · diagnostic migration, no promotion', fontsize=15)
    fig.savefig(e.OUT / 'results.png', dpi=150)
    plt.close(fig)


def implementation_bundle():
    harness = read(e.ROOT / 'frozen_harness.json')
    bundle = e.OUT / 'review_bundle'
    bundle.mkdir(exist_ok=True)
    parts, sources = [], {}
    for window in ('train', 'oos'):
        path = bundle / window / 'implementation.patch'
        parts.append(path.read_text(encoding='utf-8'))
        sources[window + '/implementation.patch'] = dict(source=str(path), sha256=sha(path), binding='native generation delta from preserved minute parent')
    delivery = ['scripts/olr_structural_results_v6.py', 'scripts/olr_structural_feedback_v6.py',
                'scripts/olr_structural_report_v6.py', 'scripts/olr_structural_delivery_v6.py']
    names = sorted(set(harness['files']) | set(delivery))
    for name in names:
        source = REPO / name
        digest = sha(source)
        if name in harness['files']:
            assert harness['files'][name] == digest, name
        relative = 'workspace/' + name
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), target
        shutil.copy2(source, target)
        parts.extend(difflib.unified_diff([], source.read_text(encoding='utf-8').splitlines(True),
                     fromfile='/dev/null', tofile='b/' + relative))
        sources[relative] = dict(source=str(source), sha256=digest,
            binding='frozen research/test harness' if name in harness['files'] else 'reporting only; no decision influence')
    path = bundle / 'implementation.patch'
    assert not path.exists()
    path.write_text(''.join(parts), encoding='utf-8')
    write(bundle / 'source_manifest.json', dict(files=sources,
        original_launch_sha256=sha(e.ROOT / 'launch_harness.json'),
        frozen_harness_sha256=sha(e.ROOT / 'frozen_harness.json'),
        test_repair_sha256=sha(e.ROOT / 'test_isolation_repair.json'),
        parent_native_bundle=str(e.PARENT_OUT / 'review_bundle/implementation.patch')))
    return {p.relative_to(e.OUT).as_posix(): sha(p) for p in bundle.rglob('*') if p.is_file()}


def render(data, feedback):
    control, migration = data['candidates']
    decision = data['training_decision']
    result = ('Training preservation gates passed; the registered OOS comparisons were evaluated.' if data['oos_evaluated'] else
              'The migration failed the training preservation gates. OOS portfolio performance was not evaluated.')
    body = '<h1>OLR structural foundation study</h1>' + para(result)
    body += para('This experiment repairs policy-dependent observation and the under-three-observations request-cap cliff. It is a migration diagnostic, not proof of new alpha or a selected successor. The original research policy remains selected; the broader authorized return/frequency program is still incomplete.')
    body += '<img src="results.png" alt="Control versus migration net return, complete fine drawdown and actual entries by scenario">'
    body += '<h2>Controls and chronology</h2>' + para('Both policies use identical original entry artifacts, ranks, sizing rules, risk thresholds and native minute execution. The treatment uses all qualified prior-session auction observations, independently of held positions or routed symbols, and requests zero when liquidity is unknown. Actual fill capacity is unchanged. The median can constrain a request with one or two observations; missing history no longer removes that bound.')
    calibration = read(PROJECT_OUT / 'forecast_calibration.json')
    body += para('Before portfolio outcomes, the normalized shrinkage forecast was rejected: it underperformed the broad median in every chronological training block and among selected candidates. It was neither retuned nor advanced. The broad median is the fixed migration comparator, not an adopted forecasting improvement. Exact source projection is recomputed per case; every consumed minute retains its native source, execution, capacity and ledger audit.')
    tests = read(e.ROOT / 'frozen_harness.json')['tests']['tests']
    body += para(f'{tests} frozen-module tests passed after an import-isolation failure was preserved. The repaired runner authenticates native modules before importing them; 88 module hashes and three import guard checks passed. Original preparation and all old studies remain unchanged. The test failure occurred before performance and did not relax an economic gate.')
    body += '<h2>Return, drawdown and meaningful activity</h2>'
    headers = ['Policy', 'Window', 'Return %', 'Common/coarse DD %', 'Complete fine DD %', 'Entries', 'Entry days', 'Filled notional KRW m']
    rows = []
    for label, values in [('Original', data['references']['original']), ('Guard', data['references']['guard']['windows']), ('Retained', data['references']['retained']['windows'])]:
        for window in ('train', 'oos'):
            rows.append(reference_row(label, window, values[window]))
    for candidate in data['candidates']:
        for window, row in candidate['windows'].items():
            rows.append(reference_row(LABELS[candidate['label']], window, row))
    body += table(headers, rows)
    body += para('Historical comparator rows are immutable saved studies. Their coarse DD is shown alongside the causal common-original grid used for new minute cases; complete fine DD is unavailable for old coarse-only policies. A dash is unavailable, never zero. Entry counts are unique filled lifecycles and not partial fills, orders or candidates. IS covers August 7–March 31; inspected OOS covers April 1–July 10. Windows are separate KRW100m scoring accounts, not a continuously funded wealth curve.')
    for group, label in GROUPS[1:]:
        rows = [reference_row('Retained', w, row) for w, row in data['references']['retained'][group].items()]
        rows += [reference_row(LABELS[c['label']], w, row) for c in data['candidates'] for w, row in c[group].items()]
        body += '<h3>' + escape(label) + '</h3>' + table(headers, rows)
    body += para('Combined stress retains native stressed costs and halves participation capacity. Capped sizing can shrink after losses but cannot grow above starting capital after gains. Capped IS preservation is tested against retained capped IS; it must not be described as preserving retained compounded IS return of 53.639%.')
    body += '<h2>Registered gates</h2>'
    for group, label in GROUPS:
        checks = decision['checks'][group]
        body += table([label + ' training gate', 'Result'], [[key, 'PASS' if value else 'FAIL'] for key, value in checks.items()])
        if data['oos_evaluated']:
            body += table([label + ' full-objective gate', 'Result'], [[key, 'PASS' if value else 'FAIL'] for key, value in migration['scenario_gates'][group].items()])
    body += para('The original return/DD/frequency/notional/stress/capped requirements remain mandatory. Nominal IS preserves original−3pp, guard−3pp, retained−1.5pp and guard DD+2pp. Full nominal OOS requires retained return+0.25pp, DD−0.75pp, at least 186/47 IS/OOS entries and 88/29 entry days, and both median added and net incremental OOS notional of KRW1m. Matched fine-control returns and strictly lower OOS DD on both grids are additional requirements. No promotion is allowed merely because the corrected source model is coherent.')
    body += '<h2>Admission, quantities and funding feedback</h2>'
    rows = []
    for group, label in GROUPS:
        for window, pair in feedback['scenarios'][group].items():
            for mode in ('control', 'migration'):
                a = pair[mode]
                rows.append([label, window, mode, a['admissions'], a['capacity_clipped'], a['allocator_clipped'], a['filled_lifecycles'], a['submitted_without_fill'], money(a['filled_notional'])])
    body += table(['Scenario', 'Window', 'Policy', 'Admission events', 'Capacity clipped', 'Allocator clipped', 'Filled entries', 'Submissions unfilled', 'Actual notional m'], rows)
    body += para('The two clipping counts describe sequential recorded constraints and can overlap; they are not separate causal loss totals. Funding decisions are separated from repeated fill and expiry events. Unknown-liquidity rejection does not establish that the forgone trade had negative expectancy. Detailed records retain forecast observations, request size, allocation, recorded cash/equity, redistribution and eventual filled economics.')
    for group, label in GROUPS:
        for window, pair in feedback['scenarios'][group].items():
            comp = migration['comparisons'][group][window]
            match = pair['comparison']
            body += '<h3>' + escape(label + ' · ' + window) + '</h3>'
            body += table(['Net P&L delta m', 'Per-unit term m', 'Quantity term m', 'Added−removed term m', 'Whole old winners retained %', 'Top five old winners retained %'], [[money(comp['pnl_delta']), money(comp['per_unit_execution_effect']), money(comp['quantity_effect']), money(comp['new_minus_removed_effect']), number(comp['whole_old_winner_pnl_retention_pct']), number(comp['top_five_original_winner_pnl_retention_pct'])]])
            body += table(['Matched admissions', 'Changed sample history', 'Changed exit cap', 'Changed approved quantity', 'Changed filled quantity'], [[match[k] for k in ('matched_count', 'changed_histories', 'changed_caps', 'changed_allocations', 'changed_fills')]])
            body += table(['Policy', 'Funding constraints'], [[mode, json.dumps(pair[mode]['funding_bindings'], sort_keys=True)] for mode in ('control', 'migration')])
            changes = match['largest_absolute_changes'][:6]
            body += table(['Entry', 'Symbol', 'Control request→fill', 'Migration request→fill', 'Control exit cap', 'Migration exit cap', 'Recorded cash delta m', 'P&L delta m'], [[r['timestamp'][:10], r['symbol'], str(r['control']['approved_qty'])+'→'+str(r['control'].get('filled_qty',0)), str(r['migration']['approved_qty'])+'→'+str(r['migration'].get('filled_qty',0)), r['control']['exit_cap'], r['migration']['exit_cap'], '—' if r['cash_change'] is None else money(r['cash_change']), money(r['net_pnl_change'])] for r in changes])
    body += para('These are exact accounting decompositions, not randomized causal effects. Per-unit terms include both entry and exit execution. Whole-winner retention includes removed winners as zero and replacement losses as losses; matched-only retention would conceal removals. The largest changes are disclosed for concentration analysis, not used to blacklist losers or tune a subsequent strategy. Unfilled and never-admitted opportunities have no observed trade-return label.')
    body += '<h2>State continuity and scope</h2>'
    body += para('The action-independent panel contains 103 names per source session and uses only completed prior-day target observations. Every native case independently recomputes its source projection and request arithmetic. Original five-minute entry features and coarse liquidity history remain exact. If OOS is permitted, its matched candidate/scenario/capped run restores actual IS terminal routed history plus the full observed panel prefix and checks that prefix against separately authenticated OOS sources before April 1. A flat account reset does not fabricate market-history continuity.')
    body += para('The inherited 99 authentic closing-clock corrections across 62 names remain bound to exact original OHLCV and source clocks. They were not rerun and no performance benefit is separately attributed to them. The original F pilot remains ineligible for union; this study creates no new entry family and does not change entry supply.')
    body += '<h2>Disposition and remaining work</h2>' + para(result)
    body += para('Next work must establish causal opportunity supply and chronological net expectancy before a new entry or allocation policy earns a native comparison. Frequency alone is not alpha. This foundation result can diagnose an unstable size mapping, but cannot by itself meet the broader requested return/frequency/DD objective. Any follow-on experiment needs a separate bounded protocol frozen before its outcomes; this report authorizes no replacement candidate or threshold search.')
    body += para(LIMITS)
    body += '<p><a href="selected_research_policy.json">Retained research profile</a> · <a href="review_bundle/implementation.patch">Implementation patch</a> · <a href="results.json">Results and gate evidence</a> · <a href="admission_feedback.json">Complete admission feedback</a></p>'
    return body


def main():
    e.configure()
    e.verify_frozen()
    for name in ('report.html', 'report.txt', 'results.png', 'report_receipt.json', 'selected_research_policy.json'):
        assert not (e.OUT / name).exists(), 'Preserve and review any reporting repair: ' + name
    data, feedback = read(e.OUT / 'results.json'), read(e.OUT / 'admission_feedback.json')
    assert feedback['results_sha256'] == sha(e.OUT / 'results.json')
    write(e.OUT / 'selected_research_policy.json', profile(data))
    chart(data)
    body = render(data, feedback)
    style = 'body{font:16px/1.6 system-ui;color:#172b3b;max-width:1450px;margin:35px auto;padding:0 24px}h1,h2,h3{line-height:1.3}img{width:100%;height:auto}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%;margin:18px 0}th,td{border:1px solid #ccd6df;padding:7px;text-align:left}th{background:#eef3f7}a{color:#1b5d87}'
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OLR structural foundation</title><style>'+style+'</style>'+body+'</html>'
    (e.OUT / 'report.html').write_text(html, encoding='utf-8')
    class Text(HTMLParser):
        def __init__(self):
            super().__init__(); self.parts=[]
        def handle_data(self, value):
            self.parts.append(value)
        def handle_starttag(self, tag, attrs):
            if tag in ('h1','h2','h3','p','tr'):self.parts.append('\n')
            if tag in ('th','td'):self.parts.append(' | ')
    text = Text(); text.feed(body)
    (e.OUT / 'report.txt').write_text(''.join(text.parts), encoding='utf-8')
    bundle = implementation_bundle()
    files = {name:sha(e.OUT / name) for name in ('results.json','admission_feedback.json','report.html','report.txt','results.png','selected_research_policy.json','results_collection_receipt.json','training_decision.json')}
    write(e.OUT / 'report_receipt.json', dict(utc=utc(), files={**files, **bundle},
        report_script_sha256=sha(__file__), qualification_complete_sha256=sha(e.ROOT / 'qualification_complete.json'),
        frozen_harness_sha256=sha(e.ROOT / 'frozen_harness.json')))
    print(e.OUT / 'report.html')


if __name__ == '__main__':
    main()
