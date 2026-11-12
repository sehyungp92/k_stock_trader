"""Read-only report authentication shared by registration and final review."""
from pathlib import Path
from .retention_research_io_v1 import read, bound, verify
from .qualified_artifacts import require, close
from .pending_funding_qualification_full_diagnostics_v1 import STATUS

OLD_STATUS = 'audited_pending_funding_screening_only_no_promotion'


def check_report(receipt_path, key, account, recipe, folder, reused=False):
    receipt_path = Path(receipt_path)
    receipt = read(receipt_path)
    require(receipt['complete'] and not receipt['economic_promotion'], 'Incomplete/promoted report')
    artifacts = [bound(receipt_path)] + [receipt[k] for k in
        ('report', 'full_diagnostics', 'integrity', 'framework_status')]
    verify(artifacts)
    full = read(receipt['full_diagnostics']['path'])
    integrity = read(receipt['integrity']['path'])
    framework = read(receipt['framework_status']['path'])
    status = OLD_STATUS if reused else STATUS
    require(full['promotion_status'] == full['diagnostics_summary']['promotion_status'] ==
        framework['payload']['promotion_status'] == status, 'Misleading report status')
    require(integrity['passed'] and integrity['native_evidence_unchanged'], 'Report integrity failure')
    ctx = full['diagnostics_summary']['pending_funding_context']
    name, period, scenario = key.split('/')
    profile = ctx['profile']
    require(profile['primary_profile'] == name and profile['period'] == period and
        ctx['scenario'] == scenario and profile['recipe'] == recipe, 'Report identity/context drift')
    require(Path(profile['result_paths'][scenario]) == Path(folder)/'result.json', 'Wrong native report source')
    require(not ctx['production_enabled'] and not ctx['canonical_replaced'], 'Unqualified promotion')
    metrics = account['metrics']
    close(full['final_metrics']['net_return_pct']*100, metrics['return_pct'], 'Report return drift')
    close(full['final_metrics']['max_drawdown_pct']*100, metrics['drawdown_pct'], 'Report DD drift')
    activity = account['activity']
    loss = abs(activity['gross_losing_net_pnl_krw'])
    require(loss > 0, 'Unsupported zero-loss profit-factor comparison')
    close(full['final_metrics']['profit_factor'], activity['gross_winning_net_pnl_krw']/loss, 'Report PF drift')
    require(ctx['profitable_activity'] == activity and ctx['actual_intervention_support'] == account['support']
        and ctx['funding'] == account['funding'], 'Report activity/support/funding drift')
    text = Path(receipt['report']['path']).read_text(encoding='utf-8')
    marker = 'AUDITED OPENING PENDING-FUNDING SCREENING' if reused else 'AUDITED OPENING PENDING-FUNDING FOUR-STRESS EVIDENCE'
    require(marker in text and status in text, 'Missing truthful report text')
    return {'case': key, 'passed': True, 'reused_v25_report': reused,
        'scope': 'Original V25 screening report reused unchanged; joint context comes from V28 assessment.' if reused else
                 'New existing-engine stress report; funded/independent/live qualification pending.',
        'return_pct': metrics['return_pct'], 'drawdown_pct': metrics['drawdown_pct'],
        'profit_factor': full['final_metrics']['profit_factor'], 'artifacts': artifacts}
