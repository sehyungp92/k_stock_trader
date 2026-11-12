"""Read-only joint qualification of native stress and funded allocation cases."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, validate_native
from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
from backtests.strategies.kalcb.fixed_allocation_reference_v1 import verify_identity_bridge
from scripts.kalcb_population_assessment_v1 import account, paired_attribution
from scripts.kalcb_removal_qualification_assessment_v1 import drawdown_episodes


def require_complete(root):
    done = read(root/'stage_completion.json')
    progress = read(root/'progress.json')
    expected = {'compounding': 8, 'identity': 4, 'fixed': 24}
    if not done['complete'] or done['native_cases'] != 36 or Counter(r['mode'] for r in progress) != expected:
        raise ValueError('Incomplete funded qualification')
    keys = [(r['mode'], r['name'], r['period'], r['scenario']) for r in progress]
    if len(set(keys)) != 36 or any(r['returncode'] != 0 for r in progress):
        raise ValueError('Duplicated or failed native qualification')
    for row in progress:
        verify([row[k] for k in ('result', 'funding', 'log')])
        if row['mode'] != 'compounding':
            verify([row['allocation']])
    return progress


def account_evidence(folder, item, scenario, *, funded=False):
    validate_native(folder, item, scenario, save=False)
    result = account(folder)
    episodes = drawdown_episodes(read(folder/'equity.json'))
    if abs((episodes[0]['drawdown_pct'] if episodes else 0) - result['metrics']['drawdown_pct']) > 1e-8:
        raise ValueError('Drawdown attribution drift')
    result['largest_MTM_drawdown_episodes'] = episodes[:5]
    review = funding_review(folder)
    if (folder/'funding_review.json').exists() and read(folder/'funding_review.json') != review:
        raise ValueError('Changed actual funding evidence')
    result['funding'] = review
    if funded:
        allocation = read(folder/'allocation_reference.json')
        if allocation['enabled'] is not True or allocation['reference'] != 100_000_000. or allocation['ledger_cash_mutations'] != 0 or allocation['external_capital_flows'] != 0:
            raise ValueError('Unsupported funded allocation interpretation')
        result['allocation'] = {k:v for k,v in allocation.items() if k != 'rows'}
        result['allocation']['binding'] = bound(folder/'allocation_reference.json')
    return result


def difference_in_deltas(fixed_candidate, fixed_parent, compound_candidate, compound_parent):
    # Four actual, independently funded paths, not summed independent trades.
    return {k:(fixed_candidate['metrics'][k]-fixed_parent['metrics'][k])-
              (compound_candidate['metrics'][k]-compound_parent['metrics'][k])
            for k in ('return_pct','drawdown_pct','trades','active_days')}


def assess(root, output):
    if output.exists():
        raise ValueError('Preserve prior assessment')
    progress = require_complete(root)
    protocol = read(root/'protocol.json')
    verify(protocol['programs']+protocol['bindings'])
    support = read(root/'prior125_support_transfer.json')
    verify([support['program']])
    for row in support['rows']:
        verify(row['bindings'])
    accounts = {}
    for mode, recipes in (('compounding',protocol['stress_recipes']),('fixed',protocol['recipes'])):
        for name, spec in recipes.items():
            for period in ('is','oos'):
                item = protocol['profiles'][name][period] if mode == 'compounding' else read(root/'cases'/name/period/'input_profile.json')
                verify([item[k] for k in ('manifest','plan','policy')])
                for scenario in spec['scenarios']:
                    folder = root/('compounding/cases' if mode == 'compounding' else 'cases')/name/period/scenario
                    accounts[mode,name,period,scenario] = account_evidence(folder,item,scenario,funded=mode=='fixed')
    bridges = []
    for name in ('control','control_expanding_atr5'):
        for period in ('is','oos'):
            parent = root/'compounding/cases'/name/period/'baseline'
            folder = root/'identity'/name/period/'baseline'
            item = read(folder.parent/'input_profile.json')
            validate_native(folder,item,'baseline',save=False)
            bridge = verify_identity_bridge(folder,parent)
            if read(folder/'identity_bridge.json') != bridge or read(folder/'allocation_reference.json')['enabled'] is not False:
                raise ValueError('Invalid disabled-overlay bridge')
            bridges.append({'name':name,'period':period,'bridge':bound(folder/'identity_bridge.json'),
                            'account':bound(folder/'independent_account_verification.json')})
    trials = []
    for name, spec in protocol['stress_recipes'].items():
        parent = 'control' if spec['parent']=='expanding' else 'control_expanding_atr5'
        row = {'recipe':name,'specification':spec,'compounding':{},'funded_fixed':{}}
        for period in ('is','oos'):
            for scenario in spec['scenarios']:
                a,b = accounts['compounding',name,period,scenario],accounts['compounding',parent,period,scenario]
                row['compounding'][period+'/'+scenario] = {'account':a,'vs_local_parent':paired_attribution(a,b)}
                if ('fixed',name,period,scenario) in accounts:
                    fa,fb = accounts['fixed',name,period,scenario],accounts['fixed',parent,period,scenario]
                    row['funded_fixed'][period+'/'+scenario] = {'account':fa,'vs_local_parent':paired_attribution(fa,fb),
                        'same_policy_vs_compounding':paired_attribution(fa,a),
                        'policy_delta_fixed_minus_compounding':difference_in_deltas(fa,fb,a,b)}
        row['global_IS_sacrifice_pp'] = {s:accounts['compounding','control','is',s]['metrics']['return_pct']-
            accounts['compounding',name,'is',s]['metrics']['return_pct'] for s in ('baseline','combined')}
        row['IS_reporting_bands'] = {str(b):all(v<=b for v in row['global_IS_sacrifice_pp'].values()) for b in (2,5,10)}
        row['all_four_stresses_tested'] = set(spec['scenarios'])=={'baseline','participation','cost','combined'}
        trials.append(row)
    result = {'created_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),
        'inputs':[bound(root/f) for f in ('protocol.json','study.json','stage_completion.json','stress_completion.json',
            'progress.json','allocation_ledger.json','compounding/qualification_ledger.json','prior125_support_transfer.json')],
        'all_cases_source_account_passed':True,'fresh_native_cases':len(progress),'compounding_cases':56,'fixed_cases':24,
        'identity_bridges':bridges,'trials':trials,'actual_parent_gate_support':support,
        'scope':'All finite stress/prior cases plus funded fixed nominal allocation comparisons. Actual native funding, borrowing and risk state retained; no replenishment/rescaling. Conditional paired blocks are descriptive, not independent or multiplicity-adjusted. Policy delta differences use four funded account paths and are not isolated causal compounding PnL. Full report and economic decision still required; no automatic promotion.'}
    verify(protocol['programs']+protocol['bindings'])
    write(output/'assessment.json',result)
    print(str(output/'assessment.json'),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assess(a.root,a.output)
