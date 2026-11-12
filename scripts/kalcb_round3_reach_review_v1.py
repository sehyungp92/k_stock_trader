"""Attribute native prefix reach to individual registered components."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import ROOT,read,write,bound
from backtests.strategies.kalcb.round3_analysis_v1 import native_reach
from backtests.strategies.kalcb.round3_contract_v1 import verify_acceptance
source=ROOT/'reach_v1/completion.json';data=read(source)
for receipt in data['receipts'].values():verify_acceptance(receipt)
arms=read(ROOT/'native_v5/arms.json');result=data.copy()
for name,arm in arms.items():
    if not arm['selectable']:continue
    cfg=arm['config'];v=cfg['entry_plan_routes'][0]['min_quality_votes'];h=cfg['research_opening_failure_min_hold_bars'];comparisons={}
    for component,parent in ([('votes',f'v6_h{h}')] if v!=6 else [])+([('hold',f'v{v}_h0')] if h else []):
        records={p:native_reach(ROOT/'reach_v1/accounts'/p/name,ROOT/'reach_v1/accounts'/p/parent,smoke=True) for p in ('is','oos')}
        effective=any(r['changed_action_records'] and r['changed_physical_fills'] for r in records.values())
        if component=='hold':effective=effective and any(r['opening_failure_exit_decisions'] for r in records.values())
        comparisons[component]=dict(counterfactual=parent,periods=records,effective=effective)
    result['arms'][name]['component_reach']=comparisons
    result['arms'][name]['eligible_for_full_evaluation']=all(v['effective'] for v in comparisons.values())
    if not result['arms'][name]['eligible_for_full_evaluation']:
        result['arms'][name].update(verdict='component_inactive_in_registered_prefix',coverage_gap=True)
result['review_registration']=dict(program=bound(__file__),source=bound(source),economic_outcomes_used=False,
    rule='A composite needs each component to change native actions/fills against its own removal; absent prefix reach is an explicit scope gap, not economic failure.')
write(ROOT/'reach_review_v1.json',result)
for n,a in result['arms'].items():print(n,a['eligible_for_full_evaluation'],a['verdict'])
