"""Small, registered native opening-risk and partial-exit experiment."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from scripts import olr_targeted_risk_study_v1 as previous
n,s,b=previous.n,previous.s,previous.b
REPO=Path(__file__).resolve().parents[1]
PARENT=previous.ROOT
ROOT=PARENT.parent/'opening_risk_v2'
OUT=REPO/'data/backtests/analysis/olr_opening_risk_20261010'
ENGINES=Path.home()/'.codex/research/olr_opening_risk_v2'
REFERENCE='cc9a4ad98fc93351e'
PATCHED=('strategy_olr/config.py','strategy_olr/research.py','strategy_olr/core/logic.py',
         'strategy_olr/core/loss_budget.py','strategy_olr/core/opening_tail.py','strategy_olr/core/weak_partial.py',
         'scripts/portfolio_synergy_study.py')

def register():
    if (ROOT/'registration.json').exists():return n.read(ROOT/'candidate_plan.json')
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    base=copy.deepcopy(n.read(previous.OUT/'selected_research_policy.json')['mutations'])
    base.update(capacity_opening_tail=False,capacity_weak_partial_fraction=0.)
    specs=[('selected_control',{}),('opening_tail_es90',{'capacity_opening_tail':True}),
           ('concentration_55',{'olr.allocation.max_position_pct':.55}),
           ('weak_partial_half',{'capacity_weak_partial_fraction':.5})]
    candidates=[]
    for label,changes in specs:
        mutations={**copy.deepcopy(base),**changes}
        candidates.append({'id':'c'+n.sig({'version':'opening_risk_v2','mutations':mutations})[:16],
            'label':label,'history_policy':'literal_carry','mutations':mutations,'mutations_hash':n.sig(mutations),
            'aliases':[],'kind':'opening_risk_v2','paths':[list(p) for p,_,_ in n.mutation_edits(base,mutations)]})
    plan={'baseline':base,'baseline_id':candidates[0]['id'],'candidates':candidates}
    n.write(ROOT/'candidate_plan.json',plan)
    protocol=copy.deepcopy(n.read(PARENT/'protocol.json'));protocol['generation_directories']={w:str(ENGINES) for w in ('train','oos')}
    n.write(ROOT/'protocol.json',protocol)
    mechanism=copy.deepcopy(n.read(PARENT/'mechanism_plan.json'));mechanism.update(plan);n.write(ROOT/'mechanism_plan.json',mechanism)
    registration={'registered_utc':s.utc(),'reference':REFERENCE,'candidates':candidates,'windows':protocol['windows'],
        'hypotheses':{
          'opening_tail_es90':'max(existing 1.25 ATR, buffered price * shrinkage expected shortfall of adverse close-to-next-eligible-open returns) + costs; original 4/6/8% position/sector/account allowances.',
          'concentration_55':'Reduce the existing strategy maximum position fraction from .65 to .55, retaining six slots and native ranks/eligibility. Existing liquidity-only redistribution remains unchanged.',
          'weak_partial_half':'Once per lifecycle, on a later session after three distinct completed bars of that session, close<=-.4R and observed MFE<.5R: sell half rounded down leaving at least one share. Existing stronger exit has priority; working partial can only be escalated after matching cancellation acknowledgment.'},
        'tail_estimator':{'sessions':60,'quantile':.9,'symbol_peer_prior_strength':20,
          'pool':'same-sector peers excluding own symbol; declared-universe peers if fewer than 20 sector samples',
          'observation':'Previous official closing match to first next-session continuous open with positive contiguous completed-bar capacity; early price proxy, not a guaranteed full-position liquidation price.',
          'availability':'Use outcomes from completed prior sessions only. Current-session opening outcomes become usable on the following session.',
          'universe':'Frozen declared 103-symbol universe; no selection by traded names or eventual P&L.',
          'corporate_actions':'Exclude declared ex-date discontinuities from price-loss estimation; native exposure guard remains unchanged.',
          'carry':'Restore actual candidate/scenario-specific terminal IS tail and original liquidity history. Require equality to independently built OOS prior-prefix tail history.'},
        'gates':{'is_return_loss_max_pp_original':3.,'is_return_loss_max_pp_guard':3.,'is_return_loss_max_pp_selected':1.5,
          'is_dd_increase_max_pp_guard':2.,'oos_return_improvement_min_pp_selected':.25,
          'oos_dd_reduction_min_pp_selected':.75,'oos_trades_min':45,'is_trades_min':180,
          'stress':'All original/guard/selected IS return and guard IS DD gates also apply; OOS return at least stressed selected and DD below stressed selected; full native audits.',
          'selection':'Among fully qualified variants maximize nominal OOS return/intraday DD, then balanced IS/OOS frequency, then IS return. Report Pareto alternatives.',
          'combination':'At most one union of the best two individually nominal-qualified mechanisms; no combination if fewer than two qualify. No new thresholds.',
          'qualification':'Stress control and at most two objective leaders: return/DD and balanced frequency. Capped comparisons control and selected. If no successor qualifies retain selected control.'},
        'maximum_primary_candidates':4,'maximum_conditional_combinations':1,'broad_search':False,
        'independent_validation':False,'promotion':False,'unchanged_signal_target':True}
    n.write(ROOT/'registration.json',registration)
    (OUT/'registration.txt').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return plan

def activate():
    b.ROOT=ROOT;b.ENGINES=ENGINES

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register']);a=p.parse_args();register()
