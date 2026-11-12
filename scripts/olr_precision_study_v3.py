"""Registered bounded successor study starting from the retained selection."""
from __future__ import annotations
import copy
from pathlib import Path
from scripts import olr_opening_risk_study_v2 as previous
n,s,b = previous.n,previous.s,previous.b
REPO=Path(__file__).resolve().parents[1]
PARENT=previous.ROOT
ROOT=PARENT.parent/'precision_v3'
OUT=REPO/'data/backtests/analysis/olr_precision_20261011'
ENGINES=Path.home()/'.codex/research/olr_precision_v3'
REFERENCE='c6718aff2b6b3bbc1'
NEW_MODULES=('strategy_olr/core/precision_v3.py','strategy_olr/core/liquidation_tail_v3.py')
PATCHED=(*NEW_MODULES,'strategy_olr/config.py','strategy_olr/research.py','strategy_olr/core/logic.py',
         'strategy_olr/core/weak_partial.py','strategy_olr/core/loss_budget.py','scripts/portfolio_synergy_study.py',
         'scripts/olr_liquidation_panel_v3.py')

def candidate(label,base,changes):
    m={**copy.deepcopy(base),**changes}
    return {'id':'c'+n.sig({'version':'precision_v3','mutations':m})[:16],'label':label,
            'history_policy':'literal_carry','mutations':m,'mutations_hash':n.sig(m),'aliases':[],
            'kind':'precision_v3','paths':[list(p) for p,_,_ in n.mutation_edits(base,m)]}

def register():
    if (ROOT/'registration.json').exists():return n.read(ROOT/'candidate_plan.json')
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    profile=n.read(previous.OUT/'selected_research_policy.json')
    assert profile['selected_id']==REFERENCE and profile['results_sha256']==n.sha(previous.OUT/'results.json')
    base=copy.deepcopy(profile['mutations'])
    base.update(capacity_liquidation_tail=False,capacity_structural_partial=False,capacity_late_confirmation=False)
    cs=[candidate(label,base,changes) for label,changes in [
        ('selected_control',{}),('liquidation_tail_es90',{'capacity_liquidation_tail':True}),
        ('structural_partial_half',{'capacity_structural_partial':True,'capacity_weak_partial_fraction':.5}),
        ('late_confirmation_9',{'capacity_late_confirmation':True})]]
    plan={'baseline':base,'baseline_id':cs[0]['id'],'candidates':cs}
    n.write(ROOT/'candidate_plan.json',plan)
    protocol=copy.deepcopy(n.read(PARENT/'protocol.json'));protocol['generation_directories']={w:str(ENGINES) for w in ('train','oos')}
    n.write(ROOT/'protocol.json',protocol)
    mechanism=copy.deepcopy(n.read(PARENT/'mechanism_plan.json'));mechanism.update(plan);n.write(ROOT/'mechanism_plan.json',mechanism)
    reg={'registered_utc':s.utc(),'reference':REFERENCE,'reference_profile_sha256':n.sha(previous.OUT/'selected_research_policy.json'),
      'reference_results_sha256':profile['results_sha256'],'windows':protocol['windows'],'candidates':cs,
      'correction_to_previous_diagnosis':'The old tail panel uses the FIRST CAPACITY-ELIGIBLE open, 09:05 for 042660/403870, and already includes their first five-minute movement. The new hypothesis concerns full-position capacity and entry-to-liquidation loss, not an omitted opening bar.',
      'hypotheses':{
        'liquidation_tail_es90':'Replace the one-share previous-close proxy with lagged all-universe decision-close-to-next-session quantity-dependent adverse liquidation scenarios; max with original 1.25 ATR plus unchanged costs; retain 4/6/8% allowances.',
        'structural_partial_half':'Existing once-only half exit at later-session close<=-.4R, MFE<.5R after >=3 bars, additionally requires two contiguous completed closes below entry-session low AND cumulative current-session VWAP. Stronger guard and pending-order safeguards retained.',
        'late_confirmation_9':'Preserve original six-bar confirmations; allow first qualifying signal in bars7-9 with unchanged price/VWAP/close-location gates, only real remaining cash, gross/risk allowance and remaining executable capacity. No retry of an earlier rejected signal.'},
      'liquidation_estimator':{'sessions':60,'quantile':.9,'peer_prior_strength':20,
        'reference':'last completed continuous close before the historical 14:30 decision, session-adjusted',
        'execution':'next-session continuous opens, capacity=floor(participation * immediately previous contiguous continuous bar volume); halt/periodic-auction/session rules inherited; no same-bar volume or guessed auction fills',
        'size':'translate upper proposed current notional to historical shares using ceil(notional/reference); quote at upper original allocation including permitted recycling, so risk cannot be bypassed by extras',
        'monotone_envelope':'maximum adverse average proceeds loss over all filled prefixes up to requested shares; unliquidated remainder after last continuous interval valued at zero for a conservative liquidity impairment scenario; no favorable extrapolation',
        'pool':'same-sector peers excluding own symbol; universe fallback if fewer than20. Symbol weight n/(n+20). Pool history only, never labels selected by eventual trade outcome.',
        'clock':'Outcome available after full following session; usable only on later dates. Actual IS terminal panel state restored and matched to independent OOS prefix.',
        'limitations':['Conservative scenario rather than forecast of actual exit policy or guaranteed loss bound.','Current KRW notional mapped to historical price and liquidity; fixed universe and taxonomy remain.','Individual tail budgets are retained; aligned-date portfolio losses are diagnostic, not a new correlation model.']},
      'gates':{'is_return_loss_max_pp_original':3.,'is_return_loss_max_pp_guard':3.,'is_return_loss_max_pp_selected':1.5,
        'is_dd_increase_max_pp_guard':2.,'oos_return_improvement_min_pp_selected':.25,'oos_dd_reduction_min_pp_selected':.75,
        'is_trades_min':186,'oos_trades_min':47,'is_entry_days_min':88,'oos_entry_days_min':29,
        'stress':'Preserve original/guard/selected IS return and guard DD gates; OOS return >= stressed control and DD strictly lower; IS/OOS unique entries >= matched stress control.',
        'combination':'At most one active risk component A/B plus C. Choose risk by IS preservation pass first, then largest nominal OOS DD reduction, then IS return, then OOS return. Active means economic trade path differs from control. No component must pass the full frequency or OOS improvement gates before this preregistered interaction.',
        'selection':'Among full nominal+stress qualifiers maximize OOS return/DD then balanced genuine frequency then IS return; report Pareto alternatives.',
        'qualification':'Stress control and up to2 nominal-qualified objective leaders; no-upward-compounding control and final selection. Retain selected control if none qualifies.'},
      'maximum_primary_candidates':4,'maximum_conditional_combinations':1,'broad_search':False,
      'history_policy':'literal actual candidate/scenario/capped IS market-history carry','promotion':False,'independent_validation':False,
      'source_policy':'Clone authenticated retained frozen engines and apply surgical patch; do not copy unrelated workspace core changes.',
      'diagnostics':['mechanism activation and quantity changes','winner preservation and cash/quantity decomposition','genuine late entries and meaningful notional','sparse auction support','full intraday DD and endpoint wealth','component versus combination interaction']}
    n.write(ROOT/'registration.json',reg);n.write(OUT/'registration.json',reg)
    return plan

def activate():
    b.ROOT=ROOT;b.ENGINES=ENGINES

if __name__=='__main__':register()
