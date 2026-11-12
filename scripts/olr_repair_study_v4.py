"""Bounded risk-model repairs and empty-afternoon opportunity experiment."""
import copy
from pathlib import Path
from scripts import olr_precision_study_v3 as previous
n,s,b=previous.n,previous.s,previous.b
REPO=Path(__file__).resolve().parents[1]
PARENT=previous.ROOT
ROOT=PARENT.parent/'repair_v4'
OUT=REPO/'data/backtests/analysis/olr_repair_20261011'
ENGINES=Path.home()/'.codex/research/olr_repair_v4'
REFERENCE='cb3762e53f4f930fb'
NEW_MODULES=('strategy_olr/core/risk_repair_v4.py','strategy_olr/flow_rescue_v4.py')
PATCHED=(*NEW_MODULES,'strategy_olr/config.py','strategy_olr/research.py','strategy_olr/core/logic.py',
         'strategy_olr/core/loss_budget.py','strategy_olr/core/allocation.py','scripts/portfolio_synergy_study.py')
FLOW='olr.afternoon.empty_sector_flow_rescue'
RISK='capacity_risk_repair_mode'

def candidate(label,base,changes):
    m={**copy.deepcopy(base),**changes}
    return dict(id='c'+n.sig({'version':'repair_v4','mutations':m})[:16],label=label,
        history_policy='literal_carry',mutations=m,mutations_hash=n.sig(m),aliases=[],kind='repair_v4',
        paths=[list(p) for p,_,_ in n.mutation_edits(base,m)])

def register():
    if (ROOT/'registration.json').exists():return n.read(ROOT/'candidate_plan.json')
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    profile=n.read(previous.OUT/'selected_research_policy.json')
    assert profile['selected_id']==REFERENCE and profile['results_sha256']==n.sha(previous.OUT/'results.json')
    base=copy.deepcopy(profile['mutations']);base.update({RISK:'off',FLOW:False})
    cs=[candidate(label,base,change) for label,change in [
        ('selected_control',{}),
        ('structural_partial_half',{'capacity_structural_partial':True,'capacity_weak_partial_fraction':.5}),
        ('quantity_consistent_tail',{'capacity_liquidation_tail':True,RISK:'quantity'}),
        ('separated_peer_price_tail',{'capacity_liquidation_tail':True,RISK:'separated'}),
        ('aligned_portfolio_shock',{'capacity_liquidation_tail':True,RISK:'portfolio'}),
        ('empty_afternoon_flow_rescue',{FLOW:True})]]
    plan=dict(baseline=base,baseline_id=cs[0]['id'],candidates=cs)
    n.write(ROOT/'candidate_plan.json',plan)
    protocol=copy.deepcopy(n.read(PARENT/'protocol.json'));protocol['generation_directories']={w:str(ENGINES) for w in ('train','oos')}
    n.write(ROOT/'protocol.json',protocol)
    mechanism=copy.deepcopy(n.read(PARENT/'mechanism_plan.json'));mechanism.update(plan);n.write(ROOT/'mechanism_plan.json',mechanism)
    gates=copy.deepcopy(n.read(PARENT/'registration.json')['gates'])
    gates.update(combination='One union: include structural exit only if all four nominal IS preservation gates pass; choose at most one economically active new risk repair passing the same IS gates, ranked by OOS DD reduction then OOS return then IS return; include empty-afternoon rescue only if IS preservation and positive OOS entry increment. Skip if fewer than two components. Do not add unsupported mechanisms or thresholds.',
        qualification='Always stress and cap control and existing structural exit. Stress up to two other active IS-preserving risk components ranked by OOS DD improvement then return, plus at most two full nominal objective leaders. Cap final selection. Component testing is not final promotion.',
        meaningful_frequency='Keep lifecycle and entry-day floors; additionally require positive net incremental OOS entry notional of at least 1% starting capital and median added OOS entry notional at least 1% starting capital. No token fills counted as economic frequency success.')
    reg=dict(registered_utc=s.utc(),reference=REFERENCE,reference_profile_sha256=n.sha(previous.OUT/'selected_research_policy.json'),
      reference_results_sha256=profile['results_sha256'],windows=protocol['windows'],candidates=cs,gates=gates,
      hypotheses={
        'structural_partial_half':'Exact existing v3 half-exit rule, no threshold edits. Always stress and cap to resolve concentration/execution sensitivity.',
        'quantity_consistent_tail':'Same v3 60-session ES90 shrink20 peer-liquidity scenarios. Binary search maximum integer entry size satisfying quantity-dependent 4/6/8% loss and native cash/capacity constraints. Requote base plus recycled extras together; held/pending orders retain admitted per-share risk until physically released.',
        'separated_peer_price_tail':'Same quantity solver, own-stock full-size liquidation tail retained as floor. Transfer only peers\' one-share executable price shocks, averaged by date then ES90. Sector requires >=3 distinct peers and >=20 dates, otherwise declared-universe price peers. Blend with n/(n+20), never dilute own tail. No peer full-notional liquidity impairment transfers.',
        'aligned_portfolio_shock':'Keep original ATR+cost 4/6/8% entry budgets. Add current-mark own-stock quantity liquidation loss on aligned prior dates, worst supported joint scenario limited to 6% sector/8% portfolio. Common-date intersection >=20; reject new risk on inadequate support. Existing/pending exposures consume limits. Admission only; no forced pre-close trims, new exit thresholds or earlier fills.',
        'empty_afternoon_flow_rescue':'Only on days the unchanged final selector is empty. Consider stage-1 names whose sole afternoon rejection is lagged_sector_flow_below_floor; require strictly positive stock flow, positive current sector return, breadth>=0.5 and >=3 effective sector names. Reapply every later score gate and original six-bar signal. No raw score or daily gate changes; real cash/risk/capacity only.'},
      tail_clock='Actual matching IS terminal liquidation history restored before April1, exact independent OOS prefix; 60 prior completed sessions, existing causal panel unchanged.',
      portfolio_rationale='Worst supported common-date stress is a separate registered conservative hypothesis, not ES90 or a guarantee. Admission-time marks protect marginal concentration; cannot retroactively protect an earlier marked peak.',
      primary_count=6,maximum_combinations=1,broad_search=False,promotion=False,independent_validation=False,
      unchanged=['full native execution/source/capacity/ledger/terminal-flat audits','original/guard/selected IS gates','original confirmation window except unchanged disabled v3 flag','fixed universe/taxonomy and cost assumptions','all old result/profile bindings'],
      diagnostics=['all native decision events including allocation rejections','quantity quotes before/after','whole-cohort/extra loss-budget proof','peer name/date support and own impairment','aligned-date sector/account worst-loss proof','rescue candidates, downstream rejections and meaningful filled notional','winner retention including removed winners and monthly concentration','stress/capped component comparisons'],
      limitations=['April-July repeatedly inspected; future source-qualified validation still needed','sparse auction histories','fixed universe/taxonomy','opening-price and lagged-volume proxies','empty-afternoon rescue has only four stage-local OOS empty-day opportunities before further gates'])
    n.write(ROOT/'registration.json',reg);n.write(OUT/'registration.json',reg)
    return plan

def activate():b.ROOT=ROOT;b.ENGINES=ENGINES
if __name__=='__main__':register()
