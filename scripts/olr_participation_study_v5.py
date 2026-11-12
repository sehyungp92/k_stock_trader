"""One bounded earlier-stage entry pilot; no threshold search or live activation."""
import copy
from pathlib import Path
from scripts import olr_repair_study_v4 as previous
n,s,b = previous.n,previous.s,previous.b
REPO = Path(__file__).resolve().parents[1]
PARENT = previous.ROOT
ROOT = PARENT.parent/'participation_v5'
OUT = REPO/'data/backtests/analysis/olr_participation_20261011'
ENGINES = Path.home()/'.codex/research/olr_participation_v5_r1'
REFERENCE = 'c4f49be01ed09b8dc'
FLAG = 'olr.research.empty_participation_rescue'
NEW_MODULES = ('strategy_olr/participation_rescue_v5.py',)
PATCHED = (*NEW_MODULES, 'strategy_olr/config.py', 'strategy_olr/research.py',
           'strategy_olr/core/logic.py', 'strategy_olr/core/allocation.py')


def candidate(label, base, changes):
    m = {**copy.deepcopy(base), **changes}
    return dict(id='c'+n.sig({'version':'participation_v5','mutations':m})[:16], label=label,
        history_policy='literal_carry', mutations=m, mutations_hash=n.sig(m), aliases=[],
        kind='participation_v5', paths=[list(p) for p,_,_ in n.mutation_edits(base,m)])


def register():
    if (ROOT/'registration.json').exists():
        return n.read(ROOT/'candidate_plan.json')
    ROOT.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    profile=n.read(previous.OUT/'selected_research_policy.json')
    assert profile['selected_id']==REFERENCE and profile['results_sha256']==n.sha(previous.OUT/'results.json')
    base={**copy.deepcopy(profile['mutations']),FLAG:False}
    cs=[candidate('selected_control',base,{}), candidate('bounded_participation_rescue',base,{FLAG:True})]
    plan=dict(baseline=base,baseline_id=cs[0]['id'],candidates=cs)
    protocol=copy.deepcopy(n.read(PARENT/'protocol.json'))
    protocol['generation_directories']={w:str(ENGINES) for w in ('train','oos')}
    n.write(ROOT/'protocol.json',protocol); n.write(ROOT/'candidate_plan.json',plan)
    mechanism=copy.deepcopy(n.read(PARENT/'mechanism_plan.json')); mechanism.update(plan)
    builder=REPO/'scripts/olr_participation_snapshots_v5.py'
    mechanism['snapshot_acceleration'].update(module=builder.name,implementation_sha256=n.sha(builder),prefix_memo=False,
        dependencies={name:n.sha(REPO/'scripts'/name) for name in ('olr_snapshot_streaming.py','olr_snapshot_acceleration.py')})
    n.write(ROOT/'mechanism_plan.json',mechanism)
    gates=copy.deepcopy(n.read(PARENT/'registration.json')['gates'])
    gates.update(combination='None in this separately frozen entry pilot. A future fixed faster-exit component requires its own clock/source controls before a single union can be registered.',
        qualification='Always compare both control and treatment in nominal, combined stress, and no-upward-compounding IS/OOS. All inherited gates remain mandatory. No further candidate or threshold.',
        capped_preservation='Capped IS >= capped control minus 1.5pp; OOS return >= capped control and DD strictly lower; IS/OOS entries and entry days >= capped control.',
        stress_entry_days='IS and OOS entry days must also be at least matched stressed control.')
    reg=dict(registered_utc=s.utc(),reference=REFERENCE,reference_profile_sha256=n.sha(previous.OUT/'selected_research_policy.json'),
        reference_results_sha256=profile['results_sha256'],windows=protocol['windows'],candidates=cs,gates=gates,
        primary_count=2,maximum_combinations=0,broad_search=False,promotion=False,independent_validation=False,
        mechanism='Only baseline-empty final selection days: remove daily sector-participation floor for reconstruction; retain native top six before filtering; require original sole rejection, positive own five-day flow, current sector return>0/breadth>=.5/effective names>=3. All other native gates, original six-bar signal, exits, costs and execution stay unchanged.',
        sizing='At each admission, cap each new-family order at 5% sizing NAV using native buffered unit cost, and total marked new-family holdings plus pending reservations plus new cohort at 10% sizing NAV. No redistribution extras. Ordinary loss/sector/cash/liquidity limits remain stricter bounds. These are admission caps, not forced mark-to-market trims; execution gaps and price appreciation are reported separately.',
        rationale='The prior late-stage rescue reached only one OOS candidate without a signal. Authenticated earlier-stage screen found 36 IS and 28 OOS six-bar signals without inspecting their future P&L. This pilot tests actual funded entries and drawdown, not signal counts.',
        clock_work='Faster exits require a separately authenticated one-minute execution clock and matched control. No counterfactual screen is reported as native performance; do not combine it with this pilot before native qualification.',
        limitations=['April-July reused development sample','Fixed universe and sector taxonomy','Sparse actual routed auction histories','Five-minute price/volume execution proxies','Additional candidates may fail native cash/risk/liquidity gates or lose money'],
        preserved=['actual matching candidate/scenario/capped IS terminal history before April1','all native source, ledger, capacity and flat-terminal audits','original/guard/selected IS gates','meaningful unique-entry, entry-day and added-notional gates','unchanged old source/result/profile hashes'])
    n.write(ROOT/'registration.json',reg); n.write(OUT/'registration.json',reg)
    return plan


def activate():
    b.ROOT=ROOT; b.ENGINES=ENGINES


if __name__=='__main__': register()
