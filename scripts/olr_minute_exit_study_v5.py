"""Two fixed policies and three native scenarios; no parameter search."""
import copy
from pathlib import Path
from scripts import olr_repair_study_v4 as previous
n,s,b=previous.n,previous.s,previous.b
REPO=Path(__file__).resolve().parents[1]
PARENT=previous.ROOT
ROOT=PARENT.parent/'minute_exit_v5'
OUT=REPO/'data/backtests/analysis/olr_minute_exit_20261011'
SOURCE_OUT=REPO/'data/backtests/analysis/olr_participation_20261011'
ENGINES=ROOT/'native_engines'
REFERENCE='c4f49be01ed09b8dc'
FLAG='olr.research.minute_exit'
NEW_MODULES=tuple('backtests/engine/'+x for x in ('olr_minute_v5.py','olr_minute_audit_v5.py','olr_minute_source_v5.py'))
PATCHED=(*NEW_MODULES,'strategy_olr/config.py','strategy_olr/research.py',
    'backtests/engine/capacity_audit.py','backtests/engine/integrity.py',
    'backtests/engine/qualification_evidence.py','scripts/portfolio_synergy_study.py')


def register():
    if (ROOT/'registration.json').exists():return n.read(ROOT/'candidate_plan.json')
    assert not (ROOT/'cases').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    selected=n.read(previous.OUT/'selected_research_policy.json')
    assert selected['selected_id']==REFERENCE and selected['results_sha256']==n.sha(previous.OUT/'results.json')
    base={**copy.deepcopy(selected['mutations']),FLAG:False};cs=[]
    for enabled,label in ((False,'minute_clock_control'),(True,'fixed_minute_loss_and_fade')):
        m={**base,FLAG:enabled}
        cs.append(dict(id='c'+n.sig(dict(version='minute_exit_v5',mutations=m))[:16],label=label,
            history_policy='literal_carry',mutations=m,mutations_hash=n.sig(m),aliases=[],kind='minute_exit_v5',
            paths=[list(p) for p,_,_ in n.mutation_edits(base,m)]))
    plan=dict(baseline=base,baseline_id=cs[0]['id'],candidates=cs)
    protocol=copy.deepcopy(n.read(PARENT/'protocol.json'))
    protocol.update(generation_directories={w:str(ENGINES) for w in ('train','oos')},
        oos_scope='April-July2026 repeatedly inspected development sample, never independent validation.')
    mechanism=copy.deepcopy(n.read(PARENT/'mechanism_plan.json'));mechanism.update(plan)
    builder=REPO/'scripts/olr_minute_exit_snapshots_v5.py'
    mechanism['snapshot_acceleration']=dict(module=builder.name,implementation_sha256=n.sha(builder),
        dependencies={name:n.sha(REPO/'scripts'/name) for name in ('olr_snapshot_streaming.py','olr_snapshot_acceleration.py')},
        cache_root=str(ROOT/'snapshot_cache'),prefix_memo=False,validated_source_hashes={},
        semantic_scope='Fresh native feature cache; unchanged legacy selection. Minute-exit flag excluded from candidate config hash.')
    gates=copy.deepcopy(n.read(PARENT.parent/'participation_v5/registration.json')['gates'])
    gates.update(combination='Forbidden: completed F pilot failed IS preservation in every scenario.',
        selection='Only fixed E may succeed. Preserve every retained policy gate and every matched minute-control comparison; otherwise retain prior selected profile.',
        qualification='Exactly12 windows: control/E x train/oos x baseline/combined/baseline_capped. Always test both policies in every scenario.',
        matched_minute_control='IS return >= matched fine control minus1.5pp; OOS return >= matched fine control; both common-grid and complete fine-event OOS DD strictly lower. Full promotion additionally preserves matched control entries and days.',
        risk_component='Evaluate same return/DD/native gates without requiring extra entries or added notional. This diagnostic status alone cannot promote a full-objective successor.',
        common_grid='Freeze retained-control5m labels for each scenario; sample fine NAV at label+5 before new opening phase. Fine closes visible at label+1 including auctions. Keep every fine opening/closing/fill-cost point for refined DD.',
        no_upward_compounding='Preserve capped retained IS floor minus1.5pp, OOS return and strictly lower common DD, entries/days; also matched capped fine-control return/DD/entries/days.')
    reg=dict(registered_utc=s.utc(),version='minute_exit_v5',reference=REFERENCE,
        reference_profile_sha256=n.sha(previous.OUT/'selected_research_policy.json'),reference_results_sha256=selected['results_sha256'],
        windows=protocol['windows'],candidates=cs,gates=gates,primary_count=2,maximum_combinations=0,
        mechanism='Keep original completed5m entry signal and sizing. On completed1m after entryday apply unchanged -1R loss with MFE<4R and MFEfade start1R/gap1.25R/floor0. Native cancellation/acknowledgement/remainder lifecycle.',
        execution='Both controls share minute market execution:1% lagged contiguous minute volume,0.5%combined stress; native auction-point volume, native fees and stress costs. No first-minute/stale-gap capacity.',
        source_clock='All99 corrected OOS closing labels use authentic15:32 minute prints with identical old OHLCV; original data kept. Separate old-to-fine source/execution change from E effect.',
        history='Actual candidate/scenario/capped IS terminal coarse history restored beforeApril1; independent OOS prefix equality required; no expansion of actually observed auctions.',
        entry_union_eligible=False,broad_search=False,promotion=False,independent_validation=False,
        limitations=['Sparse routed auction history','Fixed universe and taxonomy','Inferred retained archive clocks, not exchange queue time',
            'Minute prices and lagged-volume capacity are research proxies','April-July is repeatedly inspected development','F entry pilot failed IS preservation'])
    for name,value in [('protocol.json',protocol),('candidate_plan.json',plan),('mechanism_plan.json',mechanism),('registration.json',reg)]:n.write(ROOT/name,value)
    n.write(OUT/'registration.json',reg)
    return plan


def activate():b.ROOT=ROOT;b.ENGINES=ENGINES


if __name__=='__main__':register()
