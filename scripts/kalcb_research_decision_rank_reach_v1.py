"""Source/domain feasibility audit of fixed-set ordering, no financial replay."""
from pathlib import Path
from dataclasses import asdict,replace
from datetime import datetime,timezone
import sys,json,hashlib,ast,time
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str)
def main():
    out=ROOT/'rank_reach_v1';out.mkdir(exist_ok=False);start=time.monotonic();lr=read(ROOT/'labels_run_v5/registration.json')
    save(out/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),program=bound(__file__),labels_registration=bound(ROOT/'labels_run_v5/registration.json'),
        scope='Before any random or predictive native ranking outcome: test only permuting existing opening candidates, preserving them at checkpoint, and permuting new checkpoint candidates. No admission replacement or market-batch reorder.',
        falsifier='Any effective rank gate, physical active-symbol constraint, account-priority branch or context predicate that can change on the supported rank domain makes source-only inactivity unproved; withhold the conclusion.',financial_evaluations=0))
    code=Path(lr['profiles']['is']['manifest']['path']).parent/'files';sys.path.insert(0,str(code))
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    from scripts.kalcb_executed_value_study import load_plan,config_for
    from strategy_kalcb.core.logic import _checkpoint_config,_entry_route_config
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.engine.sim_broker import BrokerCosts
    records=[];bindings=[]
    for period in ['is','oos']:
        profile=lr['profiles'][period];plan=load_plan(Path(profile['plan']['path']));cfg=config_for(plan,'baseline')
        manifest,snapshots=verify_policy(Path(profile['policy']['path']).parent,'interaction_cost_margin')
        updates=timed_policy(Path(profile['policy']['path']).parent,manifest,snapshots)
        assert cfg.entry_plan_frontier_branch_universe and not cfg.frontier_shadow_enabled and not cfg.frontier_rotation_enabled
        assert cfg.secondary_rank_start==0 and cfg.entry_plan_quality_max_frontier_rank==20
        assert cfg.exit_mfe_floor_min_frontier_rank==cfg.exit_mfe_floor_max_frontier_rank==0
        assert cfg.exit_conditional_target_min_frontier_rank==cfg.exit_conditional_target_max_frontier_rank==0
        # Only explicit numeric gates below use rank. Other context maps must not reference it.
        for key,value in asdict(cfg).items():
            if isinstance(value,dict):assert 'rank' not in json.dumps(value),key
        routes=[_entry_route_config(cfg,r) for r in cfg.entry_plan_routes]
        checkpoint=_checkpoint_config(cfg);croutes=[_entry_route_config(checkpoint,r) for r in checkpoint.entry_plan_routes]
        openingmax=max(len(s.candidates) for s in snapshots.values());timedmax=max((len(s.candidates) for seq in updates.values() for s in seq),default=0)
        assert openingmax<=8 and timedmax<=13
        for c in routes:
            assert c.entry_plan_max_frontier_rank in (0,8)
            assert c.entry_plan_quality_max_frontier_rank>=openingmax
        for c in croutes:assert c.entry_plan_max_frontier_rank==0 and c.entry_plan_quality_max_frontier_rank>=timedmax
        # Actual adapter must retain the full candidate set despite ws_budget8.
        checked=0;opening_reorderable=0;checkpoint_reorderable=0
        for day,snapshot in snapshots.items():
            opening_reorderable+=len(snapshot.candidates)>1
            allsnaps=[snapshot]+list(updates.get(day,()))
            for s in allsnaps:
                adapter=KALCBReplayAdapter(cfg,{day:s},initial_equity=1e8,costs=BrokerCosts(0,0,0))
                assert {c.symbol for c in adapter._active_snapshot(s).candidates}=={c.symbol for c in s.candidates}
                for c in s.candidates:
                    if c.symbol in snapshot.by_symbol():assert c.to_json_dict()==snapshot.by_symbol()[c.symbol].to_json_dict()
                    else:assert c.metadata.get('entry_checkpoint_at') and c.metadata.get('feature_available_at')
                checked+=len(s.candidates)
            checkpoint_reorderable+=any(len(s.candidates)-len(snapshot.candidates)>1 for s in updates.get(day,()))
        records.append(dict(period=period,opening_max=openingmax,cumulative_max=timedmax,candidate_instances_checked=checked,
            opening_sessions_with_multiple_candidates=opening_reorderable,checkpoint_sessions_with_multiple_new_candidates=checkpoint_reorderable,
            rank_gate_truth_change_possible=False,physical_symbol_set_change_possible=False,
            native_financial_replays=0,changed_decisions='not empirically replayed',changed_fills='not empirically replayed'))
        bindings.extend([profile['plan'],profile['policy'],profile['manifest']])
    # Pin the actual dispatch and sizing implementation reviewed for rank consumers.
    files=['strategy_kalcb/core/logic.py','strategy_kalcb/risk.py','strategy_kalcb/core/candidate_updates.py',
        'backtests/strategies/kalcb/runner.py','backtests/engine/replay.py','backtests/strategies/kalcb/funded_horizon_v1.py']
    bindings.extend(bound(code/p) for p in files)
    risk=(code/'strategy_kalcb/risk.py').read_text(encoding='utf-8');assert 'candidate_rank' not in risk and 'frontier_rank' not in risk
    driver=(code/'backtests/engine/replay.py').read_text(encoding='utf-8')
    assert 'key=lambda bar: (bar.timestamp, bar.symbol)' in driver and 'for bar in selected:' in driver
    save(out/'source_bindings.json',bindings)
    save(out/'receipt.json',dict(passed=True,seconds=time.monotonic()-start,records=records,financial_evaluations=0,
        conclusion='Fixed admitted-set ordering has no identified native economic control on the exercised rank domains; not a matched-entry placebo result.',
        decision='Withhold full random/predictive ranking accounts as unsupported information tests. Do not replace them with admission expansion or broker-order reordering.',
        limitations=['No financial ranking perturbation replay performed; no empirical zero-effect estimate or CI.',
            'Existing score may still have information value for admission/holding; source rank inactivity does not reject alpha.',
            'Matched-entry replacement null remains unavailable in this fixed registered comparison.']))
    print('Fixed-set rank feasibility audit complete',flush=True)
if __name__=='__main__':main()
