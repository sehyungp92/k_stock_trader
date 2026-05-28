from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.strategies.registry import create_plugin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run phased auto-optimisation.")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize", help="Run phased greedy optimisation.")
    optimize.add_argument("--strategy", required=True, choices=["kalcb", "olr", "portfolio_synergy"])
    optimize.add_argument("--config", default=None)
    optimize.add_argument("--round-name", default="round")
    optimize.add_argument("--round", type=int, default=None)
    optimize.add_argument("--output-root", default="data/backtests/output")
    optimize.add_argument("--max-workers", type=int, default=1)
    optimize.add_argument("--num-phases", type=int, default=None)
    optimize.add_argument("--start-phase", type=int, default=None)
    optimize.add_argument("--dry-run", action="store_true")
    research = sub.add_parser("research-sweep", help="Run a research-selection sweep before phased optimisation.")
    research.add_argument("--strategy", required=True, choices=["kalcb", "olr"])
    research.add_argument("--config", default=None)
    research.add_argument("--output-root", default="data/backtests/output")
    research.add_argument("--max-workers", type=int, default=2)
    research.add_argument("--holdout-days", type=int, default=42)
    research.add_argument("--fold-days", type=int, default=None)
    research.add_argument("--fold-count", type=int, default=2)
    research.add_argument("--top-n", type=int, default=10)
    research.add_argument("--max-candidates", type=int, default=None)
    research.add_argument("--refine-top-n", type=int, default=3)
    research.add_argument("--max-refinement-candidates", type=int, default=96)
    research.add_argument("--stage1-stage2-seed-count", type=int, default=5)
    research.add_argument("--resume-stage1-artifact", default=None)
    research.add_argument("--audit-finalist-count", type=int, default=5)
    research.add_argument("--no-refinement", action="store_true")
    research.add_argument("--sweep-phase", choices=["combined"], default="combined")
    research.add_argument("--dry-run", action="store_true")
    holdout = sub.add_parser("allocation-holdout", help="Run allocation holdout audit and stress for research-derived strategies.")
    holdout.add_argument("--strategy", required=True, choices=["olr"])
    holdout.add_argument("--config", default=None)
    holdout.add_argument("--research-sweep-path", default=None)
    holdout.add_argument("--allocation-sweep-path", default=None)
    holdout.add_argument("--output-root", default="data/backtests/output")
    holdout.add_argument("--top-n", type=int, default=5)
    holdout.add_argument("--holdout-days", type=int, default=42)
    holdout.add_argument("--max-workers", type=int, default=2)
    holdout.add_argument("--max-stress-scenarios", type=int, default=80)
    holdout.add_argument("--dry-run", action="store_true")
    oos = sub.add_parser("oos-ablation", help="Run reusable train/OOS ablation for a phased optimisation round.")
    oos.add_argument("--strategy", required=True, choices=["kalcb", "olr", "portfolio_synergy"])
    oos.add_argument("--config", required=True)
    oos.add_argument("--round", type=int, default=None, dest="round_num")
    oos.add_argument("--round-dir", default=None)
    oos.add_argument("--output-root", default="data/backtests/output")
    oos.add_argument("--output-dir", default=None)
    oos.add_argument("--adapter", choices=["auto", "runner", "generic", "kalcb-fixed"], default="auto")
    oos.add_argument("--oos-start", default=None)
    oos.add_argument("--oos-end", default=None)
    oos.add_argument("--max-oos", type=int, default=0)
    oos.add_argument("--top-train", type=int, default=40)
    oos.add_argument("--no-perturbations", action="store_true")
    oos.add_argument("--include-phase-candidates", action="store_true")
    oos.add_argument("--no-targeted", action="store_true")
    oos.add_argument("--max-targeted", type=int, default=80)
    oos.add_argument("--candidate-manifest", default=None)
    args = parser.parse_args(argv)

    config = normalize_runtime_config(args.strategy, load_yaml_config(args.config))
    if args.command == "oos-ablation":
        from backtests.auto.oos_ablation import load_round_chain, run_oos_ablation

        round_dir = Path(args.round_dir) if args.round_dir else None
        artifacts = load_round_chain(args.strategy, Path(args.output_root), args.round_num, round_dir=round_dir)
        target = artifacts[-1]
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / args.strategy / f"round_{target.round_num}_oos_ablation"
        payload = run_oos_ablation(
            strategy=args.strategy,
            config=config,
            artifacts=artifacts,
            output_dir=output_dir,
            adapter_name=args.adapter,
            max_oos=args.max_oos,
            top_train=args.top_train,
            include_perturbations=not args.no_perturbations,
            include_phase_candidates=args.include_phase_candidates,
            include_targeted=not args.no_targeted,
            max_targeted=args.max_targeted,
            candidate_manifest=Path(args.candidate_manifest) if args.candidate_manifest else None,
            oos_start=args.oos_start,
            oos_end=args.oos_end,
        )
        print(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "target_round": payload["target_round"],
                    "result_path": str(output_dir / "oos_ablation_results.json"),
                    "summary_path": str(output_dir / "oos_ablation_summary.md"),
                    "oos_evaluated": payload["counts"]["oos_evaluated"],
                    "train_confirmed": payload["counts"]["train_confirmed"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "allocation-holdout":
        from backtests.strategies.olr.allocation_holdout_eval import run_allocation_holdout_eval

        payload = run_allocation_holdout_eval(
            config,
            research_sweep_path=args.research_sweep_path,
            allocation_sweep_path=args.allocation_sweep_path,
            output_dir=Path(args.output_root) / args.strategy / "allocation_holdout",
            top_n=args.top_n,
            holdout_days=args.holdout_days,
            max_workers=min(max(1, int(args.max_workers)), 2),
            max_stress_scenarios=args.max_stress_scenarios,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(json.dumps({"strategy": args.strategy, "eval_hash": payload["eval_hash"], "artifact_paths": payload["artifact_paths"], "audit_pass": payload["audit_pass"]}, indent=2))
        return 0
    if args.command == "research-sweep":
        if args.strategy == "kalcb":
            from backtests.strategies.kalcb.research_sweep import build_research_sweep_candidates, run_research_sweep
        elif args.strategy == "olr":
            from backtests.strategies.olr.research_sweep import (
                DEFAULT_EXPECTED_UNIVERSE_SIZE,
                build_afternoon_sweep_candidates,
                build_research_sweep_candidates,
                run_research_sweep,
            )
        else:
            raise ValueError(f"Unsupported research sweep strategy: {args.strategy}")
        if args.sweep_phase != "combined":
            raise ValueError("--sweep-phase is no longer supported")

        candidates = build_research_sweep_candidates()
        if args.max_candidates is not None:
            candidates = candidates[: max(0, int(args.max_candidates))]
        refine_top_n = 0 if args.no_refinement else args.refine_top_n
        max_refinement = 0 if refine_top_n <= 0 else args.max_refinement_candidates
        fold_count = max(0, min(int(args.fold_count), 2)) if args.strategy == "olr" else args.fold_count
        output_dir = Path(args.output_root) / args.strategy / "research_sweeps"
        if args.dry_run:
            coarse_count = len(candidates) + 1
            stage2_count = None
            if args.strategy == "olr":
                stage2_candidates = build_afternoon_sweep_candidates()
                stage2_cap = max_refinement if max_refinement is not None else len(stage2_candidates)
                stage2_count = max(1, int(args.stage1_stage2_seed_count)) * (1 + min(len(stage2_candidates), max(0, int(stage2_cap))))
            if args.strategy == "olr":
                sweep_type = "overnight_leader_rotation_research_training_only"
            else:
                sweep_type = "research_candidate_opportunity_training_only"
            expected_universe_size = DEFAULT_EXPECTED_UNIVERSE_SIZE if args.strategy == "olr" else None
            print(
                json.dumps(
                    {
                        "strategy": args.strategy,
                        "dry_run": True,
                        "sweep_phase": args.sweep_phase,
                        "sweep_type": sweep_type,
                        "causality_policy": (
                            "daily/flow rows use row_date < trade_date; stage-2 intraday selection uses timestamp < 14:30 KST; "
                            "close-to-next-close and next-session MFE are offline labels only"
                            if args.strategy == "olr"
                            else "selection uses only prior completed daily rows; forward intraday bars are research-opportunity scoring only"
                        ),
                        "coarse_candidate_count": coarse_count,
                        "max_refinement_candidates": max_refinement,
                        "candidate_count_estimate": coarse_count + max_refinement + (stage2_count or 0),
                        "resume_stage1_artifact": args.resume_stage1_artifact,
                        "holdout_days": args.holdout_days,
                        "fold_days": args.fold_days,
                        "fold_count": fold_count,
                        "refine_top_n": refine_top_n,
                        "top_n": args.top_n,
                        "output_dir": str(output_dir),
                        **(
                            {
                                "stage1_coarse_candidate_count": coarse_count,
                                "max_stage1_refinement_candidates": max_refinement,
                                "max_stage2_candidate_count": stage2_count,
                                "stage1_stage2_seed_count": args.stage1_stage2_seed_count,
                                "stage1_resume_enabled": bool(args.resume_stage1_artifact),
                                "implementation_lessons_contract": {
                                    "status": "research_only_thin_selector",
                                    "shared_selection_api": [
                                        "strategy_olr.research.daily_selection_from_snapshot",
                                        "strategy_olr.research.afternoon_selection_from_snapshot",
                                        "strategy_olr.research.afternoon_selection_from_contexts",
                                    ],
                                    "reference_pattern": {
                                        "live_data_builder": "_references/trading/strategies/stock/artifact_generator.py",
                                        "replay_data_builder": "_references/trading/backtests/stock/engine/research_replay.py",
                                    },
                                    "live_backtest_divergence_policy": "Only data acquisition and artifact persistence may differ.",
                                },
                                "metric_contract": {
                                    "basis": "research_selection_label_only",
                                    "stage2_basis": "fixed close-auction to next-close portfolio proxy",
                                    "headline_allowed": False,
                                    "official_performance": False,
                                },
                                "fast_replay_policy": {
                                    "enabled": True,
                                    "mode": "compiled_causal_research_replay",
                                    "full_audit_finalist_count": args.audit_finalist_count,
                                    "fill_parity_scope": "not_applicable_research_only",
                                },
                            }
                            if args.strategy == "olr"
                            else {}
                        ),
                        **(
                            {
                                "expected_stock_universe_symbols": expected_universe_size,
                                "universe_policy": "exactly_103_complete_daily_ohlcv_combined_foreign_institutional_flow_symbols",
                                "evidence_label": "research_opportunity_evidence",
                                "official_performance": False,
                            }
                            if args.strategy == "olr"
                            else {}
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        payload = run_research_sweep(
            config,
            output_dir=output_dir,
            holdout_days=args.holdout_days,
            fold_days=args.fold_days,
            fold_count=fold_count,
            top_n=args.top_n,
            max_candidates=args.max_candidates,
            refine_top_n=refine_top_n,
            max_refinement_candidates=max_refinement,
            max_workers=args.max_workers,
            **({"expected_universe_size": DEFAULT_EXPECTED_UNIVERSE_SIZE} if args.strategy == "olr" else {}),
            **({"audit_finalist_count": args.audit_finalist_count} if args.strategy == "olr" else {}),
            **({"stage1_stage2_seed_count": args.stage1_stage2_seed_count} if args.strategy == "olr" else {}),
            **({"resume_stage1_artifact": args.resume_stage1_artifact} if args.strategy == "olr" else {}),
        )
        print(json.dumps({"strategy": args.strategy, "sweep_hash": payload["sweep_hash"], "artifact_paths": payload["artifact_paths"]}, indent=2))
        return 0

    manager = RoundManager("stock", args.strategy, base_dir=Path(args.output_root))
    if args.dry_run:
        latest = manager.get_latest_round()
        round_num = args.round or (latest + 1 if latest else 1)
        archived_rounds = manager.get_archived_rounds()
        while args.round is None and round_num in archived_rounds:
            round_num += 1
        round_dir = manager.round_path(round_num)
        payload = {"strategy": args.strategy, "round": round_num, "round_dir": str(round_dir), "dry_run": True}
        if args.strategy == "kalcb":
            from strategy_kalcb.config import KALCB_CORE_VERSION

            payload["strategy_core_version"] = KALCB_CORE_VERSION
            payload["num_phases"] = args.num_phases or 6
            payload["live_parity_fill_timing"] = "next_5m_open"
            payload["auction_mode"] = "non_auction_continuous"
        if args.strategy == "olr":
            from strategy_olr.config import OLR_CORE_VERSION

            payload["strategy_core_version"] = OLR_CORE_VERSION
            payload["num_phases"] = args.num_phases or 6
            payload["live_parity_fill_timing"] = "completed_5m_signal_next_bar_or_resting_close_auction"
            payload["auction_mode"] = "resting_close_auction_after_14_30_decision"
            payload["holdout_excluded"] = True
            payload["paper_live_parity_required"] = True
        if args.strategy == "portfolio_synergy":
            payload["strategy_core_version"] = "portfolio_synergy_source_artifact_replay_v1"
            payload["num_phases"] = args.num_phases or 7
            payload["live_parity_fill_timing"] = "source_strategy_fill_contracts_preserved_completed_trade_replay"
            payload["auction_mode"] = "kalcb_next_5m_open_plus_olr_resting_close_auction"
            payload["holdout_excluded"] = True
            payload["max_workers_capped_at"] = 2
        print(json.dumps(payload, indent=2))
        return 0
    round_num, round_dir = manager.resolve_round(args.round, for_write=True, expected_phases=args.num_phases)
    plugin = create_plugin(args.strategy, config, output_dir=round_dir, max_workers=args.max_workers, capability_level=config.get("capability_level", "synthetic"))
    if args.num_phases is not None:
        plugin.num_phases = int(args.num_phases)
    runner = PhaseRunner(plugin, round_dir, round_name=args.round_name, round_manager=manager, round_num=round_num)
    state = runner.run_all_phases(start_phase=args.start_phase)
    print(json.dumps({"strategy": args.strategy, "round": round_num, "completed_phases": state.completed_phases, "round_dir": str(round_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
