from __future__ import annotations

import json
from pathlib import Path

from backtests.auto.oos_ablation import (
    apply_window_config,
    build_ablation_candidates,
    build_targeted_oos_candidates,
    clean_metric_row,
    load_round_chain,
    resolve_windows,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_round(
    output_root: Path,
    strategy: str,
    round_num: int,
    mutations: dict,
    *,
    phase_results: dict[int, dict] | None = None,
    run_summary: dict | None = None,
) -> Path:
    round_dir = output_root / strategy / f"round_{round_num}"
    _write_json(round_dir / "optimized_config.json", {"strategy": strategy, "mutations": mutations})
    summary = {"strategy": strategy, "round": round_num, "cumulative_mutations": mutations}
    summary.update(run_summary or {})
    _write_json(round_dir / "run_summary.json", summary)
    if phase_results:
        _write_json(
            round_dir / "phase_state.json",
            {
                "round_name": f"round_{round_num}",
                "current_phase": max(phase_results),
                "phase_results": {str(phase): {"new_mutations": values} for phase, values in phase_results.items()},
            },
        )
    return round_dir


def test_round_chain_and_windows_resolve_for_generic_strategy(tmp_path: Path):
    output_root = tmp_path / "output"
    _write_round(output_root, "generic_alpha", 1, {"entry_threshold": 0.7})
    _write_round(
        output_root,
        "generic_alpha",
        2,
        {"entry_threshold": 0.8},
        run_summary={"train_start": "2025-01-01", "train_end": "2025-01-31"},
    )

    artifacts = load_round_chain("generic_alpha", output_root, 2)
    config = {"date_range": {"start": "2025-01-01", "end": "2025-01-31"}, "baseline": {"holdout_start": "2025-02-01", "holdout_end": "2025-02-14"}}
    windows = resolve_windows(config, artifacts[-1])

    assert [artifact.round_num for artifact in artifacts] == [1, 2]
    assert windows.train_start == "2025-01-01"
    assert windows.train_end == "2025-01-31"
    assert windows.oos_start == "2025-02-01"
    assert windows.oos_end == "2025-02-14"
    assert apply_window_config(config, windows, "oos")["date_range"] == {"start": "2025-02-01", "end": "2025-02-14"}


def test_windows_can_derive_holdout_end_from_weeks(tmp_path: Path):
    output_root = tmp_path / "output"
    _write_round(output_root, "olr", 1, {"lookback": 20})

    windows = resolve_windows(
        {"start": "2025-01-01", "end": "2025-01-31", "holdout_start": "2025-02-01", "holdout_weeks": 2},
        load_round_chain("olr", output_root, 1)[-1],
    )

    assert windows.oos_end == "2025-02-14"
    assert apply_window_config({"start": "2025-01-01", "end": "2025-01-31"}, windows, "train")["start"] == "2025-01-01"


def test_candidate_builder_uses_all_rounds_phase_ablations_and_perturbations(tmp_path: Path):
    output_root = tmp_path / "output"
    _write_round(output_root, "generic_beta", 1, {"alpha": 1, "shared_filter": True})
    _write_round(
        output_root,
        "generic_beta",
        2,
        {"alpha": 2, "shared_filter": True, "beta": 0.25, "flag": False},
        phase_results={
            1: {"alpha": 2},
            2: {"beta": 0.25},
            3: {"flag": False},
        },
    )

    candidates = build_ablation_candidates("generic_beta", load_round_chain("generic_beta", output_root, 2), include_perturbations=True)
    by_label = {candidate.label: candidate for candidate in candidates}

    assert by_label["round_1_final"].mutations == {"alpha": 1, "shared_filter": True}
    assert by_label["round_2_final"].mutations == {"alpha": 2, "shared_filter": True, "beta": 0.25, "flag": False}
    assert by_label["round_2_phase_1_cumulative"].mutations == {"alpha": 2, "shared_filter": True}
    assert "beta" not in by_label["final_drop_round_2_phase_2"].mutations
    assert any(candidate.mutations == {"alpha": 2, "shared_filter": True, "beta": 0.25} for candidate in candidates)
    assert any(candidate.kind == "key_perturbation" and candidate.label.startswith("perturb_beta_") for candidate in candidates)


def test_candidate_manifest_can_merge_or_replace_target_mutations(tmp_path: Path):
    output_root = tmp_path / "output"
    _write_round(output_root, "generic_gamma", 1, {"alpha": 2, "beta": 0.25})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "candidates": [
                {"label": "tighten_beta", "kind": "manual", "mutations": {"beta": 0.1}},
                {"label": "standalone", "kind": "manual", "merge_with_base": False, "mutations": {"gamma": 3}},
            ]
        },
    )

    candidates = build_ablation_candidates(
        "generic_gamma",
        load_round_chain("generic_gamma", output_root, 1),
        include_perturbations=False,
        manifest_path=manifest,
    )
    by_label = {candidate.label: candidate for candidate in candidates}

    assert by_label["tighten_beta"].mutations == {"alpha": 2, "beta": 0.1}
    assert by_label["standalone"].mutations == {"gamma": 3}


def test_targeted_candidates_respond_to_oos_weakness(tmp_path: Path):
    output_root = tmp_path / "output"
    _write_round(
        output_root,
        "kalcb",
        1,
        {
            "entry_min_score": 0.6,
            "entry_require_initial_active": False,
            "risk_per_trade_pct": 0.02,
            "exit_failed_followthrough_bars": 4,
        },
        run_summary={
            "execution_contract": {
                "candidate_source": {"path": "tmp/candidates.json", "section": "top_portfolio_proxy", "rank": 0}
            }
        },
    )
    target = load_round_chain("kalcb", output_root, 1)[-1]

    candidates = build_targeted_oos_candidates(
        "kalcb",
        target.mutations,
        [target],
        target,
        {"metrics": {"broker_net_return_pct": -0.08, "trade_count": 9, "win_rate": 0.22, "broker_max_drawdown_pct": 0.12}},
        {"worst_3_loss_share_of_all_losses": 0.5},
    )
    reasons = {candidate.reason for candidate in candidates}

    assert "negative_or_drawdown_heavy_oos_risk_reduction" in reasons
    assert "low_oos_win_rate_entry_tightening" in reasons
    assert "low_oos_trade_frequency_entry_loosening" in reasons
    assert "oos_loss_concentration_exit_control" in reasons
    assert all("_kalcb.source.path" in candidate.mutations for candidate in candidates)


def test_clean_metric_row_preserves_promotion_contract_fields():
    row = clean_metric_row(
        {
            "official_mtm_net_return_pct": 0.125,
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_value": 0.125,
            "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
            "promotion_requires_audit_pass": True,
            "official_replay_pass": True,
            "audit_pass": True,
            "audit_status": "direct_shared_core_replay_holdout",
        }
    )

    assert row["primary_promotion_metric"] == "official_mtm_net_return_pct"
    assert row["primary_promotion_value"] == 0.125
    assert row["promotion_requires_audit_pass"] is True
    assert row["audit_status"] == "direct_shared_core_replay_holdout"
