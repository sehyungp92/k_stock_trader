from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from backtests.auto.shared.cache_keys import stable_signature
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.strategies.registry import create_plugin, get_backtest_runner


METRIC_KEYS = (
    "broker_net_return_pct",
    "official_mtm_net_return_pct",
    "net_return_pct",
    "primary_objective_net_return_pct",
    "official_metric_basis",
    "primary_promotion_metric",
    "primary_promotion_value",
    "primary_promotion_basis",
    "promotion_requires_audit_pass",
    "official_replay_pass",
    "audit_pass",
    "audit_status",
    "trade_count",
    "total_trades",
    "trades",
    "broker_trade_count",
    "active_days",
    "avg_trade_net_pct",
    "net_win_share",
    "win_rate",
    "profit_factor",
    "broker_max_drawdown_pct",
    "max_drawdown_pct",
    "avg_mfe_capture",
    "mfe_capture",
    "mae_le_neg_1_share",
    "avg_mfe_r",
    "avg_mae_r",
    "target_hit_share",
    "exit_reason_eod_flatten_share",
    "same_bar_fill_count",
    "forced_replay_close_count",
    "rejected_order_count",
    "end_open_position_count",
    "worst_fold_net",
    "median_fold_net",
)

SOURCE_PATH_MUTATION = "_kalcb.source.path"
SOURCE_SECTION_MUTATION = "_kalcb.source.section"
SOURCE_RANK_MUTATION = "_kalcb.source.rank"


@dataclass(slots=True)
class RoundArtifact:
    round_num: int
    path: Path
    optimized: dict[str, Any] = field(default_factory=dict)
    run_summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    phase_state: dict[str, Any] = field(default_factory=dict)
    round_final_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def mutations(self) -> dict[str, Any]:
        mutations = dict(self.optimized.get("mutations") or self.run_summary.get("cumulative_mutations") or {})
        return _enrich_strategy_mutations(str(self.run_summary.get("strategy") or self.optimized.get("strategy") or ""), mutations, self)


@dataclass(slots=True)
class AblationCandidate:
    label: str
    kind: str
    mutations: dict[str, Any]
    reason: str = ""


@dataclass(slots=True)
class WindowSpec:
    train_start: str
    train_end: str
    oos_start: str
    oos_end: str


class StrategyAblationAdapter(Protocol):
    strategy: str

    def evaluate(self, label: str, mutations: dict[str, Any], window: str) -> dict[str, Any]: ...


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_round_artifact(round_dir: Path) -> RoundArtifact:
    round_num = _round_num_from_path(round_dir)
    return RoundArtifact(
        round_num=round_num,
        path=round_dir,
        optimized=read_json(round_dir / "optimized_config.json"),
        run_summary=read_json(round_dir / "run_summary.json"),
        diagnostics=read_json(round_dir / "diagnostics_summary.json"),
        phase_state=read_json(round_dir / "phase_state.json"),
        round_final_diagnostics=read_json(round_dir / "round_final_diagnostics.json"),
    )


def load_round_chain(strategy: str, output_root: Path, target_round: int | None, round_dir: Path | None = None) -> list[RoundArtifact]:
    if round_dir is not None:
        target = load_round_artifact(round_dir)
        strategy_root = round_dir.parent
        target_round = target.round_num
    else:
        strategy_root = output_root / strategy
        if target_round is None:
            rounds = sorted(
                (_round_num_from_path(path), path)
                for path in strategy_root.glob("round_*")
                if path.is_dir() and _round_num_from_path(path) > 0
            )
            if not rounds:
                raise FileNotFoundError(f"No round_* directories found under {strategy_root}")
            target_round = rounds[-1][0]
    artifacts: list[RoundArtifact] = []
    round_paths = [path for path in strategy_root.glob("round_*") if path.is_dir() and _round_num_from_path(path) > 0]
    for path in sorted(round_paths, key=_round_num_from_path):
        round_num = _round_num_from_path(path)
        if round_num <= int(target_round):
            artifacts.append(load_round_artifact(path))
    if not artifacts:
        raise FileNotFoundError(f"No round artifacts found for {strategy} <= round {target_round}")
    return artifacts


def resolve_windows(config: dict[str, Any], target: RoundArtifact, *, oos_start: str | None = None, oos_end: str | None = None) -> WindowSpec:
    date_range = dict(config.get("date_range") or {})
    baseline = dict(config.get("baseline") or {})
    run = dict(target.run_summary or {})
    train_start = str(config.get("start") or date_range.get("start") or run.get("train_start") or run.get("training_window_start") or "")
    train_end = str(config.get("end") or date_range.get("end") or run.get("train_end") or run.get("training_window_end") or "")
    resolved_oos_start = str(
        oos_start
        or baseline.get("holdout_start")
        or config.get("holdout_start")
        or run.get("holdout_start")
        or ""
    )
    resolved_oos_end = str(
        oos_end
        or baseline.get("holdout_end")
        or config.get("holdout_end")
        or run.get("holdout_end")
        or baseline.get("data_latest_available")
        or config.get("data_latest_available")
        or ""
    )
    if not resolved_oos_end and resolved_oos_start and (config.get("holdout_weeks") or run.get("holdout_weeks")):
        start_day = _parse_date(resolved_oos_start)
        weeks = int(config.get("holdout_weeks") or run.get("holdout_weeks") or 0)
        if start_day and weeks > 0:
            resolved_oos_end = (start_day + timedelta(weeks=weeks) - timedelta(days=1)).isoformat()
    if not all((train_start, train_end, resolved_oos_start, resolved_oos_end)):
        raise ValueError(
            "Could not resolve train/OOS windows. Provide --oos-start/--oos-end, or configure start/end/date_range and holdout_start/holdout_end."
        )
    return WindowSpec(train_start=train_start, train_end=train_end, oos_start=resolved_oos_start, oos_end=resolved_oos_end)


def apply_window_config(config: dict[str, Any], windows: WindowSpec, window: str) -> dict[str, Any]:
    out = deepcopy(config)
    if window == "oos":
        start, end = windows.oos_start, windows.oos_end
    elif window == "train":
        start, end = windows.train_start, windows.train_end
    else:
        raise ValueError(f"Unknown window: {window}")
    out["use_full_available_window"] = False
    if "date_range" in out or "date_range" in config:
        out["date_range"] = dict(out.get("date_range") or {})
        out["date_range"]["start"] = _date_text(start)
        out["date_range"]["end"] = _date_text(end)
    else:
        out["start"] = start
        out["end"] = end
    return out


class BacktestRunnerAblationAdapter:
    def __init__(self, strategy: str, config: dict[str, Any], windows: WindowSpec):
        self.strategy = strategy
        self.config = dict(config)
        self.windows = windows
        self.runner = get_backtest_runner(strategy)

    def evaluate(self, label: str, mutations: dict[str, Any], window: str) -> dict[str, Any]:
        started = time.monotonic()
        config = apply_window_config(self.config, self.windows, window)
        result = self.runner(config, _strip_internal_mutations(mutations))
        metrics = dict(getattr(result, "metrics", {}) or {})
        trades = tuple(getattr(result, "trades", ()) or ())
        decisions = tuple(getattr(result, "decisions", ()) or ())
        metrics.setdefault("trade_count", _metric_trade_count(metrics, trades))
        metrics.setdefault("trades", metrics.get("trade_count", 0.0))
        metrics.setdefault("win_rate", _metric_win_rate(metrics, trades))
        metrics["window"] = window
        metrics["elapsed_seconds"] = round(time.monotonic() - started, 3)
        source = {
            "source_fingerprint": getattr(result, "source_fingerprint", ""),
            "candidate_snapshot_hash": getattr(result, "candidate_snapshot_hash", ""),
            "feature_bundle_hash": getattr(result, "feature_bundle_hash", ""),
            "capability_level": getattr(result, "capability_level", ""),
        }
        return {
            "label": label,
            "window": window,
            "mutations": dict(mutations),
            "metrics": metrics,
            "metric_row": clean_metric_row(metrics),
            "trade_rows": generic_trade_rows(trades),
            "decision_summary": generic_decision_summary(decisions),
            "source": source,
            "elapsed_seconds": metrics["elapsed_seconds"],
        }


class KALCBFixedTradePlanAblationAdapter:
    def __init__(self, config: dict[str, Any], windows: WindowSpec, target: RoundArtifact, output_dir: Path):
        self.strategy = "kalcb"
        self.config = dict(config)
        self.windows = windows
        self.target = target
        self.output_dir = Path(output_dir)
        self.default_source = _kalcb_source_from_artifact(target)
        self.plugins: dict[str, Any] = {}

    def _plugin(self, window: str) -> Any:
        from backtests.strategies.kalcb.fixed_trade_plan_phase import KALCBFixedTradePlanOptimizationPlugin

        cached = self.plugins.get(window)
        if cached is not None:
            return cached
        cfg = apply_window_config(self.config, self.windows, window)
        cfg["fixed_candidate_source"] = {
            "path": self.default_source[SOURCE_PATH_MUTATION],
            "section": self.default_source[SOURCE_SECTION_MUTATION],
            "rank": self.default_source[SOURCE_RANK_MUTATION],
        }
        cfg["skip_initial_baseline_eval"] = True
        plugin = KALCBFixedTradePlanOptimizationPlugin(
            cfg,
            output_dir=self.output_dir / f"_kalcb_fixed_{window}",
            max_workers=1,
            capability_level=str(cfg.get("capability_level") or "real_replay"),
        )
        self.plugins[window] = plugin
        return plugin

    @staticmethod
    def _window_pool_mutations(mutations: dict[str, Any], window: str) -> dict[str, Any]:
        from backtests.strategies.kalcb.fixed_trade_plan_phase import POOL_SOURCE_PATH_MUTATION

        out = dict(mutations or {})
        if window != "oos":
            return out
        raw_path = str(out.get(POOL_SOURCE_PATH_MUTATION) or "").strip()
        if not raw_path:
            return out
        path = Path(raw_path)
        if path.name != "train_guarded_prefilter_pool_rows.jsonl":
            return out
        holdout_path = path.with_name("holdout_guarded_prefilter_pool_rows.jsonl")
        if holdout_path.exists():
            out[POOL_SOURCE_PATH_MUTATION] = str(holdout_path)
        return out

    def evaluate(self, label: str, mutations: dict[str, Any], window: str) -> dict[str, Any]:
        from backtests.strategies.common.plugin_base import attach_official_metric_contract, build_execution_contract
        from backtests.strategies.kalcb.fixed_trade_plan_phase import OFFICIAL_PROMOTION_METRIC, POOL_SOURCE_PATH_MUTATION

        plugin = self._plugin(window)
        eval_mutations = self._window_pool_mutations(mutations, window)
        evaluation = plugin._evaluate(eval_mutations)
        metrics = dict(evaluation.metrics)
        metrics["window"] = window
        attach_official_metric_contract(
            metrics,
            primary_metric=OFFICIAL_PROMOTION_METRIC,
            requires_audit_pass=True,
            audit_pass=_audit_hygiene_pass(metrics),
            audit_status=f"direct_shared_core_replay_{'holdout' if window == 'oos' else 'train'}",
            official_replay_pass=True,
            execution_contract=build_execution_contract(plugin, metrics, extra={"window": window}),
        )
        source = _kalcb_source_from_mutations(eval_mutations, self.default_source)
        if eval_mutations.get(POOL_SOURCE_PATH_MUTATION):
            source["pool_source_path"] = str(eval_mutations.get(POOL_SOURCE_PATH_MUTATION))
            source["pool_source_mode"] = "guarded_prefilter_pool"
        return {
            "label": label,
            "window": window,
            "source": source,
            "mutations": dict(mutations),
            "metrics": metrics,
            "metric_row": clean_metric_row(metrics),
            "fold_rows": evaluation.fold_rows,
            "trade_rows": tuple(evaluation.trade_rows),
            "decision_summary": evaluation.decision_summary,
            "replay_digest": evaluation.replay_digest,
            "elapsed_seconds": metrics.get("elapsed_seconds", 0.0),
        }


def create_ablation_adapter(
    strategy: str,
    config: dict[str, Any],
    windows: WindowSpec,
    target: RoundArtifact,
    output_dir: Path,
    adapter: str = "auto",
) -> StrategyAblationAdapter:
    key = strategy.lower()
    chosen = adapter.lower()
    if chosen == "auto":
        chosen = "kalcb-fixed" if key == "kalcb" and _kalcb_source_from_artifact(target, required=False) else "runner"
    if chosen in {"runner", "generic"}:
        return BacktestRunnerAblationAdapter(key, config, windows)
    if chosen in {"kalcb-fixed", "kalcb_fixed", "fixed"}:
        if key != "kalcb":
            raise ValueError("The kalcb-fixed adapter can only be used with --strategy kalcb")
        return KALCBFixedTradePlanAblationAdapter(config, windows, target, output_dir)
    raise ValueError(f"Unsupported OOS ablation adapter: {adapter}")


def build_ablation_candidates(
    strategy: str,
    artifacts: list[RoundArtifact],
    *,
    include_perturbations: bool = True,
    include_phase_candidates: bool = False,
    config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> list[AblationCandidate]:
    if not artifacts:
        raise ValueError("No round artifacts supplied")
    target = artifacts[-1]
    candidates: list[AblationCandidate] = []
    final = target.mutations

    for artifact in artifacts:
        candidates.append(
            AblationCandidate(
                label=f"round_{artifact.round_num}_final",
                kind="round_final",
                mutations=artifact.mutations,
                reason=f"Final cumulative mutations from round {artifact.round_num}",
            )
        )

    for artifact in artifacts:
        base = _previous_round_mutations(artifacts, artifact.round_num)
        timeline = _phase_timeline(strategy, artifact, base)
        for phase, muts, cumulative in timeline:
            candidates.append(
                AblationCandidate(
                    label=f"round_{artifact.round_num}_phase_{phase}_cumulative",
                    kind="phase_cumulative",
                    mutations=cumulative,
                    reason=f"Cumulative mutations through round {artifact.round_num} phase {phase}",
                )
            )

    target_base = _previous_round_mutations(artifacts, target.round_num)
    target_timeline = _phase_timeline(strategy, target, target_base)
    if target_timeline:
        phase_baselines: dict[int, dict[str, Any]] = {}
        cumulative = dict(target_base)
        for phase, muts, _ in target_timeline:
            phase_baselines[phase] = dict(cumulative)
            cumulative.update(muts)
        for phase, muts, _ in target_timeline:
            reverted = dict(final)
            before = phase_baselines.get(phase, {})
            for key in muts:
                if key in before:
                    reverted[key] = before[key]
                else:
                    reverted.pop(key, None)
            candidates.append(
                AblationCandidate(
                    label=f"final_drop_round_{target.round_num}_phase_{phase}",
                    kind="phase_ablation",
                    mutations=_enrich_strategy_mutations(strategy, reverted, target),
                    reason=f"Final mutations with round {target.round_num} phase {phase} accepted mutations reverted",
                )
            )

    baseline_for_revert = _previous_round_mutations(artifacts, target.round_num)
    for key in sorted(final):
        if key.startswith("_") and not key.startswith("_kalcb.source."):
            continue
        if key not in baseline_for_revert or baseline_for_revert.get(key) != final.get(key):
            reverted = dict(final)
            if key in baseline_for_revert:
                reverted[key] = baseline_for_revert[key]
                reason = f"Revert {key} to previous-round value"
            else:
                reverted.pop(key, None)
                reason = f"Remove {key} accepted by current round"
            candidates.append(
                AblationCandidate(
                    label=f"final_revert_key_{_safe_label(key)}",
                    kind="key_ablation",
                    mutations=_enrich_strategy_mutations(strategy, reverted, target),
                    reason=reason,
                )
            )

    if include_perturbations:
        candidates.extend(_build_generic_perturbations(strategy, final, artifacts, target))

    if include_phase_candidates and config is not None:
        candidates.extend(_build_phase_spec_candidates(strategy, config, final, output_dir))

    if manifest_path:
        candidates.extend(_load_manifest_candidates(manifest_path, final))

    return _dedupe_candidates(candidates)


def run_oos_ablation(
    *,
    strategy: str,
    config: dict[str, Any],
    artifacts: list[RoundArtifact],
    output_dir: Path,
    adapter_name: str = "auto",
    max_oos: int = 0,
    top_train: int = 40,
    include_perturbations: bool = True,
    include_phase_candidates: bool = False,
    include_targeted: bool = True,
    max_targeted: int = 80,
    candidate_manifest: Path | None = None,
    oos_start: str | None = None,
    oos_end: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = artifacts[-1]
    windows = resolve_windows(config, target, oos_start=oos_start, oos_end=oos_end)
    adapter = create_ablation_adapter(strategy, config, windows, target, output_dir, adapter=adapter_name)
    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    def status(stage: str, **extra: Any) -> None:
        payload = {"ts": utc_now(), "stage": stage, **extra}
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)
        append_jsonl(progress_path, payload)

    candidates = build_ablation_candidates(
        strategy,
        artifacts,
        include_perturbations=include_perturbations,
        include_phase_candidates=include_phase_candidates,
        config=config,
        output_dir=output_dir,
        manifest_path=candidate_manifest,
    )
    if max_oos > 0:
        mandatory = {f"round_{artifact.round_num}_final" for artifact in artifacts}
        mandatory.add(f"round_{target.round_num}_final")
        kept = {item.label: item for item in candidates[:max_oos]}
        for item in candidates:
            if item.label in mandatory or item.kind in {"phase_ablation", "key_ablation"}:
                kept[item.label] = item
        candidates = list(kept.values())

    status("candidate_plan", total=len(candidates), counts=dict(Counter(item.kind for item in candidates)))
    results_by_label: dict[str, dict[str, Any]] = {}
    oos_results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        status("evaluate_oos_start", index=index, total=len(candidates), label=candidate.label, kind=candidate.kind)
        try:
            result = adapter.evaluate(candidate.label, candidate.mutations, "oos")
            result["kind"] = candidate.kind
            result["reason"] = candidate.reason
            result["oos_repair_score"] = oos_repair_score(result["metrics"])
            oos_results.append(result)
            results_by_label.setdefault(candidate.label, {})["oos"] = result
            status(
                "evaluate_oos_done",
                index=index,
                label=candidate.label,
                net_return_pct=metric_net(result["metrics"]),
                trades=metric_trades(result["metrics"]),
                win_rate=metric_win(result["metrics"]),
                drawdown=metric_drawdown(result["metrics"]),
                elapsed=result.get("elapsed_seconds", 0.0),
            )
        except Exception as exc:
            status("evaluate_oos_error", index=index, label=candidate.label, error=repr(exc))

    target_label = f"round_{target.round_num}_final"
    if include_targeted and target_label in results_by_label and "oos" in results_by_label[target_label]:
        edge = edge_case_diagnostics(results_by_label[target_label]["oos"])
        targeted_candidates = build_targeted_oos_candidates(
            strategy,
            target.mutations,
            artifacts,
            target,
            results_by_label[target_label]["oos"],
            edge,
            max_candidates=max_targeted,
        )
        seen = {stable_signature(candidate.mutations) for candidate in candidates}
        new_targeted: list[AblationCandidate] = []
        for candidate in targeted_candidates:
            signature = stable_signature(candidate.mutations)
            if signature in seen:
                continue
            seen.add(signature)
            new_targeted.append(candidate)
        candidates.extend(new_targeted)
        status("targeted_candidate_plan", total=len(new_targeted), counts=dict(Counter(item.reason for item in new_targeted)))
        for index, candidate in enumerate(new_targeted, start=1):
            status("evaluate_oos_start", index=index, total=len(new_targeted), label=candidate.label, kind=candidate.kind)
            try:
                result = adapter.evaluate(candidate.label, candidate.mutations, "oos")
                result["kind"] = candidate.kind
                result["reason"] = candidate.reason
                result["oos_repair_score"] = oos_repair_score(result["metrics"])
                oos_results.append(result)
                results_by_label.setdefault(candidate.label, {})["oos"] = result
                status(
                    "evaluate_oos_done",
                    index=index,
                    label=candidate.label,
                    net_return_pct=metric_net(result["metrics"]),
                    trades=metric_trades(result["metrics"]),
                    win_rate=metric_win(result["metrics"]),
                    drawdown=metric_drawdown(result["metrics"]),
                    elapsed=result.get("elapsed_seconds", 0.0),
                )
            except Exception as exc:
                status("evaluate_oos_error", index=index, label=candidate.label, error=repr(exc))

    mandatory_train = {f"round_{artifact.round_num}_final" for artifact in artifacts}
    mandatory_train.update(row["label"] for row in oos_results if row.get("kind") in {"phase_ablation", "key_ablation", "targeted_oos_repair"})
    top_oos = sorted(oos_results, key=lambda row: row["oos_repair_score"], reverse=True)
    train_labels = set(row["label"] for row in top_oos[: max(0, top_train)])
    train_labels.update(mandatory_train)
    by_label = {item.label: item for item in candidates}
    train_candidates = [by_label[label] for label in sorted(train_labels) if label in by_label]
    status("train_confirm_plan", total=len(train_candidates))

    train_results: list[dict[str, Any]] = []
    for index, candidate in enumerate(train_candidates, start=1):
        status("evaluate_train_start", index=index, total=len(train_candidates), label=candidate.label, kind=candidate.kind)
        try:
            result = adapter.evaluate(candidate.label, candidate.mutations, "train")
            result["kind"] = candidate.kind
            result["reason"] = candidate.reason
            train_results.append(result)
            results_by_label.setdefault(candidate.label, {})["train"] = result
            status(
                "evaluate_train_done",
                index=index,
                label=candidate.label,
                net_return_pct=metric_net(result["metrics"]),
                trades=metric_trades(result["metrics"]),
                win_rate=metric_win(result["metrics"]),
                drawdown=metric_drawdown(result["metrics"]),
                elapsed=result.get("elapsed_seconds", 0.0),
            )
        except Exception as exc:
            status("evaluate_train_error", index=index, label=candidate.label, error=repr(exc))

    if target_label not in results_by_label or "oos" not in results_by_label[target_label]:
        raise RuntimeError(f"{target_label} OOS evaluation did not complete")
    if "train" not in results_by_label[target_label]:
        target_candidate = by_label[target_label]
        results_by_label[target_label]["train"] = adapter.evaluate(target_candidate.label, target_candidate.mutations, "train")

    final_train = results_by_label[target_label]["train"]["metrics"]
    confirmed = []
    for label, windows_result in results_by_label.items():
        if "oos" not in windows_result or "train" not in windows_result:
            continue
        confirmed.append(
            {
                "label": label,
                "kind": windows_result["oos"].get("kind", ""),
                "reason": windows_result["oos"].get("reason", ""),
                "combined_score": combined_score(windows_result["oos"]["metrics"], windows_result["train"]["metrics"], final_train),
                "oos": compact_result(windows_result["oos"]),
                "train": compact_result(windows_result["train"]),
            }
        )
    confirmed.sort(key=lambda row: row["combined_score"], reverse=True)

    ablation_impacts = {
        label: impact_vs(results_by_label, label, target_label)
        for label, row in results_by_label.items()
        if row.get("oos", {}).get("kind") in {"phase_ablation", "key_ablation", "phase_cumulative", "round_final"}
        and "oos" in row
        and "train" in row
    }
    edge = edge_case_diagnostics(results_by_label[target_label]["oos"])
    payload = {
        "generated_at_utc": utc_now(),
        "strategy": strategy,
        "adapter": adapter_name,
        "target_round": target.round_num,
        "round_dir": str(target.path),
        "windows": {
            "train": {"start": windows.train_start, "end": windows.train_end},
            "oos": {"start": windows.oos_start, "end": windows.oos_end},
        },
        "counts": {
            "candidate_count": len(candidates),
            "oos_evaluated": len(oos_results),
            "train_confirmed": len(train_results),
        },
        "oos_ranked": [compact_result(row) | {"score": row.get("oos_repair_score", 0.0)} for row in top_oos],
        "confirmed_train_ranked": confirmed,
        "ablation_impacts": ablation_impacts,
        "edge_case_diagnostics": edge,
        "results_by_label": {
            label: {window: compact_result(result) for window, result in windows_result.items()}
            for label, windows_result in results_by_label.items()
        },
        "top_recommendation": confirmed[0] if confirmed else {},
    }
    write_json(output_dir / "oos_ablation_results.json", payload)
    write_json(output_dir / "oos_trade_diagnostics.json", edge)
    (output_dir / "oos_ablation_summary.md").write_text(render_markdown(payload, target_label), encoding="utf-8")
    if confirmed:
        write_json(output_dir / "recommended_mutations.json", confirmed[0])
    status("complete", result_path=str(output_dir / "oos_ablation_results.json"), summary_path=str(output_dir / "oos_ablation_summary.md"))
    return payload


def clean_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def _audit_hygiene_pass(metrics: dict[str, Any]) -> bool:
    return all(
        _num(metrics.get(key)) == 0.0
        for key in (
            "same_bar_fill_count",
            "forced_replay_close_count",
            "rejected_order_count",
            "end_open_position_count",
        )
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": result.get("label", ""),
        "kind": result.get("kind", ""),
        "reason": result.get("reason", ""),
        "metrics": clean_metric_row(dict(result.get("metrics") or {})),
        "source": result.get("source", {}),
        "mutations": result.get("mutations", {}),
        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
    }


def oos_repair_score(metrics: dict[str, Any]) -> float:
    net = metric_net(metrics)
    trades = metric_trades(metrics)
    avg_trade = _num(metrics.get("avg_trade_net_pct"))
    win = metric_win(metrics)
    dd = abs(metric_drawdown(metrics))
    hygiene = _num(metrics.get("same_bar_fill_count")) + _num(metrics.get("end_open_position_count"))
    return (
        1000.0 * net
        + 160.0 * math.tanh(avg_trade / 0.008)
        + 55.0 * math.tanh((trades - 12.0) / 12.0)
        + 50.0 * (win - 0.40)
        - 650.0 * dd
        - 80.0 * hygiene
    )


def combined_score(oos_metrics: dict[str, Any], train_metrics: dict[str, Any], final_train: dict[str, Any]) -> float:
    score = oos_repair_score(oos_metrics)
    train_net = metric_net(train_metrics)
    final_train_net = max(abs(metric_net(final_train)), 0.01)
    train_ratio = train_net / final_train_net
    if train_ratio < 0.75:
        score -= 150.0 * (0.75 - train_ratio)
    if metric_trades(train_metrics) < max(10.0, 0.55 * metric_trades(final_train)):
        score -= max(0.0, 0.55 * metric_trades(final_train) - metric_trades(train_metrics))
    if abs(metric_drawdown(train_metrics)) > max(abs(metric_drawdown(final_train)) * 1.25, 0.08):
        score -= 600.0 * (abs(metric_drawdown(train_metrics)) - max(abs(metric_drawdown(final_train)) * 1.25, 0.08))
    score += 160.0 * math.tanh(train_net / 0.50)
    return score


def impact_vs(results_by_label: dict[str, dict[str, Any]], label: str, baseline_label: str) -> dict[str, Any]:
    row = results_by_label[label]
    base = results_by_label[baseline_label]
    out: dict[str, Any] = {}
    for window in ("oos", "train"):
        if window not in row or window not in base:
            continue
        rm = row[window]["metrics"]
        bm = base[window]["metrics"]
        out[window] = {
            "net_delta_pct_points": 100.0 * (metric_net(rm) - metric_net(bm)),
            "trade_delta": metric_trades(rm) - metric_trades(bm),
            "win_rate_delta_pct_points": 100.0 * (metric_win(rm) - metric_win(bm)),
            "dd_delta_pct_points": 100.0 * (metric_drawdown(rm) - metric_drawdown(bm)),
        }
    return out


def edge_case_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    rows = list(result.get("trade_rows") or [])
    metrics = dict(result.get("metrics") or {})
    initial_equity = _infer_initial_equity(rows, metrics)
    total_pnl = sum(_num(row.get("net_pnl")) for row in rows)
    worst = sorted(rows, key=lambda row: _num(row.get("net_pnl")))[:12]
    removals = []
    for k in (1, 2, 3, 5, 8, 10):
        removed = worst[: min(k, len(worst))]
        adjusted_pnl = total_pnl - sum(_num(row.get("net_pnl")) for row in removed)
        remaining = [row for row in rows if row not in removed]
        removals.append(
            {
                "remove_worst_k": k,
                "adjusted_net_return_pct": adjusted_pnl / max(initial_equity, 1.0),
                "adjusted_win_rate": sum(1 for row in remaining if _num(row.get("net_pnl")) > 0) / max(len(remaining), 1),
                "remaining_trades": len(remaining),
            }
        )
    loss_rows = [row for row in rows if _num(row.get("net_pnl")) < 0]
    total_losses = abs(sum(_num(row.get("net_pnl")) for row in loss_rows))
    return {
        "final_oos_metrics": clean_metric_row(metrics),
        "trade_count": len(rows),
        "total_closed_trade_pnl": total_pnl,
        "worst_trades": worst,
        "worst_3_loss_share_of_all_losses": abs(sum(_num(row.get("net_pnl")) for row in worst[:3])) / max(total_losses, 1.0),
        "remove_worst_impacts": removals,
        "by_symbol_worst": group_trade_rows(rows, "symbol"),
        "by_exit_reason_worst": group_trade_rows(rows, "exit_reason"),
        "by_entry_date_worst": group_trade_rows(rows, "entry_date"),
    }


def group_trade_rows(rows: Iterable[dict[str, Any]], key: str, *, top_n: int = 12) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "UNKNOWN")].append(row)
    out = []
    for name, group in groups.items():
        out.append(
            {
                key: name,
                "trades": len(group),
                "net_pnl": sum(_num(row.get("net_pnl")) for row in group),
                "win_rate": sum(1 for row in group if _num(row.get("net_pnl")) > 0) / max(len(group), 1),
                "avg_r": statistics.fmean(_num(row.get("r")) for row in group) if group else 0.0,
            }
        )
    out.sort(key=lambda row: (row["net_pnl"], -row["trades"]))
    return out[:top_n]


def render_markdown(payload: dict[str, Any], target_label: str) -> str:
    final_oos = payload["results_by_label"][target_label]["oos"]["metrics"]
    final_train = payload["results_by_label"][target_label]["train"]["metrics"]
    lines = [
        f"# {payload['strategy'].upper()} OOS Ablation",
        "",
        f"- Target round: {payload['target_round']}",
        f"- Train window: {payload['windows']['train']['start']} to {payload['windows']['train']['end']}",
        f"- OOS window: {payload['windows']['oos']['start']} to {payload['windows']['oos']['end']}",
        f"- Evaluated OOS candidates: {payload['counts']['oos_evaluated']}",
        f"- Train-confirmed candidates: {payload['counts']['train_confirmed']}",
        "",
        "## Target Baseline",
        f"- Train: {_pct(metric_net(final_train))} net, {metric_trades(final_train):.0f} trades, {_pct(metric_win(final_train))} win, {_pct(metric_drawdown(final_train))} max DD.",
        f"- OOS: {_pct(metric_net(final_oos))} net, {metric_trades(final_oos):.0f} trades, {_pct(metric_win(final_oos))} win, {_pct(metric_drawdown(final_oos))} max DD.",
        "",
        "## Top Train-Confirmed Candidates",
    ]
    lines.extend(
        markdown_table(
            [
                {
                    "label": row["label"],
                    "kind": row["kind"],
                    "oos_net": 100.0 * metric_net(row["oos"]["metrics"]),
                    "oos_trades": metric_trades(row["oos"]["metrics"]),
                    "oos_win": 100.0 * metric_win(row["oos"]["metrics"]),
                    "train_net": 100.0 * metric_net(row["train"]["metrics"]),
                    "train_trades": metric_trades(row["train"]["metrics"]),
                    "score": row["combined_score"],
                }
                for row in payload["confirmed_train_ranked"][:25]
            ],
            [("Label", "label"), ("Kind", "kind"), ("OOS Net %", "oos_net"), ("OOS Trades", "oos_trades"), ("OOS Win %", "oos_win"), ("Train Net %", "train_net"), ("Train Trades", "train_trades"), ("Score", "score")],
        )
    )
    lines.extend(["", "## Ablation Impacts"])
    ablations = []
    for label, impact in payload["ablation_impacts"].items():
        oos = impact.get("oos", {})
        train = impact.get("train", {})
        ablations.append(
            {
                "label": label,
                "oos_net_delta": oos.get("net_delta_pct_points", 0.0),
                "oos_trade_delta": oos.get("trade_delta", 0.0),
                "train_net_delta": train.get("net_delta_pct_points", 0.0),
                "train_trade_delta": train.get("trade_delta", 0.0),
            }
        )
    ablations.sort(key=lambda row: row["oos_net_delta"], reverse=True)
    lines.extend(markdown_table(ablations[:30], [("Label", "label"), ("OOS Net Delta pp", "oos_net_delta"), ("OOS Trade Delta", "oos_trade_delta"), ("Train Net Delta pp", "train_net_delta"), ("Train Trade Delta", "train_trade_delta")]))
    edge = payload.get("edge_case_diagnostics") or {}
    lines.extend(["", "## Edge-Case Check"])
    lines.append(f"- Worst 3 trades explain {_pct(edge.get('worst_3_loss_share_of_all_losses'))} of OOS losses.")
    for row in edge.get("remove_worst_impacts", [])[:6]:
        lines.append(f"- Remove worst {row['remove_worst_k']}: adjusted OOS net {_pct(row['adjusted_net_return_pct'])} over {row['remaining_trades']} trades.")
    return "\n".join(lines) + "\n"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    out = ["| " + " | ".join(label for label, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def metric_net(metrics: dict[str, Any]) -> float:
    for key in ("broker_net_return_pct", "official_mtm_net_return_pct", "primary_objective_net_return_pct", "net_return_pct", "return_pct"):
        if metrics.get(key) is not None:
            return _num(metrics.get(key))
    return 0.0


def metric_trades(metrics: dict[str, Any]) -> float:
    for key in ("trade_count", "total_trades", "trades", "broker_trade_count"):
        if metrics.get(key) is not None:
            return _num(metrics.get(key))
    return 0.0


def metric_win(metrics: dict[str, Any]) -> float:
    for key in ("net_win_share", "win_rate"):
        if metrics.get(key) is not None:
            return _num(metrics.get(key))
    return 0.0


def metric_drawdown(metrics: dict[str, Any]) -> float:
    for key in ("broker_max_drawdown_pct", "max_drawdown_pct", "portfolio_equivalent_max_drawdown_pct"):
        if metrics.get(key) is not None:
            return abs(_num(metrics.get(key)))
    return 0.0


def generic_trade_rows(trades: Iterable[Any]) -> tuple[dict[str, Any], ...]:
    rows = []
    for trade in trades:
        entry_time = getattr(trade, "entry_fill_time", None) or getattr(trade, "entry_time", None)
        exit_time = getattr(trade, "exit_fill_time", None) or getattr(trade, "exit_time", None)
        net_pnl = _num(getattr(trade, "net_pnl", 0.0))
        qty = max(int(getattr(trade, "qty", 0) or 0), 1)
        entry_price = _num(getattr(trade, "entry_price", 0.0))
        exit_price = _num(getattr(trade, "exit_price", entry_price))
        notional = max(entry_price * qty, 1e-9)
        rows.append(
            {
                "symbol": str(getattr(trade, "symbol", "")),
                "qty": qty,
                "entry_time": entry_time.isoformat() if hasattr(entry_time, "isoformat") else "",
                "entry_date": entry_time.date().isoformat() if hasattr(entry_time, "date") else "",
                "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else "",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": _num(getattr(trade, "gross_pnl", net_pnl)),
                "net_pnl": net_pnl,
                "net_return_pct": net_pnl / notional,
                "r": _num(getattr(trade, "r", 0.0)) or _trade_r(trade),
                "mfe": _num(getattr(trade, "mfe", 0.0)),
                "mae": _num(getattr(trade, "mae", 0.0)),
                "exit_reason": str(getattr(trade, "exit_reason", "") or "unknown"),
            }
        )
    return tuple(rows)


def generic_decision_summary(decisions: Iterable[Any]) -> dict[str, Any]:
    codes = Counter(str(getattr(decision, "decision_code", "") or "") for decision in decisions)
    reasons = Counter(str(getattr(decision, "reason", "") or "") for decision in decisions if getattr(decision, "reason", ""))
    return {"decision_code_counts": codes.most_common(), "reason_counts": reasons.most_common(20)}


def _build_generic_perturbations(strategy: str, final: dict[str, Any], artifacts: list[RoundArtifact], target: RoundArtifact) -> list[AblationCandidate]:
    previous_values: dict[str, set[Any]] = defaultdict(set)
    for artifact in artifacts:
        for key, value in artifact.mutations.items():
            try:
                hash(value)
            except TypeError:
                continue
            previous_values[key].add(value)
    candidates: list[AblationCandidate] = []
    for key, value in sorted(final.items()):
        if key.startswith("_") and not key.startswith("_kalcb.source."):
            continue
        variants: list[Any] = []
        if isinstance(value, bool):
            variants.append(not value)
        elif isinstance(value, int) and not isinstance(value, bool):
            variants.extend(sorted({0, value - 1, value + 1, int(round(value * 0.75)), int(round(value * 1.25))}))
        elif isinstance(value, float):
            variants.extend(sorted({0.0, value * 0.5, value * 0.75, value * 1.25, value * 1.5}))
        elif isinstance(value, str):
            variants.extend(sorted(str(item) for item in previous_values.get(key, set()) if str(item) != value))
        for variant in variants:
            if variant == value:
                continue
            mutated = dict(final)
            mutated[key] = variant
            candidates.append(
                AblationCandidate(
                    label=f"perturb_{_safe_label(key)}_{_safe_label(str(variant))}",
                    kind="key_perturbation",
                    mutations=_enrich_strategy_mutations(strategy, mutated, target),
                    reason=f"Generic perturbation of {key}",
                )
            )
    return candidates


def _build_phase_spec_candidates(strategy: str, config: dict[str, Any], final: dict[str, Any], output_dir: Path | None) -> list[AblationCandidate]:
    try:
        plugin = create_plugin(strategy, config, output_dir=output_dir, max_workers=1, capability_level=config.get("capability_level", "synthetic"))
    except Exception:
        return []
    from backtests.auto.shared.phase_state import PhaseState

    state = PhaseState(cumulative_mutations=dict(final))
    candidates: list[AblationCandidate] = []
    for phase in range(1, int(getattr(plugin, "num_phases", 0) or 0) + 1):
        try:
            spec = plugin.get_phase_spec(phase, state)
        except Exception:
            continue
        for experiment in spec.candidates:
            mutated = dict(final)
            mutated.update(dict(experiment.mutations or {}))
            candidates.append(
                AblationCandidate(
                    label=f"phase_spec_{phase}_{_safe_label(experiment.name)}",
                    kind="phase_spec_candidate",
                    mutations=mutated,
                    reason=f"Candidate from {strategy} phase {phase} spec applied to target final",
                )
            )
    close = getattr(plugin, "close_pool", None)
    if callable(close):
        close()
    return candidates


def _load_manifest_candidates(path: Path, base: dict[str, Any]) -> list[AblationCandidate]:
    payload = read_json(path)
    raw = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise ValueError(f"Candidate manifest must be a list or contain candidates list: {path}")
    out = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        mutations = dict(base if bool(item.get("merge_with_base", True)) else {})
        mutations.update(dict(item.get("mutations") or {}))
        out.append(AblationCandidate(str(item.get("label") or f"manifest_{index}"), str(item.get("kind") or "manifest"), mutations, str(item.get("reason") or "")))
    return out


def build_targeted_oos_candidates(
    strategy: str,
    final: dict[str, Any],
    artifacts: list[RoundArtifact],
    target: RoundArtifact,
    baseline_oos_result: dict[str, Any],
    edge: dict[str, Any] | None = None,
    *,
    max_candidates: int = 80,
) -> list[AblationCandidate]:
    del artifacts
    metrics = dict(baseline_oos_result.get("metrics") or {})
    edge = dict(edge or {})
    net = metric_net(metrics)
    trades = metric_trades(metrics)
    win = metric_win(metrics)
    drawdown = metric_drawdown(metrics)
    worst_loss_share = _num(edge.get("worst_3_loss_share_of_all_losses"))
    risk_trigger = net < 0.0 or drawdown > 0.08 or worst_loss_share > 0.35
    win_trigger = 0.0 < win < 0.38
    frequency_trigger = trades < 15.0
    loss_shape_trigger = drawdown > 0.08 or worst_loss_share > 0.35
    candidates: list[AblationCandidate] = []

    def add(action: str, key: str, original: Any, variant: Any, reason: str) -> None:
        if variant == original:
            return
        mutated = dict(final)
        mutated[key] = variant
        candidates.append(
            AblationCandidate(
                label=f"targeted_{action}_{_safe_label(key)}_{_safe_label(str(variant))}",
                kind="targeted_oos_repair",
                mutations=_enrich_strategy_mutations(strategy, mutated, target),
                reason=reason,
            )
        )

    for key, value in sorted(final.items()):
        if key.startswith("_"):
            continue
        lowered = key.lower()
        if isinstance(value, bool):
            if win_trigger and not value and _key_looks_like_entry_requirement(lowered):
                add("entry_require_on", key, value, True, "low_oos_win_rate_entry_tightening")
            if frequency_trigger and value and _key_looks_like_entry_requirement(lowered):
                add("entry_require_off", key, value, False, "low_oos_trade_frequency_entry_loosening")
            continue
        if not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        if risk_trigger and _key_looks_like_risk_control(lowered):
            for scale in (0.50, 0.75):
                add("risk_down", key, value, _scaled_numeric_variant(value, scale), "negative_or_drawdown_heavy_oos_risk_reduction")
        if win_trigger and _key_looks_like_min_entry_gate(lowered):
            for scale in (1.10, 1.25):
                add("entry_tighter", key, value, _scaled_numeric_variant(value, scale), "low_oos_win_rate_entry_tightening")
        if win_trigger and _key_looks_like_max_entry_gate(lowered):
            for scale in (0.75, 0.90):
                add("entry_cap_lower", key, value, _scaled_numeric_variant(value, scale), "low_oos_win_rate_entry_tightening")
        if frequency_trigger and _key_looks_like_min_entry_gate(lowered):
            for scale in (0.75, 0.90):
                add("entry_looser", key, value, _scaled_numeric_variant(value, scale), "low_oos_trade_frequency_entry_loosening")
        if frequency_trigger and _key_looks_like_max_entry_gate(lowered):
            for scale in (1.10, 1.25):
                add("entry_cap_higher", key, value, _scaled_numeric_variant(value, scale), "low_oos_trade_frequency_entry_loosening")
        if loss_shape_trigger and _key_looks_like_exit_control(lowered):
            for scale in (0.50, 0.75):
                add("exit_tighter", key, value, _scaled_numeric_variant(value, scale), "oos_loss_concentration_exit_control")

    deduped = _dedupe_candidates(candidates)
    if max_candidates > 0:
        return deduped[:max_candidates]
    return deduped


def _phase_timeline(strategy: str, artifact: RoundArtifact, base: dict[str, Any]) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    phase_mutations = _phase_mutations(artifact)
    cumulative = dict(base)
    rows = []
    for phase in sorted(phase_mutations):
        muts = _enrich_strategy_mutations(strategy, dict(phase_mutations[phase]), artifact)
        cumulative.update(muts)
        rows.append((phase, muts, _enrich_strategy_mutations(strategy, dict(cumulative), artifact)))
    return rows


def _phase_mutations(artifact: RoundArtifact) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    sources = []
    if isinstance(artifact.phase_state.get("phase_results"), dict):
        sources.append(artifact.phase_state["phase_results"])
    if isinstance(artifact.round_final_diagnostics.get("phase_results"), dict):
        sources.append(artifact.round_final_diagnostics["phase_results"])
    if isinstance(artifact.diagnostics.get("accepted_phase_mutations"), dict):
        sources.append({key: {"new_mutations": value} for key, value in artifact.diagnostics["accepted_phase_mutations"].items()})
    for source in sources:
        for key, value in source.items():
            try:
                phase = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            muts = value.get("new_mutations") if isinstance(value.get("new_mutations"), dict) else value
            muts = dict(muts or {})
            if muts:
                out.setdefault(phase, {}).update(muts)
    return out


def _previous_round_mutations(artifacts: list[RoundArtifact], round_num: int) -> dict[str, Any]:
    previous = [artifact for artifact in artifacts if artifact.round_num < round_num]
    return dict(previous[-1].mutations) if previous else {}


def _enrich_strategy_mutations(strategy: str, mutations: dict[str, Any], artifact: RoundArtifact) -> dict[str, Any]:
    out = dict(mutations or {})
    if str(strategy or artifact.run_summary.get("strategy") or "").lower() == "kalcb":
        source = _kalcb_source_from_artifact(artifact, required=False)
        if source:
            out.setdefault(SOURCE_PATH_MUTATION, source[SOURCE_PATH_MUTATION])
            out.setdefault(SOURCE_SECTION_MUTATION, source[SOURCE_SECTION_MUTATION])
            out.setdefault(SOURCE_RANK_MUTATION, source[SOURCE_RANK_MUTATION])
    return out


def _kalcb_source_from_artifact(artifact: RoundArtifact, *, required: bool = True) -> dict[str, Any]:
    for raw in (artifact.optimized.get("mutations"), artifact.run_summary.get("cumulative_mutations")):
        if not isinstance(raw, dict):
            continue
        if raw.get(SOURCE_PATH_MUTATION):
            return {
                SOURCE_PATH_MUTATION: str(raw.get(SOURCE_PATH_MUTATION)),
                SOURCE_SECTION_MUTATION: str(raw.get(SOURCE_SECTION_MUTATION) or "top_portfolio_proxy"),
                SOURCE_RANK_MUTATION: int(raw.get(SOURCE_RANK_MUTATION) or 0),
            }
    source = {}
    for payload in (artifact.diagnostics, artifact.optimized.get("execution_contract") or {}, artifact.run_summary.get("execution_contract") or {}):
        raw = (payload.get("source") or payload.get("candidate_source")) if isinstance(payload, dict) else {}
        if isinstance(raw, dict) and (raw.get("path") or raw.get("source_path")):
            source = raw
            break
    if not source:
        metric_contract = artifact.optimized.get("metric_contract") if isinstance(artifact.optimized.get("metric_contract"), dict) else {}
        execution = metric_contract.get("execution_contract") if isinstance(metric_contract.get("execution_contract"), dict) else {}
        raw = execution.get("candidate_source") if isinstance(execution.get("candidate_source"), dict) else {}
        if raw.get("path") or raw.get("source_path"):
            source = raw
    if not source:
        if required:
            raise ValueError(f"Could not resolve KALCB fixed candidate source from {artifact.path}")
        return {}
    return {
        SOURCE_PATH_MUTATION: str(source.get("path") or source.get("source_path")),
        SOURCE_SECTION_MUTATION: str(source.get("section") or source.get("source_section") or "top_portfolio_proxy"),
        SOURCE_RANK_MUTATION: int(source.get("rank") if source.get("rank") is not None else source.get("source_rank") or 0),
    }


def _kalcb_source_from_mutations(mutations: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    return {
        SOURCE_PATH_MUTATION: str(mutations.get(SOURCE_PATH_MUTATION, default[SOURCE_PATH_MUTATION])),
        SOURCE_SECTION_MUTATION: str(mutations.get(SOURCE_SECTION_MUTATION, default[SOURCE_SECTION_MUTATION])),
        SOURCE_RANK_MUTATION: int(mutations.get(SOURCE_RANK_MUTATION, default[SOURCE_RANK_MUTATION])),
    }


def _strip_internal_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    out = dict(mutations or {})
    out.pop(SOURCE_PATH_MUTATION, None)
    out.pop(SOURCE_SECTION_MUTATION, None)
    out.pop(SOURCE_RANK_MUTATION, None)
    return out


def _dedupe_candidates(candidates: Iterable[AblationCandidate]) -> list[AblationCandidate]:
    out: list[AblationCandidate] = []
    seen: set[str] = set()
    seen_labels: set[str] = set()
    for candidate in candidates:
        key = stable_signature(candidate.mutations)
        if key in seen:
            continue
        seen.add(key)
        if candidate.label in seen_labels:
            base_label = candidate.label
            suffix = 2
            while f"{base_label}_{suffix}" in seen_labels:
                suffix += 1
            candidate.label = f"{base_label}_{suffix}"
        seen_labels.add(candidate.label)
        out.append(candidate)
    return out


def _round_num_from_path(path: Path) -> int:
    name = Path(path).name
    if name.startswith("round_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def _parse_date(text: str) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(str(text).split("T", 1)[0])
    except ValueError:
        return None


def _date_text(text: str) -> str:
    return str(text).split("T", 1)[0] if "T" in str(text) else str(text)


def _scaled_numeric_variant(value: int | float, scale: float) -> int | float:
    raw = float(value) * scale
    if isinstance(value, int) and not isinstance(value, bool):
        variant = int(round(raw))
        if variant == value:
            variant = value + (1 if scale > 1.0 else -1)
        return max(0, variant) if value >= 0 else variant
    return raw


def _key_looks_like_risk_control(key: str) -> bool:
    return any(token in key for token in ("risk", "position", "notional", "heat", "leverage", "allocation", "exposure", "capital", "size"))


def _key_looks_like_entry_requirement(key: str) -> bool:
    return ("entry" in key or "signal" in key or "setup" in key) and ("require" in key or key.startswith("use_"))


def _key_looks_like_min_entry_gate(key: str) -> bool:
    if "max" in key or "cap" in key:
        return False
    return ("entry" in key or "signal" in key or "setup" in key or "filter" in key) and any(
        token in key for token in ("min", "threshold", "score", "volume", "ret", "momentum", "confidence", "quality", "rank", "z")
    )


def _key_looks_like_max_entry_gate(key: str) -> bool:
    return ("entry" in key or "signal" in key or "setup" in key or "filter" in key) and ("max" in key or "cap" in key or "limit" in key)


def _key_looks_like_exit_control(key: str) -> bool:
    return any(token in key for token in ("exit", "stop", "loss", "mae", "mfe", "hold", "bars", "trail", "followthrough"))


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")[:90]


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pct(value: Any) -> str:
    return f"{100.0 * _num(value):.2f}%"


def _metric_trade_count(metrics: dict[str, Any], trades: Iterable[Any]) -> float:
    count = metric_trades(metrics)
    if count > 0:
        return count
    return float(len(tuple(trades)))


def _metric_win_rate(metrics: dict[str, Any], trades: Iterable[Any]) -> float:
    win = metric_win(metrics)
    if win > 0:
        return win
    rows = tuple(trades)
    return sum(1 for trade in rows if _num(getattr(trade, "net_pnl", 0.0)) > 0) / max(len(rows), 1)


def _trade_r(trade: Any) -> float:
    route = dict(getattr(trade, "route_metadata", {}) or {})
    qty = max(int(getattr(trade, "qty", 0) or 0), 1)
    risk = max(_num(route.get("risk_per_share")), 1e-9)
    return _num(getattr(trade, "net_pnl", 0.0)) / max(risk * qty, 1e-9)


def _infer_initial_equity(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> float:
    net = metric_net(metrics)
    pnl = sum(_num(row.get("net_pnl")) for row in rows)
    if abs(net) > 1e-9 and abs(pnl) > 1e-9:
        return abs(pnl / net)
    return _num(metrics.get("initial_equity")) or 100_000_000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reusable OOS ablation framework for phased auto-optimization rounds.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--round", type=int, default=None, dest="round_num")
    parser.add_argument("--round-dir", default=None)
    parser.add_argument("--output-root", default="data/backtests/output")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter", choices=["auto", "runner", "generic", "kalcb-fixed"], default="auto")
    parser.add_argument("--oos-start", default=None)
    parser.add_argument("--oos-end", default=None)
    parser.add_argument("--max-oos", type=int, default=0)
    parser.add_argument("--top-train", type=int, default=40)
    parser.add_argument("--no-perturbations", action="store_true")
    parser.add_argument("--include-phase-candidates", action="store_true")
    parser.add_argument("--no-targeted", action="store_true")
    parser.add_argument("--max-targeted", type=int, default=80)
    parser.add_argument("--candidate-manifest", default=None)
    args = parser.parse_args(argv)

    strategy = args.strategy.lower()
    config = normalize_runtime_config(strategy, load_yaml_config(args.config))
    output_root = Path(args.output_root)
    round_dir = Path(args.round_dir) if args.round_dir else None
    artifacts = load_round_chain(strategy, output_root, args.round_num, round_dir=round_dir)
    target = artifacts[-1]
    output_dir = Path(args.output_dir) if args.output_dir else output_root / strategy / f"round_{target.round_num}_oos_ablation"
    run_oos_ablation(
        strategy=strategy,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
