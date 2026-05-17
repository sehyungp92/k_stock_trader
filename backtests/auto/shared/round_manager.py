from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .phase_state import _atomic_write_json, _utc_now_iso, load_phase_state

RUN_SPEC_FILENAME = "run_spec.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
OPTIMIZED_CONFIG_FILENAME = "optimized_config.json"
ROUND_DIAGNOSTICS_FILENAME = "round_final_diagnostics.txt"
ROUND_EVALUATION_FILENAME = "round_evaluation.txt"
PHASE_STATE_FILENAME = "phase_state.json"
MANIFEST_FILENAME = "rounds_manifest.json"
_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")


def canonicalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    raw = metrics or {}
    return {
        "total_trades": raw.get("total_trades", raw.get("trades")),
        "win_rate": raw.get("win_rate"),
        "profit_factor": raw.get("profit_factor", raw.get("pf")),
        "max_drawdown_pct": raw.get("max_drawdown_pct", raw.get("max_dd_pct")),
        "net_return_pct": raw.get("net_return_pct", raw.get("return_pct")),
        "sharpe_ratio": raw.get("sharpe_ratio", raw.get("sharpe")),
    }


class RoundManager:
    def __init__(self, family: str, strategy: str, base_dir: Path | None = None):
        self.family = family
        self.strategy = strategy
        self.base_dir = Path(base_dir or Path("data/backtests/output"))
        self.strategy_dir = self.base_dir / strategy
        self.manifest_path = self.strategy_dir / MANIFEST_FILENAME

    def round_path(self, round_num: int) -> Path:
        return self.strategy_dir / f"round_{round_num}"

    def get_round_dir(self, round_num: int) -> Path:
        path = self.round_path(round_num)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def phase_state_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / PHASE_STATE_FILENAME

    def diagnostics_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / ROUND_DIAGNOSTICS_FILENAME

    def evaluation_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / ROUND_EVALUATION_FILENAME

    def run_summary_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / RUN_SUMMARY_FILENAME

    def optimized_config_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / OPTIMIZED_CONFIG_FILENAME

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"family": self.family, "strategy": self.strategy, "rounds": []}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data.setdefault("family", self.family)
        data.setdefault("strategy", self.strategy)
        data.setdefault("rounds", [])
        return data

    def get_latest_round(self) -> int:
        latest = 0
        for item in self.load_manifest().get("rounds", []):
            if item.get("archived"):
                continue
            latest = max(latest, int(item.get("round", 0) or 0))
        return latest

    def resolve_round(self, requested_round: int | None, *, for_write: bool, expected_phases: int | None = None) -> tuple[int, Path]:
        latest = self.get_latest_round()
        if requested_round is not None:
            if requested_round < 1:
                raise ValueError("Round numbers must be positive")
            if for_write:
                return requested_round, self.get_round_dir(requested_round)
            path = self.round_path(requested_round)
            if not path.exists():
                raise FileNotFoundError(path)
            return requested_round, path
        if not for_write:
            if latest < 1:
                raise FileNotFoundError(f"No rounds exist under {self.strategy_dir}")
            return latest, self.round_path(latest)
        if latest == 0:
            return 1, self.get_round_dir(1)
        latest_dir = self.round_path(latest)
        state_path = self.phase_state_path(latest_dir)
        if state_path.exists() and expected_phases is not None:
            state = load_phase_state(state_path)
            if len(state.completed_phases) < expected_phases:
                return latest, self.get_round_dir(latest)
        if not self.run_summary_path(latest_dir).exists():
            return latest, self.get_round_dir(latest)
        return latest + 1, self.get_round_dir(latest + 1)

    def get_previous_mutations(self, current_round: int | None = None) -> dict[str, Any]:
        previous = self.get_latest_round() if current_round is None else current_round - 1
        if previous < 1:
            return {}
        path = self.optimized_config_path(self.round_path(previous))
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data.get("mutations", data)) if isinstance(data, dict) else {}

    def write_run_spec(self, round_dir: Path, round_num: int, strategy_name: str, *, description: str = "", baseline_mutations: dict[str, Any] | None = None, baseline_source: str | Path | None = None, execution_context: dict[str, Any] | None = None, overwrite: bool = False) -> Path:
        path = Path(round_dir) / RUN_SPEC_FILENAME
        if path.exists() and not overwrite:
            return path
        _atomic_write_json(
            {
                "family": self.family,
                "strategy": self.strategy,
                "strategy_name": strategy_name,
                "round": round_num,
                "description": description,
                "generated_at_utc": _utc_now_iso(),
                "baseline_source": str(baseline_source) if baseline_source else None,
                "baseline_mutations": dict(baseline_mutations or {}),
                "execution_context": execution_context or {},
            },
            path,
        )
        return path

    def write_run_summary(
        self,
        round_dir: Path,
        cumulative_mutations: dict[str, Any],
        final_metrics: dict[str, Any] | None,
        completed_phases: list[int],
        *,
        round_num: int | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> Path:
        resolved_round = round_num if round_num is not None else self._round_num_from_dir(round_dir)
        path = self.run_summary_path(round_dir)
        metadata = dict(artifact_metadata or {})
        _atomic_write_json(
            {
                "family": self.family,
                "strategy": self.strategy,
                "round": resolved_round,
                "generated_at_utc": _utc_now_iso(),
                "completed_phases": completed_phases,
                "cumulative_mutations": cumulative_mutations,
                "headline_metrics": canonicalize_metrics(final_metrics),
                "final_metrics": final_metrics or {},
                **metadata,
            },
            path,
        )
        return path

    def write_optimized_config(
        self,
        round_dir: Path,
        cumulative_mutations: dict[str, Any],
        *,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.optimized_config_path(round_dir)
        metadata = dict(artifact_metadata or {})
        _atomic_write_json(
            {
                "mutations": dict(cumulative_mutations),
                "generated_at_utc": _utc_now_iso(),
                **metadata,
            },
            path,
        )
        return path

    def append_to_manifest(self, round_num: int, cumulative_mutations: dict[str, Any], final_metrics: dict[str, Any] | None) -> Path:
        manifest = self.load_manifest()
        entry = {
            "round": round_num,
            "timestamp": _utc_now_iso(),
            "mutations_count": len(cumulative_mutations),
            "mutations": dict(cumulative_mutations),
            **canonicalize_metrics(final_metrics),
        }
        rounds = manifest.setdefault("rounds", [])
        rounds[:] = [item for item in rounds if int(item.get("round", 0) or 0) != round_num]
        rounds.append(entry)
        rounds.sort(key=lambda item: int(item.get("round", 0) or 0))
        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest, self.manifest_path)
        return self.manifest_path

    @staticmethod
    def _round_num_from_dir(round_dir: Path) -> int:
        match = _ROUND_DIR_RE.match(Path(round_dir).name)
        if not match:
            raise ValueError(f"Could not infer round number from {round_dir}")
        return int(match.group(1))
