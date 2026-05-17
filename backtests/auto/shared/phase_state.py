from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _SafeJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        isoformat = getattr(obj, "isoformat", None)
        if callable(isoformat):
            return isoformat()
        return str(obj)


@dataclass(slots=True)
class PhaseState:
    current_phase: int = 0
    completed_phases: list[int] = field(default_factory=list)
    cumulative_mutations: dict[str, Any] = field(default_factory=dict)
    phase_results: dict[int, dict] = field(default_factory=dict)
    phase_gate_results: dict[int, dict] = field(default_factory=dict)
    retry_count: dict[int, int] = field(default_factory=dict)
    scoring_retries: dict[int, int] = field(default_factory=dict)
    diagnostic_retries: dict[int, int] = field(default_factory=dict)
    phase_timestamps: dict[int, dict[str, str]] = field(default_factory=dict)
    round_name: str = ""

    def start_phase(self, phase: int) -> None:
        self.current_phase = phase
        self.phase_timestamps.setdefault(phase, {})
        self.phase_timestamps[phase].setdefault("started", _utc_now_iso())

    def complete_phase(self, phase: int) -> None:
        self.phase_timestamps.setdefault(phase, {})
        self.phase_timestamps[phase]["completed"] = _utc_now_iso()

    def advance_phase(self, phase: int, mutations: dict[str, Any], result: dict) -> None:
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
            self.completed_phases.sort()
        self.cumulative_mutations.update(mutations)
        self.phase_results[phase] = result
        self.current_phase = phase
        self.complete_phase(phase)

    def record_gate(self, phase: int, gate_result: dict) -> None:
        self.phase_gate_results[phase] = gate_result

    def increment_retry(self, phase: int) -> int:
        self.retry_count[phase] = self.retry_count.get(phase, 0) + 1
        return self.retry_count[phase]

    def increment_scoring_retry(self, phase: int) -> int:
        self.scoring_retries[phase] = self.scoring_retries.get(phase, 0) + 1
        return self.scoring_retries[phase]

    def increment_diagnostic_retry(self, phase: int) -> int:
        self.diagnostic_retries[phase] = self.diagnostic_retries.get(phase, 0) + 1
        return self.diagnostic_retries[phase]

    def get_phase_metrics(self, phase: int) -> dict | None:
        result = self.phase_results.get(phase)
        if not result:
            return None
        value = result.get("final_metrics")
        return dict(value) if isinstance(value, dict) else None


def _atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    last_error: PermissionError | None = None
    for attempt in range(8):
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, cls=_SafeJsonEncoder)
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    if last_error:
        raise last_error


def save_phase_state(state: PhaseState, path: Path) -> None:
    _atomic_write_json(
        {
            "current_phase": state.current_phase,
            "completed_phases": state.completed_phases,
            "cumulative_mutations": state.cumulative_mutations,
            "phase_results": {str(key): value for key, value in state.phase_results.items()},
            "phase_gate_results": {str(key): value for key, value in state.phase_gate_results.items()},
            "retry_count": {str(key): value for key, value in state.retry_count.items()},
            "scoring_retries": {str(key): value for key, value in state.scoring_retries.items()},
            "diagnostic_retries": {str(key): value for key, value in state.diagnostic_retries.items()},
            "phase_timestamps": {str(key): value for key, value in state.phase_timestamps.items()},
            "round_name": state.round_name,
        },
        path,
    )


def _int_key_dict(raw: dict[str, Any] | None) -> dict[int, Any]:
    return {int(key): value for key, value in (raw or {}).items()}


def load_phase_state(path: Path) -> PhaseState:
    if not Path(path).exists():
        return PhaseState()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhaseState(
        current_phase=int(data.get("current_phase", 0)),
        completed_phases=[int(item) for item in data.get("completed_phases", [])],
        cumulative_mutations=dict(data.get("cumulative_mutations", {})),
        phase_results=_int_key_dict(data.get("phase_results")),
        phase_gate_results=_int_key_dict(data.get("phase_gate_results")),
        retry_count=_int_key_dict(data.get("retry_count")),
        scoring_retries=_int_key_dict(data.get("scoring_retries")),
        diagnostic_retries=_int_key_dict(data.get("diagnostic_retries")),
        phase_timestamps=_int_key_dict(data.get("phase_timestamps")),
        round_name=str(data.get("round_name", "")),
    )

