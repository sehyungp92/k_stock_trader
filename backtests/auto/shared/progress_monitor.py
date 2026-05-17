from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def build_progress_snapshot(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    state = _read_json(output_dir / "phase_state.json")
    progress = _read_json(output_dir / "progress.json")
    activity = _read_jsonl(output_dir / "phase_activity_log.jsonl")
    current_phase = int(progress.get("current_phase") or state.get("current_phase") or 0)
    phase_progress = (progress.get("phases") or {}).get(str(current_phase), {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "round_completed": (output_dir / "round_evaluation.txt").exists(),
        "current_phase": current_phase,
        "completed_phases": state.get("completed_phases", []),
        "status": phase_progress.get("status", "unknown"),
        "phase_progress": phase_progress,
        "recent_activity": [row for row in activity if int(row.get("phase", 0) or 0) == current_phase][-8:],
    }


def render_progress_text(snapshot: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Generated: {snapshot.get('generated_at', '')}",
            f"Round dir: {snapshot.get('output_dir', '')}",
            f"Status: {snapshot.get('status', 'unknown')} | phase={snapshot.get('current_phase', 0)} | completed={snapshot.get('completed_phases', [])}",
        ]
    ) + "\n"

