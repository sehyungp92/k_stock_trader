from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .phase_state import _atomic_write_json, _utc_now_iso

_PHASE_FILE_RE = re.compile(r"^phase_(\d+)(?:_|\.|$)")


class PhaseLogger:
    def __init__(self, output_dir: Path, round_name: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_name = round_name
        self._phase_loggers: dict[int, logging.Logger] = {}

    def get_phase_logger(self, phase: int) -> logging.Logger:
        if phase in self._phase_loggers:
            return self._phase_loggers[phase]
        key = hashlib.md5(str(self.output_dir.resolve()).lower().encode("utf-8")).hexdigest()[:12]
        logger = logging.getLogger(f"k_stock.backtests.phase.{key}.{phase}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        handler = logging.FileHandler(self.output_dir / f"phase_{phase}.log", mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        self._phase_loggers[phase] = logger
        return logger

    def log_activity(self, phase: int, action: str, details: dict) -> None:
        entry = json.dumps({"timestamp": _utc_now_iso(), "phase": phase, "action": action, **details}, default=str)
        with open(self.output_dir / "phase_activity_log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(entry + "\n")

    def update_progress(self, phase: int, summary: dict) -> None:
        path = self.output_dir / "progress.json"
        current = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current.setdefault("phases", {})
        current["last_updated"] = _utc_now_iso()
        current["current_phase"] = phase
        current["phases"][str(phase)] = summary
        _atomic_write_json(current, path)

    def save_phase_output(self, phase: int, kind: str, data: str | dict) -> None:
        if isinstance(data, str):
            (self.output_dir / f"phase_{phase}_{kind}.txt").write_text(data, encoding="utf-8")
        else:
            _atomic_write_json(data, self.output_dir / f"phase_{phase}_{kind}.json")

    def backup_state(self, state_path: Path, label: str) -> None:
        if not state_path.exists():
            return
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")[:80] or "backup"
        target = state_path.with_name(f"phase_state_pre_{safe}_backup.json")
        if target.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            target = state_path.with_name(f"phase_state_pre_{safe}_{stamp}_backup.json")
        shutil.copy2(state_path, target)

    def clear_generated_outputs(self, from_phase: int) -> None:
        for path in self.output_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.endswith("_greedy_checkpoint.json"):
                continue
            match = _PHASE_FILE_RE.match(path.name)
            if match and int(match.group(1)) >= from_phase:
                path.unlink(missing_ok=True)

    def prune_progress(self, keep_phases: set[int], *, current_phase: int = 0) -> None:
        path = self.output_dir / "progress.json"
        current = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current["phases"] = {
            str(int(key)): value
            for key, value in (current.get("phases") or {}).items()
            if int(key) in keep_phases
        }
        current["current_phase"] = current_phase
        current["last_updated"] = _utc_now_iso()
        _atomic_write_json(current, path)

