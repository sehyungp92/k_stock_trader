"""Config watcher — detects parameter changes via checksum comparison."""
from __future__ import annotations

import json
import hashlib
import importlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("instrumentation.config_watcher")


@dataclass
class ParameterChangeEvent:
    """Records a detected parameter change."""
    bot_id: str
    param_name: str
    old_value: Any
    new_value: Any
    change_source: str = "manual"    # "pr_merge", "manual", "hot_reload", "experiment"
    timestamp: str = ""
    config_file: str = ""            # which config module/file changed
    commit_sha: Optional[str] = None
    pr_url: Optional[str] = None
    event_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.event_id:
            raw = f"{self.bot_id}|{self.timestamp}|parameter_change|{self.param_name}"
            self.event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


class ConfigWatcher:
    """Monitors config modules for parameter changes via checksum comparison.

    Usage:
        watcher = ConfigWatcher(
            bot_id="k_stock_trader",
            config_modules=["strategy_kmp.config.constants", "strategy_kpr.config.constants"],
            data_dir=Path("instrumentation/data"),
        )
        watcher.take_baseline()       # call on startup
        changes = watcher.check()     # call periodically (every 5 min)
    """

    def __init__(
        self,
        bot_id: str,
        config_modules: list[str],
        data_dir: Path,
    ) -> None:
        self._bot_id = bot_id
        self._config_modules = config_modules
        self._data_dir = data_dir / "config_changes"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._baseline: dict[str, dict[str, Any]] = {}  # module_name -> {param: value}
        self._checksum_path = self._data_dir / "config_checksums.json"

    def _snapshot_module(self, module_name: str) -> dict[str, Any]:
        """Import module and extract all public uppercase constants."""
        try:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)  # pick up file changes
            result = {}
            for name in dir(mod):
                if not name.isupper() or name.startswith("_"):
                    continue
                val = getattr(mod, name)
                if callable(val) and not isinstance(val, type):
                    continue
                result[name] = self._make_json_safe(val)
            return result
        except Exception as e:
            logger.debug("Failed to snapshot %s: %s", module_name, e)
            return {}

    def _make_json_safe(self, val: Any) -> Any:
        """Convert value to JSON-serializable form."""
        if isinstance(val, (int, float, str, bool, type(None))):
            return val
        if isinstance(val, (list, tuple)):
            return [self._make_json_safe(v) for v in val]
        if isinstance(val, dict):
            return {str(k): self._make_json_safe(v) for k, v in val.items()}
        if isinstance(val, set):
            return sorted(str(v) for v in val)
        return str(val)

    def take_baseline(self) -> None:
        """Snapshot all config modules as baseline. Call on startup."""
        for module_name in self._config_modules:
            self._baseline[module_name] = self._snapshot_module(module_name)

        # Persist checksums
        checksums = {
            mod: hashlib.sha256(
                json.dumps(snap, sort_keys=True).encode()
            ).hexdigest()
            for mod, snap in self._baseline.items()
        }
        try:
            self._checksum_path.write_text(json.dumps(checksums, indent=2))
        except Exception:
            pass

    def check(self) -> list[ParameterChangeEvent]:
        """Compare current config to baseline. Returns list of changes."""
        if not self._baseline:
            return []

        changes: list[ParameterChangeEvent] = []

        for module_name in self._config_modules:
            current = self._snapshot_module(module_name)
            baseline = self._baseline.get(module_name, {})

            # Find changed params
            all_keys = set(baseline.keys()) | set(current.keys())
            for key in all_keys:
                old_val = baseline.get(key)
                new_val = current.get(key)
                if old_val != new_val:
                    event = ParameterChangeEvent(
                        bot_id=self._bot_id,
                        param_name=key,
                        old_value=old_val,
                        new_value=new_val,
                        change_source="manual",
                        config_file=module_name,
                    )
                    changes.append(event)

            # Update baseline
            self._baseline[module_name] = current

        # Write changes to JSONL
        if changes:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self._data_dir / f"config_changes_{today}.jsonl"
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    for c in changes:
                        f.write(json.dumps(c.to_dict(), default=str) + "\n")
            except Exception as e:
                logger.debug("Failed to write config changes: %s", e)

        return changes
