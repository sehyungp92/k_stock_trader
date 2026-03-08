"""Tests for ConfigWatcher and ParameterChangeEvent."""
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from instrumentation.src.config_watcher import ConfigWatcher, ParameterChangeEvent


class TestParameterChangeEvent:
    def test_event_id_deterministic(self):
        """Same inputs produce same event_id."""
        e1 = ParameterChangeEvent(
            bot_id="bot1", param_name="HARD_STOP",
            old_value=1.2, new_value=1.0,
            timestamp="2026-03-15T14:00:00",
        )
        e2 = ParameterChangeEvent(
            bot_id="bot1", param_name="HARD_STOP",
            old_value=1.2, new_value=1.0,
            timestamp="2026-03-15T14:00:00",
        )
        assert e1.event_id == e2.event_id

    def test_timestamp_auto_generated(self):
        """Timestamp auto-generated if not provided."""
        e = ParameterChangeEvent(bot_id="b", param_name="P", old_value=1, new_value=2)
        assert e.timestamp != ""

    def test_to_dict(self):
        """to_dict returns all fields."""
        e = ParameterChangeEvent(
            bot_id="b", param_name="P", old_value=1, new_value=2,
            change_source="manual", config_file="mod.py",
        )
        d = e.to_dict()
        assert d["param_name"] == "P"
        assert d["old_value"] == 1
        assert d["new_value"] == 2


class TestConfigWatcher:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_module(self, name, constants):
        """Create a fake module with uppercase constants."""
        mod = types.ModuleType(name)
        for k, v in constants.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    def _cleanup_module(self, name):
        sys.modules.pop(name, None)

    @patch("importlib.reload", side_effect=lambda m: m)
    def test_baseline_captures_uppercase_constants(self, _mock_reload):
        """Baseline snapshot captures all uppercase constants."""
        mod_name = "_test_config_watcher_mod1"
        self._make_module(mod_name, {"RISK_PCT": 0.01, "STOP_MULT": 1.2, "_private": 99})
        try:
            watcher = ConfigWatcher(
                bot_id="test", config_modules=[mod_name], data_dir=Path(self.tmpdir),
            )
            watcher.take_baseline()
            baseline = watcher._baseline[mod_name]
            assert "RISK_PCT" in baseline
            assert "STOP_MULT" in baseline
            assert "_private" not in baseline
        finally:
            self._cleanup_module(mod_name)

    @patch("importlib.reload", side_effect=lambda m: m)
    def test_change_detected(self, _mock_reload):
        """old vs new value in event."""
        mod_name = "_test_config_watcher_mod2"
        mod = self._make_module(mod_name, {"STOP_MULT": 1.2})
        try:
            watcher = ConfigWatcher(
                bot_id="test", config_modules=[mod_name], data_dir=Path(self.tmpdir),
            )
            watcher.take_baseline()

            # Modify the constant
            mod.STOP_MULT = 1.0

            changes = watcher.check()
            assert len(changes) == 1
            assert changes[0].param_name == "STOP_MULT"
            assert changes[0].old_value == 1.2
            assert changes[0].new_value == 1.0
        finally:
            self._cleanup_module(mod_name)

    @patch("importlib.reload", side_effect=lambda m: m)
    def test_no_event_when_unchanged(self, _mock_reload):
        """Empty list returned when nothing changed."""
        mod_name = "_test_config_watcher_mod3"
        self._make_module(mod_name, {"STOP_MULT": 1.2})
        try:
            watcher = ConfigWatcher(
                bot_id="test", config_modules=[mod_name], data_dir=Path(self.tmpdir),
            )
            watcher.take_baseline()
            changes = watcher.check()
            assert len(changes) == 0
        finally:
            self._cleanup_module(mod_name)

    @patch("importlib.reload", side_effect=lambda m: m)
    def test_jsonl_persistence(self, _mock_reload):
        """Changes written to config_changes_YYYY-MM-DD.jsonl."""
        mod_name = "_test_config_watcher_mod4"
        mod = self._make_module(mod_name, {"X": 1})
        try:
            watcher = ConfigWatcher(
                bot_id="test", config_modules=[mod_name], data_dir=Path(self.tmpdir),
            )
            watcher.take_baseline()
            mod.X = 2
            watcher.check()

            files = list(Path(self.tmpdir).joinpath("config_changes").glob("config_changes_*.jsonl"))
            assert len(files) == 1
            data = json.loads(files[0].read_text().strip())
            assert data["param_name"] == "X"
        finally:
            self._cleanup_module(mod_name)

    @patch("importlib.reload", side_effect=lambda m: m)
    def test_json_safe_conversion(self, _mock_reload):
        """Tuples, sets converted correctly."""
        mod_name = "_test_config_watcher_mod5"
        self._make_module(mod_name, {
            "TUPLE_VAL": (1, 2, 3),
            "SET_VAL": {"a", "b"},
            "DICT_VAL": {"key": "val"},
        })
        try:
            watcher = ConfigWatcher(
                bot_id="test", config_modules=[mod_name], data_dir=Path(self.tmpdir),
            )
            watcher.take_baseline()
            baseline = watcher._baseline[mod_name]
            assert baseline["TUPLE_VAL"] == [1, 2, 3]
            assert isinstance(baseline["SET_VAL"], list)
            assert baseline["DICT_VAL"] == {"key": "val"}
        finally:
            self._cleanup_module(mod_name)
