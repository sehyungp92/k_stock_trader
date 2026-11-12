"""A final report's receipt must bind its actual bytes on every host platform."""
import hashlib
from types import SimpleNamespace

from backtests.strategies.olr.final_diagnostics import render_full_diagnostics


def test_final_report_hash_matches_written_bytes(tmp_path):
    plugin = SimpleNamespace(training_config={}, config={})
    state = SimpleNamespace(cumulative_mutations={})
    payload = render_full_diagnostics(plugin, state, tmp_path, round_num=2,
        round_name="retained control", cases={}, training_text="First line\nSecond line")
    report = (tmp_path / "round_final_diagnostics.txt").read_bytes()
    assert b"\r\n" not in report
    assert payload["final_diagnostics_text_sha256"] == hashlib.sha256(report).hexdigest()
