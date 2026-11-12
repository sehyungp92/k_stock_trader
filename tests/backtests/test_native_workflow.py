from copy import deepcopy
import json

import pytest

from kis_core.input_manifest import InputManifest
from backtests.strategies.portfolio_synergy.native_workflow import validate_study, run_native_study, RESERVATION_CONTRACT


def study():
    return {"version": "native-policy-study-v1", "reservation_contract": RESERVATION_CONTRACT,
        "qualification_usage": "regression", "initial_equity": 100000000,
        "train": {"start": "2025-05-12", "end": "2026-03-31"},
        "qualification": {"start": "2026-04-01", "end": "2026-05-12"},
        "candidates": [{"name": "baseline", "mutations": {}}]}


@pytest.mark.parametrize("bad", ["overlap", "synthetic_mutation", "reservation", "unseen", "duplicate"])
def test_native_study_rejects_ambiguous_or_unsupported_contracts(bad):
    payload = study()
    if bad == "overlap":
        payload["qualification"]["start"] = "2026-03-31"
    elif bad == "synthetic_mutation":
        payload["candidates"][0]["mutations"] = {"portfolio.max_daily_gross": 2}
    elif bad == "reservation":
        payload["reservation_contract"] = "live-equivalent"
    elif bad == "unseen":
        payload["qualification_usage"] = "newly_unseen"
    else:
        payload["candidates"] *= 2
    with pytest.raises(ValueError):
        validate_study(payload)


def test_missing_evidence_stops_before_research_and_never_writes_a_winner(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    file = root / "source.json"
    file.write_text("{}")
    manifest = InputManifest.create([file], source_root=root, destination=tmp_path / "generation")
    payload = {**study(), "input_manifest_path": str(manifest.path)}
    def forbidden(*args):
        pytest.fail("Unqualified evidence must stop before expensive selection")
    output = tmp_path / "study"
    result = run_native_study(payload, output, input_loader=forbidden)
    assert result["status"] == "unqualified" and not result["historical_qualification"]
    assert result["blocked_before_selection"] == ["references", "basis", "capacity"]
    assert (output / "study.json").is_file() and not (output / "finalist.json").exists()
    with pytest.raises(FileExistsError):
        run_native_study(payload, output, input_loader=forbidden)


def test_native_olr_window_does_not_reapply_training_holdout_split(monkeypatch):
    from datetime import date
    from pathlib import Path
    from types import SimpleNamespace
    from backtests.strategies.olr import replay_cache as module
    requested = {"start": "2026-04-01", "end": "2026-05-12"}
    dataset = SimpleNamespace(train_start=date(2026, 4, 1), train_end=date(2026, 5, 12),
        trading_dates=(date(2026, 4, 1),), bars_by_key={}, source_fingerprint="fixture", data_root=Path("."))
    calls = []
    def prepare(config, **kwargs):
        calls.append((config, kwargs))
        return dataset
    monkeypatch.setattr(module, "prepare_research_sweep_dataset", prepare)
    monkeypatch.setattr(module, "_load_or_build_stage2_snapshots", lambda *args: {})
    monkeypatch.setattr(module, "compile_olr_replay_bundle", lambda *args, **kwargs: SimpleNamespace(metadata={}))
    result = module.load_olr_declared_window_bundle({"date_range": requested})
    assert calls[0][0]["date_range"] == requested and calls[0][1]["include_holdout"] is True
    assert result.metadata["declared_window"] == requested
    dataset.train_end = date(2026, 3, 31)
    with pytest.raises(ValueError, match="changed the declared"):
        module.load_olr_declared_window_bundle({"date_range": requested})
