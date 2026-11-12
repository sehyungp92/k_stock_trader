from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.evaluate_olr_kalcb_shared import management_replay_pairs


@pytest.mark.parametrize("verify_only", [True, False])
def test_input_verification_opts_into_changed_market_diagnostics_only(tmp_path, monkeypatch, verify_only):
    from types import SimpleNamespace
    from scripts import evaluate_olr_kalcb_shared as module
    spec = {"initial_equity": 100000000, "kalcb": {"optimization_config": "config.yaml", "optimized_result": "optimized.json",
            "candidate_snapshot_root": "snapshots", "market_bar_digest": "digest.json"}}
    monkeypatch.setattr(module, "active_manifest", lambda: SimpleNamespace(root=tmp_path, path=tmp_path / "manifest.json"))
    monkeypatch.setattr(module, "market_input_contract", lambda: {})
    monkeypatch.setattr(module, "_repo_path", lambda path: tmp_path / path)
    monkeypatch.setattr(module, "_read_json", lambda path: spec if path.name == "spec.json" else {})
    monkeypatch.setattr(module, "load_yaml_config", lambda path: {})
    class ReachedLoader(Exception):
        pass
    def loader(*args, **kwargs):
        assert kwargs["allow_digest_mismatch"] is verify_only
        raise ReachedLoader
    monkeypatch.setattr(module, "load_promoted_kalcb_replay", loader)
    args = SimpleNamespace(diagnostic_paired=False, verify_inputs_only=verify_only,
                           output=tmp_path / "result.json", spec="spec.json")
    with pytest.raises(ReachedLoader):
        module.run(args)
from scripts.recover_integrity_session import align_to_archived_labels


def test_management_scope_survives_removal_reentry_and_empty_days():
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(5)]
    candidate = SimpleNamespace(symbol="005930")
    snapshots = {days[1]: SimpleNamespace(candidates=[candidate]),
                 days[2]: SimpleNamespace(candidates=[]),
                 days[4]: SimpleNamespace(candidates=[candidate])}
    assert management_replay_pairs(snapshots, days) == {(d, "005930") for d in days[1:]}
    with pytest.raises(ValueError, match="outside"):
        management_replay_pairs(snapshots, days[:2])


def frames():
    anchor = pd.DataFrame({"timestamp": pd.date_range("2025-11-13 10:00", periods=31, freq="min", tz="Asia/Seoul"),
                           "open": 100, "high": 110, "low": 90, "close": 100, "volume": range(100, 131)})
    raw = anchor.copy()
    raw.timestamp -= pd.Timedelta(seconds=44)
    return raw, anchor


def test_timestamp_reconstruction_requires_exact_uniform_observed_agreement():
    raw, anchor = frames()
    result, receipt = align_to_archived_labels(raw, anchor, date(2025, 11, 13))
    pd.testing.assert_frame_equal(result, anchor)
    assert receipt["matching_unique_ohlcv_anchors"] == 31
    assert receipt["offset_seconds"] == 44
    assert not receipt["vendor_timestamp_contract_verified"]
    assert raw.iloc[0].timestamp.second == 16


@pytest.mark.parametrize("failure", ["insufficient", "nonuniform", "overlap_conflict", "ambiguous"])
def test_timestamp_reconstruction_rejects_unproven_alignment(failure):
    raw, anchor = frames()
    if failure == "insufficient":
        anchor = anchor.iloc[:19]
    elif failure == "nonuniform":
        raw.loc[5, "timestamp"] += pd.Timedelta(seconds=1)
    elif failure == "overlap_conflict":
        raw.loc[5, "volume"] = 999
    else:
        raw["volume"] = 100
        anchor["volume"] = 100
    with pytest.raises(ValueError):
        align_to_archived_labels(raw, anchor, date(2025, 11, 13))


def test_reconstruction_preserves_late_auction_time_across_timezone_implementations():
    from zoneinfo import ZoneInfo
    from scripts.recover_flagged_sessions import reconstruct_session
    raw, anchor = frames()
    raw.timestamp += pd.Timedelta(seconds=44)
    raw.timestamp = raw.timestamp.dt.tz_convert(ZoneInfo("Asia/Seoul"))
    late = raw.iloc[[-1]].copy()
    late["timestamp"] = pd.Timestamp("2025-11-13 16:32", tz=ZoneInfo("Asia/Seoul"))
    late["volume"] = 999
    raw = pd.concat([raw, late], ignore_index=True)
    result, proof = reconstruct_session(raw, anchor, date(2025, 11, 13))
    assert result.timestamp.dt.tz is not None
    assert len(result) == 32
    assert proof["added_rows"] == 1
    assert result.timestamp.iloc[-1].hour == 16
    assert result.timestamp.iloc[-1].minute == 32
    assert proof["revised_rows"] == 0
