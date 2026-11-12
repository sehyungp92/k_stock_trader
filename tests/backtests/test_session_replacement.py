import hashlib
import json

import pandas as pd
import pytest

from kis_core.input_manifest import InputManifest
from backtests.core.session_replacement import apply_session_replacements
from strategy_common.clock import KST


@pytest.fixture
def replacement(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    stamps = list(pd.date_range("2026-03-30 09:00", periods=76, freq="5min", tz=KST)) + [pd.Timestamp("2026-03-30 15:30", tz=KST)]
    frame = pd.DataFrame({"timestamp": stamps, "open": 100., "high": 110., "low": 90., "close": 101., "volume": 1000.})
    full = root / "full.parquet"
    frame.to_parquet(full, index=False)
    bundle = root / "recovery"
    archive = InputManifest.create([full], source_root=root, destination=bundle)
    derived = bundle / "005930_session.parquet"
    frame.to_parquet(derived, index=False)
    before = frame.iloc[:1].copy()
    before.loc[0, "close"] = 100
    previous_day = frame.iloc[:1].copy()
    previous_day.timestamp -= pd.Timedelta(days=1)
    primary = root / "primary.parquet"
    pd.concat([previous_day, before]).to_parquet(primary, index=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    row = {"symbol": "005930", "date": "2026-03-30", "path": derived.relative_to(root).as_posix(),
        "sha256": sha(derived), "source_path": "full.parquet", "source_sha256": sha(full),
        "source_manifest_hash": archive.identity, "anchor_files": {"primary.parquet": sha(primary)}, "rows": 77,
        "added_timestamps": 76, "revisions": [{"timestamp": stamps[0].isoformat(),
            "archived": {key: float(before.iloc[0][key]) for key in ("open", "high", "low", "close", "volume")},
            "later_snapshot": {key: float(frame.iloc[0][key]) for key in ("open", "high", "low", "close", "volume")}}]}
    proof = bundle / "recovery.json"
    proof.write_text(json.dumps({"source_manifest_hash": archive.identity, "sessions": [row]}))
    return root, proof, primary, row


def test_replacement_preserves_other_sessions_and_records_every_change(replacement):
    root, proof, primary, row = replacement
    selected, paths, receipts = apply_session_replacements(root, {"005930": ["primary.parquet"]}, proof)
    composed = pd.read_parquet(root / selected["005930"][0])
    assert len(composed) == 78 and not composed.timestamp.duplicated().any()
    assert composed.iloc[0]["close"] == 101 and composed.iloc[1]["close"] == 101
    assert pd.read_parquet(primary).iloc[1]["close"] == 100
    assert receipts[0]["added_timestamps"] == 76 and len(receipts[0]["revised_rows"]) == 1
    assert not receipts[0]["historical_qualification"]
    assert apply_session_replacements(root, {"005930": ["primary.parquet"]}, proof)[0] == selected


@pytest.mark.parametrize("bad", ["anchor", "revision", "overlap", "source"])
def test_replacement_rejects_incorrect_evidence(replacement, bad):
    root, proof, primary, row = replacement
    payload = json.loads(proof.read_text())
    if bad == "anchor":
        primary.write_bytes(b"changed")
    elif bad == "revision":
        payload["sessions"][0]["revisions"][0]["archived"]["close"] = 999
    elif bad == "overlap":
        payload["sessions"] *= 2
    else:
        payload["sessions"][0]["source_sha256"] = "wrong"
    proof.write_text(json.dumps(payload))
    with pytest.raises((ValueError, OSError)):
        apply_session_replacements(root, {"005930": ["primary.parquet"]}, proof)
