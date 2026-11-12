import hashlib
import json

import pandas as pd
import pytest

from backtests.core.anchored_recovery import apply_anchored_recoveries, verified_reconstruction
from scripts.update_kis_intraday import aggregate_timeframe


@pytest.fixture
def recovery(tmp_path):
    stamps = [*pd.date_range("2025-11-13 10:00", periods=380, freq="min", tz="Asia/Seoul"),
              pd.Timestamp("2025-11-13 16:30", tz="Asia/Seoul")]
    full = pd.DataFrame(dict(timestamp=stamps, open=100., high=110., low=90., close=105., volume=range(100, 481)))
    anchor = full.iloc[:331].copy()
    raw = full.iloc[270:].copy().reset_index(drop=True)
    raw.timestamp -= pd.Timedelta(seconds=44)
    anchor.to_parquet(tmp_path / "anchor.parquet", index=False)
    raw.to_parquet(tmp_path / "raw.parquet", index=False)
    aggregate_timeframe(anchor, "5m").to_parquet(tmp_path / "primary.parquet", index=False)
    payload = {"symbol": "005930", "market": "J", "endpoint": "inquire-time-dailychartprice",
        "volume_field": "cntg_vol", "rows": [dict(stck_bsop_date=r.timestamp.strftime("%Y%m%d"),
            stck_cntg_hour=r.timestamp.strftime("%H%M%S"), open=r.open, high=r.high, low=r.low,
            close=r.close, cntg_vol=r.volume) for r in raw.itertuples()]}
    (tmp_path / "response.json").write_text(json.dumps(payload))
    row = {"symbol": "005930", "date": "2025-11-13", "anchor": "anchor.parquet", "raw": "raw.parquet",
        "provider_responses": ["response.json"], "files": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ("anchor.parquet", "raw.parquet", "response.json")}}
    return tmp_path, row


def test_recovery_recomputes_raw_parser_anchor_and_session(recovery):
    root, row = recovery
    old, new, proof, _ = verified_reconstruction(root, row)
    assert len(old) == 67 and len(new) == 77
    assert proof["offset_seconds"] == 44
    assert proof["matching_unique_ohlcv_anchors"] == 61
    path = root / "bundle.json"
    path.write_text(json.dumps({"version": "anchored-session-recoveries-v1", "sessions": [row]}))
    selected, _, receipts = apply_anchored_recoveries(root, {"005930": ["primary.parquet"]}, path)
    assert len(pd.read_parquet(root / selected["005930"][0])) == 77
    assert len(pd.read_parquet(root / "primary.parquet")) == 67
    assert not receipts[0]["historical_qualification"]
    assert apply_anchored_recoveries(root, {"005930": ["primary.parquet"]}, path)[0] == selected


@pytest.mark.parametrize("bad", ["bytes", "parser", "basis", "selected_anchor"])
def test_recovery_rejects_unbound_or_incompatible_observations(recovery, bad):
    root, row = recovery
    if bad in {"bytes", "parser"}:
        raw = pd.read_parquet(root / "raw.parquet")
        raw.loc[0, "volume"] += 1
        raw.to_parquet(root / "raw.parquet", index=False)
        if bad == "parser":
            row["files"]["raw.parquet"] = hashlib.sha256((root / "raw.parquet").read_bytes()).hexdigest()
    elif bad == "basis":
        anchor = pd.read_parquet(root / "anchor.parquet")
        anchor["open"] *= .5
        anchor.to_parquet(root / "anchor.parquet", index=False)
        row["files"]["anchor.parquet"] = hashlib.sha256((root / "anchor.parquet").read_bytes()).hexdigest()
    else:
        selected = pd.read_parquet(root / "primary.parquet")
        selected.loc[0, "volume"] += 1
        selected.to_parquet(root / "primary.parquet", index=False)
    path = root / "bundle.json"
    path.write_text(json.dumps({"version": "anchored-session-recoveries-v1", "sessions": [row]}))
    with pytest.raises((ValueError, AssertionError)):
        apply_anchored_recoveries(root, {"005930": ["primary.parquet"]}, path)


@pytest.mark.parametrize("bad", ["late_auction", "conflicting_page", "wrong_market"])
def test_raw_evidence_cannot_waive_execution_timing_or_source_conflicts(recovery, bad):
    root, row = recovery
    payload = json.loads((root / "response.json").read_text())
    if bad == "late_auction":
        raw = pd.read_parquet(root / "raw.parquet")
        raw.loc[len(raw)-1, "timestamp"] += pd.Timedelta(minutes=2)
        raw.to_parquet(root / "raw.parquet", index=False)
        row["files"]["raw.parquet"] = hashlib.sha256((root / "raw.parquet").read_bytes()).hexdigest()
        payload["rows"][-1]["stck_cntg_hour"] = raw.timestamp.iloc[-1].strftime("%H%M%S")
    elif bad == "conflicting_page":
        payload["rows"].append({**payload["rows"][0], "cntg_vol": 999999})
    else:
        payload["market"] = "UN"
    (root / "response.json").write_text(json.dumps(payload))
    row["files"]["response.json"] = hashlib.sha256((root / "response.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verified_reconstruction(root, row)


def test_empty_archive_accepts_independent_daily_conservation_and_rejects_volume_mismatch(recovery):
    root, row = recovery
    anchor = pd.read_parquet(root / "anchor.parquet")
    raw = pd.read_parquet(root / "raw.parquet")
    raw.timestamp += pd.Timedelta(seconds=44)
    full = pd.concat([anchor, raw]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # A legitimate sparse minute needs no synthetic zero-volume bar.
    full = full.drop(index=10).reset_index(drop=True)
    full.to_parquet(root / "raw.parquet", index=False)
    anchor.iloc[:0].to_parquet(root / "anchor.parquet", index=False)
    payload = json.loads((root / "response.json").read_text())
    payload["rows"] = [dict(stck_bsop_date=r.timestamp.strftime("%Y%m%d"), stck_cntg_hour=r.timestamp.strftime("%H%M%S"),
        open=r.open, high=r.high, low=r.low, close=r.close, cntg_vol=r.volume) for r in full.itertuples()]
    (root / "response.json").write_text(json.dumps(payload))
    daily = pd.DataFrame([dict(date="2025-11-13", open=100, high=110, low=90, close=105, volume=full.volume.sum())])
    daily.to_parquet(root / "daily.parquet", index=False)
    row.update(method="daily_reconciled_empty_session", daily_source="daily.parquet")
    row["files"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ("anchor.parquet", "raw.parquet", "response.json", "daily.parquet")}
    old, new, proof, _ = verified_reconstruction(root, row)
    assert old.empty and not new.empty
    assert proof["daily_comparison"]["status"] == "exact_daily_ohlcv_match"
    daily.loc[0, "volume"] += 1
    daily.to_parquet(root / "daily.parquet", index=False)
    row["files"]["daily.parquet"] = hashlib.sha256((root / "daily.parquet").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="independent daily"):
        verified_reconstruction(root, row)
