from __future__ import annotations

from datetime import date

from strategy_common.lrs import LRSDatabase


def test_research_artifact_save_is_idempotent_for_same_hash(tmp_path):
    lrs = LRSDatabase(tmp_path / "lrs.db")
    payload = {"artifact_hash": "abc", "value": 1}

    first = lrs.save_artifact(date(2026, 1, 5), payload, strategy_id="GAMMA", artifact_type="watchlist_snapshot")
    second = lrs.save_artifact(date(2026, 1, 5), payload, strategy_id="GAMMA", artifact_type="watchlist_snapshot")

    assert first.version == 1
    assert second.version == 1
