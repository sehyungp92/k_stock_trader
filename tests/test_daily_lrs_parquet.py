from __future__ import annotations

from datetime import date

from scripts import backfill_lrs
from strategy_common.daily_lrs_parquet import (
    available_daily_symbols,
    export_lrs_sqlite_to_parquet,
    load_daily_flow,
    load_daily_foreign_flow,
    load_daily_institutional_flow,
    load_daily_ohlcv,
    load_index_ohlcv,
    load_sector_map,
)


def test_lrs_sqlite_exports_to_shared_daily_parquet(tmp_path):
    db_path = tmp_path / "lrs.db"
    root = tmp_path / "krx_daily_parquet"
    conn = backfill_lrs.init_db(db_path)
    try:
        conn.executemany(
            "INSERT INTO daily_ohlcv (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("005930", "2024-01-02", 100, 110, 90, 105, 1000),
                ("005930", "2024-01-03", 106, 112, 101, 109, 1200),
            ],
        )
        conn.executemany(
            "INSERT INTO daily_flow (ticker, date, foreign_net, inst_net) VALUES (?, ?, ?, ?)",
            [
                ("005930", "2024-01-02", 2000, 1000),
                ("005930", "2024-01-03", 3000, -500),
            ],
        )
        conn.execute(
            "INSERT INTO index_ohlcv (index_code, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("KOSPI", "2024-01-02", 2500, 2520, 2490, 2510, 10000),
        )
        conn.execute("INSERT INTO sector_map (ticker, sector) VALUES (?, ?)", ("005930", "Semiconductors"))
        conn.commit()
    finally:
        conn.close()

    manifest = export_lrs_sqlite_to_parquet(
        db_path,
        root,
        start=date(2024, 1, 2),
        end=date(2024, 1, 3),
        universe=["005930"],
    )

    assert manifest["dataset_version"] == "krx-daily-lrs-parquet-v1"
    assert manifest["tables"]["daily_ohlcv"]["rows"] == 2
    assert manifest["tables"]["daily_flow"]["rows"] == 2
    assert manifest["tables"]["daily_foreign_flow"]["rows"] == 2
    assert manifest["tables"]["daily_institutional_flow"]["rows"] == 2
    assert manifest["tables"]["index_ohlcv"]["rows"] == 1
    assert (root / "tables" / "daily_ohlcv.parquet").exists()
    assert (root / "tables" / "daily_foreign_flow.parquet").exists()
    assert (root / "tables" / "daily_institutional_flow.parquet").exists()
    assert (root / "daily_ohlcv" / "005930").exists()
    assert (root / "daily_foreign_flow" / "005930").exists()
    assert (root / "daily_institutional_flow" / "005930").exists()
    assert available_daily_symbols(root) == ["005930"]
    assert len(load_daily_ohlcv(root, "005930")) == 2
    assert load_daily_flow(root, "005930").iloc[0]["foreign_net"] == 2000
    assert load_daily_foreign_flow(root, "005930").iloc[0]["foreign_net"] == 2000
    assert load_daily_institutional_flow(root, "005930").iloc[0]["institutional_net"] == 1000
    assert load_index_ohlcv(root, "KOSPI").iloc[0]["close"] == 2510
    assert load_sector_map(root) == {"005930": "Semiconductors"}
