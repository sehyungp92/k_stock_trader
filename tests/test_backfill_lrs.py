from __future__ import annotations

import pandas as pd
from bs4 import BeautifulSoup

from scripts import backfill_lrs


class FakePykrxStock:
    def get_market_ohlcv_by_date(self, *_args, **_kwargs):
        return pd.DataFrame(
            [
                {"시가": 100, "고가": 110, "저가": 90, "종가": 105, "거래량": 1000},
                {"시가": 106, "고가": 112, "저가": 101, "종가": 109, "거래량": 1200},
            ],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

    def get_market_trading_value_by_date(self, *_args, **_kwargs):
        return pd.DataFrame(
            [
                {"기관합계": 1000, "외국인합계": 2000},
                {"기관합계": -500, "외국인합계": 3000},
            ],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

    def get_index_ohlcv_by_date(self, *_args, **_kwargs):
        return pd.DataFrame(
            [
                {"시가": 2500, "고가": 2520, "저가": 2490, "종가": 2510, "거래량": 10_000},
            ],
            index=pd.to_datetime(["2024-01-02"]),
        )


def test_backfill_lrs_normalizes_pykrx_and_writes_no_fx(tmp_path):
    db_path = tmp_path / "lrs.db"
    conn = backfill_lrs.init_db(db_path)
    stock = FakePykrxStock()

    try:
        assert backfill_lrs.upsert_sector_map(conn, {"005930": "Semiconductors"}) == 1
        assert backfill_lrs.backfill_stock_daily(conn, stock, "005930", pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()) == 2
        assert backfill_lrs.backfill_stock_flow(conn, stock, "005930", pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()) == 2
        assert backfill_lrs.backfill_index(conn, stock, "KOSPI", "1001", pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()) == 1

        ohlcv = conn.execute("SELECT * FROM daily_ohlcv WHERE ticker = '005930' ORDER BY date").fetchall()
        flow = conn.execute("SELECT * FROM daily_flow WHERE ticker = '005930' ORDER BY date").fetchall()
        index = conn.execute("SELECT * FROM index_ohlcv WHERE index_code = 'KOSPI'").fetchall()
        fx_count = conn.execute("SELECT COUNT(*) AS cnt FROM fx_rates").fetchone()["cnt"]

        assert len(ohlcv) == 2
        assert ohlcv[0]["close"] == 105
        assert len(flow) == 2
        assert flow[0]["foreign_net"] == 2000
        assert flow[0]["inst_net"] == 1000
        assert len(index) == 1
        assert fx_count == 0
    finally:
        conn.close()


def test_naver_flow_parser_uses_row_shape_not_localized_headers():
    soup = BeautifulSoup(
        """
        <table class="type2">
          <tr><th>ignored</th></tr>
          <tr>
            <td>2024.01.02</td><td>10</td><td>11</td><td>12</td><td>13</td>
            <td>1,000</td><td>-2,000</td>
          </tr>
        </table>
        """,
        "html.parser",
    )
    rows: dict[str, dict] = {}

    backfill_lrs._collect_naver_flow_rows(
        soup,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-01-31").date(),
        rows,
    )

    assert rows["2024-01-02"]["inst_net"] == 1000
    assert rows["2024-01-02"]["foreign_net"] == -2000
