from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from kis_core.kis_client import KoreaInvestAPI, get_paper_tr_id
from oms.adapter import KISExecutionAdapter


def response(rows, continuation="", cursor=("", "")):
    return SimpleNamespace(is_ok=lambda: True, get_body=lambda: SimpleNamespace(output1=rows,
        ctx_area_fk100=cursor[0], ctx_area_nk100=cursor[1]),
        get_header=lambda: SimpleNamespace(tr_cont=continuation))


def test_history_reads_every_page_and_paper_mapping():
    api = KoreaInvestAPI.__new__(KoreaInvestAPI)
    api.account_num = "test"
    api._url_fetch = MagicMock(side_effect=[response([{"odno": "1"}], "M", ("a", "b")), response([{"odno": "2"}])])
    assert len(api.get_execution_orders("20260105", "20260105")) == 2
    assert api._url_fetch.call_args.kwargs["tr_cont"] == "N"
    assert get_paper_tr_id("TTTC0081R", strict=True) == "VTTC0081R"


@pytest.mark.parametrize("pages", [[None], [response([], "M", ("a", "b")), response([], "M", ("a", "b"))]])
def test_failed_or_repeated_history_page_never_returns_partial_success(pages):
    api = KoreaInvestAPI.__new__(KoreaInvestAPI)
    api.account_num = "test"
    api._url_fetch = MagicMock(side_effect=pages)
    with pytest.raises(RuntimeError):
        api.get_execution_orders("20260105", "20260105")


@pytest.mark.asyncio
async def test_execution_history_uses_actual_notional_and_explicit_cancellation():
    api = MagicMock()
    api.get_execution_orders.return_value = pd.DataFrame([{"ord_dt": "20260105", "odno": "1",
        "pdno": "005930", "sll_buy_dvsn_cd": "02", "ord_qty": "10", "tot_ccld_qty": "4",
        "tot_ccld_amt": "424", "ord_unpr": "999", "ord_tmd": "090000", "cncl_yn": "Y"}])
    result = await KISExecutionAdapter(api).get_execution_orders("20260105", "20260105")
    assert result.ok and result.data[0].price == 106 and result.data[0].filled_qty == 4
    assert result.data[0].status == "CANCELLED" and result.data[0].execution_price_observed
    api.get_execution_orders.return_value.loc[0, "tot_ccld_amt"] = "nan"
    assert not (await KISExecutionAdapter(api).get_execution_orders("20260105", "20260105")).ok
