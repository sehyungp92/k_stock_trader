from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from scripts.prepare_olr_forward_daily import fetch_flow, parse_flow_row, validate_price_rows


def row(day, foreign="0", institution="-23"):
    return {"stck_bsop_date": day.strftime("%Y%m%d"), "frgn_ntby_qty": foreign,
            "orgn_ntby_qty": institution, "frgn_ntby_tr_pbmn": "999999999"}


def test_uses_signed_share_quantities_and_preserves_real_zero():
    parsed = parse_flow_row(row(date(2026, 10, 7), "1,234"))
    assert parsed == {"date": "2026-10-07", "foreign_net": 1234, "inst_net": -23}
    assert parse_flow_row(row(date(2026, 10, 7)))["foreign_net"] == 0


@pytest.mark.parametrize("value", [None, "", "NaN", "Infinity", "1.5"])
def test_missing_or_invalid_quantity_never_becomes_zero(value):
    with pytest.raises(ValueError):
        parse_flow_row(row(date(2026, 10, 7), value))


class API:
    def __init__(self, pages):
        self.pages, self.dates = iter(pages), []

    def _url_fetch(self, endpoint, transaction, params, **kwargs):
        assert endpoint.startswith("/uapi/domestic-stock/v1/quotations/")
        assert transaction == "FHPTJ04160001"
        self.dates.append(params["FID_INPUT_DATE_1"])
        page = next(self.pages)
        return SimpleNamespace(is_ok=lambda: True, get_body=lambda: SimpleNamespace(output2=page))


def test_pagination_crosses_weekend_and_archives_original_rows(tmp_path):
    api = API([[row(date(2026, 10, 7)), row(date(2026, 10, 6))], [row(date(2026, 10, 2))]])
    required = ["2026-10-02", "2026-10-06", "2026-10-07"]
    result = fetch_flow(api, "005930", required, date(2026, 10, 7), tmp_path)
    assert [r["date"] for r in result] == required
    assert api.dates == ["20261007", "20261005"]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_missing_actual_price_date_fails_even_when_endpoints_are_present(tmp_path):
    api = API([[row(date(2026, 10, 7)), row(date(2026, 10, 2))]])
    with pytest.raises(ValueError, match="Missing investor observations"):
        fetch_flow(api, "005930", ["2026-10-02", "2026-10-06", "2026-10-07"], date(2026, 10, 7), tmp_path)


def test_future_response_is_rejected(tmp_path):
    api = API([[row(date(2026, 10, 8))]])
    with pytest.raises(ValueError, match="exceeds requested date"):
        fetch_flow(api, "005930", ["2026-10-07"], date(2026, 10, 7), tmp_path)


def test_fresh_prices_and_complete_feature_lookback_are_required():
    end = date(2026, 10, 7)
    rows = [{"date": (end - timedelta(days=i)).isoformat(), "open": 10, "high": 11,
             "low": 9, "close": 10, "volume": 100} for i in reversed(range(45))]
    required = validate_price_rows(rows, end)
    assert len(required) == 40
    assert required[-1] == end.isoformat()
    with pytest.raises(ValueError, match="Latest price"):
        validate_price_rows(rows[:-1], end)
    with pytest.raises(ValueError, match="Insufficient price"):
        validate_price_rows(rows[-39:], end)
    # Two validation observations plus all 40 pre-validation observations.
    forward_required = validate_price_rows(rows, end, end - timedelta(days=1))
    assert len(forward_required) == 42
    assert forward_required[-1] == end.isoformat()
