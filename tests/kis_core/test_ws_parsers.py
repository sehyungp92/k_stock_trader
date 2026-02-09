"""Tests for KIS WebSocket message parsers."""
import pytest
from datetime import datetime
from kis_core.ws_client import (
    parse_ws_message,
    parse_tick_message,
    parse_askbid_message,
    TickMessage,
    AskBidMessage,
)


class TestParseWsMessage:
    """Tests for parse_ws_message: KIS WebSocket header parsing."""

    def test_valid_message(self):
        raw = "header0^H0STCNT0|002|data_type|tick_data_here"
        result = parse_ws_message(raw)
        assert result is not None
        assert result[0] == "H0STCNT0"
        assert result[1] == "002"
        assert result[2] == "tick_data_here"

    def test_no_pipe(self):
        assert parse_ws_message("no pipes here") is None

    def test_too_few_parts(self):
        assert parse_ws_message("a|b|c") is None

    def test_exactly_four_parts(self):
        raw = "h0^TR001|field1|field2|data"
        result = parse_ws_message(raw)
        assert result is not None
        assert result[0] == "TR001"
        assert result[1] == "field1"
        assert result[2] == "data"

    def test_no_header_caret(self):
        raw = "nosubfield|002|data_type|data"
        result = parse_ws_message(raw)
        assert result is not None
        assert result[0] == ""  # no caret -> empty tr_id

    def test_multiple_pipes_extra_data(self):
        raw = "h0^TR001|field1|field2|data|extra"
        result = parse_ws_message(raw)
        assert result is not None
        assert result[0] == "TR001"
        # parts[3] is "data" (index 3)
        assert result[2] == "data"

    def test_empty_string(self):
        assert parse_ws_message("") is None

    def test_single_pipe(self):
        assert parse_ws_message("a|b") is None

    def test_multiple_carets_in_header(self):
        raw = "a^b^c|field1|field2|data"
        result = parse_ws_message(raw)
        assert result is not None
        # header_parts[1] is "b"
        assert result[0] == "b"

    def test_empty_data_field(self):
        raw = "h0^TR001|type|count|"
        result = parse_ws_message(raw)
        assert result is not None
        assert result[2] == ""


class TestParseTickMessage:
    """Tests for parse_tick_message: H0STCNT0 tick data parsing."""

    def _make_fields(self, n=46, **overrides):
        """Build caret-delimited tick data string with n fields."""
        fields = [""] * n
        fields[0] = overrides.get("ticker", "005930")
        fields[1] = overrides.get("timestamp", "093000")
        fields[2] = overrides.get("price", "72000")
        fields[12] = overrides.get("volume", "1500")
        fields[13] = overrides.get("cum_vol", "500000")
        fields[14] = overrides.get("cum_val", "36000000000")
        if n > 45:
            fields[45] = overrides.get("vi_ref", "75000")
        return "^".join(fields)

    def test_valid_tick(self):
        data = self._make_fields()
        now = datetime(2024, 1, 15, 9, 30, 0)
        result = parse_tick_message(data, now_kst=now)
        assert result is not None
        assert result.ticker == "005930"
        assert result.price == 72000.0
        assert result.volume == 1500.0
        assert result.cum_vol == 500000.0
        assert result.cum_val == 36000000000.0
        assert result.vi_ref == 75000.0

    def test_timestamp_parsed(self):
        data = self._make_fields(timestamp="101530")
        now = datetime(2024, 1, 15, 9, 0, 0)
        result = parse_tick_message(data, now_kst=now)
        assert result is not None
        assert result.timestamp.hour == 10
        assert result.timestamp.minute == 15
        assert result.timestamp.second == 30

    def test_too_few_fields(self):
        assert parse_tick_message("a^b^c") is None

    def test_exactly_14_fields_rejected(self):
        # Need at least 15 fields (index 0..14)
        data = "^".join(["x"] * 14)
        assert parse_tick_message(data) is None

    def test_exactly_15_fields_accepted(self):
        data = self._make_fields(n=15)
        now = datetime(2024, 1, 15, 9, 30, 0)
        result = parse_tick_message(data, now_kst=now)
        assert result is not None
        # vi_ref defaults to 0.0 since len(fields) <= 45
        assert result.vi_ref == 0.0

    def test_empty_ticker(self):
        data = self._make_fields(ticker="")
        assert parse_tick_message(data) is None

    def test_zero_price(self):
        data = self._make_fields(price="0")
        assert parse_tick_message(data) is None

    def test_negative_price(self):
        data = self._make_fields(price="-100")
        assert parse_tick_message(data) is None

    def test_invalid_float_price(self):
        data = self._make_fields(price="abc")
        assert parse_tick_message(data) is None

    def test_invalid_float_volume(self):
        data = self._make_fields(volume="xyz")
        assert parse_tick_message(data) is None

    def test_empty_price_defaults_zero(self):
        data = self._make_fields(price="")
        # price=0.0, which is <= 0, so returns None
        assert parse_tick_message(data) is None

    def test_empty_volume_defaults_zero(self):
        data = self._make_fields(volume="")
        now = datetime(2024, 1, 15, 9, 30, 0)
        result = parse_tick_message(data, now_kst=now)
        assert result is not None
        assert result.volume == 0.0

    def test_no_vi_ref_short_fields(self):
        data = self._make_fields(n=15)
        now = datetime(2024, 1, 15, 9, 30, 0)
        result = parse_tick_message(data, now_kst=now)
        assert result is not None
        assert result.vi_ref == 0.0

    def test_short_timestamp_uses_now(self):
        data = self._make_fields(timestamp="09")
        now = datetime(2024, 1, 15, 9, 30, 0)
        result = parse_tick_message(data, now_kst=now)
        assert result is not None
        # Short timestamp -> uses now_kst directly
        assert result.timestamp == now

    def test_returns_tick_message_dataclass(self):
        data = self._make_fields()
        now = datetime(2024, 1, 15, 9, 30, 0)
        result = parse_tick_message(data, now_kst=now)
        assert isinstance(result, TickMessage)


class TestParseAskBidMessage:
    """Tests for parse_askbid_message: H0STASP0 bid/ask data parsing."""

    def _make_fields(self, n=14, **overrides):
        """Build caret-delimited bid/ask data string with n fields."""
        fields = [""] * n
        fields[0] = overrides.get("ticker", "005930")
        if n > 3:
            fields[3] = overrides.get("ask", "72100")
        if n > 13:
            fields[13] = overrides.get("bid", "72000")
        return "^".join(fields)

    def test_valid_askbid(self):
        data = self._make_fields()
        result = parse_askbid_message(data)
        assert result is not None
        assert result.ticker == "005930"
        assert result.ask == 72100.0
        assert result.bid == 72000.0

    def test_too_few_fields(self):
        assert parse_askbid_message("a^b^c") is None

    def test_exactly_three_fields_rejected(self):
        # Need at least 4 fields (len >= 4)
        data = "005930^x^y"
        assert parse_askbid_message(data) is None

    def test_exactly_four_fields_accepted(self):
        # 4 fields: ticker, _, _, ask
        data = "005930^^x^72100"
        result = parse_askbid_message(data)
        assert result is not None
        assert result.ask == 72100.0
        # bid defaults to 0.0 since len(fields) <= 13
        assert result.bid == 0.0

    def test_empty_ticker(self):
        data = self._make_fields(ticker="")
        assert parse_askbid_message(data) is None

    def test_bid_only_no_ask(self):
        data = self._make_fields(ask="")
        result = parse_askbid_message(data)
        assert result is not None
        assert result.ask == 0.0
        assert result.bid == 72000.0

    def test_ask_only_no_bid(self):
        data = self._make_fields(bid="")
        result = parse_askbid_message(data)
        assert result is not None
        assert result.ask == 72100.0
        assert result.bid == 0.0

    def test_returns_askbid_message_dataclass(self):
        data = self._make_fields()
        result = parse_askbid_message(data)
        assert isinstance(result, AskBidMessage)

    def test_float_prices(self):
        data = self._make_fields(ask="72150.5", bid="72050.5")
        result = parse_askbid_message(data)
        assert result is not None
        assert result.ask == 72150.5
        assert result.bid == 72050.5

    def test_invalid_ask_value(self):
        data = self._make_fields(ask="not_a_number")
        assert parse_askbid_message(data) is None
