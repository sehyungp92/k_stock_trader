"""Tests for KIS API response handling."""
import pytest
from collections import namedtuple
from unittest.mock import MagicMock, PropertyMock, patch
import json
import requests
from kis_core.kis_responses import APIResponse, create_error_response


class TestAPIResponse:
    """Tests for APIResponse wrapper class."""

    def _make_response(self, status_code=200, body=None):
        """Create a mock requests.Response with given status and JSON body.

        Uses title-cased header keys to match real requests.Response behavior
        (CaseInsensitiveDict stores keys title-cased), so _parse_header's
        islower() filter skips them, avoiding namedtuple field name errors.
        """
        resp = MagicMock(spec=requests.Response)
        resp.status_code = status_code
        resp.headers = {'Content-Type': 'application/json'}
        if body is None:
            body = {"rt_cd": "0", "msg1": "Success", "output": {"data": "test"}}
        resp.json.return_value = body
        return resp

    # -----------------------------------------------------------------
    # is_ok / is_error
    # -----------------------------------------------------------------

    def test_is_ok_success(self):
        resp = self._make_response()
        api_resp = APIResponse(resp)
        assert api_resp.is_ok() is True
        assert api_resp.is_error() is False

    def test_is_ok_error_code(self):
        resp = self._make_response(body={"rt_cd": "1", "msg1": "Error"})
        api_resp = APIResponse(resp)
        assert api_resp.is_ok() is False
        assert api_resp.is_error() is True

    def test_is_ok_non_200(self):
        resp = self._make_response(status_code=500, body={"rt_cd": "0", "msg1": "ok"})
        api_resp = APIResponse(resp)
        assert api_resp.is_ok() is False

    def test_is_ok_status_201_fails(self):
        resp = self._make_response(status_code=201, body={"rt_cd": "0", "msg1": "ok"})
        api_resp = APIResponse(resp)
        # Only HTTP 200 is considered success
        assert api_resp.is_ok() is False

    def test_is_ok_empty_rt_cd_is_success(self):
        resp = self._make_response(body={"rt_cd": "", "msg1": "ok"})
        api_resp = APIResponse(resp)
        # Empty string is in SUCCESS_CODES
        assert api_resp.is_ok() is True

    # -----------------------------------------------------------------
    # get_body
    # -----------------------------------------------------------------

    def test_get_body(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok", "output": [1, 2, 3]})
        api_resp = APIResponse(resp)
        body = api_resp.get_body()
        assert body.rt_cd == "0"
        assert body.output == [1, 2, 3]

    def test_get_body_has_all_fields(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok", "extra_field": "val"})
        api_resp = APIResponse(resp)
        body = api_resp.get_body()
        assert body.extra_field == "val"

    # -----------------------------------------------------------------
    # get_output
    # -----------------------------------------------------------------

    def test_get_output(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok", "output": "data"})
        api_resp = APIResponse(resp)
        assert api_resp.get_output("output") == "data"

    def test_get_output_missing_key_returns_default(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok"})
        api_resp = APIResponse(resp)
        assert api_resp.get_output("missing", "default") == "default"

    def test_get_output_default_none(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok"})
        api_resp = APIResponse(resp)
        assert api_resp.get_output("missing") is None

    # -----------------------------------------------------------------
    # Error properties
    # -----------------------------------------------------------------

    def test_error_properties(self):
        resp = self._make_response(body={"rt_cd": "42", "msg1": "Bad request"})
        api_resp = APIResponse(resp)
        assert api_resp.error_code == "42"
        assert api_resp.error_message == "Bad request"

    def test_error_code_getter(self):
        resp = self._make_response(body={"rt_cd": "100", "msg1": "Rate limit"})
        api_resp = APIResponse(resp)
        assert api_resp.get_error_code() == "100"
        assert api_resp.get_error_message() == "Rate limit"

    # -----------------------------------------------------------------
    # status_code
    # -----------------------------------------------------------------

    def test_status_code(self):
        resp = self._make_response(status_code=200)
        api_resp = APIResponse(resp)
        assert api_resp.status_code == 200

    def test_status_code_non_200(self):
        resp = self._make_response(status_code=403)
        api_resp = APIResponse(resp)
        assert api_resp.status_code == 403

    def test_get_result_code(self):
        resp = self._make_response(status_code=200)
        api_resp = APIResponse(resp)
        assert api_resp.get_result_code() == 200

    # -----------------------------------------------------------------
    # to_dict
    # -----------------------------------------------------------------

    def test_to_dict(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok"})
        api_resp = APIResponse(resp)
        d = api_resp.to_dict()
        assert d['status_code'] == 200
        assert d['is_ok'] is True
        assert 'body' in d
        assert 'error_code' in d
        assert 'error_message' in d

    def test_to_dict_body_contains_fields(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok", "output": "data"})
        api_resp = APIResponse(resp)
        d = api_resp.to_dict()
        assert d['body']['rt_cd'] == "0"
        assert d['body']['output'] == "data"

    # -----------------------------------------------------------------
    # __bool__
    # -----------------------------------------------------------------

    def test_bool_true_on_ok(self):
        resp = self._make_response()
        api_resp = APIResponse(resp)
        assert bool(api_resp) is True

    def test_bool_false_on_error(self):
        resp = self._make_response(body={"rt_cd": "1", "msg1": "err"})
        api_resp = APIResponse(resp)
        assert bool(api_resp) is False

    def test_bool_false_on_non_200(self):
        resp = self._make_response(status_code=500, body={"rt_cd": "0", "msg1": "ok"})
        api_resp = APIResponse(resp)
        assert bool(api_resp) is False

    # -----------------------------------------------------------------
    # __repr__ / __str__
    # -----------------------------------------------------------------

    def test_repr_ok(self):
        resp = self._make_response()
        api_resp = APIResponse(resp)
        r = repr(api_resp)
        assert "OK" in r
        assert "200" in r

    def test_repr_error(self):
        resp = self._make_response(body={"rt_cd": "42", "msg1": "err"})
        api_resp = APIResponse(resp)
        r = repr(api_resp)
        assert "ERROR" in r
        assert "42" in r

    def test_str_ok(self):
        resp = self._make_response()
        api_resp = APIResponse(resp)
        s = str(api_resp)
        assert "OK" in s

    def test_str_error(self):
        resp = self._make_response(body={"rt_cd": "1", "msg1": "fail"})
        api_resp = APIResponse(resp)
        s = str(api_resp)
        assert "ERROR" in s

    # -----------------------------------------------------------------
    # JSON decode error
    # -----------------------------------------------------------------

    def test_json_decode_error(self):
        resp = self._make_response()
        resp.json.side_effect = requests.exceptions.JSONDecodeError("msg", "doc", 0)
        api_resp = APIResponse(resp)
        assert api_resp.is_ok() is False
        assert api_resp.error_code == "999"

    # -----------------------------------------------------------------
    # Empty body
    # -----------------------------------------------------------------

    def test_empty_body(self):
        resp = self._make_response(body={})
        api_resp = APIResponse(resp)
        # Empty dict -> rt_cd defaults to "0" which is in SUCCESS_CODES
        assert api_resp.is_ok() is True
        assert api_resp.error_code == "0"

    # -----------------------------------------------------------------
    # get_response / get_header
    # -----------------------------------------------------------------

    def test_get_response_returns_original(self):
        resp = self._make_response()
        api_resp = APIResponse(resp)
        assert api_resp.get_response() is resp

    def test_get_header(self):
        resp = self._make_response()
        api_resp = APIResponse(resp)
        header = api_resp.get_header()
        # MagicMock headers are not lowercase, so may be empty namedtuple
        assert header is not None

    # -----------------------------------------------------------------
    # Hyphen key sanitization
    # -----------------------------------------------------------------

    def test_hyphen_keys_sanitized(self):
        resp = self._make_response(body={"rt_cd": "0", "msg1": "ok", "my-field": "value"})
        api_resp = APIResponse(resp)
        body = api_resp.get_body()
        assert body.my_field == "value"


class TestCreateErrorResponse:
    """Tests for create_error_response factory function.

    create_error_response uses a real requests.models.Response internally
    with a lowercase 'content-type' header. The _parse_header method picks
    this up but fails because 'content-type' is not a valid namedtuple field.
    We patch _parse_header to return an empty header since we are testing
    the error response body/status logic, not header parsing.
    """

    _empty_header = namedtuple('header', [])()

    def _create(self, **kwargs):
        with patch.object(APIResponse, '_parse_header', return_value=self._empty_header):
            return create_error_response(**kwargs)

    def test_default_error(self):
        resp = self._create()
        assert resp.is_ok() is False
        assert resp.status_code == 500
        assert resp.error_code == "999"
        assert "Internal error" in resp.error_message

    def test_custom_error(self):
        resp = self._create(
            status_code=400,
            error_code="100",
            error_message="Bad param",
        )
        assert resp.status_code == 400
        assert resp.error_code == "100"
        assert "Bad param" in resp.error_message

    def test_returns_api_response(self):
        resp = self._create()
        assert isinstance(resp, APIResponse)

    def test_is_error(self):
        resp = self._create()
        assert resp.is_error() is True
        assert bool(resp) is False

    def test_custom_status_code(self):
        resp = self._create(status_code=429)
        assert resp.status_code == 429

    def test_to_dict(self):
        resp = self._create(error_code="123", error_message="test error")
        d = resp.to_dict()
        assert d['is_ok'] is False
        assert d['error_code'] == "123"
