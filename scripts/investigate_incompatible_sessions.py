"""Collect quotation-only basis evidence for selected session dispositions.

Keeps provider responses separate from research inputs. No fitted conversion or
download in this script is automatically admitted to a backtest.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.update_kis_intraday import (
    KST, KoreaInvestAPI, KoreaInvestEnv, _load_dotenv, build_kis_config_from_env,
)


def collect(report_path, destination, use_pykrx=True, dispositions=None):
    dispositions = set(dispositions or ["returned_source_incompatible_with_archive"])
    rows = [r for r in json.loads(report_path.read_text(encoding="utf-8"))["sessions"]
            if r["disposition"] in dispositions]
    destination.mkdir(parents=True, exist_ok=True)
    _load_dotenv(ROOT / ".env")
    from loguru import logger
    logger.remove()
    api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    results = []
    for row in rows:
        symbol, day = row["symbol"], row["date"]
        for adjusted in (False, True):
            path = destination / "kis_daily" / f"{symbol}_{day}_{'adjusted' if adjusted else 'original'}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("status") == "ok":
                    results.append(previous)
                    continue
            params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol,
                      "FID_INPUT_DATE_1": day.replace("-", ""), "FID_INPUT_DATE_2": day.replace("-", ""),
                      "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0" if adjusted else "1"}
            receipt = {"symbol": symbol, "date": day, "adjusted": adjusted,
                       "queried_at": datetime.now(KST).isoformat(), "params": params,
                       "endpoint": "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"}
            try:
                response = api._url_fetch(receipt["endpoint"], "FHKST03010100", params)
                if response is None or not response.is_ok():
                    receipt["status"] = "provider_error"
                else:
                    body = response.get_body()
                    receipt.update(status="ok", output1=getattr(body, "output1", None),
                                   output2=getattr(body, "output2", []))
            except Exception as exc:
                receipt.update(status="query_failed", error_type=type(exc).__name__)
            path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
            results.append(receipt)
            print(json.dumps({"provider": "kis_daily", "symbol": symbol, "date": day,
                              "adjusted": adjusted, "status": receipt["status"]}), flush=True)
            time.sleep(.6)
    if use_pykrx:
        import pykrx
        from pykrx import stock
        import requests
        original_request = requests.sessions.Session.request
        # Bound network waits and preserve public source responses, never headers.
        captured = []
        def request(session, method, url, **kwargs):
            kwargs.setdefault("timeout", 25)
            response = original_request(session, method, url, **kwargs)
            captured.append({"url": str(url), "method": method, "status_code": response.status_code,
                             "body": response.text})
            return response
        requests.sessions.Session.request = request
        try:
            for symbol in sorted({r["symbol"] for r in rows}):
                days = sorted(r["date"] for r in rows if r["symbol"] == symbol)
                for adjusted in (False, True):
                    path = destination / "pykrx" / f"{symbol}_{'adjusted' if adjusted else 'original'}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "ok":
                        continue
                    captured.clear()
                    receipt = {"symbol": symbol, "dates": days, "adjusted": adjusted,
                               "version": pykrx.__version__, "queried_at": datetime.now(KST).isoformat()}
                    try:
                        frame = stock.get_market_ohlcv_by_date(days[0], days[-1], symbol, adjusted=adjusted)
                        receipt.update(status="ok" if not frame.empty else "empty", rows=json.loads(frame.to_json(orient="table", date_format="iso")))
                    except Exception as exc:
                        receipt.update(status="query_failed", error_type=type(exc).__name__)
                    receipt["public_responses"] = list(captured)
                    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                    print(json.dumps({"provider": "pykrx", "symbol": symbol, "adjusted": adjusted,
                                      "status": receipt["status"]}), flush=True)
                    time.sleep(1)
        finally:
            requests.sessions.Session.request = original_request
    manifest = {"quotation_only": True, "input_report": str(report_path),
                "input_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "files": {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in destination.rglob("*.json") if p.name != "sources.json"}}
    (destination / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "data/backtests/integrity/session-recoveries-20261004-v10/report.json")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--skip-pykrx", action="store_true")
    parser.add_argument("--disposition", action="append",
                        help="Collect these inventory dispositions instead of incompatible sessions")
    args = parser.parse_args()
    collect(args.report, args.destination, not args.skip_pykrx, args.disposition)
