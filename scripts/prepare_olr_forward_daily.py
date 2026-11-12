"""Acquire the frozen OLR universe's daily inputs without running the strategy.

Uses the existing LRS schema/exporter. KIS investor quantities are shares, the
units consumed by OLR's flow/volume features. No value-to-quantity substitution,
zero filling, universe reduction, broker orders, or historical-file replacement.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backfill_lrs as lrs
from scripts.olr_prospective_validation import verified_contract
from scripts.olr_rule_study import digest, load_study, write

FLOW_ENDPOINT = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
FLOW_TR_ID = "FHPTJ04160001"
FLOW_SOURCE = "https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/investor_trade_by_stock_daily/chk_investor_trade_by_stock_daily.py"
# _flow_metrics uses the most recent 40 price observations, including its
# acceleration baseline. Fetch their actual dates, not a fixed calendar span.
FLOW_OBSERVATIONS = 40


class DailyInputError(ValueError):
    """Safe local diagnostic; never contains transport headers or credentials."""


def parse_flow_row(row):
    day = datetime.strptime(row["stck_bsop_date"], "%Y%m%d").date()
    result = {"date": day.isoformat()}
    for target, source in (("foreign_net", "frgn_ntby_qty"), ("inst_net", "orgn_ntby_qty")):
        raw = row.get(source)
        if raw is None or str(raw).strip() == "":
            raise DailyInputError("Missing investor quantity: " + source)
        value = float(str(raw).replace(",", ""))
        if not math.isfinite(value) or not value.is_integer():
            raise DailyInputError("Invalid investor share quantity: " + source)
        result[target] = int(value)
    return result


def fetch_flow(api, symbol, required_dates, end, raw_root, max_pages=20):
    """Page backwards by the oldest returned date; retain original quote rows."""
    if not required_dates:
        raise ValueError("No price observations to bind flow inputs")
    first = date.fromisoformat(min(required_dates))
    cursor, values = end, {}
    for _ in range(max_pages):
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol,
                  "FID_INPUT_DATE_1": cursor.strftime("%Y%m%d"),
                  "FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": ""}
        response = api._url_fetch(FLOW_ENDPOINT, FLOW_TR_ID, params, retry_on_failure=True)
        if response is None or not response.is_ok():
            raise DailyInputError("Investor quotation request failed")
        rows = getattr(response.get_body(), "output2", None)
        if not isinstance(rows, list) or not rows:
            raise DailyInputError("Investor quotation returned no history")
        write(raw_root / (symbol + "_" + cursor.strftime("%Y%m%d") + ".json"),
              {"endpoint": FLOW_ENDPOINT, "params": params, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
               "quantity_basis": "net_shares", "rows": rows})
        parsed = [parse_flow_row(row) for row in rows]
        if any(date.fromisoformat(row["date"]) > cursor for row in parsed):
            raise DailyInputError("Investor response exceeds requested date")
        for row in parsed:
            if row["date"] in values and values[row["date"]] != row:
                raise DailyInputError("Conflicting investor observations")
            values[row["date"]] = row
        oldest = min(date.fromisoformat(row["date"]) for row in parsed)
        if oldest <= first:
            break
        cursor = oldest - timedelta(days=1)
    missing = set(required_dates) - values.keys()
    if missing:
        raise DailyInputError("Missing investor observations for price dates: " + ",".join(sorted(missing)))
    return [values[day] for day in sorted(values) if first.isoformat() <= day <= end.isoformat()]


def validate_price_rows(rows, end, validation_start=None):
    if len(rows) < FLOW_OBSERVATIONS:
        raise DailyInputError("Insufficient price history for frozen OLR flow features")
    days = [row["date"] for row in rows]
    if len(set(days)) != len(days) or days != sorted(days):
        raise ValueError("Duplicate or unsorted daily observations")
    if days[-1] != end.isoformat():
        raise DailyInputError("Latest price date differs from requested completed session; resolve source or suspension")
    for row in rows:
        if any(not math.isfinite(float(row[key])) for key in ("open", "high", "low", "close", "volume")):
            raise ValueError("Nonfinite daily observation")
        if row["volume"] < 0 or row["low"] > min(row["open"], row["close"]) or row["high"] < max(row["open"], row["close"]):
            raise ValueError("Inconsistent daily OHLCV")
    if validation_start is None:
        return days[-FLOW_OBSERVATIONS:]
    prior = [day for day in days if day < validation_start.isoformat()]
    if len(prior) < FLOW_OBSERVATIONS:
        raise DailyInputError("Insufficient pre-validation flow warm-up")
    # Later downloads must retain the beginning of the prospective period too;
    # a rolling 40-row tail alone would invalidate the final continuous replay.
    return [day for day in days if day >= prior[-FLOW_OBSERVATIONS]]


def collect(contract_path, end, destination, workers=2, resume=False):
    contract = verified_contract(contract_path)
    if end >= datetime.now(ZoneInfo("Asia/Seoul")).date():
        raise ValueError("Use a completed prior KST date; final investor data may arrive after the close")
    study = load_study(Path(contract_path).parent / "study.json")
    if study["study_hash"] != contract["study_hash"]:
        raise ValueError("Wrong frozen study")
    engine = Path(study["engine_root"])
    for name, expected in contract["frozen_inputs"].items():
        if digest(engine / name) != expected:
            raise ValueError("Frozen universe or sector file changed")
    symbols = lrs.load_universe(engine / "config/universe_103.yaml")
    destination = Path(destination).resolve()
    if resume:
        prior = json.loads((destination / "acquisition_status.json").read_text())
        if (prior["contract_hash"] != contract["contract_hash"] or prior["as_of"] != end.isoformat()
                or prior["status"] != "incomplete_no_artifact_publication" or (destination / "daily").exists()):
            raise ValueError("Resume only an incomplete acquisition of this exact date and contract")
        attempts = destination / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        attempts.mkdir(parents=True)
        import shutil
        shutil.copyfile(destination / "acquisition_status.json", attempts / "acquisition_status.json")
        # Preserve any response versions belonging to failed symbols before retry.
        for symbol in prior["errors"]:
            for source in (destination / "provider_flows").glob(symbol + "_*.json"):
                shutil.copyfile(source, attempts / source.name)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        prior = {"completed": {}}
    start = end - timedelta(days=420)
    status = {"status": "running", "contract_hash": contract["contract_hash"], "as_of": end.isoformat(),
              "price_start": start.isoformat(), "symbols": len(symbols), "flow_observations_required": FLOW_OBSERVATIONS,
              "flow_units": "net_shares", "flow_source": FLOW_SOURCE, "performance_evaluated": False,
              "errors": {}, "completed": prior["completed"], "script_sha256": digest(__file__)}
    write(destination / "acquisition_status.json", status)
    from scripts.update_kis_intraday import _load_dotenv, KoreaInvestAPI, KoreaInvestEnv, build_kis_config_from_env, configure_kis_request_rate
    _load_dotenv(ROOT / ".env")
    configure_kis_request_rate(2.0)
    api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    stock = lrs.ensure_pykrx()
    if stock is None:
        raise RuntimeError("Daily price provider unavailable")
    connection = lrs.init_db(destination / "lrs.db")
    lrs.upsert_sector_map(connection, lrs.load_sector_map(engine / "config/olr/sector_map.yaml"))

    def one(symbol):
        raw = stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), symbol, adjusted=True)
        raw_path = destination / "provider_prices" / (symbol + ".parquet")
        raw_path.parent.mkdir(exist_ok=True)
        raw.to_parquet(raw_path)
        prices = lrs.normalize_ohlcv_df(raw)
        required = validate_price_rows(prices, end, date.fromisoformat(contract["earliest_eligible_kst_date"]))
        flows = fetch_flow(api, symbol, required, end, destination / "provider_flows")
        return prices, flows

    try:
        for code, provider_code in (("KOSPI", "1001"), ("KOSDAQ", "2001")):
            lrs.backfill_index(connection, stock, code, provider_code, start, end)
            last = connection.execute("SELECT MAX(date) FROM index_ohlcv WHERE index_code=?", (code,)).fetchone()[0]
            if last != end.isoformat():
                status["errors"][code] = "Index does not reach requested completed session"
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
            futures = {pool.submit(one, symbol): symbol for symbol in symbols if symbol not in status["completed"]}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    prices, flows = future.result()
                    connection.executemany("INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?)",
                        [(symbol, row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"]) for row in prices])
                    connection.executemany("INSERT INTO daily_flow VALUES (?,?,?,?)",
                        [(symbol, row["date"], row["foreign_net"], row["inst_net"]) for row in flows])
                    connection.commit()
                    status["completed"][symbol] = {"prices": len(prices), "flows": len(flows), "latest": prices[-1]["date"]}
                except Exception as exc:
                    connection.rollback()
                    # Source error text is deliberately omitted: network exceptions may contain credentials.
                    status["errors"][symbol] = str(exc) if isinstance(exc, DailyInputError) else type(exc).__name__
                write(destination / "acquisition_status.json", status)
                print(f"Daily inputs {len(status['completed'])}/{len(symbols)}, errors {len(status['errors'])}", flush=True)
    finally:
        connection.close()
    if not status["errors"] and len(status["completed"]) == len(symbols):
        lrs.export_lrs_sqlite_to_parquet(destination / "lrs.db", destination / "daily", start=start, end=end,
            universe=symbols, source_label="olr_forward_pykrx_adjusted_prices_kis_net_shares")
        status["status"] = "daily_inputs_ready"
    else:
        status["status"] = "incomplete_no_artifact_publication"
    write(destination / "acquisition_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Retry only failed inputs; preserve prior failures and response versions")
    args = parser.parse_args()
    result = collect(args.contract, args.as_of, args.destination, args.workers, args.resume)
    print(json.dumps({key: result[key] for key in ("status", "as_of", "symbols", "errors", "performance_evaluated")}), flush=True)
    return 0 if result["status"] == "daily_inputs_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
