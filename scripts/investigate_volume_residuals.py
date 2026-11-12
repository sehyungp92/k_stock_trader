"""Collect complete quote streams for sparse sessions; never alter market bars."""
import argparse
from datetime import datetime, time
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.update_kis_intraday import (
    KST, KoreaInvestAPI, KoreaInvestEnv, _load_dotenv, build_kis_config_from_env,
    fetch_kis_1m_bars,
)


def collect(inventory, destination):
    destination = destination.resolve()
    rows = json.loads(inventory.read_text(encoding="utf-8"))["sessions"]
    _load_dotenv(ROOT / ".env")
    from loguru import logger
    logger.remove()
    api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    for row in rows:
        if row["disposition"] != "repeat_source_sparse_daily_unreconciled":
            continue
        symbol, day = row["symbol"], datetime.fromisoformat(row["date"]).date()
        folder = destination / "full_minutes" / symbol / day.isoformat()
        receipt = folder / "receipt.json"
        if receipt.exists() and json.loads(receipt.read_text())["status"] == "ok":
            continue
        folder.mkdir(parents=True, exist_ok=True)
        start, end = (datetime.combine(day, t, KST) for t in (time(8), time(18, 10)))
        result = {"symbol": symbol, "date": day.isoformat(), "start": start.isoformat(),
                  "end": end.isoformat(), "queried_at": datetime.now(KST).isoformat(),
                  "quotation_only": True}
        try:
            if (folder / "raw_1m.parquet").exists():
                frame = pd.read_parquet(folder / "raw_1m.parquet")
                pages = len(list((folder / "provider_responses").rglob("*.json")))
            else:
                frame, pages = fetch_kis_1m_bars(api, symbol, start, end, market_code="J",
                    extra_sleep_sec=.6, max_pages=8, raw_page_root=folder / "provider_responses")
            frame.to_parquet(folder / "raw_1m.parquet", index=False)
            result.update(status="ok", rows=len(frame), pages=pages,
                          earliest=str(frame.timestamp.min()), latest=str(frame.timestamp.max()))
        except Exception as exc:
            result.update(status="query_failed", error_type=type(exc).__name__)
        result["files"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in folder.rglob("*") if p.is_file() and p != receipt}
        receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: result.get(k) for k in ("symbol", "date", "status", "rows", "pages")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    collect(args.inventory, args.destination)
