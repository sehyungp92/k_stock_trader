"""Source-only repair generation. Frozen portfolio outcomes remain unopened."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.portfolio_targeted_validation import ROOT as PARENT, read, bound
from backtests.strategies.kalcb.executed_value import write_json

ROOT = Path.home() / ".codex/research/portfolio_source_repair_20261009"


def collect(root, *, volumes=False):
    from scripts.update_kis_intraday import _load_dotenv, KST, fetch_kis_1m_bars
    from kis_core import KoreaInvestAPI, KoreaInvestEnv, build_kis_config_from_env
    from loguru import logger
    root.mkdir(parents=True, exist_ok=True)
    registration = root / "repair_registration.json"
    if not registration.exists():
        write_json(registration, {
            "version": "portfolio-source-repair-v1", "registered_at": datetime.now(KST).isoformat(),
            "protocol": bound(PARENT / "protocol.json"),
            "selection": bound(PARENT / "development_selection.json"),
            "prior_incident": bound(PARENT / "source_repair_diagnosis.json"),
            "initial_equity": 25_000_000, "window": ["2026-06-01", "2026-07-10"],
            "comparisons": ["control", "incumbent", "k_marginal"],
            "scenarios": ["baseline", "combined"],
            "candidate_reselection_permitted": False, "oos_performance_opened": False,
            "repair_rule": "Original integer prices/shares uniquely supported by retained anchors, rounded source observations and turnover; no invented bars or residual allocation.",
        })
    _load_dotenv(REPO / ".env"); logger.remove()
    api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    jobs = [("068270", day) for day in ("2026-06-01", "2026-06-02")]
    if volumes:
        jobs = [(row["symbol"], row["date"]) for row in read(PARENT / "source_basis_review.json")["exceptions"]
                if row["classification"] == "volume_only_revision"]
    for symbol, day in jobs:
        folder = root / "minute_sources" / symbol / day
        folder.mkdir(parents=True, exist_ok=True)
        completion = folder / "completion.json"
        if completion.exists():
            continue
        first = datetime.fromisoformat(day + "T09:00:00").replace(tzinfo=KST)
        last = datetime.fromisoformat(day + "T15:35:00").replace(tzinfo=KST)
        frame, pages = fetch_kis_1m_bars(api, symbol, first, last, market_code="J",
            extra_sleep_sec=.1, max_pages=8, raw_page_root=folder / "raw")
        frame.to_parquet(folder / "current_1m.parquet", index=False)
        write_json(completion, {"rows": len(frame), "pages": pages,
            "queried_at": datetime.now(KST).isoformat(), "source": bound(folder / "current_1m.parquet"),
            "receipts": [bound(p) for p in (folder / "raw").rglob("*.json")]})
        print(json.dumps({"symbol": symbol, "collected": day, "rows": len(frame), "pages": pages}), flush=True)


def repair_sessions(root):
    import pandas as pd
    from datetime import date
    from backtests.core.anchored_basis_repair import reconstruct_original_session
    from backtests.core.adjusted_recovery import DAILY_FIELDS
    from scripts.update_kis_intraday import aggregate_timeframe
    original_path = REPO / "data/kis_intraday_parquet/068270/068270_1m_20250529_20260529.parquet"
    original = pd.read_parquet(original_path)
    original = original[original.timestamp.dt.strftime("%Y-%m-%d") == "2026-05-29"].reset_index(drop=True)
    adjusted = pd.read_parquet(PARENT / "warmup_sources/current_1m.parquet")
    outputs = root / "repaired_sessions"; outputs.mkdir(parents=True, exist_ok=False)
    proofs = []
    for day in ("2026-05-29", "2026-06-01", "2026-06-02"):
        folder = PARENT / "warmup_sources" if day == "2026-05-29" else root / "minute_sources/068270" / day
        raw = pd.read_parquet(folder / "current_1m.parquet")
        controls, bindings = [], [bound(original_path), bound(PARENT / "warmup_sources/current_1m.parquet"), bound(folder / "current_1m.parquet")]
        for label in ("original", "adjusted"):
            path = PARENT / "provider_basis_audit" / ("068270_" + label + ".json")
            receipt = read(path); bindings.append(bound(path))
            if (receipt["adjusted"] is not (label == "adjusted")
                    or receipt["params"]["FID_ORG_ADJ_PRC"] != ("0" if label == "adjusted" else "1")):
                raise ValueError("Wrong daily control convention")
            rows = [x for x in receipt["rows"] if x["stck_bsop_date"] == day.replace("-", "")]
            if len(rows) != 1: raise ValueError("Ambiguous daily control")
            controls.append({**{k: int(rows[0][v]) for k, v in DAILY_FIELDS.items()}, "turnover": int(rows[0]["acml_tr_pbmn"])})
        provider = []
        for path in (folder / ("provider_responses" if day == "2026-05-29" else "raw")).rglob("*.json"):
            provider.extend(read(path)["rows"]); bindings.append(bound(path))
        decoded, proof = reconstruct_original_session(raw, original, adjusted, *controls, provider, date.fromisoformat(day), "068270")
        files = {}
        for tf in ("1m", "5m", "15m", "30m", "1h"):
            path = outputs / ("068270_" + day + "_" + tf + ".parquet")
            aggregate_timeframe(decoded, tf).to_parquet(path, index=False); files[tf] = bound(path)
        proofs.append({**proof, "inputs": bindings, "outputs": files})
    write_json(outputs / "proof.json", {"pass": True, "sessions": proofs,
        "program": bound(Path(__file__)), "verifier": bound(REPO / "backtests/core/anchored_basis_repair.py"),
        "performance_opened": False})
    print(json.dumps({"repaired_sessions": [{k: x[k] for k in ("date", "rows", "methods", "daily_excess_shares")} for x in proofs]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "collect-volumes", "repair-sessions"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {"collect": collect, "collect-volumes": lambda root: collect(root, volumes=True), "repair-sessions": repair_sessions}[args.command](args.root)
