"""Observer-only price-path attribution for completed, audited OLR trades.

Uses realised positions and frozen market bytes. It neither modifies selection
nor treats hindsight path components as an executable alternative strategy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import gc
from pathlib import Path
import sys

try:
    from scripts.olr_rule_study import digest, load_study, read, write
    from scripts.olr_rule_analysis import rows
except ModuleNotFoundError:
    from olr_rule_study import digest, load_study, read, write
    from olr_rule_analysis import rows


def fill_lifecycles(fills):
    """Inventory-based entry lifecycles, preserving actual broker fill order."""
    active, completed = {}, {}
    for fill in fills:
        symbol = fill["symbol"]
        if symbol not in active:
            if fill["side"] != "BUY":
                raise ValueError("Exit without opening inventory")
            active[symbol] = {"qty": 0, "first": fill["timestamp"], "fills": []}
        state = active[symbol]
        state["fills"].append(fill)
        state["qty"] += fill["qty"] if fill["side"] == "BUY" else -fill["qty"]
        if state["qty"] < 0:
            raise ValueError("Fill lifecycle became short")
        if state["qty"] == 0:
            key = (symbol, state["first"])
            if key in completed:
                raise ValueError("Ambiguous lifecycle identity")
            completed[key] = state["fills"]
            del active[symbol]
    return completed


def decompose_filled_trade(trade, sessions, fills):
    """Price-path attribution with actual inventory across every overnight gap."""
    by_day = defaultdict(list)
    for fill in fills:
        by_day[fill["timestamp"][:10]].append(fill)
    qty, previous_close = 0, None
    afternoon = overnight = continuation = 0.0
    for index, session in enumerate(sessions):
        start_qty = qty
        if previous_close is not None:
            overnight += start_qty * (session["open"] - previous_close)
        cashflow = 0.0
        for fill in by_day.pop(session["date"], []):
            if fill["side"] == "BUY":
                qty += fill["qty"]
                cashflow -= fill["qty"] * fill["price"]
            elif fill["side"] == "SELL":
                qty -= fill["qty"]
                cashflow += fill["qty"] * fill["price"]
            else:
                raise ValueError("Unsupported fill side")
            if qty < 0:
                raise ValueError("Invalid intraday inventory")
        intraday = qty * session["close"] - start_qty * session["open"] + cashflow
        if index == 0:
            afternoon += intraday
        else:
            continuation += intraday
        previous_close = session["close"]
    if by_day or qty != 0 or sum(f["qty"] for f in fills if f["side"] == "SELL") != trade["qty"]:
        raise ValueError("Closed trade and fill lifecycle coverage differ")
    if abs(afternoon + overnight + continuation - trade["gross_pnl"]) > .001:
        raise ValueError("Actual fill cashflows do not reconcile to trade gross PnL")
    return {"entry_session_pnl_krw": afternoon, "overnight_gap_pnl_krw": overnight,
            "subsequent_intraday_pnl_krw": continuation,
            "explicit_fees_krw": trade["gross_pnl"] - trade["net_pnl"], "net_pnl_krw": trade["net_pnl"]}


def analyse(study_path, names):
    spec = load_study(study_path)
    engine_root = Path(spec["engine_root"])
    sys.path.insert(0, str(engine_root))
    import os
    os.chdir(engine_root)
    from kis_core.input_manifest import InputManifest, _VERIFIED_BYTES
    from backtests.strategies.olr.research_sweep import _load_symbol_frame, _FRAME_CACHE
    from kis_core.trading_calendar import get_trading_calendar
    generation = InputManifest(spec["input_manifest"])
    selected, lifecycles = {}, {}
    source_hashes = {}
    for name in names:
        path = Path(spec["output"]) / "cases" / name / "nominal" / "trades.jsonl.gz"
        receipt = read(path.parent / "completed.json")
        source_hashes[name] = digest(path)
        if receipt["outputs"][path.name] != source_hashes[name]:
            raise ValueError("Completed trade export changed")
        selected[name] = rows(path)
        fill_path = path.parent / "fills.jsonl.gz"
        if digest(fill_path) != receipt["outputs"][fill_path.name]:
            raise ValueError("Completed fill export changed")
        source_hashes[name + "/fills"] = digest(fill_path)
        lifecycles[name] = fill_lifecycles(rows(fill_path))
        if set(lifecycles[name]) != {(t["symbol"], t["entry_fill_time"]) for t in selected[name]}:
            raise ValueError("Closed trade identities differ from reconstructed fill lifecycles")
    by_symbol = defaultdict(list)
    for name, trades in selected.items():
        for trade in trades:
            by_symbol[trade["symbol"]].append((name, trade))
    attributed, missing = defaultdict(list), []
    with generation.activate():
        calendar = get_trading_calendar()
        for symbol, trades in sorted(by_symbol.items()):
            end = max(date.fromisoformat(t["exit_fill_time"][:10]) for _, t in trades)
            frame = _load_symbol_frame(generation.root / "data/kis_intraday_parquet", symbol, "5m", end)
            session_rows = {}
            for day, group in frame.groupby(frame.timestamp.dt.date):
                session_rows[day] = {"date": day.isoformat(), "open": float(group.iloc[0]["open"]),
                                     "close": float(group.iloc[-1]["close"])}
            for name, trade in trades:
                cursor = date.fromisoformat(trade["entry_fill_time"][:10])
                last = date.fromisoformat(trade["exit_fill_time"][:10])
                expected = []
                while cursor <= last:
                    expected.append(cursor)
                    cursor = calendar.next_trading_day(cursor)
                if any(day not in session_rows for day in expected):
                    missing.append({"case": name, "symbol": symbol, "entry": trade["entry_fill_time"],
                                    "reason": "missing_held_session_mark"})
                    continue
                parts = decompose_filled_trade(trade, [session_rows[day] for day in expected],
                                               lifecycles[name][(symbol, trade["entry_fill_time"])])
                attributed[name].append({"symbol": symbol, "entry": trade["entry_fill_time"],
                    "sector": trade["route_metadata"].get("sector"),
                    "band": trade["route_metadata"].get("afternoon_score_band_rule", "no_band"), **parts})
            del frame
            _FRAME_CACHE.clear()
            _VERIFIED_BYTES.clear()
            gc.collect()
    summary = {}
    components = ("entry_session_pnl_krw", "overnight_gap_pnl_krw", "subsequent_intraday_pnl_krw", "explicit_fees_krw", "net_pnl_krw")
    for name in names:
        rr = attributed[name]
        summary[name] = {"attributed_trades": len(rr), "closed_trades": len(selected[name]),
                         **{key: sum(r[key] for r in rr) for key in components}}
        summary[name]["bands"] = {
            band: {"trades": sum(r["band"] == band for r in rr),
                   **{key: sum(r[key] for r in rr if r["band"] == band) for key in components}}
            for band in sorted({r["band"] for r in rr})}
    result = {"study_hash": spec["study_hash"], "observer_sha256": digest(__file__), "trade_export_sha256": source_hashes,
              "status": "complete" if not missing else "partial", "summary": summary,
              "missing": missing, "trades": dict(attributed),
              "interpretation": "Accounting decomposition using actual fill inventory across each night, first observed session open / last observed close. Components are gross except explicit fees and net total. Entry/exit slippage remains in intraday components. Not benchmark-relative alpha, independent observations, or a trading alternative."}
    target = Path(spec["output"]) / "analysis/price_path_attribution.json"
    write(target, result)
    print("Wrote", target, "coverage", result["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--cases", nargs="+", default=["control"])
    args = parser.parse_args()
    analyse(args.study, args.cases)


if __name__ == "__main__":
    main()
