from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_VERSION = "krx-daily-lrs-parquet-v1"


SQLITE_TABLE_COLUMNS = {
    "daily_ohlcv": ["ticker", "date", "open", "high", "low", "close", "volume"],
    "daily_flow": ["ticker", "date", "foreign_net", "inst_net"],
    "index_ohlcv": ["index_code", "date", "open", "high", "low", "close", "volume"],
    "fx_rates": ["pair", "date", "close"],
    "sector_map": ["ticker", "sector"],
}

TABLE_COLUMNS = {
    **SQLITE_TABLE_COLUMNS,
    "daily_foreign_flow": ["ticker", "date", "foreign_net"],
    "daily_institutional_flow": ["ticker", "date", "institutional_net"],
}


def export_lrs_sqlite_to_parquet(
    db_path: str | Path,
    output_root: str | Path,
    *,
    start: date | None = None,
    end: date | None = None,
    universe: list[str] | None = None,
    source_label: str = "nulrimok_lrs_sqlite",
) -> dict[str, Any]:
    """Export historical LRS tables to a strategy-neutral parquet mirror."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    db = Path(db_path)
    universe_set = {str(symbol).zfill(6) for symbol in universe or []}
    source_frames = {
        table: _read_table(db, table, start=start, end=end, universe=universe_set)
        for table in SQLITE_TABLE_COLUMNS
    }
    frames = {
        **source_frames,
        "daily_foreign_flow": _foreign_flow_frame(source_frames["daily_flow"]),
        "daily_institutional_flow": _institutional_flow_frame(source_frames["daily_flow"]),
    }

    _clear_managed_outputs(root, frames.keys())

    paths: dict[str, Any] = {}
    for table, frame in frames.items():
        paths[table] = _write_table(root, table, frame)
    paths["daily_ohlcv_by_symbol"] = _write_grouped(
        root,
        "daily_ohlcv",
        frames["daily_ohlcv"],
        key_col="ticker",
        file_label="daily_ohlcv",
    )
    paths["daily_flow_by_symbol"] = _write_grouped(
        root,
        "daily_flow",
        frames["daily_flow"],
        key_col="ticker",
        file_label="daily_flow",
    )
    paths["daily_foreign_flow_by_symbol"] = _write_grouped(
        root,
        "daily_foreign_flow",
        frames["daily_foreign_flow"],
        key_col="ticker",
        file_label="daily_foreign_flow",
    )
    paths["daily_institutional_flow_by_symbol"] = _write_grouped(
        root,
        "daily_institutional_flow",
        frames["daily_institutional_flow"],
        key_col="ticker",
        file_label="daily_institutional_flow",
    )
    paths["index_ohlcv_by_code"] = _write_grouped(
        root,
        "index_ohlcv",
        frames["index_ohlcv"],
        key_col="index_code",
        file_label="daily_ohlcv",
    )

    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_label": source_label,
        "source_db_path": str(db.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "start": start.isoformat() if start else _min_date(frames),
        "end": end.isoformat() if end else _max_date(frames),
        "tables": {table: _table_summary(frame) for table, frame in frames.items()},
        "source_fingerprint": _dataset_fingerprint(frames),
        "paths": paths,
        "usage": {
            "daily_ohlcv": "load_daily_ohlcv(root, symbol) or tables/daily_ohlcv.parquet",
            "daily_flow": "load_daily_flow(root, symbol) or tables/daily_flow.parquet",
            "daily_foreign_flow": "load_daily_foreign_flow(root, symbol) or tables/daily_foreign_flow.parquet",
            "daily_institutional_flow": "load_daily_institutional_flow(root, symbol) or tables/daily_institutional_flow.parquet",
            "index_ohlcv": "load_index_ohlcv(root, index_code) or tables/index_ohlcv.parquet",
            "sector_map": "load_sector_map(root) or tables/sector_map.parquet",
        },
    }
    _atomic_write_text(root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return manifest


def load_daily_ohlcv(root: str | Path, symbol: str | None = None, *, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    frame = _load_group_or_table(root, "daily_ohlcv", "ticker", symbol, "daily_ohlcv")
    return _filter_dates(frame, start, end)


def load_daily_flow(root: str | Path, symbol: str | None = None, *, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    frame = _load_group_or_table(root, "daily_flow", "ticker", symbol, "daily_flow")
    return _filter_dates(frame, start, end)


def load_daily_foreign_flow(root: str | Path, symbol: str | None = None, *, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    frame = _load_group_or_table(root, "daily_foreign_flow", "ticker", symbol, "daily_foreign_flow")
    return _filter_dates(frame, start, end)


def load_daily_institutional_flow(root: str | Path, symbol: str | None = None, *, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    frame = _load_group_or_table(root, "daily_institutional_flow", "ticker", symbol, "daily_institutional_flow")
    return _filter_dates(frame, start, end)


def load_index_ohlcv(root: str | Path, index_code: str | None = None, *, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    frame = _load_group_or_table(root, "index_ohlcv", "index_code", index_code, "daily_ohlcv")
    return _filter_dates(frame, start, end)


def load_sector_map(root: str | Path) -> dict[str, str]:
    path = Path(root) / "tables" / "sector_map.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    return {str(row.ticker).zfill(6): str(row.sector) for row in frame.itertuples(index=False)}


def available_daily_symbols(root: str | Path) -> list[str]:
    base = Path(root) / "daily_ohlcv"
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def load_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_table(
    db_path: Path,
    table: str,
    *,
    start: date | None,
    end: date | None,
    universe: set[str],
) -> pd.DataFrame:
    columns = TABLE_COLUMNS[table]
    if not db_path.exists():
        return pd.DataFrame(columns=columns)
    clauses: list[str] = []
    params: list[Any] = []
    if "date" in columns:
        if start is not None:
            clauses.append("date >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("date <= ?")
            params.append(end.isoformat())
    if universe and "ticker" in columns:
        placeholders = ",".join("?" for _ in sorted(universe))
        clauses.append(f"ticker IN ({placeholders})")
        params.extend(sorted(universe))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT {', '.join(columns)} FROM {table}{where}"
    order_cols = [col for col in ("ticker", "index_code", "pair", "date") if col in columns]
    if order_cols:
        query = f"{query} ORDER BY {', '.join(order_cols)}"
    with sqlite3.connect(db_path) as conn:
        try:
            frame = pd.read_sql_query(query, conn, params=params)
        except (sqlite3.DatabaseError, pd.errors.DatabaseError):
            frame = pd.DataFrame(columns=columns)
    return _normalize_frame(frame, table)


def _normalize_frame(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in TABLE_COLUMNS[table]:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[TABLE_COLUMNS[table]]
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    return frame


def _foreign_flow_frame(daily_flow: pd.DataFrame) -> pd.DataFrame:
    if daily_flow.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS["daily_foreign_flow"])
    return daily_flow[["ticker", "date", "foreign_net"]].copy()


def _institutional_flow_frame(daily_flow: pd.DataFrame) -> pd.DataFrame:
    if daily_flow.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS["daily_institutional_flow"])
    frame = daily_flow[["ticker", "date", "inst_net"]].copy()
    return frame.rename(columns={"inst_net": "institutional_net"})


def _clear_managed_outputs(root: Path, tables: Any) -> None:
    table_set = set(tables)
    tables_dir = root / "tables"
    for table in table_set:
        table_path = tables_dir / f"{table}.parquet"
        if table_path.exists():
            table_path.unlink()
    for grouped_dir in (
        "daily_ohlcv",
        "daily_flow",
        "daily_foreign_flow",
        "daily_institutional_flow",
        "index_ohlcv",
    ):
        path = root / grouped_dir
        if grouped_dir in table_set and path.exists():
            shutil.rmtree(path)


def _write_table(root: Path, table: str, frame: pd.DataFrame) -> str:
    path = root / "tables" / f"{table}.parquet"
    _write_parquet(frame, path)
    return str(path)


def _write_grouped(root: Path, table: str, frame: pd.DataFrame, *, key_col: str, file_label: str) -> dict[str, str]:
    if frame.empty:
        return {}
    out: dict[str, str] = {}
    for key, group in frame.groupby(key_col, sort=True):
        key_text = str(key).zfill(6) if key_col == "ticker" else str(key)
        start, end = _frame_date_range(group)
        filename = f"{key_text}_{file_label}_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
        path = root / table / key_text / filename
        _write_parquet(group.reset_index(drop=True), path)
        out[key_text] = str(path)
    return out


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_group_or_table(root: str | Path, table: str, key_col: str, key: str | None, file_label: str) -> pd.DataFrame:
    base = Path(root)
    if key:
        key_text = str(key).zfill(6) if key_col == "ticker" else str(key)
        files = sorted((base / table / key_text).glob(f"{key_text}_{file_label}_*.parquet"))
        if files:
            return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        return pd.DataFrame(columns=TABLE_COLUMNS[table])
    path = base / "tables" / f"{table}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=TABLE_COLUMNS[table])


def _filter_dates(frame: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["date"]).dt.date
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return frame.loc[mask].reset_index(drop=True)


def _table_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(frame))}
    if "date" in frame.columns and not frame.empty:
        summary["min_date"], summary["max_date"] = _frame_date_range(frame)
    for column in ("ticker", "index_code", "pair"):
        if column in frame.columns:
            summary[f"{column}_count"] = int(frame[column].nunique())
    return summary


def _frame_date_range(frame: pd.DataFrame) -> tuple[str, str]:
    values = pd.to_datetime(frame["date"]).dt.date.astype(str)
    return str(values.min()), str(values.max())


def _min_date(frames: dict[str, pd.DataFrame]) -> str | None:
    dates = [summary["min_date"] for summary in (_table_summary(frame) for frame in frames.values()) if "min_date" in summary]
    return min(dates) if dates else None


def _max_date(frames: dict[str, pd.DataFrame]) -> str | None:
    dates = [summary["max_date"] for summary in (_table_summary(frame) for frame in frames.values()) if "max_date" in summary]
    return max(dates) if dates else None


def _dataset_fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    payload = {
        table: {
            "columns": list(frame.columns),
            "rows": int(len(frame)),
            "data_hash": _frame_hash(frame),
        }
        for table, frame in sorted(frames.items())
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    normalized = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    raw = normalized.to_json(orient="records", date_format="iso")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export LRS SQLite daily data to shared parquet")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output-root", default="data/krx_daily_parquet")
    parser.add_argument("--from", dest="start", default=None)
    parser.add_argument("--to", dest="end", default=None)
    args = parser.parse_args(argv)
    manifest = export_lrs_sqlite_to_parquet(
        args.db_path,
        args.output_root,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
