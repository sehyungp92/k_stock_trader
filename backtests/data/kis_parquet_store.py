from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from backtests.auto.shared.cache_keys import fingerprint_paths
from strategy_common.market import MarketBar


@dataclass(frozen=True, slots=True)
class KISParquetStore:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def discover_files(
        self,
        *,
        symbols: Iterable[str] | None = None,
        timeframes: Iterable[str] | None = None,
    ) -> list[Path]:
        files = sorted(self.root.rglob("*.parquet"))
        symbol_filter = {str(item) for item in symbols or []}
        timeframe_filter = {item.lower() for item in timeframes or []}
        selected: list[Path] = []
        for path in files:
            symbol = path.parent.name
            timeframe = _timeframe_from_path(path)
            if symbol_filter and symbol not in symbol_filter:
                continue
            if timeframe_filter and timeframe not in timeframe_filter:
                continue
            selected.append(path)
        return selected

    def fingerprint(self, files: Iterable[Path] | None = None) -> str:
        return fingerprint_paths(files or self.discover_files(), root=self.root)

    def load_bars(
        self,
        symbol: str,
        timeframe: str = "1m",
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[MarketBar]:
        files = self.discover_files(symbols=[symbol], timeframes=[timeframe])
        frames: list[pd.DataFrame] = []
        for path in files:
            frame = pd.read_parquet(path)
            frame["__source_path"] = str(path)
            frames.append(frame)
        if not frames:
            return []
        data = pd.concat(frames, ignore_index=True)
        if "timestamp" not in data.columns:
            raise ValueError("KIS parquet data must contain a timestamp column")
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        data = data.drop_duplicates(subset=["timestamp"], keep="last")
        if start is not None:
            data = data[data["timestamp"].dt.date >= start]
        if end is not None:
            data = data[data["timestamp"].dt.date <= end]
        fingerprint = self.fingerprint(files)
        bars: list[MarketBar] = []
        for row in data.itertuples(index=False):
            values = row._asdict()
            ts = values["timestamp"]
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=ts,
                    timeframe=timeframe,
                    open=float(values["open"]),
                    high=float(values["high"]),
                    low=float(values["low"]),
                    close=float(values["close"]),
                    volume=float(values.get("volume", 0.0)),
                    source="kis_parquet",
                    source_fingerprint=fingerprint,
                    metadata={"source_path": values.get("__source_path", "")},
                )
            )
        return bars


def _timeframe_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1].lower() if len(parts) >= 2 else ""
