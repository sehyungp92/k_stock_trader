"""One content identity and file-selection policy for OLR and KALCB inputs."""
from pathlib import Path
from datetime import date

from kis_core.input_manifest import active_manifest, signature


def market_input_contract(manifest=None):
    manifest = manifest or active_manifest()
    if manifest is None:
        return None
    contract = manifest.payload.get("evidence", {}).get("shared_market_inputs")
    if contract is None:
        return None
    content = {key: value for key, value in contract.items() if key != "market_input_hash"}
    if signature(content) != contract.get("market_input_hash"):
        raise ValueError("Shared market input identity mismatch")
    for name, digest in contract["files"].items():
        if manifest.payload["files"].get(name) != digest:
            raise ValueError(f"Shared market input is not bound to the manifest: {name}")
    return contract


def selected_intraday_paths(data_root, symbol, timeframe):
    """None preserves legacy fixtures; a bound generation never falls back to globbing."""
    manifest = active_manifest()
    contract = market_input_contract(manifest)
    if contract is None:
        return None
    if timeframe != contract["timeframe"]:
        raise ValueError("Timeframe differs from shared market inputs")
    expected_root = (manifest.root / contract["data_root"]).resolve()
    if manifest.resolve(data_root) != expected_root:
        raise ValueError("Intraday root differs from shared market inputs")
    names = contract["intraday_files"].get(str(symbol).zfill(6))
    if not names:
        raise FileNotFoundError(f"No shared intraday input for {symbol}")
    paths = [manifest.root / name for name in names]
    if any(name not in contract["files"] for name in names):
        raise ValueError("Selected intraday file is absent from shared market identity")
    return paths


def bind_market_config(config, manifest):
    contract = market_input_contract(manifest)
    if contract is None:
        return config
    result = dict(config)
    for key in ("data_root", "daily_data_root"):
        expected = (manifest.root / contract[key]).resolve()
        if key in result and manifest.resolve(result[key]) != expected:
            raise ValueError(f"{key} differs from shared market inputs")
        result[key] = str(expected)
    expected_sector = (manifest.root / contract["sector_map_path"]).resolve()
    if result.get("sector_map_path") and manifest.resolve(result["sector_map_path"]) != expected_sector:
        raise ValueError("Sector map differs from shared market inputs")
    result["sector_map_path"] = str(expected_sector)
    if result.get("sector_map"):
        raise ValueError("Shared market inputs require their declared sector map file")
    # Subwindows support training/holdout evaluation, but cannot silently expand
    # the immutable market contract's declared date coverage.
    lower, upper = date.fromisoformat(contract["train_start"]), date.fromisoformat(contract["data_end"])
    dates = {key: result[key] for key in ("start", "end", "holdout_start", "data_snapshot_end") if result.get(key)}
    dates.update({f"date_range.{k}": v for k, v in (result.get("date_range") or {}).items() if v})
    for key, value in dates.items():
        day = date.fromisoformat(str(value)[:10])
        if not lower <= day <= upper:
            raise ValueError(f"{key} is outside shared market input coverage")
    result["market_input_hash"] = contract["market_input_hash"]
    result.setdefault("holdout_start", contract["holdout_start"])
    result.setdefault("data_snapshot_end", contract["data_end"])
    # Both strategies may choose different candidates, but from one declared universe.
    expected_universe = (manifest.root / contract["universe"]).resolve()
    universe = result.get("universe")
    if isinstance(universe, (str, Path)):
        candidate = Path(universe)
        if not candidate.is_absolute():
            candidate = Path(manifest.payload["source_root"]) / (candidate if candidate.parts[0] == "config" else Path("config") / candidate)
        if manifest.resolve(candidate) != expected_universe:
            raise ValueError("Universe differs from shared market inputs")
    elif universe is not None:
        raise ValueError("Shared market inputs require their declared universe file")
    result["universe"] = str(expected_universe)
    return result


def market_bar_values(bar):
    return (bar.timeframe, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.is_completed)


def compare_consumed_bars(kalcb_bars, olr_bars):
    """Different symbol coverage is allowed; overlapping market observations must agree."""
    seen = {}
    hashes = {}
    keys = {}
    for strategy, bars in (("KALCB", kalcb_bars), ("OLR", olr_bars)):
        rows = []
        keys[strategy] = set()
        for bar in bars:
            key = (bar.timestamp.isoformat(), bar.symbol)
            values = market_bar_values(bar)
            if key in seen and seen[key] != values:
                raise ValueError(f"Conflicting shared market bar: {key}")
            seen[key] = values
            keys[strategy].add(key)
            rows.append((key, values))
        hashes[strategy] = {"bar_count": len(rows), "market_bar_hash": signature(sorted(rows))}
    return {"overlapping_bars": len(keys["KALCB"] & keys["OLR"]), "conflicting_bars": 0,
            "strategies": hashes}
