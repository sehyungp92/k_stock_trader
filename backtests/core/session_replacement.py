"""Explicit replacement of an entire session from a retained immutable archive."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

import pandas as pd

from kis_core.input_manifest import InputManifest, signature
from strategy_common.session_coverage import session_bar_coverage


def apply_session_replacements(root, selected, bundle_path):
    root, bundle_path = Path(root).resolve(), Path(bundle_path).resolve()
    if not bundle_path.is_relative_to(root):
        raise ValueError("Replacement bundle must be inside the research root")
    proof = json.loads(bundle_path.read_text(encoding="utf-8"))
    archive = InputManifest(bundle_path.parent / "manifest.json")
    if archive.identity != proof["source_manifest_hash"]:
        raise ValueError("Replacement archive identity mismatch")
    paths = {bundle_path, archive.path}
    result = {symbol: list(names) for symbol, names in selected.items()}
    receipts = []
    seen = set()
    groups = {}
    for row in proof["sessions"]:
        key = (row["symbol"], row["date"])
        if key in seen:
            raise ValueError("Ambiguous session replacement")
        seen.add(key)
        groups.setdefault(row["symbol"], []).append(row)
    for symbol, replacements in groups.items():
        names = selected.get(symbol)
        if not names:
            raise ValueError("Replacement symbol is outside selected inputs")
        originals = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
        frames = [pd.read_parquet(root / name) for name in names]
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if frame.timestamp.duplicated().any():
            raise ValueError("Selected primary inputs contain duplicate timestamps")
        for row in replacements:
            if row["anchor_files"] != originals:
                raise ValueError("Replacement anchor file identity mismatch")
            path = root / row["path"]
            if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Replacement content identity mismatch")
            source_path = archive.root / row["source_path"]
            if archive.digest(source_path) != row["source_sha256"] or row["source_manifest_hash"] != archive.identity:
                raise ValueError("Replacement source identity mismatch")
            source = pd.read_parquet(BytesIO(archive.read(source_path)))
            day = pd.Timestamp(row["date"]).date()
            expected = source[source.timestamp.dt.date == day].reset_index(drop=True)
            replacement = pd.read_parquet(path).reset_index(drop=True)
            pd.testing.assert_frame_equal(replacement, expected)
            if len(replacement) != row["rows"] or session_bar_coverage(replacement.timestamp, day, 5, symbol=symbol)["status"] != "pass":
                raise ValueError("Replacement is not the complete declared session")
            old = frame[frame.timestamp.dt.date == day]
            changed = []
            indexed = old.set_index("timestamp")
            columns = ["open", "high", "low", "close", "volume"]
            for _, bar in replacement.iterrows():
                if bar.timestamp in indexed.index:
                    prior = indexed.loc[bar.timestamp]
                    before, after = {k: float(prior[k]) for k in columns}, {k: float(bar[k]) for k in columns}
                    if before != after:
                        changed.append({"timestamp": bar.timestamp.isoformat(), "before": before, "after": after})
            added = len(set(replacement.timestamp) - set(old.timestamp))
            # The operation carries every replaced old row, including deletions.
            receipt = {"symbol": symbol, "date": row["date"], "reason": "complete_retained_endpoint_session",
                "source_sha256": row["source_sha256"], "replacement_sha256": row["sha256"],
                "anchor_files": originals, "added_timestamps": added, "revised_rows": changed,
                "removed_timestamps": [t.isoformat() for t in sorted(set(old.timestamp) - set(replacement.timestamp))],
                "old_session_hash": signature(old.to_dict("records")),
                "new_session_hash": signature(replacement.to_dict("records")),
                "historical_qualification": False}
            declared_revisions = [{"timestamp": r["timestamp"], "before": r["archived"], "after": r["later_snapshot"]}
                                  for r in row["revisions"]]
            if added != row["added_timestamps"] or changed != declared_revisions:
                raise ValueError("Replacement delta differs from recovery evidence")
            receipts.append(receipt)
            frame = pd.concat([frame[frame.timestamp.dt.date != day], replacement], ignore_index=True)
            paths.update({path, source_path, *(root / name for name in names)})
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        identity = signature([r for r in receipts if r["symbol"] == symbol])
        output = bundle_path.parent / "composed" / f"{symbol}_5m_{identity}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(output), frame)
        else:
            frame.to_parquet(output, index=False)
        paths.add(output)
        result[symbol] = [output.relative_to(root).as_posix()]
    return result, paths, receipts
