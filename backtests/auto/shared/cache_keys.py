from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_signature(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def mutation_subset(
    mutations: dict[str, Any],
    *,
    exact_keys: Iterable[str] = (),
    prefixes: Iterable[str] = (),
) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key in exact_keys:
        if key in mutations:
            subset[key] = mutations[key]
    for prefix in prefixes:
        dotted = prefix if prefix.endswith(".") else f"{prefix}."
        for key, value in mutations.items():
            if key == prefix or key.startswith(dotted):
                subset[key] = value
    return {key: subset[key] for key in sorted(subset)}


def build_cache_key(
    namespace: str,
    *,
    source_fingerprint: str = "",
    mutations: dict[str, Any] | None = None,
    mutation_exact_keys: Iterable[str] = (),
    mutation_prefixes: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> str:
    if mutations is None:
        mutation_payload = None
    elif tuple(mutation_exact_keys) or tuple(mutation_prefixes):
        mutation_payload = mutation_subset(
            mutations,
            exact_keys=mutation_exact_keys,
            prefixes=mutation_prefixes,
        )
    else:
        mutation_payload = {key: mutations[key] for key in sorted(mutations)}
    return stable_signature(
        {
            "namespace": namespace,
            "source_fingerprint": source_fingerprint,
            "mutations": mutation_payload,
            "extra": extra or {},
        }
    )


def fingerprint_paths(paths: Iterable[Path], *, root: Path | None = None) -> str:
    entries: list[dict[str, Any]] = []
    for path in sorted({Path(item) for item in paths}, key=lambda item: str(item).lower()):
        exists = path.exists()
        stat = path.stat() if exists else None
        try:
            label = str(path.relative_to(root)) if root else str(path)
        except ValueError:
            label = str(path)
        entries.append(
            {
                "path": label.replace("\\", "/"),
                "exists": exists,
                "size": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
            }
        )
    return stable_signature(entries)


def fingerprint_tree(
    root: Path,
    *,
    patterns: Iterable[str] = ("*.parquet", "*.json", "*.yaml", "*.yml"),
    recursive: bool = True,
) -> str:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(root).rglob(pattern) if recursive else Path(root).glob(pattern))
    return fingerprint_paths(paths, root=Path(root))


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)

