from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from backtests.engine.contracts import CONTRACT_VERSIONS


def stable_signature(value: Any) -> str:
    # Full replay ledgers can exceed memory if both their JSON text and UTF-8
    # copy are materialized. Stream the same canonical bytes into the digest.
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=_default)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


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
    from kis_core.input_manifest import active_manifest
    generation = active_manifest()
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
            "input_generation": generation.identity if generation else None,
            "contract_versions": CONTRACT_VERSIONS,
            "source_fingerprint": source_fingerprint,
            "mutations": mutation_payload,
            "extra": extra or {},
        }
    )


def fingerprint_paths(paths: Iterable[Path], *, root: Path | None = None) -> str:
    entries: list[dict[str, Any]] = []
    for path in sorted({Path(item) for item in paths}, key=lambda item: str(item).lower()):
        exists = path.exists()
        try:
            label = str(path.relative_to(root)) if root else str(path)
        except ValueError:
            label = str(path)
        entries.append(
            {
                "path": label.replace("\\", "/"),
                "exists": exists,
                "sha256": file_content_hash(path) if exists else None,
            }
        )
    return stable_signature(entries)


def file_content_hash(path: Path) -> str:
    from kis_core.input_manifest import active_manifest
    generation = active_manifest()
    if generation and generation.digest(path):
        return generation.digest(path)
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_generation_hash(*, root: Path | None = None) -> str:
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
    paths = [path for directory in ("backtests", "strategy_common", "strategy_kalcb", "strategy_olr", "strategy_pcim", "deployment", "instrumentation", "oms", "oms_client", "kis_core", "scripts")
        for path in (root / directory).rglob("*.py")]
    paths.extend((root / "kis_core/data").glob("*.yaml"))
    # Code identity must describe the executing checkout even inside a frozen
    # input context. Input manifests must never substitute old code digests.
    return stable_signature([{ "path": p.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)])


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

