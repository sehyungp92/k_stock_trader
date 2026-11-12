"""Immutable research generations, copied once and verified when bytes are read."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
import hashlib
from io import BytesIO
import json
import tempfile
from pathlib import Path

_ACTIVE = ContextVar("research_generation", default=None)
_VERIFIED_BYTES = {}


def signature(value):
    # Bar-alignment inputs can contain millions of rows. Preserve canonical
    # bytes without allocating their complete JSON text and UTF-8 copy.
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=str)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


class InputManifest:
    def __init__(self, path):
        self.path = Path(path).resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.identity = payload.pop("manifest_hash")
        if signature(payload) != self.identity:
            raise ValueError("Input manifest identity mismatch")
        self.payload = payload
        self.root = self.path.parent / "files"
        for name in payload["files"]:
            if not (self.root / name).resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Input manifest path escapes its generation")
        self.files = {str((self.root / name).resolve()): digest for name, digest in payload["files"].items()}
        self.loaded = set()

    @classmethod
    def create(cls, paths, *, source_root, destination, evidence=None, execution_source_root=None):
        from backtests.auto.shared.cache_keys import source_generation_hash
        source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
        if (destination / "manifest.json").exists():
            raise FileExistsError("Input generation already exists; use a new destination")
        files = {}
        for path in sorted({Path(path).resolve() for path in paths}):
            name = path.relative_to(source_root).as_posix()
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            target = destination / "files" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
            files[name] = digest
        payload = {"version": "research-inputs-v1", "source_root": str(source_root),
            "source_generation_hash": source_generation_hash(root=execution_source_root), "files": files, "evidence": evidence or {}}
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "manifest.json"
        path.write_text(json.dumps({**payload, "manifest_hash": signature(payload)}, indent=2), encoding="utf-8")
        return cls(path)

    def remap(self, config):
        original = Path(self.payload["source_root"])
        result = dict(config)
        for key, value in config.items():
            if not isinstance(value, (str, Path)) or key in {"artifact_root", "input_manifest_path", "output_dir", "output_root"}:
                continue
            if key.endswith(("_root", "_path")):
                path = Path(value)
                absolute = path.resolve() if path.is_absolute() else (original / path).resolve()
                if not path.is_absolute() and not (self.root / path).exists() and (self.root / "config" / path).exists():
                    absolute = (original / "config" / path).resolve()
                if absolute.is_relative_to(self.root.resolve()):
                    result[key] = str(absolute)
                    continue
                try:
                    relative = absolute.relative_to(original)
                except ValueError:
                    raise ValueError(f"Input {key} is outside the frozen generation") from None
                mapped = self.root / relative
                if not mapped.exists():
                    raise ValueError(f"Input {key} is absent from the frozen generation")
                result[key] = str(mapped)
        for key, default in (("data_root", "data/kis_intraday_parquet"), ("daily_data_root", "data/krx_daily_parquet")):
            if key not in result and (self.root / default).exists():
                result[key] = str(self.root / default)
        result["artifact_root"] = str(self.path.parent / "derived_artifacts")
        result["input_generation"] = self.identity
        from backtests.core.shared_market_inputs import bind_market_config
        return bind_market_config(result, self)

    def digest(self, path):
        return self.files.get(str(self.resolve(path)))

    def resolve(self, path):
        path = Path(path).resolve()
        if path.is_relative_to(self.root.resolve()):
            return path
        original = Path(self.payload["source_root"])
        if path.is_relative_to(original):
            return (self.root / path.relative_to(original)).resolve()
        return path

    def read(self, path):
        path = self.resolve(path)
        expected = self.digest(path)
        if expected is None:
            raise ValueError(f"Unmanifested research input: {path}")
        key = (self.identity, str(path))
        content = _VERIFIED_BYTES.get(key)
        if content is None:
            content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Frozen input bytes changed: {path}")
        self.loaded.add(str(path.relative_to(self.root)))
        _VERIFIED_BYTES[key] = content
        return content

    def receipt(self):
        from backtests.core.shared_market_inputs import market_input_contract
        contract = market_input_contract(self)
        return {"manifest_hash": self.identity, "manifest_path": str(self.path),
            "source_generation_hash": self.payload["source_generation_hash"], "files": len(self.files),
            "market_input_hash": contract["market_input_hash"] if contract else None,
            "loaded_files": sorted(self.loaded), "evidence": self.payload.get("evidence", {})}

    def semantic_identity(self):
        return {"manifest_hash": self.identity, "source_generation_hash": self.payload["source_generation_hash"]}

    @contextmanager
    def open_verified(self, path):
        path = self.resolve(path)
        expected = self.digest(path)
        if expected is None:
            raise ValueError(f"Unmanifested research input: {path}")
        # Snapshot the exact verified bytes without retaining large source rows in RAM.
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as snapshot:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    snapshot.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Frozen input bytes changed: {path}")
            self.loaded.add(path.relative_to(self.root).as_posix())
            snapshot.seek(0)
            yield snapshot

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)


def active_manifest():
    return _ACTIVE.get()


@contextmanager
def input_stream(path):
    current = active_manifest()
    if current:
        with current.open_verified(path) as stream:
            yield stream
    else:
        with Path(path).open("rb") as stream:
            yield stream


def manifest_method(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        generation = getattr(self, "_input_generation", None)
        if generation is None:
            return function(self, *args, **kwargs)
        from backtests.auto.shared.cache_keys import source_generation_hash
        if generation.payload["source_generation_hash"] != source_generation_hash():
            raise ValueError("Frozen research generation does not match current source")
        with generation.activate():
            return function(self, *args, **kwargs)
    return wrapped


def read_input_bytes(path):
    current = active_manifest()
    if current:
        path = Path(path).resolve()
        if current.digest(path):
            return current.read(path)
        try:
            mapped = current.root / path.relative_to(Path(current.payload["source_root"]))
        except ValueError:
            mapped = path
        if current.digest(mapped):
            return current.read(mapped)
        if (current.path.parent / "derived_artifacts") in path.parents:
            return path.read_bytes()
        raise ValueError(f"Unmanifested research input: {path}")
    return Path(path).read_bytes()


def parquet_input(path):
    return BytesIO(read_input_bytes(path))


def frozen_research_loader(function):
    @wraps(function)
    def wrapped(config=None, mutations=None, **kwargs):
        config = dict(config or {})
        path = config.get("input_manifest_path")
        if active_manifest() is not None:
            if path and InputManifest(path).identity != active_manifest().identity:
                raise ValueError("Requested input generation differs from active generation")
            bundle = function(active_manifest().remap(config), mutations, **kwargs)
            return replace(bundle, metadata={**bundle.metadata, "input_manifest": active_manifest().receipt()})
        if not path:
            return function(config, mutations, **kwargs)
        manifest = InputManifest(path)
        from backtests.auto.shared.cache_keys import source_generation_hash
        if manifest.payload["source_generation_hash"] != source_generation_hash():
            raise ValueError("Frozen research generation does not match current source")
        with manifest.activate():
            bundle = function(manifest.remap(config), mutations, **kwargs)
        return replace(bundle, metadata={**bundle.metadata, "input_manifest": manifest.receipt()})
    return wrapped


def frozen_input_loader(function):
    """Bind dataset builders which do not return an EventReplayBundle."""
    @wraps(function)
    def wrapped(config, *args, **kwargs):
        config = dict(config or {})
        path = config.get("input_manifest_path")
        current = active_manifest()
        manifest = InputManifest(path) if path else current
        if manifest is None:
            return function(config, *args, **kwargs)
        if current is not None and current.identity != manifest.identity:
            raise ValueError("Requested input generation differs from active generation")
        if current is not None:
            manifest = current
        from backtests.auto.shared.cache_keys import source_generation_hash
        if manifest.payload["source_generation_hash"] != source_generation_hash():
            raise ValueError("Frozen research generation does not match current source")
        with manifest.activate():
            return function(manifest.remap(config), *args, **kwargs)
    return wrapped


def main():
    """python -m backtests.core.input_manifest --destination ... --include ..."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--include", action="append", required=True)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    paths = []
    for value in args.include:
        path = root / value
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            paths.append(path)
        else:
            parser.error(f"Missing input: {path}")
    manifest = InputManifest.create(paths, source_root=root, destination=args.destination)
    print(json.dumps(manifest.receipt(), indent=2))


if __name__ == "__main__":
    main()
