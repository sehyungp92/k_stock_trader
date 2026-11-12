"""Build the same verified manifest index with one resolution per entry."""
import json
from pathlib import Path


def initialize_manifest(instance, path, signature):
    instance.path = Path(path).resolve()
    payload = json.loads(instance.path.read_text(encoding='utf-8'))
    instance.identity = payload.pop('manifest_hash')
    if signature(payload) != instance.identity:
        raise ValueError('Input manifest identity mismatch')
    instance.payload = payload
    instance.root = instance.path.parent / 'files'
    root_resolved = instance.root.resolve()
    indexed = {}
    for name, expected in payload['files'].items():
        resolved = (instance.root / name).resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError('Input manifest path escapes its generation')
        indexed[str(resolved)] = expected
    if instance.root.resolve() != root_resolved:
        raise ValueError('Input generation root changed while indexing')
    instance.files = indexed
    instance.loaded = set()


def install_manifest_index_controls():
    from backtests.core import input_manifest
    counters = {'constructions': 0, 'indexed_files': 0}

    def initialize(instance, path):
        initialize_manifest(instance, path, input_manifest.signature)
        counters['constructions'] += 1
        counters['indexed_files'] += len(instance.files)

    # The native read, open_verified, resolve and digest methods remain unchanged.
    # Every actual read still resolves its current path and verifies file bytes.
    input_manifest.InputManifest.__init__ = initialize
    return counters
