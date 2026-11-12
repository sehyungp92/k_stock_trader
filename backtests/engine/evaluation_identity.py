"""Bind receipts to the request supplied at an evaluation boundary."""
from functools import wraps
from contextlib import nullcontext

from backtests.auto.shared.cache_keys import stable_signature


def request_identity(config, mutations):
    config = dict(config or {})
    manifest = None
    if config.get("input_manifest_path"):
        from kis_core.input_manifest import InputManifest
        manifest = InputManifest(config["input_manifest_path"]).identity
    return stable_signature({"config": config, "mutations": dict(mutations or {}), "input_generation": manifest})


def bound_evaluation(function):
    @wraps(function)
    def wrapped(config=None, mutations=None, **kwargs):
        expected = request_identity(config, mutations)
        config = dict(config or {})
        frozen = None
        if config.get("input_manifest_path"):
            from kis_core.input_manifest import InputManifest
            from backtests.auto.shared.cache_keys import source_generation_hash
            frozen = InputManifest(config["input_manifest_path"])
            if frozen.payload["source_generation_hash"] != source_generation_hash():
                raise ValueError("Frozen research generation does not match current source")
            supplied = kwargs.get("replay_bundle")
            if supplied is not None and supplied.metadata.get("input_manifest", {}).get("manifest_hash") != frozen.identity:
                raise ValueError("Supplied replay bundle does not match the requested input generation")
        with frozen.activate() if frozen else nullcontext():
            result = function(frozen.remap(config) if frozen else config, mutations, **kwargs)
        broker = result.replay_result.broker
        identity = dict(getattr(broker, "evaluation_identity", {}))
        identity.update(request=expected, mutations=stable_signature(dict(mutations or {})))
        identity["configuration"] = config
        bundle = kwargs.get("replay_bundle")
        manifest = frozen.receipt() if frozen else (getattr(bundle, "metadata", {}) or {}).get("input_manifest")
        if not manifest:
            manifest = identity.get("input_manifest")
        identity["input_manifest"] = manifest
        from .integrity import audit_broker
        with frozen.activate() if frozen else nullcontext():
            result.metrics.update(audit_broker(broker, identity=identity))
        result.metrics["evaluation_request"] = expected
        result.metrics["evaluation_mutations"] = identity["mutations"]
        return result
    return wrapped
