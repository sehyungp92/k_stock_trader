import json
from copy import deepcopy

import pytest

from backtests.strategies.kalcb.qualified_artifacts import digest
from backtests.strategies.kalcb.recovery_full_diagnostics_v2 import adapt_feature_lineage


@pytest.fixture
def bundle(tmp_path):
    files = {}

    def save(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        files[str(path)] = digest(path)
        return {"path": str(path), "sha256": digest(path)}

    preparations = [dict(save(f"prepared{i}.json", {"stage": i}), original_path=f"stage{i}/prepared.json") for i in range(4)]
    forecast = save("forecast.json", {"passed": True})
    reference = save("lineage.json", {
        "schema": "kalcb-recovery-feature-preparation-lineage-v1",
        "preparations": preparations, "forecast_reconstruction": forecast,
    })
    manifest = tmp_path / "acceptance.json"
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    return {"manifest": str(manifest), "profile": {"diagnostic_feature_lineage": reference,
        "forecast_reconstruction": forecast["path"]}, "policy": {"recipe": "decay60"}}


def test_report_fingerprint_uses_bound_complete_feature_lineage_without_native_mutation(bundle):
    before = deepcopy(bundle)
    adapted = adapt_feature_lineage(bundle)
    assert bundle == before
    assert adapted["policy"]["prepared_sha256"] == bundle["profile"]["diagnostic_feature_lineage"]["sha256"]
    assert adapted["policy"]["recipe"] == "decay60"
    assert "compatibility metadata only" in adapted["policy"]["diagnostic_prepared_hash_basis"]


@pytest.mark.parametrize("change", ["unbound_lineage", "changed_lineage", "changed_preparation", "unbound_preparation", "wrong_forecast", "existing_identity"])
def test_lineage_compatibility_fails_closed(bundle, change):
    from pathlib import Path
    reference = bundle["profile"]["diagnostic_feature_lineage"]
    path = Path(reference["path"])
    lineage = json.loads(path.read_text())
    manifest = Path(bundle["manifest"])
    capsule = json.loads(manifest.read_text())
    if change == "unbound_lineage":
        del capsule["files"][str(path)]
        manifest.write_text(json.dumps(capsule))
    elif change == "changed_lineage":
        path.write_text("{}")
    elif change == "changed_preparation":
        Path(lineage["preparations"][0]["path"]).write_text("{}")
    elif change == "unbound_preparation":
        del capsule["files"][lineage["preparations"][0]["path"]]
        manifest.write_text(json.dumps(capsule))
    elif change == "wrong_forecast":
        bundle["profile"]["forecast_reconstruction"] = str(path)
    else:
        bundle["policy"]["prepared_sha256"] = "existing_real_identity"
    with pytest.raises(ValueError):
        adapt_feature_lineage(bundle)
