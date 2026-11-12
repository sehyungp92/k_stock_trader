"""Create a new immutable runtime revision without changing prior generations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.executed_value import digest, signature, write_json


def bind_calendar_evidence(repo, root, files):
    """Calendar YAML is executable input and includes hashed source documents."""
    import yaml
    for path in (root / "kis_core/data").glob("*.yaml"):
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, expected in content.get("evidence_files", {}).items():
            source, target = (repo / name).resolve(), (root / name).resolve()
            if not source.is_relative_to(repo.resolve()) or not target.is_relative_to(root.resolve()):
                raise ValueError("Calendar source escapes generation")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if digest(target) != expected:
                raise ValueError("Calendar source hash mismatch: " + name)
            files[name] = expected


def bind_shared_calendar_contract(evidence, root, files):
    """Rebind a calendar revision while requiring identical price/volume inputs."""
    import yaml
    contract = evidence.get("shared_market_inputs")
    if contract is None:
        return
    original = contract["market_input_hash"]
    for name, expected in list(contract["files"].items()):
        if files.get(name) != expected:
            if not name.startswith("kis_core/data/"):
                raise ValueError("Research runtime changed a shared market file: " + name)
            contract["files"][name] = files[name]
    for path in (root / "kis_core/data").glob("*.yaml"):
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name in content.get("evidence_files", {}):
            contract["files"][name] = files[name]
    contract.pop("market_input_hash")
    contract["market_input_hash"] = signature(contract)
    evidence["calendar_runtime_parent_market_hash"] = original


def refresh(study_path, output, register_current_policies=False, calendar_manifest=None):
    repo = Path(__file__).resolve().parents[1]
    study = json.loads(study_path.read_text())
    manifest = Path(study["manifest"])
    parent = json.loads(manifest.read_text())
    original_hash = parent.pop("manifest_hash")
    if signature(parent) != original_hash:
        raise ValueError("Changed parent manifest")
    output.mkdir(parents=True, exist_ok=False)
    root = output / "inputs/files"
    root.mkdir(parents=True)
    files = {}
    for name, expected in parent["files"].items():
        current, previous, target = repo / name, manifest.parent / "files" / name, root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if current.is_file() and name.endswith(".py"):
            shutil.copy2(current, target)
            files[name] = digest(target)
        else:
            # Prior generation was fully hashed at creation. A same-volume
            # hardlink preserves those bytes; runtime verified reads recheck.
            try:
                os.link(previous, target)
                files[name] = expected
            except OSError:
                shutil.copy2(previous, target)
                files[name] = digest(target)
                if files[name] != expected:
                    raise ValueError("Changed parent dependency")
    for folder in ("backtests", "strategy_common", "strategy_kalcb", "strategy_olr", "strategy_pcim", "deployment", "instrumentation", "oms", "oms_client", "kis_core", "scripts"):
        for current in (repo / folder).rglob("*.py"):
            name = current.relative_to(repo).as_posix()
            if name not in files:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, target)
                files[name] = digest(target)
    if calendar_manifest:
        calendar_parent = json.loads(calendar_manifest.read_text())
        expected_identity = calendar_parent.pop("manifest_hash")
        if signature(calendar_parent) != expected_identity:
            raise ValueError("Changed calendar parent")
        for name, expected in calendar_parent["files"].items():
            if name.startswith("kis_core/data/"):
                target = root / name
                if target.exists():
                    target.unlink()
                shutil.copy2(calendar_manifest.parent / "files" / name, target)
                if digest(target) != expected:
                    raise ValueError("Changed pinned calendar")
                files[name] = expected
        parent["evidence"]["shared_market_inputs"] = calendar_parent["evidence"]["shared_market_inputs"]
        parent["evidence"]["calendar_development_pin"] = expected_identity
    bind_calendar_evidence(repo, root, files)
    bind_shared_calendar_contract(parent["evidence"], root, files)
    spec = json.loads((root / study["spec_path"]).read_text())
    original_request = json.loads((root / spec["olr_request"]).read_text())
    spec["configuration"]["require_complete_feature_prefix"] = bool(original_request["config"]["require_complete_feature_prefix"])
    if register_current_policies:
        from scripts.portfolio_synergy_study import policy_recipes
        spec["candidates"] = policy_recipes(spec["initial_equity"])
        spec["registration_revision"] = "candidate-priority-before-reservation; before first portfolio outcome"
    spec["configuration"]["market_input_hash"] = parent["evidence"]["shared_market_inputs"]["market_input_hash"]
    new_path = "research/portfolio_current/study_" + signature(spec)[:16] + ".json"
    if not (root / new_path).exists():
        write_json(root / new_path, spec)
    files[new_path] = digest(root / new_path)
    study = {**study, **spec, "spec_path": new_path}
    parent["evidence"]["portfolio_current_study"] = new_path
    program = "from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())"
    source = subprocess.check_output([sys.executable, "-B", "-c", program], cwd=root, text=True).strip()
    payload = {**parent, "files": files, "source_generation_hash": source}
    new_manifest = output / "inputs/manifest.json"
    write_json(new_manifest, {**payload, "manifest_hash": signature(payload)})
    from backtests.core.input_manifest import InputManifest
    InputManifest(new_manifest).receipt()
    write_json(output / "study.json", {**study, "manifest": str(new_manifest.resolve()),
        "code": str(root.resolve()), "source_generation_hash": source})
    write_json(output / "runtime_revision.json", {"parent_manifest_hash": original_hash, "source_generation_hash": source,
        "parent_study": str(study_path), "purpose": "implementation revision; candidate and data registration unchanged"})
    print(json.dumps({"refreshed": str(output), "source": source}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-current-policies", action="store_true")
    parser.add_argument("--calendar-manifest", type=Path)
    args = parser.parse_args()
    refresh(args.study, args.output, args.register_current_policies, args.calendar_manifest)
