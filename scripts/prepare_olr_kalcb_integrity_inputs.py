"""Freeze retained replay inputs and executing source without changing round artifacts."""
import argparse
import json
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtests.core.input_manifest import InputManifest, signature


def verified_supplement(root, symbol, entry, primary_paths):
    """Restore only an exact archived session, with no competing timestamps."""
    import pandas as pd
    from io import BytesIO
    from datetime import date
    from strategy_common.session_coverage import session_bar_coverage
    path, proof_path = root / entry["path"], root / entry["provenance"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    source = InputManifest(proof["source_manifest"])
    source_path = source.root / proof["source_file"]
    if source.identity != proof["source_manifest_hash"] or source.digest(source_path) != proof["source_sha256"]:
        raise ValueError("Supplement source provenance does not match")
    if hashlib.sha256(path.read_bytes()).hexdigest() != proof["derived_sha256"]:
        raise ValueError("Supplement content does not match provenance")
    frame = pd.read_parquet(path)
    original = pd.read_parquet(BytesIO(source.read(source_path)))
    day = date.fromisoformat(entry["date"])
    expected = original[original.timestamp.dt.date == day].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)
    if not path.name.startswith(symbol + "_") or session_bar_coverage(frame.timestamp, day, 5, symbol=symbol)["status"] != "pass":
        raise ValueError("Supplement does not contain a complete declared symbol/session")
    occupied = set()
    for primary in primary_paths:
        occupied.update(pd.read_parquet(primary, columns=["timestamp"]).timestamp)
    if occupied.intersection(frame.timestamp):
        raise ValueError("Supplement overlaps selected primary data")
    return path, {path, proof_path, source.path, source_path}


def shared_market_contract(root, spec):
    policy = dict(spec["market_inputs"])
    intraday_root = root / policy["data_root"]
    selected = {}
    symbols = sorted({p.parent.name for p in intraday_root.glob(f"*/*_5m_*_{policy['snapshot_end']}.parquet")})
    for symbol in symbols:
        end = policy.get("snapshot_end_overrides", {}).get(symbol, policy["snapshot_end"])
        paths = sorted((intraday_root / symbol).glob(f"{symbol}_5m_*_{end}.parquet"))
        if len(paths) != 1:
            raise ValueError(f"Expected one declared intraday snapshot for {symbol}, found {len(paths)}")
        selected[symbol] = [p.relative_to(root).as_posix() for p in paths]
    supplemental_paths = set()
    for symbol, entries in policy.get("intraday_supplements", {}).items():
        if symbol not in selected:
            raise ValueError(f"Supplement has no selected primary input: {symbol}")
        for entry in entries:
            path, proof_paths = verified_supplement(root, symbol, entry, [root / n for n in selected[symbol]])
            selected[symbol].append(path.relative_to(root).as_posix())
            supplemental_paths.update(proof_paths)
    import yaml
    universe = yaml.safe_load((root / policy["universe"]).read_text(encoding="utf-8"))["symbols"]
    missing = {str(s).zfill(6) for s in universe} - selected.keys()
    if missing:
        raise ValueError(f"Shared universe missing intraday inputs: {sorted(missing)}")
    paths = {root / name for names in selected.values() for name in names}
    paths.update(supplemental_paths)
    replacement_receipts = []
    if policy.get("session_replacement_bundle"):
        from backtests.core.session_replacement import apply_session_replacements
        selected, replacement_paths, replacement_receipts = apply_session_replacements(
            root, selected, root / policy["session_replacement_bundle"])
        paths.update(replacement_paths)
    anchored_receipts = []
    if policy.get("anchored_session_recovery"):
        from backtests.core.anchored_recovery import apply_anchored_recoveries
        selected, recovery_paths, anchored_receipts = apply_anchored_recoveries(
            root, selected, root / policy["anchored_session_recovery"])
        paths.update(recovery_paths)
    daily_pattern = f"*_{policy['daily_snapshot_end']}.parquet" if policy.get("daily_snapshot_end") else "*.parquet"
    paths.update((root / policy["daily_data_root"]).rglob(daily_pattern))
    if policy.get("session_resolution_report"):
        resolution_path = root / policy["session_resolution_report"]
        resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
        if resolution.get("version") != "session-recovery-dispositions-v1" or not resolution.get("all_original_flags_accounted"):
            raise ValueError("Incomplete session-resolution inventory")
        paths.add(resolution_path)
        for name, digest in resolution["source_files"].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("Session-resolution source identity mismatch")
            paths.add(source)
    paths.update((root / policy["daily_data_root"]).rglob("*.json"))
    paths.update((root / "kis_core/data").glob("*.yaml"))
    halt_contract = yaml.safe_load((root / "kis_core/data/krx_intraday_halts.yaml").read_text(encoding="utf-8"))
    for name, digest in halt_contract["evidence_files"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Halt source bytes changed")
        paths.add(path)
    paths.update(root / policy[key] for key in ("universe", "sector_map_path"))
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    contract = {"version": "olr-kalcb-shared-market-inputs-v1", **policy,
                "intraday_files": selected, "files": files,
                "session_replacements": replacement_receipts,
                "anchored_session_recoveries": anchored_receipts,
                "verification_scope": "content_identity_and_loader_alignment",
                "historical_qualification": False}
    return {**contract, "market_input_hash": signature(contract)}, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--qualification-evidence", type=Path,
                        help="Versioned source/reference/phase observations and their source-file inventory")
    parser.add_argument("--validation-contract", type=Path, help="OLR-only generation from this frozen validation recipe and engine")
    parser.add_argument("--market-spec", type=Path, help="Explicit market generation, including daily_snapshot_end")
    parser.add_argument("--reference-revision", type=Path, help="Reviewed YAML-only source revision: reason, sources and files (relative path to SHA256)")
    parser.add_argument("--inspect-window", nargs=2, metavar=("START", "END"), help="Inspect actual new data coverage/basis without evaluating strategy outcomes")
    args = parser.parse_args()
    if args.inspect_window:
        if not args.market_spec:
            parser.error("Window inspection requires --market-spec")
        spec = json.loads(args.market_spec.read_text(encoding="utf-8"))
        result = inspect_validation_window(ROOT, spec, *args.inspect_window)
        args.destination.mkdir(parents=True, exist_ok=True)
        (args.destination / "window_input_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "issues"}, indent=2))
        return
    if args.validation_contract:
        if not args.market_spec:
            parser.error("OLR validation requires --market-spec")
        manifest = prepare_validation_inputs(args.validation_contract, args.market_spec, args.destination,
                                             args.qualification_evidence, args.reference_revision)
        print(json.dumps({"manifest": str(manifest.path), "files": len(manifest.files), "historical_qualification": False}))
        return
    spec_path = ROOT / "config/olr_kalcb/shared_replay.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    contract, market_paths = shared_market_contract(ROOT, spec)
    qualification, qualification_paths = prepare_qualification_sources(ROOT, contract, args.qualification_evidence, research_spec=spec)
    paths = {spec_path, ROOT / "requirements.txt"}
    for directory in ("backtests", "strategy_common", "strategy_kalcb", "strategy_olr", "strategy_pcim",
                      "deployment", "instrumentation", "oms", "oms_client", "kis_core", "scripts", "tests"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update((ROOT / "kis_core/data").glob("*.yaml"))
    paths.update((ROOT / "config").rglob("*.yaml"))
    paths.update((ROOT / "config").rglob("*.json"))
    for section in (spec["kalcb"], spec["olr"]):
        for key in ("optimization_config", "optimized_result", "market_bar_digest"):
            if key in section:
                paths.add(ROOT / section[key])
        paths.update((ROOT / section["candidate_snapshot_root"]).glob("*.json"))
    paths.update(market_paths)
    paths.update(qualification_paths)
    pool = ROOT / "data/backtests/output/kalcb/round_1/guarded_q3_full_diagnostics"
    paths.update(pool.glob("*guarded_prefilter_pool_rows.jsonl"))
    optimized = json.loads((ROOT / spec["kalcb"]["optimized_result"]).read_text(encoding="utf-8"))
    for key in ("_kalcb.source.path", "_kalcb.pool_source.path"):
        path = Path(optimized["mutations"][key])
        paths.add(path if path.is_absolute() else ROOT / path)
    manifest = InputManifest.create(paths, source_root=ROOT, destination=args.destination,
        evidence={"kalcb_round": 3, "olr_round": 5, "shared_market_inputs": contract,
                  "qualification_path": qualification.relative_to(ROOT).as_posix(),
                  "capacity_evidence": "five-minute bars only; execution-phase liquidity unverified"})
    print(json.dumps({"manifest": str(manifest.path), "manifest_hash": manifest.identity,
                      "market_input_hash": contract["market_input_hash"], "files": len(manifest.files)}, indent=2))


def prepare_validation_inputs(contract_path, market_spec_path, destination, qualification_evidence=None, reference_revision=None):
    """Freeze new data with the original engine; only explicit reference revisions may change it."""
    import shutil
    from scripts.olr_prospective_validation import verified_contract
    from backtests.auto.shared.cache_keys import source_generation_hash
    contract = verified_contract(contract_path)
    parent = InputManifest(contract["parent_input_manifest"])
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new input-generation directory")
    spec = json.loads(Path(market_spec_path).read_text(encoding="utf-8"))
    if not spec["market_inputs"].get("daily_snapshot_end"):
        raise ValueError("Validation requires explicit daily and intraday generations")
    market, market_paths = shared_market_contract(ROOT, spec)
    for name, expected in contract["frozen_inputs"].items():
        if market["files"].get(name) != expected:
            raise ValueError("Validation universe or sector map differs from freeze: " + name)
    qualification, qualification_paths = prepare_qualification_sources(ROOT, market, qualification_evidence)
    revision = json.loads(Path(reference_revision).read_text()) if reference_revision else {"files": {}}
    if revision["files"] and (not revision.get("reason") or not revision.get("sources")):
        raise ValueError("Reference revision requires reason and source evidence")
    if any(not (name.startswith("kis_core/data/") and name.endswith(".yaml")) for name in revision["files"]):
        raise ValueError("Only explicit reference YAML changes are allowed; strategy/engine changes require a new study")
    source = destination / "source"
    original_sources = {name: sha for name, sha in parent.payload["files"].items()
                        if name.endswith(".py") or name.startswith("config/") or name.startswith("kis_core/data/")}
    selected = {name: parent.root / name for name in original_sources}
    selected.update({p.relative_to(ROOT).as_posix(): p for p in market_paths | qualification_paths})
    for name, path in selected.items():
        if name in original_sources and name.startswith("kis_core/data/"):
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != original_sources[name] and revision["files"].get(name) != actual:
                raise ValueError("Unreviewed reference change: " + name)
        if name in original_sources and path == parent.root / name and hashlib.sha256(path.read_bytes()).hexdigest() != original_sources[name]:
            raise ValueError("Frozen parent source changed: " + name)
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    source_hash = source_generation_hash(root=source)
    bound_contract = dict(contract)
    if source_hash != contract["source_generation_hash"]:
        if not revision["files"]:
            raise ValueError("Source differs without a reviewed reference revision")
        bound_contract.pop("contract_hash")
        bound_contract.update(parent_contract_hash=contract["contract_hash"], source_generation_hash=source_hash,
                              reference_revision=revision)
        bound_contract["contract_hash"] = signature(bound_contract)
    # Keep the revised contract and its frozen evaluator together, preserving the old contract.
    protocol_dir = destination / "protocol"
    protocol_dir.mkdir(parents=True)
    for name in ("study.json", "prospective_validator_manifest.json"):
        shutil.copyfile(Path(contract_path).parent / name, protocol_dir / name)
    shutil.copytree(Path(contract_path).parent / "prospective_harness", protocol_dir / "prospective_harness")
    (protocol_dir / "prospective_validation_contract.json").write_text(json.dumps(bound_contract, indent=2), encoding="utf-8")
    return InputManifest.create(source.rglob("*.*"), source_root=source, execution_source_root=source, destination=destination / "inputs",
        evidence={"olr_round": 1, "validation_contract_hash": bound_contract["contract_hash"],
                  "reference_revision": revision, "shared_market_inputs": market,
                  "qualification_path": qualification.relative_to(ROOT).as_posix(),
                  "independence": spec.get("independence", "unconfirmed"), "historical_qualification": False})


def inspect_validation_window(root, spec, start, end):
    """Locate actionable missing sessions and inconsistent price bases, without PnL."""
    import pandas as pd
    import yaml
    from datetime import date, datetime, timedelta
    from kis_core.trading_calendar import get_trading_calendar
    policy = spec["market_inputs"]
    calendar = get_trading_calendar()
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    days = []
    cursor = first
    while cursor <= last:
        if not calendar.has_coverage(cursor):
            raise ValueError("No holiday coverage for " + str(cursor))
        if calendar.is_trading_day(cursor):
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    symbols = [str(s).zfill(6) for s in yaml.safe_load((root / policy["universe"]).read_text())["symbols"]]
    # These are screening assumptions only. Missing intervals need dated exchange
    # exceptions; regular-grid agreement does not certify the reference calendar.
    regular = {f"{minute // 60:02d}:{minute % 60:02d}" for minute in range(9 * 60, 15 * 60 + 20, 5)} | {"15:30"}
    issues, rows, close_pairs, close_mismatches, sourced_intervals = [], 0, 0, 0, 0
    for symbol in symbols:
        files = list((root / policy["data_root"] / symbol).glob(f"{symbol}_5m_*_{policy['snapshot_end']}.parquet"))
        if len(files) != 1:
            issues.append({"symbol": symbol, "reason": "intraday_generation_missing_or_ambiguous"})
            continue
        frame = pd.read_parquet(files[0], columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["day"] = frame.timestamp.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
        frame = frame[(frame.day >= start) & (frame.day <= end)].copy()
        frame["time"] = frame.timestamp.dt.tz_convert("Asia/Seoul").dt.strftime("%H:%M")
        rows += len(frame)
        if frame.timestamp.duplicated().any() or (frame[["open", "high", "low", "close"]] <= 0).any().any():
            issues.append({"symbol": symbol, "reason": "duplicate_timestamp_or_nonpositive_price"})
        grouped = {day: group for day, group in frame.groupby("day")}
        for day in days:
            observed = set(grouped[day].time) if day in grouped else set()
            expected = set(regular)
            for first_halt, last_halt, _ in calendar.intraday_halts_for(symbol, date.fromisoformat(day)):
                halted = {label for label in expected if
                    first_halt <= datetime.fromisoformat(day + "T" + label).time() and
                    (datetime.fromisoformat(day + "T" + label) + timedelta(minutes=5)).time() <= last_halt}
                sourced_intervals += len(halted)
                expected -= halted
                if observed & halted:
                    issues.append({"symbol": symbol, "date": day, "reason": "observations_during_sourced_halt", "times": sorted(observed & halted)})
            missing = sorted(expected - observed)
            if missing:
                issues.append({"symbol": symbol, "date": day, "reason": "regular_grid_gap_requires_source_resolution", "missing": missing})
        for table in ("daily_ohlcv", "daily_flow", "daily_foreign_flow", "daily_institutional_flow"):
            daily_paths = list((root / policy["daily_data_root"] / table / symbol).glob(f"*_{policy['daily_snapshot_end']}.parquet"))
            if len(daily_paths) != 1:
                issues.append({"symbol": symbol, "reason": "daily_generation_missing_or_ambiguous", "table": table})
                continue
            daily = pd.read_parquet(daily_paths[0])
            daily["day"] = pd.to_datetime(daily.date).dt.strftime("%Y-%m-%d")
            missing_days = sorted(set(days) - set(daily.day))
            if missing_days:
                issues.append({"symbol": symbol, "reason": "missing_daily_context", "table": table, "dates": missing_days})
            if table == "daily_ohlcv":
                closes = daily.set_index("day").close
                for day, group in grouped.items():
                    auction = group[group.time == "15:30"]
                    if day in closes and len(auction) == 1:
                        close_pairs += 1
                        if float(auction.close.iloc[0]) != float(closes[day]):
                            close_mismatches += 1
                            issues.append({"symbol": symbol, "date": day, "reason": "daily_intraday_close_basis_mismatch"})
    return {"window": {"start": start, "end": end}, "symbols": len(symbols), "sessions": len(days),
            "intraday_rows": rows, "close_pairs": close_pairs, "close_mismatches": close_mismatches,
            "sourced_halt_intervals_excluded": sourced_intervals,
            "issue_count": len(issues), "issues": issues,
            "source_session_reference_covers_window": all(calendar.has_session_coverage(date.fromisoformat(day)) for day in days),
            "qualification": "screening_only; source calendar, halts, basis and native management coverage remain mandatory",
            "strategy_outcomes_opened": False}


def prepare_qualification_sources(root, contract, evidence_path=None, *, research_spec=None):
    """Freeze a common source registry; unavailable semantics remain explicit."""
    supplied, paths = {}, set()
    if evidence_path:
        evidence_path = evidence_path.resolve()
        supplied = json.loads(evidence_path.read_text(encoding="utf-8"))
        if supplied.get("version") != "qualification-observations-v2":
            raise ValueError("Qualification evidence must use version 2")
        paths.add(evidence_path)
        for name, digest in supplied["source_files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Qualification source file identity mismatch")
            paths.add(path)
    default_capacity = {"unavailable_reason": "supported_execution_phase_capacity_evidence_unavailable"}
    if research_spec and research_spec.get("capacity_scenario"):
        default_capacity = declared_capacity_accounts(root, research_spec, supplied.get("basis"))
    payload = {"version": "qualification-observations-v2",
        "market_files": [{"symbol": symbol, "path": name, "timeframe": contract["timeframe"],
                          "parser_version": "completed_ohlcv_parquet_v1"}
                         for symbol, names in sorted(contract["intraday_files"].items()) for name in names],
        "references": supplied.get("references", {"unavailable_reason": "dated_daily_source_basis_not_verified"}),
        "basis": supplied.get("basis", {"unavailable_reason": "vendor_clock_price_and_share_basis_not_verified"}),
        "capacity": supplied.get("capacity", default_capacity)}
    if contract.get("session_resolution_report"):
        payload["session_resolution_report"] = contract["session_resolution_report"]
    path = root / "data/backtests/integrity/qualification-sources" / (signature(payload) + ".json")
    content = json.dumps(payload, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError("Qualification registry identity collision")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path, paths | {path}


def declared_capacity_accounts(root, spec, basis):
    """Register assumptions before replay; this does not assert missing source facts."""
    from dataclasses import replace
    from backtests.config import load_yaml_config
    from backtests.engine.capacity import BarParticipationScenario, CAPACITY_VERSION, CAPACITY_SCENARIOS
    from backtests.engine.sim_broker import SimBroker
    from backtests.engine.qualification_evidence import execution_policy
    from backtests.strategies.portfolio_synergy.source_replay import NativeRiskSharedBroker, kalcb_costs, olr_costs
    from strategy_kalcb.config import KALCBConfig
    from strategy_olr.config import OLRConfig
    configs, equities = {}, {}
    for label, cls in (("kalcb", KALCBConfig), ("olr", OLRConfig)):
        raw = load_yaml_config(root / spec[label]["optimization_config"])
        saved = json.loads((root / spec[label]["optimized_result"]).read_text(encoding="utf-8"))
        configs[label] = cls.from_mapping(raw, saved.get("mutations") or {})
        equities[label] = float((saved.get("execution_contract") or {}).get("initial_equity") or raw["initial_equity"])
    registrations = []
    for name, participation, extra in CAPACITY_SCENARIOS:
        policy = BarParticipationScenario(participation, name)
        k = replace(configs["kalcb"], slippage_bps=configs["kalcb"].slippage_bps + extra)
        o = replace(configs["olr"], slippage_bps=configs["olr"].slippage_bps + extra)
        brokers = (("KALCB", SimBroker(equities["kalcb"], kalcb_costs(k), buying_power_leverage=k.intraday_leverage, capacity_policy=policy)),
            ("OLR", SimBroker(equities["olr"], olr_costs(o), capacity_policy=policy)),
            ("shared", NativeRiskSharedBroker(float(spec["initial_equity"]),
                strategy_costs={"KALCB": kalcb_costs(k), "OLR": olr_costs(o)},
                strategy_leverage={"KALCB": k.intraday_leverage, "OLR": 1}, capacity_policy=policy)))
        for label, broker in brokers:
            registrations.append({"account": label, "policy": execution_policy(broker),
                "initial_equity": broker.initial_equity, "buying_power_leverage": broker.buying_power_leverage})
    normalized = None
    if basis and basis.get("source_documents"):
        from backtests.engine.source_basis import target_basis
        targets = [target_basis(json.loads((root / p).read_text(encoding="utf-8"))) for p in basis["source_documents"]]
        if any(t != targets[0] for t in targets):
            raise ValueError("Capacity registration requires one normalized market basis")
        normalized = targets[0]
    return {"kind": "declared_bar_participation_model", "model_version": CAPACITY_VERSION,
        "qualification_scope": "historical_research_model", "phase_liquidity_verified": False,
        "market_basis": normalized, "registered_execution_policies": registrations}


if __name__ == "__main__":
    main()
