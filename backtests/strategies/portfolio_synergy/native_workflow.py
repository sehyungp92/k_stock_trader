"""Registered native policy selection, frozen finalist, and separate qualification.

The account admits at execution from available cash. This is an explicit research
contract; it does not claim deployment intent-reservation parity.
"""
from dataclasses import asdict
from datetime import date
import json
import math
from pathlib import Path

from backtests.auto.shared.cache_keys import source_generation_hash, stable_signature
from backtests.auto.shared.types import Experiment
from kis_core.input_manifest import InputManifest, read_input_bytes
from backtests.engine.contracts import execution_valid
from .native_evaluator import NativePortfolioPolicyEvaluator
from .native_policy import native_policy_config

RESERVATION_CONTRACT = "admit_at_execution_phase_from_current_cash"


def validate_study(study):
    if study.get("version") != "native-policy-study-v1" or study.get("reservation_contract") != RESERVATION_CONTRACT:
        raise ValueError("Native policy study requires its explicit account contract")
    if study.get("qualification_usage") != "regression":
        raise ValueError("Previously examined history must remain labeled regression")
    train, qualification = study["train"], study["qualification"]
    a, b, c, d = [date.fromisoformat(w[k][:10]) for w, k in ((train, "start"), (train, "end"),
                                                             (qualification, "start"), (qualification, "end"))]
    if not a <= b < c <= d:
        raise ValueError("Training and qualification windows must be ordered and disjoint")
    candidates = [Experiment(row["name"], row["mutations"]) for row in study["candidates"]]
    if not candidates or len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Native policy candidates require unique names")
    for candidate in candidates:
        native_policy_config(candidate.mutations)
    if not math.isfinite(float(study["initial_equity"])) or float(study["initial_equity"]) <= 0:
        raise ValueError("Positive initial equity required")
    return candidates


def load_research_inputs(study, window, generation):
    """Use current research builders, then a full shared management feed."""
    from backtests.config import load_yaml_config
    from backtests.core.shared_market_inputs import market_input_contract
    from backtests.strategies.kalcb.replay_cache import load_kalcb_real_replay_bundle
    from backtests.strategies.olr.replay_cache import load_olr_declared_window_bundle
    from backtests.strategies.kalcb.features import snapshots_from_bundle as kalcb_snapshots
    from backtests.strategies.olr.runner import snapshots_from_bundle as olr_snapshots
    from backtests.strategies.kalcb.trade_plan_sweep import CompiledCoreReplay
    from strategy_kalcb.config import KALCBConfig
    from strategy_olr.config import OLRConfig
    from kis_core.trading_calendar import get_trading_calendar
    from backtests.engine.qualification_evidence import required_sessions
    from .source_replay import PromotedReplayInputs, _load_pinned_parquet_bars
    contract = market_input_contract(generation)
    if contract is None:
        raise ValueError("Native research requires a common frozen market generation")
    configurations, mutations, bundles = {}, {}, {}
    for strategy, loader in (("kalcb", load_kalcb_real_replay_bundle), ("olr", load_olr_declared_window_bundle)):
        item = study[strategy]
        raw = load_yaml_config(generation.root / item["config_path"])
        mutations[strategy] = dict(item.get("mutations") or {})
        if any(key.startswith("_") for key in mutations[strategy]):
            raise ValueError("Native research builder cannot silently reuse private fixed-pool mutations")
        raw.update(window, date_range=window, input_manifest_path=str(generation.path), capability_level="real_replay")
        configurations[strategy] = generation.remap(raw)
        bundles[strategy] = loader(configurations[strategy], mutations[strategy])
    request = {**window, "universe": contract["universe"], "timeframe": contract["timeframe"]}
    if study.get("capacity_scenario"):
        request["capacity_scenario"] = dict(study["capacity_scenario"])
    required = required_sessions(request, generation, get_trading_calendar())
    pairs = {(day, symbol) for symbol, day, _ in required}
    bars = tuple(_load_pinned_parquet_bars(pairs, intraday_root=generation.root / contract["data_root"],
                                         data_snapshot_end=contract["snapshot_end"]))
    k, o = kalcb_snapshots(bundles["kalcb"]), olr_snapshots(bundles["olr"])
    allowed_dates = {day for _, day, _ in required}
    k, o = ({day: row for day, row in snapshots.items() if day in allowed_dates} for snapshots in (k, o))
    if set(k) != allowed_dates or set(o) != allowed_dates:
        raise ValueError("Research builders did not produce every requested candidate session")
    compiled = CompiledCoreReplay(bars, k, tuple(sorted(allowed_dates)),
        {day: len(row.candidates) for day, row in k.items()}, float(study["initial_equity"]),
        bundles["kalcb"].source_fingerprint, stable_signature([s.to_json_dict() for s in k.values()]))
    return PromotedReplayInputs(KALCBConfig.from_mapping(configurations["kalcb"], mutations["kalcb"]), compiled,
        OLRConfig.from_mapping(configurations["olr"], mutations["olr"]), o, bars,
        {"candidate_basis": "current_research_builders", "capability_level": "real_replay"}), request


def run_native_study(study, output_dir, *, input_loader=load_research_inputs):
    candidates = validate_study(study)
    generation = InputManifest(study["input_manifest_path"])
    if generation.payload["source_generation_hash"] != source_generation_hash():
        raise ValueError("Native study source differs from frozen generation")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    declared = {"study": study, "input_generation": generation.semantic_identity()}
    study_hash = stable_signature(declared)
    (output / "study.json").write_text(json.dumps({**declared, "study_hash": study_hash}, indent=2), encoding="utf-8")
    report = {"study_hash": study_hash, "status": "unqualified", "historical_qualification": False,
              "deployment_qualified": False, "reservation_contract": RESERVATION_CONTRACT}
    with generation.activate():
        proof_path = generation.payload.get("evidence", {}).get("qualification_path")
        proof = json.loads(generation.read(generation.root / proof_path)) if proof_path else {}
        missing = [name for name in ("references", "basis", "capacity") if not proof.get(name) or proof[name].get("unavailable_reason")]
        if missing:
            report["blocked_before_selection"] = missing
            report["qualification_limitations"] = {name: proof.get(name, {}).get("unavailable_reason", "evidence_not_supplied") for name in missing}
            if proof.get("session_resolution_report"):
                resolution = json.loads(generation.read(generation.root / proof["session_resolution_report"]))
                report["source_recovery"] = {"inventory_accounted": resolution["all_original_flags_accounted"],
                    "dispositions": resolution["counts"], "evidence_path": proof["session_resolution_report"]}
        else:
            inputs, request = input_loader(study, study["train"], generation)
            evaluator = NativePortfolioPolicyEvaluator(inputs, initial_equity=study["initial_equity"], configuration=request)
            scored = evaluator(candidates, {})
            report["training_candidates"] = [asdict(row) for row in scored]
            valid = [row for row in scored if not row.rejected and execution_valid(row.metrics)
                     and row.metrics.get("capability_level") == "real_replay"]
            if valid:
                winner = max(valid, key=lambda row: (row.score, row.name))
                policy = next(c.mutations for c in candidates if c.name == winner.name)
                finalist = {"study_hash": study_hash, "name": winner.name, "mutations": policy,
                            "input_generation": generation.semantic_identity(), "training_request": winner.metrics["evaluation_request"]}
                finalist_hash = stable_signature(finalist)
                path = output / "finalist.json"
                path.write_text(json.dumps({**finalist, "finalist_hash": finalist_hash}, indent=2), encoding="utf-8")
                # Reload the saved artifact before qualification; never mutate it using held-out results.
                saved = json.loads(path.read_text(encoding="utf-8"))
                digest = saved.pop("finalist_hash")
                if stable_signature(saved) != digest or digest != finalist_hash:
                    raise ValueError("Saved native finalist identity mismatch")
                confirmed = evaluator.evaluate_mutations(saved["mutations"])
                if not execution_valid(confirmed, expected_request=winner.metrics["evaluation_request"]) or confirmed["immutable_score"] != winner.score:
                    raise ValueError("Reloaded native finalist failed training confirmation")
                validation_inputs, validation_request = input_loader(study, study["qualification"], generation)
                qualification = NativePortfolioPolicyEvaluator(validation_inputs, initial_equity=study["initial_equity"],
                    configuration=validation_request).evaluate_mutations(saved["mutations"])
                report.update(finalist_hash=finalist_hash, saved_finalist_confirmation=confirmed,
                              qualification=qualification, qualification_usage=study["qualification_usage"])
                report["historical_qualification"] = execution_valid(qualification) and qualification.get("capability_level") == "real_replay"
                report["status"] = "historical_regression_qualified" if report["historical_qualification"] else "finalist_unqualified"
            else:
                report["blocked_before_finalist"] = "no_audit_qualified_native_candidate"
    if generation.payload["source_generation_hash"] != source_generation_hash():
        raise ValueError("Source changed during native study")
    (output / "result.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
