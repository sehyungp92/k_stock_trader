"""Fixtures prove the qualification mechanism, not historical market capacity."""
from dataclasses import replace, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile

import pytest

from backtests.auto.shared.types import Experiment
from backtests.auto.shared.plugin_utils import CachedBatchEvaluator
from backtests.auto.shared.phase_runner import _metric_contract_criteria
from kis_core.input_manifest import InputManifest, read_input_bytes
from backtests.core.replay_bundle import EventReplayBundle
from backtests.core.replay_events import ReplayEvent
from backtests.engine.contracts import execution_valid
from backtests.strategies.common.synthetic import make_kalcb_synthetic_bars
from backtests.strategies.kalcb.runner import _synthetic_snapshot
from backtests.strategies.kalcb import worker
from strategy_common.market import MarketBar
from strategy_kalcb.models import KALCBDailySnapshot


@pytest.fixture
def evidence_fixture(tmp_path):
    root = Path(__file__).resolve().parents[2]
    # InputManifest preserves source-relative names; keep fixture inputs in a
    # unique test-owned directory and the immutable generation in pytest's temp.
    cache = root / ".pytest_cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qualification-", dir=cache) as directory:
        folder = Path(directory)
        prefix = folder.relative_to(root).as_posix()
        bars = make_kalcb_synthetic_bars("005930")
        last = bars[-1]
        for i in range(12, 79):
            bars.append(replace(last, timestamp=bars[0].timestamp + timedelta(minutes=5*i)))
        snapshot = _synthetic_snapshot(bars, "qualification-test-fixture")
        snapshot = replace(snapshot, candidates=tuple(replace(c, metadata={**c.metadata, "prior_day_date": "2026-01-02"}) for c in snapshot.candidates))
        events = [bar.to_json_dict() for bar in bars]
        candidate = snapshot.candidates[0]
        payloads = {
            "events.json": events, "raw.json": events, "snapshot.json": snapshot.to_json_dict(),
            "daily.json": [{"symbol": candidate.symbol, "date": "2026-01-02", "high": candidate.prior_day_high,
                "low": candidate.prior_day_low, "close": candidate.prior_day_close}],
            "source.json": {"kind": "unit_test_fixture", "notice": "Invented observations for contract tests only"},
            "basis.json": [{"symbol": e["symbol"], "timestamp": e["timestamp"], "price_unit": "KRW", "volume_unit": "shares",
                "volume_kind": "interval", "venue": "KRX", "price_factor": 1, "share_factor": 1,
                "source_path": f"{prefix}/raw.json", "source_index": i} for i, e in enumerate(events)],
            "capacity.json": [{"symbol": e["symbol"], "timestamp": e["timestamp"], "phase": phase,
                "volume_kind": "phase_executable_shares", "volume": 100_000_000,
                "price_low": e["low"], "price_high": e["high"]} for e in events for phase in ("open", "intrabar", "auction")],
        }
        from backtests.engine.qualification_evidence import execution_policy
        from backtests.engine.sim_broker import SimBroker, BrokerCosts
        payloads["phase-raw.json"] = [dict(row) for row in payloads["capacity.json"]]
        for i, row in enumerate(payloads["capacity.json"]):
            row.update(source_path=f"{prefix}/phase-raw.json", source_index=i)
        payloads["source.json"] = {"kind": "market_basis_contract_v1", "provider": "synthetic-test-only",
            "source_url": "fixture://invented-market-observations", "parser_version": "fixture-v1",
            "venue": "KRX", "raw_paths": [f"{prefix}/raw.json"], "timestamp_convention": "completed_interval_start",
            "timezone": "Asia/Seoul", "price_unit": "KRW", "volume_unit": "shares", "volume_kind": "interval"}
        payloads["phase-source.json"] = {"kind": "phase_liquidity_contract_v1", "provider": "synthetic-test-only",
            "source_url": "fixture://invented-phase-observations", "parser_version": "fixture-v1",
            "venue": "KRX", "raw_paths": [f"{prefix}/phase-raw.json"], "volume_unit": "shares",
            "phases": ["open", "intrabar", "auction"]}
        payloads["daily-raw.json"] = [dict(r) for r in payloads["daily.json"]]
        for i, row in enumerate(payloads["daily.json"]):
            row.update(source_path=f"{prefix}/daily-raw.json", source_index=i)
        payloads["daily-source.json"] = {"kind": "daily_reference_contract_v1", "provider": "synthetic-test-only",
            "source_url": "fixture://invented-daily-observations", "parser_version": "fixture-v1", "venue": "KRX",
            "raw_paths": [f"{prefix}/daily-raw.json"], "price_unit": "KRW", "date_convention": "KRX_session_date"}
        for name in ("source.json", "daily-source.json", "phase-source.json"):
            payloads[name]["normalization"] = {"source_basis_id": "fixture-shares-20260105",
                "target_basis": {"id": "fixture-shares-20260105", "price_unit": "KRW", "volume_unit": "shares",
                                 "adjustment_as_of": "2026-01-05"}, "transforms": []}
        policy = execution_policy(SimBroker(100_000_000, BrokerCosts(commission_bps=2, slippage_bps=3)))
        proof = {"version": "qualification-observations-v2", "market_events": f"{prefix}/events.json",
            "sessions": [{"symbol": "005930", "date": "2026-01-05", "timeframe": "5m"}],
            "references": {"snapshots": [f"{prefix}/snapshot.json"], "daily_rows": f"{prefix}/daily.json", "source_documents": [f"{prefix}/daily-source.json"]},
            "basis": {"source_documents": [f"{prefix}/source.json"], "observations": f"{prefix}/basis.json"},
            "capacity": {"kind": "observed_phase_liquidity", "source_documents": [f"{prefix}/phase-source.json"],
                "registered_execution_policies": [{"policy": policy, "participation": 1.0}], "observations": f"{prefix}/capacity.json"}}
        payloads["qualification.json"] = proof
        for name, payload in payloads.items():
            (folder / name).write_text(json.dumps(payload), encoding="utf-8")
        import yaml
        calendar_files = list((root / "kis_core/data").glob("*.yaml"))
        halt_sources = yaml.safe_load((root / "kis_core/data/krx_intraday_halts.yaml").read_text(encoding="utf-8"))["evidence_files"]
        generation = InputManifest.create([*folder.glob("*.json"), *calendar_files, *(root / p for p in halt_sources)],
            source_root=root, destination=tmp_path/"generation", evidence={"qualification_path": f"{prefix}/qualification.json"})
        with generation.activate():
            loaded = json.loads(read_input_bytes(folder/"events.json"))
            loaded_bars = [MarketBar(**{**r, "timestamp": datetime.fromisoformat(r["timestamp"])}) for r in loaded]
            loaded_snapshot = KALCBDailySnapshot.from_json_dict(json.loads(read_input_bytes(folder/"snapshot.json")))
        bundle = EventReplayBundle(events=tuple(ReplayEvent.from_bar(b) for b in loaded_bars), source_fingerprint="qualification-test-fixture",
            metadata={"capability_level": "synthetic", "input_manifest": generation.receipt(), "kalcb_candidate_snapshots": {loaded_snapshot.trade_date.isoformat(): loaded_snapshot.to_json_dict()}})
        yield generation, bundle, prefix


def test_complete_fixture_passes_real_worker_cache_and_saved_request(evidence_fixture, monkeypatch):
    generation, bundle, _ = evidence_fixture
    config = {"capability_level": "synthetic", "input_manifest_path": str(generation.path),
              "symbols": ["005930"], "start": "2026-01-05", "end": "2026-01-05"}
    monkeypatch.setattr(worker, "_worker_config", config)
    monkeypatch.setattr(worker, "_worker_replay_bundle", bundle)
    monkeypatch.setattr(worker, "_worker_hard_rejects", {"min_trades": 1})
    candidate = worker.score_candidate(("baseline", {}, {}))
    assert candidate.metrics["audit_pass"], candidate.metrics["audit_receipt"]["qualification"]
    assert candidate.metrics["entry_count"] > 0
    assert not candidate.rejected
    evaluator = CachedBatchEvaluator(lambda items, base: [worker.score_candidate((item.name, item.mutations, base)) for item in items])
    cold = evaluator([Experiment("baseline", {})], {})[0]
    warm = evaluator([Experiment("baseline", {})], {})[0]
    assert execution_valid(cold.metrics) and execution_valid(warm.metrics)
    assert cold.metrics["evaluation_request"] == warm.metrics["evaluation_request"]
    # A passing execution fixture remains ineligible as historical evidence.
    gates = {c.name: c.passed for c in _metric_contract_criteria({**warm.metrics, "promotion_requires_audit_pass": True})}
    assert gates["hard_audit_pass"] and not gates["hard_non_synthetic_capability"]


@pytest.mark.parametrize("name", ["events", "basis", "daily", "capacity"])
def test_modified_frozen_evidence_fails_actual_worker(evidence_fixture, monkeypatch, name):
    generation, bundle, prefix = evidence_fixture
    path = generation.root / prefix / f"{name}.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(worker, "_worker_config", {"capability_level": "synthetic", "input_manifest_path": str(generation.path)})
    monkeypatch.setattr(worker, "_worker_replay_bundle", bundle)
    from kis_core.input_manifest import _VERIFIED_BYTES
    _VERIFIED_BYTES.clear()
    result = worker.score_candidate(("tampered", {}, {}))
    assert result.rejected and not result.metrics["audit_pass"]
