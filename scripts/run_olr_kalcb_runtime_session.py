from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from check_live_readiness_manifest import check_manifest
from deployment.olr_kalcb.hashing import canonical_json_hash
from deployment.olr_kalcb.offline_replay import load_market_bars_for_replay, rebuild_offline_replay_from_session
from deployment.olr_kalcb.market_data_coordinator import KISMarketDataCoordinator, KISWebSocketCompletedBarSource
from deployment.olr_kalcb.portfolio import PortfolioPolicyConfig
from deployment.olr_kalcb.readiness import DEFAULT_ARTIFACT_ROOTS
from deployment.olr_kalcb.replay import replay_paper_session
from deployment.olr_kalcb.runtime import EXECUTION_MODES, OMS_HEALTH_PAYLOAD_KEYS, prepare_runtime_session
from deployment.olr_kalcb.session_capture import PaperSessionRecorder, market_bar_hash
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLRArtifactStore

DEFAULT_BASELINE_MANIFEST = Path(
    os.environ.get("OLR_KALCB_BASELINE_MANIFEST", "data/live_readiness/olr_kalcb/2026-05-28/baseline_manifest.json")
)
DEFAULT_PORTFOLIO_POLICY = Path("config/olr_kalcb/portfolio_policy.conservative.json")
DEFAULT_SECTOR_MAP = Path("config/olr/sector_map.yaml")
DEFAULT_SESSION_ROOT = Path("data/paper_live/olr_kalcb")
DEFAULT_STRATEGY_IDS = ("KALCB", "OLR")
DRY_RUN_HEALTH_CHECKS = ("artifact_only_gate_passed", "market_data_ok", "risk_limits_loaded")
MARKET_DATA_SOURCES = ("auto", "external_completed_bars", "kis_websocket")
KST = ZoneInfo("Asia/Seoul")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return _run_preflight(args)
        if args.command == "dry-run-bars":
            return _run_dry_run_bars(args)
        if args.command == "watch-bars":
            return _run_watch_bars(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _emit({"passed": False, "error_type": type(exc).__name__, "error": str(exc)})
        return 2
    parser.error(f"unsupported command {args.command!r}")
    return 2


def _run_preflight(args: argparse.Namespace) -> int:
    trade_date = date.fromisoformat(args.trade_date)
    mode = str(args.mode)
    manifest_check = check_manifest(_resolve_path(args.baseline_manifest))
    recorder = None
    if mode in EXECUTION_MODES and getattr(args, "session_root", None):
        recorder = _new_session_recorder(args, trade_date)
    plan = _prepare_plan(args, trade_date, mode=mode, recorder=recorder)
    payload = {
        "command": "preflight",
        "passed": bool(manifest_check["passed"] and plan.ready_to_start),
        "baseline_manifest": manifest_check,
        "runtime_plan": _plan_output(plan, full=bool(args.full_plan)),
    }
    _emit(payload, args.output_json)
    return 0 if payload["passed"] else 1


def _run_dry_run_bars(args: argparse.Namespace) -> int:
    trade_date = date.fromisoformat(args.trade_date)
    recorder = _new_session_recorder(args, trade_date)
    plan = _prepare_plan(args, trade_date, mode="dry_run", recorder=recorder)
    if not plan.ready_to_start:
        payload = {"command": "dry-run-bars", "passed": False, "runtime_plan": _plan_output(plan, full=bool(args.full_plan))}
        _emit(payload, args.output_json)
        return 1

    bars = load_market_bars_for_replay(_resolve_path(args.bars_parquet))
    bar_count = asyncio.run(_process_bars_with_operator_source(args, plan, recorder, bars))
    closeout_manifest = plan.close_session(
        _end_of_day_positions(plan),
        session_metrics={
            "operator_command": "dry-run-bars",
            "market_bar_count": bar_count,
            "market_data_source": _resolve_market_data_source(args, "dry_run"),
            "portfolio_policy_file": str(_resolve_path(args.portfolio_policy)) if args.portfolio_policy else "",
            "health_checks_file": str(_resolve_path(args.health_checks_json)) if args.health_checks_json else "",
        },
        closeout_reason="normal_eod_dry_run",
    )

    replay_report = None
    offline_replay_root = None
    if args.build_offline_replay:
        offline_replay_root = rebuild_offline_replay_from_session(recorder.paths.root)
        replay_report = replay_paper_session(recorder.paths.root)
    payload = {
        "command": "dry-run-bars",
        "passed": replay_report["paper_gate_passed"] if replay_report else True,
        "session_root": str(recorder.paths.root),
        "closeout_manifest": str(closeout_manifest),
        "market_bar_count": bar_count,
        "market_data_source": _resolve_market_data_source(args, "dry_run"),
        "offline_replay_root": str(offline_replay_root) if offline_replay_root else "",
        "parity_report": replay_report,
    }
    _emit(payload, args.output_json)
    return 0 if payload["passed"] else 1


def _run_watch_bars(args: argparse.Namespace) -> int:
    trade_date = date.fromisoformat(args.trade_date)
    mode = str(args.mode)
    recorder = _new_session_recorder(args, trade_date)
    oms_client = _make_cli_oms_client(args, mode)
    try:
        plan = _prepare_plan(args, trade_date, mode=mode, recorder=recorder, oms_client=oms_client)
        if not plan.ready_to_start:
            payload = {"command": "watch-bars", "passed": False, "runtime_plan": _plan_output(plan, full=bool(args.full_plan))}
            _emit(payload, args.output_json)
            return 1
        payload = asyncio.run(_watch_bars(args, plan, recorder))
        _emit(payload, args.output_json if args.once else None)
        return 0 if payload["passed"] else 1
    finally:
        close = getattr(oms_client, "close", None)
        if callable(close):
            try:
                asyncio.run(close())
            except RuntimeError:
                pass


async def _process_bars_with_operator_source(args: argparse.Namespace, plan: Any, recorder: PaperSessionRecorder, bars: Sequence[Any]) -> int:
    coordinator = await _make_market_data_coordinator(args, plan, recorder)
    try:
        return await _process_bars(plan, bars, coordinator=coordinator)
    finally:
        await _close_market_data_coordinator(coordinator)


async def _process_bars(plan: Any, bars: Sequence[Any], *, coordinator: KISMarketDataCoordinator | None = None) -> int:
    count = 0
    active_coordinator = coordinator or (
        KISMarketDataCoordinator(resource_plan=plan.kis_resource_plan, recorder=plan.session_recorder)
        if getattr(plan, "kis_resource_plan", None) is not None
        else None
    )
    for bar in bars:
        if active_coordinator is not None:
            await active_coordinator.route_completed_bar(plan, bar)
        else:
            await plan.handle_bar(bar)
        count += 1
    return count


async def _watch_bars(args: argparse.Namespace, plan: Any, recorder: PaperSessionRecorder) -> dict[str, Any]:
    seen = _existing_market_bar_hashes(recorder)
    coordinator = await _make_market_data_coordinator(args, plan, recorder)
    processed_total = 0
    poll_count = 0
    source = _resolve_market_data_source(args, plan.mode)
    try:
        if source == "kis_websocket":
            return await _watch_kis_websocket_bars(args, plan, recorder, coordinator)
        olr_final_enabled = False
        bars_path = _required_completed_bars_path(args, source)
        while True:
            poll_count += 1
            olr_final_enabled = _maybe_enable_olr_final(args, plan, coordinator=coordinator) or olr_final_enabled
            processed = 0
            loaded = 0
            if bars_path.is_file():
                bars = load_market_bars_for_replay(bars_path)
                loaded = len(bars)
                processed = await _process_unseen_bars(plan, bars, seen, coordinator=coordinator)
                processed_total += processed
            elif args.once:
                return {
                    "command": "watch-bars",
                    "passed": False,
                    "mode": plan.mode,
                    "market_data_source": source,
                    "session_root": str(recorder.paths.root),
                    "bars_parquet": str(bars_path),
                    "error": "bars parquet file is missing",
                }
            if args.once:
                closeout_manifest = ""
                if args.close_session_after_once:
                    closeout_manifest = str(
                        plan.close_session(
                            _end_of_day_positions(plan),
                            session_metrics={
                                "operator_command": "watch-bars --once",
                                "market_bar_count": processed_total,
                                "market_data_source": source,
                                "olr_final_enabled_during_session": olr_final_enabled,
                                "source_bars_parquet": str(bars_path),
                            },
                            closeout_reason="normal_eod_watch_bars_once",
                        )
                    )
                return {
                    "command": "watch-bars",
                    "passed": processed_total > 0,
                    "mode": plan.mode,
                    "market_data_source": source,
                    "session_root": str(recorder.paths.root),
                    "bars_parquet": str(bars_path),
                    "loaded_bar_count": loaded,
                    "processed_bar_count": processed_total,
                    "olr_final_enabled_during_session": olr_final_enabled,
                    "closeout_manifest": closeout_manifest,
                }
            print(
                json.dumps(
                    {
                        "event": "watch-bars-poll",
                        "mode": plan.mode,
                        "market_data_source": source,
                        "poll_count": poll_count,
                        "loaded_bar_count": loaded,
                        "processed_new_bar_count": processed,
                        "processed_bar_count": processed_total,
                        "olr_final_enabled_during_session": olr_final_enabled,
                        "bars_parquet": str(bars_path),
                        "session_root": str(recorder.paths.root),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            await asyncio.sleep(float(args.poll_seconds))
    finally:
        await _close_market_data_coordinator(coordinator)


async def _watch_kis_websocket_bars(
    args: argparse.Namespace,
    plan: Any,
    recorder: PaperSessionRecorder,
    coordinator: KISMarketDataCoordinator | None,
) -> dict[str, Any]:
    if coordinator is None or getattr(coordinator, "websocket_client", None) is None:
        raise RuntimeError("kis_websocket market-data source requires a coordinator-owned KISWebSocketClient")
    source = _resolve_market_data_source(args, plan.mode)
    bar_source = KISWebSocketCompletedBarSource(coordinator.websocket_client)
    processed_total = 0
    poll_count = 0
    olr_final_enabled = False
    await bar_source.start()
    try:
        while True:
            poll_count += 1
            olr_final_enabled = _maybe_enable_olr_final(args, plan, coordinator=coordinator) or olr_final_enabled
            await coordinator.activate_due_windows(datetime.now(tz=KST), runtime_plan=plan)
            bar = await bar_source.next_bar(timeout_s=float(args.poll_seconds))
            processed = 0
            if bar is not None:
                await coordinator.route_completed_bar(plan, bar)
                processed = 1
                processed_total += 1
            if args.once:
                closeout_manifest = ""
                if args.close_session_after_once:
                    closeout_manifest = str(
                        plan.close_session(
                            _end_of_day_positions(plan),
                            session_metrics={
                                "operator_command": "watch-bars --once",
                                "market_bar_count": processed_total,
                                "market_data_source": source,
                                "dropped_completed_bar_count": bar_source.dropped_bar_count,
                                "olr_final_enabled_during_session": olr_final_enabled,
                            },
                            closeout_reason="normal_eod_watch_bars_once",
                        )
                    )
                return {
                    "command": "watch-bars",
                    "passed": processed_total > 0,
                    "mode": plan.mode,
                    "market_data_source": source,
                    "session_root": str(recorder.paths.root),
                    "processed_bar_count": processed_total,
                    "dropped_completed_bar_count": bar_source.dropped_bar_count,
                    "olr_final_enabled_during_session": olr_final_enabled,
                    "closeout_manifest": closeout_manifest,
                }
            print(
                json.dumps(
                    {
                        "event": "watch-bars-poll",
                        "mode": plan.mode,
                        "market_data_source": source,
                        "poll_count": poll_count,
                        "processed_new_bar_count": processed,
                        "processed_bar_count": processed_total,
                        "dropped_completed_bar_count": bar_source.dropped_bar_count,
                        "olr_final_enabled_during_session": olr_final_enabled,
                        "session_root": str(recorder.paths.root),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        await bar_source.stop()


async def _process_unseen_bars(
    plan: Any,
    bars: Sequence[Any],
    seen: set[str],
    *,
    coordinator: KISMarketDataCoordinator | None = None,
) -> int:
    processed = 0
    for bar in bars:
        key = market_bar_hash(bar)
        if key in seen:
            continue
        if coordinator is not None:
            await coordinator.route_completed_bar(plan, bar)
        else:
            await plan.handle_bar(bar)
        seen.add(key)
        processed += 1
    return processed


async def _make_market_data_coordinator(
    args: argparse.Namespace,
    plan: Any,
    recorder: PaperSessionRecorder,
) -> KISMarketDataCoordinator | None:
    resource_plan = getattr(plan, "kis_resource_plan", None)
    if resource_plan is None:
        return None
    source = _resolve_market_data_source(args, plan.mode)
    websocket_client = None
    if source == "kis_websocket":
        websocket_client = _make_kis_websocket_client()
        url = str(getattr(args, "kis_ws_url", "") or _kis_websocket_url(websocket_client))
        if not url:
            raise RuntimeError("KIS WebSocket source requires a KIS WebSocket URL")
        if not await websocket_client.connect(url):
            raise RuntimeError(f"failed to connect KIS WebSocket market-data source: {url}")
    return KISMarketDataCoordinator(
        resource_plan=resource_plan,
        recorder=recorder,
        websocket_client=websocket_client,
        ledger_path=_ws_ledger_path(args, recorder),
        market_data_source=source,
    )


async def _close_market_data_coordinator(coordinator: KISMarketDataCoordinator | None) -> None:
    if coordinator is None:
        return
    await coordinator.release_all()
    websocket_client = getattr(coordinator, "websocket_client", None)
    disconnect = getattr(websocket_client, "disconnect", None)
    if callable(disconnect):
        await disconnect()


def _resolve_market_data_source(args: argparse.Namespace, mode: str) -> str:
    source = str(getattr(args, "market_data_source", "auto") or "auto")
    if source != "auto":
        return source
    return "kis_websocket" if str(mode or "").lower() in {"paper", "live"} else "external_completed_bars"


def _maybe_enable_olr_final(
    args: argparse.Namespace,
    plan: Any,
    *,
    coordinator: KISMarketDataCoordinator | None = None,
) -> bool:
    if not _olr_final_enablement_pending(plan):
        return False
    root = _olr_artifact_root(args)
    path = OLRArtifactStore(root).path_for(getattr(plan, "trade_date"), artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    if not path.is_file():
        return False
    plan.enable_olr_final(artifact_root=root)
    if coordinator is not None:
        coordinator.sync_runtime_plan(plan)
    return True


def _olr_final_enablement_pending(plan: Any) -> bool:
    if str(getattr(plan, "mode", "") or "").lower() not in EXECUTION_MODES:
        return False
    summaries = {str(key).upper().strip() for key in dict(getattr(plan, "strategy_config_summaries", {}) or {})}
    if "OLR" not in summaries:
        return False
    if "OLR" in {str(key).upper().strip() for key in dict(getattr(plan, "drivers", {}) or {})}:
        return False
    return callable(getattr(plan, "enable_olr_final", None))


def _olr_artifact_root(args: argparse.Namespace) -> Path:
    raw = getattr(args, "olr_artifact_root", None)
    if raw:
        return _resolve_path(raw)
    return _resolve_path(DEFAULT_ARTIFACT_ROOTS["OLR"])


def _required_completed_bars_path(args: argparse.Namespace, source: str) -> Path:
    raw = getattr(args, "bars_parquet", None)
    if not raw:
        raise ValueError(f"--bars-parquet is required when market-data-source={source}")
    return _resolve_path(raw)


def _make_kis_websocket_client() -> Any:
    from kis_core.kis_auth import KoreaInvestEnv, build_kis_config_from_env
    from kis_core.kis_client import KoreaInvestAPI
    from kis_core.ws_client import KISWebSocketClient

    env = KoreaInvestEnv(build_kis_config_from_env())
    return KISWebSocketClient(KoreaInvestAPI(env))


def _kis_websocket_url(websocket_client: Any) -> str:
    api = getattr(websocket_client, "api", None)
    env = getattr(api, "env", None)
    return str(getattr(env, "ws_url", "") or "")


def _ws_ledger_path(args: argparse.Namespace, recorder: PaperSessionRecorder) -> Path:
    raw = getattr(args, "ws_ledger_path", None) or os.environ.get("KIS_WS_LEDGER_PATH")
    return _resolve_path(raw) if raw else recorder.paths.root / "kis_ws_ledger.json"


def _existing_market_bar_hashes(recorder: PaperSessionRecorder) -> set[str]:
    path = recorder.paths.root / "decision_stream.jsonl"
    if not path.is_file():
        return set()
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("record_type") or "") != "runtime_event_input":
            continue
        if str(row.get("event_type") or "") != "bar":
            continue
        bar_hash = str(row.get("bar_hash") or "")
        if bar_hash:
            hashes.add(bar_hash)
    return hashes


def _prepare_plan(
    args: argparse.Namespace,
    trade_date: date,
    *,
    mode: str,
    recorder: PaperSessionRecorder | None,
    oms_client: Any | None = None,
) -> Any:
    strategy_config_source = _resolve_path(args.strategy_config_source or args.baseline_manifest)
    sector_map = _load_sector_map(args.sector_map)
    portfolio_policy = _load_portfolio_policy(args.portfolio_policy)
    health_checks = _load_health_checks(args.health_checks_json, mode=mode, fixture_health_ok=bool(args.fixture_health_ok))
    health_checks = _with_oms_health_payload(health_checks, args, mode=mode)
    initial_account_state = _load_mapping(args.account_state_json, required=False) if args.account_state_json else None
    initial_positions = _load_mapping(args.positions_json, required=False) if args.positions_json else None
    return prepare_runtime_session(
        args.strategy_ids,
        trade_date=trade_date,
        mode=mode,
        artifact_roots=_artifact_roots(args),
        health_checks=health_checks,
        oms_client=oms_client,
        session_recorder=recorder,
        portfolio_config=portfolio_policy,
        strategy_config_source=strategy_config_source,
        completed_bar_source=_resolve_market_data_source(args, mode),
        sector_map=sector_map,
        initial_account_state=initial_account_state,
        initial_positions=initial_positions,
        assistant_event_dir=_assistant_event_dir(args) if recorder is not None else None,
        deployment_metadata_path=getattr(args, "deployment_metadata_json", None),
        deployment_metadata_contract_path=getattr(args, "strategy_plugin_contract", None),
        deployment_metadata_environment=getattr(args, "deployment_metadata_environment", None),
        runtime_entrypoint=f"scripts/run_olr_kalcb_runtime_session.py:{args.command}",
    )


def _make_cli_oms_client(args: argparse.Namespace, mode: str) -> Any | None:
    if mode not in {"paper", "live"}:
        return None
    from oms_client.client import OMSClient

    return OMSClient(str(args.oms_url or os.environ.get("OMS_URL") or "http://oms:8000"))


def _load_mapping(path: str | Path | None, *, required: bool = True) -> dict[str, Any]:
    if path is None:
        if required:
            raise ValueError("required JSON/YAML path was not provided")
        return {}
    resolved = _resolve_path(path)
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(resolved)
        return {}
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML operator inputs") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text or "{}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved} must contain a JSON/YAML object")
    return dict(payload)


def _load_sector_map(path: str | Path | None) -> dict[str, str]:
    payload = _load_mapping(path, required=True) if path else {}
    raw = payload.get("sector_map", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("sector map file must contain a mapping or a top-level sector_map object")
    return {
        str(symbol).zfill(6): str(sector).upper().strip()
        for symbol, sector in raw.items()
        if str(sector or "").strip()
    }


def _load_portfolio_policy(path: str | Path | None) -> PortfolioPolicyConfig:
    if path is None:
        return PortfolioPolicyConfig()
    payload = _load_mapping(path, required=True)
    raw = payload.get("portfolio_policy", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("portfolio policy file must contain a mapping or a top-level portfolio_policy object")
    return _portfolio_config_from_payload(raw)


def _portfolio_config_from_payload(payload: Mapping[str, Any]) -> PortfolioPolicyConfig:
    allowed = set(PortfolioPolicyConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown portfolio policy fields: {', '.join(unknown)}")
    data = {key: value for key, value in payload.items() if key in allowed}
    if "strategy_priority" in data:
        data["strategy_priority"] = tuple(str(item).upper().strip() for item in data["strategy_priority"] or ())
    return PortfolioPolicyConfig(**data)


def _load_health_checks(path: str | Path | None, *, mode: str, fixture_health_ok: bool) -> dict[str, Any]:
    checks = _load_mapping(path, required=False) if path else {}
    if fixture_health_ok:
        if mode != "dry_run":
            raise ValueError("--fixture-health-ok is only valid for non-promotional dry-run rehearsals")
        for name in DRY_RUN_HEALTH_CHECKS:
            checks[name] = {
                "passed": True,
                "detail": "fixture-only operator override; not paper/live promotional evidence",
            }
    return checks


def _with_oms_health_payload(checks: Mapping[str, Any], args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    augmented = dict(checks or {})
    if str(mode or "").lower() not in {"paper", "live"}:
        return augmented
    if any(key in augmented for key in OMS_HEALTH_PAYLOAD_KEYS):
        return augmented
    try:
        augmented["oms_health_payload"] = _fetch_oms_health_payload(_oms_health_url(args))
    except Exception as exc:
        augmented["oms_health_payload_error"] = str(exc)
    return augmented


def _oms_health_url(args: argparse.Namespace) -> str:
    base = str(getattr(args, "oms_url", None) or os.environ.get("OMS_URL") or "http://oms:8000").rstrip("/")
    if base.endswith("/health"):
        return base
    return f"{base}/health"


def _fetch_oms_health_payload(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=5) as response:
        text = response.read().decode("utf-8")
    payload = json.loads(text or "{}")
    if not isinstance(payload, Mapping):
        raise ValueError("OMS /health response must be a JSON object")
    return dict(payload)


def _artifact_roots(args: argparse.Namespace) -> dict[str, Path] | None:
    roots: dict[str, Path] = {}
    if args.kalcb_artifact_root:
        roots["KALCB"] = _resolve_path(args.kalcb_artifact_root)
    if args.olr_artifact_root:
        roots["OLR"] = _resolve_path(args.olr_artifact_root)
    return roots or None


def _end_of_day_positions(plan: Any) -> dict[str, Any]:
    positions: list[dict[str, Any]] = []
    for strategy_id, descriptor in plan.descriptors.items():
        engine_state = getattr(getattr(descriptor, "engine", None), "state", None)
        symbols = getattr(engine_state, "symbols", {}) or {}
        for symbol, state in dict(symbols).items():
            position = getattr(state, "position", None)
            if position is None or int(getattr(position, "qty_open", 0) or 0) <= 0:
                continue
            payload = asdict(position) if is_dataclass(position) else dict(position)
            positions.append({"strategy_id": strategy_id, "symbol": str(symbol).zfill(6), **payload})
    return {"positions": positions}


def _plan_output(plan: Any, *, full: bool) -> dict[str, Any]:
    payload = plan.to_json_dict()
    if full:
        return payload
    sector_map = dict(payload.get("sector_map") or {})
    payload["sector_map"] = {
        "count": len(sector_map),
        "hash": canonical_json_hash(sector_map) if sector_map else "",
    }
    return payload


def _session_root(args: argparse.Namespace, trade_date: date) -> Path:
    raw = getattr(args, "session_root", None)
    if raw:
        return _resolve_path(raw)
    return _resolve_path(DEFAULT_SESSION_ROOT / trade_date.isoformat())


def _new_session_recorder(args: argparse.Namespace, trade_date: date) -> PaperSessionRecorder:
    return PaperSessionRecorder(
        _session_root(args, trade_date),
        trade_date,
        assistant_event_dir=_assistant_event_dir(args),
    )


def _assistant_event_dir(args: argparse.Namespace) -> Path | None:
    raw = getattr(args, "assistant_event_data_dir", None)
    if raw is None:
        raw = os.environ.get("ASSISTANT_EVENT_DATA_DIR", "instrumentation/data")
    if str(raw).strip().lower() in {"", "off", "none", "disabled"}:
        return None
    return _resolve_path(raw)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def _emit(payload: Mapping[str, Any], output_json: str | Path | None = None) -> None:
    if output_json:
        target = _resolve_path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict OLR/KALCB artifact, dry-run, and replay deployment gates."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Verify baseline manifest and runtime readiness.")
    _add_common_args(preflight)
    preflight.add_argument("--mode", choices=("artifact_only", "artifact_only_stage1", "dry_run", "paper", "live"), default="artifact_only")
    preflight.add_argument("--session-root", help="Optional exact session directory for execution-mode preflight.")

    dry_run = subparsers.add_parser("dry-run-bars", help="Process captured 5m bars through routed dry-run OMS capture.")
    _add_common_args(dry_run)
    _add_market_data_args(dry_run)
    dry_run.add_argument("--bars-parquet", required=True, help="Completed 5m bars to feed through the deployment market-data coordinator.")
    dry_run.add_argument("--session-root", help="Exact session directory. Defaults to data/paper_live/olr_kalcb/YYYY-MM-DD.")
    dry_run.add_argument("--build-offline-replay", action="store_true", help="Rebuild offline replay and write parity_report.json after closeout.")

    watch = subparsers.add_parser("watch-bars", help="Continuously process new completed bars from a parquet file through the routed runtime.")
    _add_common_args(watch)
    _add_market_data_args(watch)
    watch.add_argument("--mode", choices=("dry_run", "paper", "live"), default="dry_run")
    watch.add_argument("--bars-parquet", help="Completed 5m bars parquet to poll when using external_completed_bars.")
    watch.add_argument("--session-root", help="Exact session directory. Defaults to data/paper_live/olr_kalcb/YYYY-MM-DD.")
    watch.add_argument("--poll-seconds", type=float, default=15.0)
    watch.add_argument("--once", action="store_true", help="Process the current file once and exit.")
    watch.add_argument("--close-session-after-once", action="store_true", help="Close and seal the session after a one-shot watch-bars run.")
    watch.add_argument("--oms-url", default=os.environ.get("OMS_URL", "http://oms:8000"), help="OMS base URL for paper/live modes.")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trade-date", required=True, help="KRX trade date, YYYY-MM-DD.")
    parser.add_argument("--strategy-ids", nargs="+", default=list(DEFAULT_STRATEGY_IDS), help="Strategy ids to stage; defaults to KALCB OLR.")
    parser.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST), help="Frozen approved optimized-config manifest.")
    parser.add_argument("--strategy-config-source", help="Override strategy config manifest; defaults to --baseline-manifest.")
    parser.add_argument("--portfolio-policy", default=str(DEFAULT_PORTFOLIO_POLICY), help="PortfolioPolicyConfig JSON/YAML.")
    parser.add_argument("--sector-map", default=str(DEFAULT_SECTOR_MAP), help="Approved sector map JSON/YAML.")
    parser.add_argument("--kalcb-artifact-root", help="Override KALCB candidate artifact root.")
    parser.add_argument("--olr-artifact-root", help="Override OLR candidate artifact root.")
    parser.add_argument("--health-checks-json", help="Health-check status JSON/YAML for execution modes.")
    parser.add_argument("--account-state-json", help="Captured initial AccountState JSON/YAML for execution modes.")
    parser.add_argument("--positions-json", help="Captured initial positions JSON/YAML for execution modes.")
    parser.add_argument("--fixture-health-ok", action="store_true", help="Dry-run fixture rehearsal only; not promotional evidence.")
    parser.add_argument("--full-plan", action="store_true", help="Include full runtime plan detail, including the complete sector map.")
    parser.add_argument("--output-json", help="Optional path to save the command result JSON.")
    parser.add_argument(
        "--assistant-event-data-dir",
        default=os.environ.get("ASSISTANT_EVENT_DATA_DIR", "instrumentation/data"),
        help="Canonical assistant telemetry directory; use 'off' to disable local export.",
    )
    parser.add_argument(
        "--deployment-metadata-json",
        default=os.environ.get("OLR_KALCB_DEPLOYMENT_METADATA_PATH"),
        help="Optional approval-grade deployment_metadata.json output path for paper/live VPS runs.",
    )
    parser.add_argument(
        "--strategy-plugin-contract",
        default=os.environ.get("OLR_KALCB_STRATEGY_PLUGIN_CONTRACT"),
        help="Strategy plugin contract JSON whose SHA256 is recorded in deployment metadata.",
    )
    parser.add_argument(
        "--deployment-metadata-environment",
        default=os.environ.get("OLR_KALCB_DEPLOYMENT_METADATA_ENV"),
        help="Optional deployment metadata environment: live_bot, vps, paper_vps, or production_vps.",
    )


def _add_market_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--market-data-source",
        choices=MARKET_DATA_SOURCES,
        default="auto",
        help="auto uses external completed bars for dry-run and coordinator-owned KIS WebSocket subscriptions for paper/live.",
    )
    parser.add_argument("--kis-ws-url", help="Override the KIS WebSocket URL used when --market-data-source=kis_websocket.")
    parser.add_argument(
        "--ws-ledger-path",
        help="Shared WebSocket registration ledger. Defaults to KIS_WS_LEDGER_PATH or the session directory ledger.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
