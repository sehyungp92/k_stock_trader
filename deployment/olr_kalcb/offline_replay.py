from __future__ import annotations

import json
import shutil
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.engine import KALCBEngine
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_olr.config import OLRConfig
from strategy_olr.engine import OLREngine
from strategy_olr.models import OLRDailySnapshot

from .action_router import RuntimeActionRouter
from .coordinator import StrategyRuntimeDescriptor
from .dry_run_oms import RecordingOMSClient
from .hashing import canonical_json_hash
from .portfolio import PortfolioArbitrationPolicy, PortfolioPolicyConfig
from .portfolio_context import PortfolioContextProvider
from .session_capture import PaperSessionRecorder, market_bar_hash, missing_artifact_evidence, session_hashes
from .session_driver import RuntimeSessionDriver, handle_combined_bar

OFFLINE_REPLAY_ENGINE_VERSION = "olr-kalcb-offline-replay-v1"
OFFLINE_REPLAY_MANIFEST = "replay_manifest.json"
OFFLINE_REPLAY_DIR = "offline_replay"


def rebuild_offline_replay_from_session(session_root: str | Path) -> Path:
    """Regenerate offline replay streams from captured artifacts and 5m bars."""

    return asyncio.run(_rebuild_offline_replay_from_session(Path(session_root)))


def load_market_bars_for_replay(path: str | Path) -> list[MarketBar]:
    """Load completed 5m market bars using the same parser as offline replay."""

    return _load_market_bars(Path(path))


async def _rebuild_offline_replay_from_session(root: Path) -> Path:
    loader = ReplayInputLoader(root)
    trade_date = loader.trade_date
    bars = loader.load_market_bars()
    snapshots = loader.load_snapshots()
    if not snapshots:
        raise FileNotFoundError("no KALCB/OLR candidate snapshots found in session bundle")
    configs = loader.load_configs(tuple(snapshots))
    states = loader.load_initial_states(tuple(snapshots))

    offline_root = root / OFFLINE_REPLAY_DIR
    if offline_root.exists():
        shutil.rmtree(offline_root)
    recorder = PaperSessionRecorder(offline_root, trade_date)
    initial_account = loader.initial_account_state()
    initial_positions = loader.initial_positions()
    startup_snapshot = loader.initial_working_order_snapshot()
    if startup_snapshot["working_orders"]:
        initial_positions = _positions_with_startup_working_orders(initial_positions, startup_snapshot["working_orders"])
    portfolio_enabled = loader.portfolio_enabled()
    portfolio_config = loader.portfolio_config() if portfolio_enabled else None
    oms = RecordingOMSClient(recorder, account_state=initial_account, positions=dict(initial_positions))
    router = RuntimeActionRouter(
        recorder=recorder,
        oms_client=oms,
        portfolio_policy=PortfolioArbitrationPolicy(portfolio_config) if portfolio_enabled else None,
        portfolio_enabled=portfolio_enabled,
        dry_run=True,
    )
    context = PortfolioContextProvider(oms_client=oms, sector_map=loader.sector_map())
    context.account_state = initial_account
    context.positions = dict(initial_positions)
    startup_working_orders = startup_snapshot["working_orders"] or context.iter_working_orders()
    descriptors = _descriptors_for_snapshots(snapshots, configs=configs, states=states)

    drivers = {
        strategy_id: RuntimeSessionDriver(
            descriptor=descriptor,
            action_router=router,
            recorder=recorder,
            portfolio_context=context,
            mode="offline_replay",
            evidence_mode=loader.capture_mode,
        )
        for strategy_id, descriptor in descriptors.items()
    }

    for strategy_id, descriptor in descriptors.items():
        router.record_state_snapshot(
            strategy_id,
            descriptor.engine.state,
            metadata={
                "record_reason": "runtime_session_pre_start",
                "mode": loader.capture_mode,
                "trade_date": trade_date.isoformat(),
                "artifact_stage": descriptor.artifact_stage,
                "artifact_hash": descriptor.artifact_hash,
            },
        )

    timer_events = loader.load_timer_events()
    runtime_events = loader.load_runtime_events()
    if not runtime_events:
        raise ValueError("offline replay requires driver-recorded runtime_event_input rows")
    events = runtime_events
    fill_replayed = False
    timer_replayed = False
    for _timestamp, kind, payload in events:
        if kind == "combined_bar":
            strategy_ids, bar = payload
            await handle_combined_bar(drivers, bar, target_strategy_ids=strategy_ids)
        elif kind == "strategy_bar":
            strategy_id, bar = payload
            if strategy_id not in drivers:
                raise ValueError(f"bar event references unavailable strategy {strategy_id}")
            await drivers[strategy_id].handle_bar(bar)
        elif kind == "timer":
            strategy_id, timestamp = payload
            if strategy_id not in drivers:
                raise ValueError(f"timer event references unavailable strategy {strategy_id}")
            await drivers[strategy_id].handle_timer(timestamp)
            timer_replayed = True
        elif kind == "fill":
            strategy_id, fill = payload
            if strategy_id not in drivers:
                raise ValueError(f"fill event references unavailable strategy {strategy_id}")
            await drivers[strategy_id].handle_fill(fill)
            fill_replayed = True
        elif kind == "order_event":
            strategy_id, event = payload
            if strategy_id not in drivers:
                raise ValueError(f"order event references unavailable strategy {strategy_id}")
            await drivers[strategy_id].handle_order_event(event)

    recorder.write_end_of_day_positions(_end_positions({sid: descriptor.engine for sid, descriptor in descriptors.items()}))
    write_offline_replay_manifest(
        offline_root,
        source_session=root,
        metadata={
            "input_hashes": session_hashes(root),
            "strategy_ids": sorted(descriptors),
            "portfolio_enabled": portfolio_enabled,
            "portfolio_policy_config": asdict(portfolio_config) if portfolio_config is not None else None,
            "portfolio_policy_hash": PortfolioArbitrationPolicy(portfolio_config).policy_hash if portfolio_config is not None else None,
            "sector_map": loader.sector_map(),
            "startup_working_order_count": len(startup_working_orders),
            "startup_working_order_source": startup_snapshot["source"] or "",
            "startup_working_order_hash": canonical_json_hash(startup_working_orders),
            "market_bar_count": len(bars),
            "timer_event_count": len(timer_events),
            "fill_replay_status": "replayed" if fill_replayed else "not_applicable_no_fill_events",
            "timer_replay_status": "replayed" if timer_replayed else "not_applicable_no_timer_events",
            "order_event_replay_status": "replayed_or_absent",
            "driver_replay": True,
        },
    )
    return offline_root


@dataclass(slots=True)
class ReplayInputLoader:
    root: Path
    _manifest: dict[str, Any] = field(init=False, repr=False)
    _market_bars: list[MarketBar] | None = field(default=None, init=False, repr=False)
    _market_bars_by_hash_cache: dict[str, MarketBar] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._manifest = _load_session_manifest(self.root)

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def trade_date(self) -> date:
        return date.fromisoformat(str(self.manifest.get("trade_date")))

    @property
    def capture_mode(self) -> str:
        return str(self.manifest.get("mode") or self.manifest.get("runtime_mode") or "dry_run")

    def load_market_bars(self) -> list[MarketBar]:
        if self._market_bars is None:
            self._market_bars = _load_market_bars(self.root / "market_bars_5m.parquet")
        return list(self._market_bars)

    def load_snapshots(self) -> dict[str, KALCBDailySnapshot | OLRDailySnapshot]:
        missing = missing_artifact_evidence(self.root, self.manifest.get("strategy_ids"))
        if missing:
            raise FileNotFoundError(f"incomplete artifact evidence for offline replay: {', '.join(missing)}")
        return _load_snapshots(self.root, self.trade_date)

    def load_configs(self, strategy_ids: tuple[str, ...]) -> dict[str, KALCBConfig | OLRConfig]:
        raw_configs = dict(
            self.manifest.get("strategy_configs")
            or self.manifest.get("captured_strategy_configs")
            or {}
        )
        configs: dict[str, KALCBConfig | OLRConfig] = {}
        for strategy_id in strategy_ids:
            if strategy_id in raw_configs:
                raw = raw_configs[strategy_id]
            elif strategy_id.lower() in raw_configs:
                raw = raw_configs[strategy_id.lower()]
            else:
                raw = self._load_config_file(strategy_id)
            if raw is None:
                raise ValueError(f"missing captured strategy config for {strategy_id}")
            wrapper = _load_config_reference(self.root, raw)
            expected_hash = _config_expected_hash(wrapper)
            payload = _config_mutation_payload(wrapper)
            if not payload and not _config_uses_defaults(wrapper):
                raise ValueError(f"missing captured strategy config payload for {strategy_id}")
            if expected_hash and canonical_json_hash(payload) != expected_hash:
                raise ValueError(f"captured strategy config hash mismatch for {strategy_id}")
            if strategy_id == "KALCB":
                configs[strategy_id] = KALCBConfig.from_mapping(payload, {})
            elif strategy_id == "OLR":
                configs[strategy_id] = OLRConfig.from_mapping(payload, {})
        return configs

    def portfolio_enabled(self) -> bool:
        if "portfolio_enabled" not in self.manifest:
            raise ValueError("missing captured portfolio_enabled policy")
        return bool(self.manifest.get("portfolio_enabled"))

    def portfolio_config(self) -> PortfolioPolicyConfig:
        payload = self.manifest.get("portfolio_policy_config")
        if not isinstance(payload, dict) or not payload:
            raise ValueError("missing captured portfolio policy config")
        allowed = set(PortfolioPolicyConfig.__dataclass_fields__)
        data = {key: value for key, value in payload.items() if key in allowed}
        if "strategy_priority" in data:
            data["strategy_priority"] = tuple(data["strategy_priority"] or ())
        config = PortfolioPolicyConfig(**data)
        expected_hash = str(self.manifest.get("portfolio_policy_hash") or "")
        if expected_hash and PortfolioArbitrationPolicy(config).policy_hash != expected_hash:
            raise ValueError("captured portfolio policy hash mismatch")
        return config

    def sector_map(self) -> dict[str, str]:
        payload = self.manifest.get("sector_map")
        if payload is None:
            return {}
        return {str(symbol).zfill(6): str(sector).upper().strip() for symbol, sector in dict(payload or {}).items() if str(sector).strip()}

    def _load_config_file(self, strategy_id: str) -> dict[str, Any] | None:
        names = (
            f"{strategy_id}.json",
            f"{strategy_id.lower()}.json",
            f"{strategy_id.lower()}_config.json",
        )
        for dirname in ("strategy_configs", "configs", ""):
            for name in names:
                path = self.root / dirname / name if dirname else self.root / name
                if path.is_file():
                    return json.loads(path.read_text(encoding="utf-8") or "{}")
        return None

    def load_fill_events(self) -> list[tuple[str, Any]]:
        rows = _read_jsonl(self.root / "fill_events.jsonl")
        events: list[tuple[str, Any]] = []
        for row in rows:
            if not row:
                continue
            strategy_id = str(row.get("strategy_id") or (row.get("event") or {}).get("strategy_id") or "").upper().strip()
            if not strategy_id:
                raise ValueError("fill event missing strategy_id")
            events.append((strategy_id, _fill_event(strategy_id, row.get("event") or row)))
        return events

    def load_order_events(self) -> list[tuple[str, Any]]:
        rows = _read_jsonl(self.root / "order_events.jsonl")
        events: list[tuple[str, Any]] = []
        for row in rows:
            record_type = str(row.get("record_type") or "")
            if record_type in {"dry_run_order_result", "oms_order_result"}:
                continue
            if not row:
                continue
            strategy_id = str(row.get("strategy_id") or (row.get("event") or {}).get("strategy_id") or "").upper().strip()
            if not strategy_id:
                raise ValueError("order event missing strategy_id")
            events.append((strategy_id, _order_event(strategy_id, row.get("event") or row)))
        return events

    def load_runtime_events(self) -> list[tuple[datetime, str, Any]]:
        indexed_rows = [
            (index, row)
            for index, row in enumerate(_read_jsonl(self.root / "decision_stream.jsonl"))
            if str(row.get("record_type") or "") == "runtime_event_input"
        ]
        if not indexed_rows:
            return []
        if any("event_sequence" not in row for _index, row in indexed_rows):
            raise ValueError("offline replay requires event_sequence on driver runtime_event_input rows")
        rows = [
            row
            for _index, row in sorted(
                indexed_rows,
                key=lambda item: (
                    int(item[1].get("event_sequence") or item[0] + 1),
                    item[0],
                ),
            )
        ]
        fills_by_ref = self._fill_events_by_ref()
        orders_by_ref = self._order_events_by_ref()
        events: list[tuple[datetime, str, Any]] = []
        index = 0
        while index < len(rows):
            row = rows[index]
            event_type = str(row.get("event_type") or "").strip()
            payload = dict(row.get("payload") or {})
            timestamp = _parse_timestamp(payload.get("timestamp") or row.get("timestamp"))
            event_ref = str(row.get("event_ref") or "")
            if event_type in {"bar", "combined_bar"}:
                grouped_rows = self._combined_bar_group(rows, index)
                if grouped_rows:
                    strategy_ids = tuple(str(item.get("strategy_id") or "").upper().strip() for item in grouped_rows)
                    bar = self._bar_for_runtime_event(grouped_rows[0], dict(grouped_rows[0].get("payload") or {}))
                    events.append((timestamp, "combined_bar", (strategy_ids, bar)))
                    index += len(grouped_rows)
                    continue
                target_ids = tuple(str(item).upper().strip() for item in row.get("target_strategy_ids") or () if str(item).strip())
                if event_type == "combined_bar" or len(target_ids) > 1:
                    if not target_ids:
                        raise ValueError("combined bar runtime event input missing target_strategy_ids")
                    events.append((timestamp, "combined_bar", (target_ids, self._bar_for_runtime_event(row, payload))))
                else:
                    strategy_id = str(row.get("strategy_id") or "").upper().strip()
                    if not strategy_id:
                        raise ValueError("runtime event input missing strategy_id")
                    events.append((timestamp, "strategy_bar", (strategy_id, self._bar_for_runtime_event(row, payload))))
            elif event_type == "timer":
                strategy_id = str(row.get("strategy_id") or "").upper().strip()
                if not strategy_id:
                    raise ValueError("runtime event input missing strategy_id")
                events.append((timestamp, "timer", (strategy_id, timestamp)))
            elif event_type == "fill":
                strategy_id = str(row.get("strategy_id") or "").upper().strip()
                if not strategy_id:
                    raise ValueError("runtime event input missing strategy_id")
                if event_ref not in fills_by_ref:
                    raise FileNotFoundError(f"fill runtime event input has no matching fill_events row: {event_ref}")
                events.append((timestamp, "fill", fills_by_ref[event_ref]))
            elif event_type == "order_event":
                strategy_id = str(row.get("strategy_id") or "").upper().strip()
                if not strategy_id:
                    raise ValueError("runtime event input missing strategy_id")
                if event_ref not in orders_by_ref:
                    raise FileNotFoundError(f"order runtime event input has no matching order_events row: {event_ref}")
                events.append((timestamp, "order_event", orders_by_ref[event_ref]))
            else:
                raise ValueError(f"unsupported runtime event input type {event_type!r}")
            index += 1
        return events

    def _combined_bar_group(self, rows: list[Mapping[str, Any]], index: int) -> list[Mapping[str, Any]]:
        row = rows[index]
        if str(row.get("event_type") or "").strip() != "bar":
            return []
        strategy_id = str(row.get("strategy_id") or "").upper().strip()
        if not strategy_id:
            return []
        payload = dict(row.get("payload") or {})
        timestamp = _parse_timestamp(payload.get("timestamp") or row.get("timestamp"))
        bar_key = str(row.get("bar_hash") or row.get("bar_row_key") or "")
        if not bar_key:
            return []
        group = [row]
        seen = {strategy_id}
        cursor = index + 1
        while cursor < len(rows):
            candidate = rows[cursor]
            if str(candidate.get("event_type") or "").strip() != "bar":
                break
            candidate_id = str(candidate.get("strategy_id") or "").upper().strip()
            if not candidate_id or candidate_id in seen:
                break
            candidate_payload = dict(candidate.get("payload") or {})
            candidate_timestamp = _parse_timestamp(candidate_payload.get("timestamp") or candidate.get("timestamp"))
            candidate_key = str(candidate.get("bar_hash") or candidate.get("bar_row_key") or "")
            if candidate_timestamp != timestamp or candidate_key != bar_key:
                break
            group.append(candidate)
            seen.add(candidate_id)
            cursor += 1
        return group if len(group) > 1 else []

    def _bar_for_runtime_event(self, row: Mapping[str, Any], payload: Mapping[str, Any]) -> MarketBar:
        bar_file = self.root / "market_bars_5m.parquet"
        if not bar_file.is_file():
            return _row_to_bar(dict(payload))
        expected_hash = str(row.get("bar_hash") or row.get("bar_row_key") or "")
        if not expected_hash:
            raise ValueError("bar runtime event input is missing bar_hash")
        bars_by_hash = self._market_bars_by_hash()
        try:
            return bars_by_hash[expected_hash]
        except KeyError as exc:
            raise FileNotFoundError(f"bar runtime event references missing market_bars_5m row hash {expected_hash}") from exc

    def _market_bars_by_hash(self) -> dict[str, MarketBar]:
        if self._market_bars_by_hash_cache is None:
            self._market_bars_by_hash_cache = {market_bar_hash(bar): bar for bar in self.load_market_bars()}
        return self._market_bars_by_hash_cache

    def load_timer_events(self) -> list[tuple[str, datetime]]:
        events: list[tuple[str, datetime]] = []
        for row in _read_jsonl(self.root / "decision_stream.jsonl"):
            if str(row.get("record_type") or "") != "runtime_event_input":
                continue
            if str(row.get("event_type") or "") != "timer":
                continue
            strategy_id = str(row.get("strategy_id") or "").upper().strip()
            if not strategy_id:
                raise ValueError("timer event missing strategy_id")
            payload = dict(row.get("payload") or {})
            events.append((strategy_id, _parse_timestamp(payload.get("timestamp") or row.get("timestamp"))))
        return events

    def _fill_events_by_ref(self) -> dict[str, tuple[str, Any]]:
        events: dict[str, tuple[str, Any]] = {}
        for row in _read_jsonl(self.root / "fill_events.jsonl"):
            event_ref = str(row.get("event_ref") or "")
            if not event_ref:
                continue
            strategy_id = str(row.get("strategy_id") or (row.get("event") or {}).get("strategy_id") or "").upper().strip()
            if not strategy_id:
                raise ValueError("fill event missing strategy_id")
            events[event_ref] = (strategy_id, _fill_event(strategy_id, row.get("event") or row))
        return events

    def _order_events_by_ref(self) -> dict[str, tuple[str, Any]]:
        events: dict[str, tuple[str, Any]] = {}
        for row in _read_jsonl(self.root / "order_events.jsonl"):
            record_type = str(row.get("record_type") or "")
            if record_type in {"dry_run_order_result", "oms_order_result"}:
                continue
            event_ref = str(row.get("event_ref") or "")
            if not event_ref:
                continue
            strategy_id = str(row.get("strategy_id") or (row.get("event") or {}).get("strategy_id") or "").upper().strip()
            if not strategy_id:
                raise ValueError("order event missing strategy_id")
            events[event_ref] = (strategy_id, _order_event(strategy_id, row.get("event") or row))
        return events

    def load_initial_states(self, strategy_ids: tuple[str, ...]) -> dict[str, Any]:
        rows = _read_jsonl(self.root / "state_snapshots.jsonl")
        states: dict[str, Any] = {}
        for strategy_id in strategy_ids:
            matching = [
                row
                for row in rows
                if str(row.get("strategy_id") or "").upper().strip() == strategy_id
                and str((row.get("metadata") or {}).get("record_reason") or "").endswith("pre_start")
            ]
            if not matching:
                raise ValueError(f"missing decodable initial state for {strategy_id}")
            payload = dict(matching[0].get("state") or {})
            if not payload or "symbols" not in payload:
                raise ValueError(f"missing decodable initial state for {strategy_id}")
            state = _restore_state(strategy_id, payload)
            expected_hash = str(matching[0].get("state_hash") or "")
            if expected_hash and canonical_json_hash(_snapshot_state(strategy_id, state)) != expected_hash:
                raise ValueError(f"initial state hash mismatch for {strategy_id}")
            states[strategy_id] = state
        return states

    def initial_account_state(self):
        from oms_client.client import AccountState

        if "initial_account_state" in self.manifest:
            account = dict(self.manifest.get("initial_account_state") or {})
        elif "account_state" in self.manifest:
            account = dict(self.manifest.get("account_state") or {})
        else:
            raise ValueError("missing captured initial account state")
        return AccountState(
            buyable_cash=_required_float(account, "buyable_cash", "cash"),
            equity=_required_float(account, "equity", "buyable_cash", "cash"),
            daily_pnl=float(account.get("daily_pnl", 0.0) or 0.0),
            daily_pnl_pct=float(account.get("daily_pnl_pct", 0.0) or 0.0),
        )

    def initial_positions(self):
        from .portfolio_context import _coerce_positions

        if "initial_positions" in self.manifest:
            return _coerce_positions(self.manifest.get("initial_positions") or {})
        if "positions" in self.manifest:
            return _coerce_positions(self.manifest.get("positions") or {})
        raise ValueError("missing captured initial positions")

    def initial_working_orders(self) -> list[dict[str, Any]]:
        return list(self.initial_working_order_snapshot()["working_orders"])

    def initial_working_order_snapshot(self) -> dict[str, Any]:
        for row in _read_jsonl(self.root / "portfolio_arbitration.jsonl"):
            if str(row.get("record_type") or "") != "pending_reservations_rehydrated":
                continue
            orders = row.get("working_orders")
            if isinstance(orders, list):
                return {
                    "source": str(row.get("source") or ""),
                    "working_orders": [dict(item) for item in orders if isinstance(item, Mapping)],
                }
        return {"source": "", "working_orders": []}


def write_offline_replay_manifest(
    offline_root: str | Path,
    *,
    source_session: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Stamp an offline replay directory as engine-regenerated evidence."""

    root = Path(offline_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "replay_engine_version": OFFLINE_REPLAY_ENGINE_VERSION,
        "replay_source": "engine_rebuild",
        "generated_by": "deployment.olr_kalcb.offline_replay",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_session": str(source_session) if source_session is not None else "",
        **dict(metadata or {}),
    }
    path = root / OFFLINE_REPLAY_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def load_offline_replay_manifest(offline_root: str | Path) -> dict[str, Any]:
    path = Path(offline_root) / OFFLINE_REPLAY_MANIFEST
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def is_engine_replay_manifest(manifest: dict[str, Any]) -> bool:
    return (
        str(manifest.get("replay_engine_version") or "") == OFFLINE_REPLAY_ENGINE_VERSION
        and str(manifest.get("replay_source") or "") == "engine_rebuild"
        and str(manifest.get("generated_by") or "") == "deployment.olr_kalcb.offline_replay"
    )


def _load_session_manifest(root: Path) -> dict[str, Any]:
    path = root / "session_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def _load_snapshots(root: Path, trade_date: date) -> dict[str, KALCBDailySnapshot | OLRDailySnapshot]:
    snapshots: dict[str, KALCBDailySnapshot | OLRDailySnapshot] = {}
    kalcb = _load_first_json(root / "daily_snapshots", trade_date)
    if kalcb is not None:
        snapshots["KALCB"] = KALCBDailySnapshot.from_json_dict(kalcb)
    olr = _load_first_json(root / "olr_final_snapshots", trade_date)
    if olr is not None:
        snapshots["OLR"] = OLRDailySnapshot.from_json_dict(olr)
    return snapshots


def _load_first_json(directory: Path, trade_date: date) -> dict[str, Any] | None:
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(f"*{trade_date.isoformat()}*.json")) or sorted(directory.glob("*.json"))
    if not candidates:
        return None
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def _descriptors_for_snapshots(
    snapshots: dict[str, KALCBDailySnapshot | OLRDailySnapshot],
    *,
    configs: dict[str, KALCBConfig | OLRConfig],
    states: dict[str, Any],
) -> dict[str, StrategyRuntimeDescriptor]:
    descriptors: dict[str, StrategyRuntimeDescriptor] = {}
    if "KALCB" in snapshots:
        snapshot = snapshots["KALCB"]
        engine = KALCBEngine(config=configs["KALCB"], state=states["KALCB"], candidate_snapshot=snapshot)
        descriptors["KALCB"] = StrategyRuntimeDescriptor(
            "KALCB",
            str(snapshot.metadata.get("artifact_stage") or ""),
            snapshot.artifact_hash,
            engine,
            snapshot,
            priority=10,
        )
    if "OLR" in snapshots:
        snapshot = snapshots["OLR"]
        engine = OLREngine(config=configs["OLR"], state=states["OLR"], candidate_snapshot=snapshot)
        descriptors["OLR"] = StrategyRuntimeDescriptor(
            "OLR",
            str(snapshot.metadata.get("artifact_stage") or ""),
            snapshot.artifact_hash,
            engine,
            snapshot,
            priority=20,
        )
    return descriptors


def _positions_with_startup_working_orders(
    positions: dict[str, Any],
    working_orders: list[Mapping[str, Any]],
) -> dict[str, Any]:
    from oms_client.client import PositionInfo, WorkingOrderInfo

    updated = dict(positions)
    grouped: dict[str, list[WorkingOrderInfo]] = {}
    for row in working_orders:
        symbol = str(row.get("symbol") or "").zfill(6)
        if not symbol:
            continue
        remaining_qty = int(row.get("remaining_qty") or row.get("qty") or 0)
        if remaining_qty <= 0:
            continue
        grouped.setdefault(symbol, []).append(
            WorkingOrderInfo(
                order_id=str(row.get("order_id") or row.get("intent_id") or row.get("idempotency_key") or f"startup:{symbol}"),
                symbol=symbol,
                side=str(row.get("side") or "").upper().strip(),
                qty=int(row.get("qty") or remaining_qty),
                filled_qty=int(row.get("filled_qty") or 0),
                remaining_qty=remaining_qty,
                price=float(row.get("price") or 0.0),
                status=str(row.get("status") or "WORKING").upper().strip(),
                strategy_id=str(row.get("strategy_id") or "").upper().strip(),
                intent_id=row.get("intent_id"),
                idempotency_key=row.get("idempotency_key"),
                submit_ref=row.get("submit_ref"),
            )
        )
    for symbol, orders in grouped.items():
        position = updated.get(symbol)
        if position is None:
            position = PositionInfo(symbol=symbol, real_qty=0, avg_price=max(float(orders[0].price or 0.0), 0.0), allocations={})
            updated[symbol] = position
        position.working_orders = orders
        position.working_order_count = len(orders)
    return updated


def _load_market_bars(path: Path) -> list[MarketBar]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to rebuild offline replay bars") from exc
    rows = pd.read_parquet(path).to_dict("records")
    bars = [_row_to_bar(row) for row in rows]
    incomplete = [bar for bar in bars if not bar.is_completed]
    if incomplete:
        first = incomplete[0]
        raise ValueError(f"incomplete market bar in replay input: {first.symbol} {first.timestamp}")
    return sorted(bars, key=lambda item: (item.timestamp, item.symbol))


def _row_to_bar(row: dict[str, Any]) -> MarketBar:
    timestamp = row.get("timestamp") or row.get("datetime") or row.get("bar_time")
    if not isinstance(timestamp, datetime):
        timestamp = datetime.fromisoformat(str(timestamp))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    metadata = _bar_metadata(row.get("metadata"))
    metadata.update({key: _json_scalar(value) for key, value in row.items() if key not in _BAR_COLUMNS})
    source = row.get("source", "")
    return MarketBar(
        symbol=str(row.get("symbol") or row.get("ticker") or "").zfill(6),
        timestamp=timestamp,
        timeframe=str(row.get("timeframe") or "5m"),
        open=float(row.get("open")),
        high=float(row.get("high")),
        low=float(row.get("low")),
        close=float(row.get("close")),
        volume=float(row.get("volume") or 0.0),
        is_completed=_bool_field(row.get("is_completed", True)),
        source="" if source is None else str(source),
        source_fingerprint=str(row.get("source_fingerprint") or ""),
        metadata=metadata,
    )


def _bar_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return dict(parsed or {}) if isinstance(parsed, dict) else {}
    return {}


def _end_positions(engines: dict[str, KALCBEngine | OLREngine]) -> dict[str, Any]:
    positions = []
    for strategy_id, engine in engines.items():
        for symbol, state in engine.state.symbols.items():
            position = getattr(state, "position", None)
            if position is not None and int(getattr(position, "qty_open", 0) or 0) > 0:
                positions.append({"strategy_id": strategy_id, "symbol": symbol, **asdict(position)})
    return {"positions": positions}


def _fill_event(strategy_id: str, payload: dict[str, Any]) -> Any:
    data = dict(payload or {})
    metadata = dict(data.get("metadata") or {})
    provisional = str(data.get("provisional_order_ref") or metadata.get("provisional_order_ref") or "")
    broker_order_id = str(data.get("order_id") or data.get("broker_order_id") or metadata.get("broker_order_id") or "")
    if broker_order_id and provisional and broker_order_id != provisional:
        metadata.setdefault("broker_order_id", broker_order_id)
        metadata.setdefault("original_order_id", broker_order_id)
    if provisional:
        metadata["provisional_order_ref"] = provisional
    timestamp = _parse_timestamp(data.get("timestamp"))
    kwargs = {
        "order_id": provisional or broker_order_id,
        "symbol": str(data.get("symbol") or "").zfill(6),
        "side": str(data.get("side") or "").upper().strip(),
        "qty": int(data.get("qty", data.get("filled_qty", 0)) or 0),
        "price": float(data.get("price", data.get("fill_price", 0.0)) or 0.0),
        "timestamp": timestamp,
        "reason": str(data.get("reason") or data.get("reason_code") or ""),
        "metadata": metadata,
    }
    if strategy_id == "KALCB":
        from strategy_kalcb.core.core_models import KALCBFillEvent

        return KALCBFillEvent(**kwargs)
    if strategy_id == "OLR":
        from strategy_olr.core.core_models import OLRFillEvent

        return OLRFillEvent(**kwargs)
    raise ValueError(f"unsupported fill strategy {strategy_id}")


def _order_event(strategy_id: str, payload: dict[str, Any]) -> Any:
    data = dict(payload or {})
    metadata = dict(data.get("metadata") or {})
    provisional = str(data.get("provisional_order_ref") or metadata.get("provisional_order_ref") or "")
    broker_order_id = str(data.get("order_id") or data.get("broker_order_id") or metadata.get("broker_order_id") or "")
    if broker_order_id and provisional and broker_order_id != provisional:
        metadata.setdefault("broker_order_id", broker_order_id)
        metadata.setdefault("original_order_id", broker_order_id)
    if provisional:
        metadata["provisional_order_ref"] = provisional
    timestamp = _parse_timestamp(data.get("timestamp"))
    order_id = provisional or broker_order_id
    status = str(data.get("status") or "").upper().strip()
    if strategy_id == "KALCB":
        from strategy_kalcb.core.core_models import KALCBOrderUpdateEvent

        return KALCBOrderUpdateEvent(
            order_id=order_id,
            symbol=str(data.get("symbol") or "").zfill(6),
            status=status or "ORDER_EVENT",
            timestamp=timestamp,
            role=str(data.get("role") or metadata.get("role") or ""),
            reason=str(data.get("reason") or data.get("reason_code") or ""),
            metadata=metadata,
        )
    if strategy_id == "OLR":
        if status == "EXPIRED":
            from strategy_olr.core.core_models import OLRExpiredOrderEvent

            return OLRExpiredOrderEvent(
                order_id=order_id,
                symbol=str(data.get("symbol") or "").zfill(6),
                side=str(data.get("side") or "").upper().strip(),
                order_type=str(data.get("order_type") or ""),
                qty=int(data["qty"]) if data.get("qty") not in (None, "") else None,
                timestamp=timestamp,
                reason=str(data.get("reason") or data.get("reason_code") or "order_event_replay"),
                metadata=metadata,
            )
        if status:
            from strategy_olr.core.core_models import OLROrderUpdateEvent

            return OLROrderUpdateEvent(
                order_id=order_id,
                symbol=str(data.get("symbol") or "").zfill(6),
                status=status,
                timestamp=timestamp,
                side=str(data.get("side") or "").upper().strip(),
                order_type=str(data.get("order_type") or ""),
                qty=int(data["qty"]) if data.get("qty") not in (None, "") else None,
                reason=str(data.get("reason") or data.get("reason_code") or "order_event_replay"),
                metadata=metadata,
            )
        from strategy_olr.core.core_models import OLRExpiredOrderEvent

        return OLRExpiredOrderEvent(
            order_id=order_id,
            symbol=str(data.get("symbol") or "").zfill(6),
            side=str(data.get("side") or "").upper().strip(),
            order_type=str(data.get("order_type") or ""),
            qty=int(data["qty"]) if data.get("qty") not in (None, "") else None,
            timestamp=timestamp,
            reason=str(data.get("reason") or data.get("reason_code") or "order_event_replay"),
            metadata=metadata,
        )
    raise ValueError(f"unsupported order event strategy {strategy_id}")


def _parse_timestamp(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        timestamp = raw
    elif raw not in (None, ""):
        timestamp = datetime.fromisoformat(str(raw))
    else:
        raise ValueError("captured event missing timestamp")
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _restore_state(strategy_id: str, payload: dict[str, Any]) -> Any:
    if strategy_id == "KALCB":
        from strategy_kalcb.core.serializers import restore_state

        return restore_state(payload)
    if strategy_id == "OLR":
        from strategy_olr.core.serializers import restore_state

        return restore_state(payload)
    raise ValueError(f"unsupported strategy_id={strategy_id!r}")


def _snapshot_state(strategy_id: str, state: Any) -> dict[str, Any]:
    if strategy_id == "KALCB":
        from strategy_kalcb.core.serializers import snapshot_state

        return snapshot_state(state)
    if strategy_id == "OLR":
        from strategy_olr.core.serializers import snapshot_state

        return snapshot_state(state)
    raise ValueError(f"unsupported strategy_id={strategy_id!r}")


def _load_config_reference(root: Path, raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    wrapper = dict(raw or {})
    ref = wrapper.get("path") or wrapper.get("file")
    if ref:
        path = Path(str(ref))
        if not path.is_absolute():
            path = root / path
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
        return {**wrapper, "payload": payload}
    return wrapper


def _config_expected_hash(wrapper: Mapping[str, Any]) -> str:
    return str(wrapper.get("payload_hash") or wrapper.get("mutation_hash") or wrapper.get("config_hash") or wrapper.get("hash") or "")


def _config_uses_defaults(wrapper: Mapping[str, Any]) -> bool:
    return wrapper.get("uses_defaults") is True


def _config_mutation_payload(wrapper: Mapping[str, Any]) -> dict[str, Any]:
    candidate: Any = wrapper.get("payload") if "payload" in wrapper else wrapper
    while isinstance(candidate, Mapping):
        if "mutations" in candidate and isinstance(candidate.get("mutations"), Mapping):
            candidate = candidate["mutations"]
            continue
        if "payload" in candidate and isinstance(candidate.get("payload"), Mapping):
            candidate = candidate["payload"]
            continue
        break
    payload = dict(candidate or {}) if isinstance(candidate, Mapping) else {}
    metadata_only = {
        "config_hash",
        "file",
        "hash",
        "hydrated_config_hash",
        "mutation_hash",
        "path",
        "payload_hash",
        "sha256",
        "source_label",
        "source_manifest",
        "source_path",
        "source_sha256",
        "uses_defaults",
    }
    return {} if payload and set(payload).issubset(metadata_only) else payload


def _required_float(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if mapping.get(key) not in (None, ""):
            return float(mapping[key])
    raise ValueError(f"missing numeric account field; expected one of {', '.join(keys)}")


def _bool_field(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "n", ""}
    return bool(value)


def _json_scalar(value: Any) -> Any:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return value


_BAR_COLUMNS = {
    "symbol",
    "ticker",
    "timestamp",
    "datetime",
    "bar_time",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_completed",
    "source",
    "source_fingerprint",
    "metadata",
}
