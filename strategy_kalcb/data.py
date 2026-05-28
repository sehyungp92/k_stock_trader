from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from kis_core.ws_client import WS_MAX_REGS_DEFAULT


@dataclass(slots=True)
class WebSocketRegistrationBudget:
    """Shared KIS websocket allocation guard for KALCB hot symbols."""

    max_registrations: int = WS_MAX_REGS_DEFAULT
    reserved_execution_regs: int = 1
    hot_regs_per_symbol: int = 1
    strategy_symbol_budget: int = 10
    ledger_path: Path | None = None
    strategy_id: str = "KALCB"
    rest_egw00201_cooldown_s: float = 30.0
    allocations: dict[str, int] = field(default_factory=dict)
    last_rate_limit_hit_monotonic: float = 0.0

    @property
    def used_regs(self) -> int:
        local = sum(self.allocations.values())
        if self.ledger_path is None:
            return local
        ledger = _read_ledger(self.ledger_path)
        return max(_ledger_used_regs(ledger), local)

    @property
    def available_regs(self) -> int:
        return max(0, self.max_registrations - self.reserved_execution_regs - self.used_regs)

    @property
    def active_symbol_capacity(self) -> int:
        reg_capacity = self.available_regs // max(self.hot_regs_per_symbol, 1)
        remaining_slice = max(0, self.strategy_symbol_budget - self._strategy_allocation_count())
        return min(reg_capacity, remaining_slice)

    def record_rate_limit_hit(self, code: str = "EGW00201") -> None:
        if code == "EGW00201":
            self.last_rate_limit_hit_monotonic = time.monotonic()

    def rest_cooldown_active(self) -> bool:
        if self.last_rate_limit_hit_monotonic <= 0:
            return False
        return (time.monotonic() - self.last_rate_limit_hit_monotonic) < self.rest_egw00201_cooldown_s

    def allocate_hot(self, symbol: str, reason: str = "") -> tuple[bool, str]:
        symbol = str(symbol)
        if self.rest_cooldown_active():
            return False, "rest_rate_limit_cooldown"
        if symbol in self.allocations:
            return True, "already_hot"
        if len(self.allocations) >= self.strategy_symbol_budget:
            return False, "strategy_ws_slice_exhausted"
        required = self.hot_regs_per_symbol
        if self.ledger_path is not None:
            with _locked_ledger(self.ledger_path) as ledger:
                owner = ledger.setdefault("allocations", {}).setdefault(self.strategy_id, {})
                if symbol in owner:
                    self.allocations[symbol] = int(owner[symbol])
                    return True, "already_hot"
                if len(owner) >= self.strategy_symbol_budget:
                    return False, "strategy_ws_slice_exhausted"
                used = _ledger_used_regs(ledger)
                if self.max_registrations - self.reserved_execution_regs - used < required:
                    return False, "ws_budget_exhausted"
                owner[symbol] = required
                self.allocations[symbol] = required
                return True, reason or "hot_allocated"
        if self.available_regs < required:
            return False, "ws_budget_exhausted"
        self.allocations[symbol] = required
        return True, reason or "hot_allocated"

    def release_hot(self, symbol: str) -> None:
        symbol = str(symbol)
        self.allocations.pop(symbol, None)
        if self.ledger_path is not None:
            with _locked_ledger(self.ledger_path) as ledger:
                owner = ledger.setdefault("allocations", {}).setdefault(self.strategy_id, {})
                owner.pop(symbol, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_registrations": self.max_registrations,
            "reserved_execution_regs": self.reserved_execution_regs,
            "hot_regs_per_symbol": self.hot_regs_per_symbol,
            "strategy_symbol_budget": self.strategy_symbol_budget,
            "used_regs": self.used_regs,
            "available_regs": self.available_regs,
            "active_symbol_capacity": self.active_symbol_capacity,
            "hot_symbols": sorted(self.allocations),
            "rest_cooldown_active": self.rest_cooldown_active(),
        }

    def _strategy_allocation_count(self) -> int:
        if self.ledger_path is None:
            return len(self.allocations)
        ledger = _read_ledger(self.ledger_path)
        owner = dict(dict(ledger.get("allocations", {}) or {}).get(self.strategy_id, {}) or {})
        return max(len(owner), len(self.allocations))


@contextmanager
def _locked_ledger(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps({"allocations": {}}), encoding="utf-8")
    with path.open("r+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            try:
                ledger = json.load(handle)
            except json.JSONDecodeError:
                ledger = {"allocations": {}}
            yield ledger
            handle.seek(0)
            handle.truncate()
            json.dump(ledger, handle, indent=2, sort_keys=True)
        finally:
            _unlock_file(handle)


def _read_ledger(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"allocations": {}}


def _ledger_used_regs(ledger: dict[str, Any]) -> int:
    total = 0
    for rows in dict(ledger.get("allocations", {}) or {}).values():
        total += sum(int(value) for value in dict(rows).values())
    return total


if os.name == "nt":
    import msvcrt

    def _lock_file(handle) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(handle) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_file(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def _unlock_file(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
