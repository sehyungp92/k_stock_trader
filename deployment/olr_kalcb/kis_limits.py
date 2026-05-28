from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from kis_core.ws_client import KIS_WS_EXECUTION_NOTIFICATION_RESERVE, WS_MAX_REGS_DEFAULT


PAPER_REST_MIN_INTERVAL_S = 0.50
LIVE_REST_MIN_INTERVAL_S = 0.07
DEFAULT_WS_RESERVED_EXECUTION_REGS = KIS_WS_EXECUTION_NOTIFICATION_RESERVE
DEFAULT_ORDER_REST_RESERVE_PER_5M = 12
DEFAULT_OMS_RECONCILE_RESERVE_PER_5M = 6
LIMIT_PROFILE_SOURCE = (
    "kis_core.kis_client throttle constants + "
    "kis_core.ws_client.WS_MAX_REGS_DEFAULT; official KIS limit verification required before live promotion"
)


@dataclass(frozen=True, slots=True)
class KISLimitProfile:
    mode: str
    kis_is_paper: bool
    rest_min_interval_s: float
    rest_calls_per_5m: int
    ws_max_registrations: int
    ws_reserved_execution_regs: int
    order_rest_reserve_per_5m: int
    oms_reconcile_reserve_per_5m: int
    source: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def detected_kis_mode(env: Mapping[str, str] | None = None) -> tuple[bool, float, str]:
    """Return the KIS mode seen by the current process without importing the client."""

    client = sys.modules.get("kis_core.kis_client")
    if client is not None and hasattr(client, "_PAPER_MODE") and hasattr(client, "_MIN_INTERVAL"):
        return (
            bool(getattr(client, "_PAPER_MODE")),
            float(getattr(client, "_MIN_INTERVAL")),
            "kis_core.kis_client imported process constants",
        )
    env_map = os.environ if env is None else env
    kis_is_paper = _truthy(env_map.get("KIS_IS_PAPER", "true"))
    return (
        kis_is_paper,
        PAPER_REST_MIN_INTERVAL_S if kis_is_paper else LIVE_REST_MIN_INTERVAL_S,
        "KIS_IS_PAPER environment with deployment defaults",
    )


def limit_profile_for_runtime(
    mode: str,
    *,
    kis_is_paper: bool | None = None,
    rest_min_interval_s: float | None = None,
    ws_max_registrations: int = WS_MAX_REGS_DEFAULT,
    ws_reserved_execution_regs: int = DEFAULT_WS_RESERVED_EXECUTION_REGS,
    order_rest_reserve_per_5m: int = DEFAULT_ORDER_REST_RESERVE_PER_5M,
    oms_reconcile_reserve_per_5m: int = DEFAULT_OMS_RECONCILE_RESERVE_PER_5M,
    env: Mapping[str, str] | None = None,
) -> KISLimitProfile:
    detected_paper, detected_interval, detected_source = detected_kis_mode(env)
    effective_paper = detected_paper if kis_is_paper is None else bool(kis_is_paper)
    interval = float(
        rest_min_interval_s
        if rest_min_interval_s is not None
        else (detected_interval if kis_is_paper is None else (PAPER_REST_MIN_INTERVAL_S if effective_paper else LIVE_REST_MIN_INTERVAL_S))
    )
    return KISLimitProfile(
        mode=str(mode or "").strip().lower(),
        kis_is_paper=effective_paper,
        rest_min_interval_s=interval,
        rest_calls_per_5m=max(1, int(300.0 / max(interval, 1e-9))),
        ws_max_registrations=int(ws_max_registrations),
        ws_reserved_execution_regs=int(ws_reserved_execution_regs),
        order_rest_reserve_per_5m=int(order_rest_reserve_per_5m),
        oms_reconcile_reserve_per_5m=int(oms_reconcile_reserve_per_5m),
        source=f"{LIMIT_PROFILE_SOURCE}; detected={detected_source}",
    )


def kis_mode_matches_runtime(mode: str, profile: KISLimitProfile) -> bool:
    mode_name = str(mode or "").strip().lower()
    if mode_name == "paper":
        return profile.kis_is_paper
    if mode_name == "live":
        return not profile.kis_is_paper
    return True


def _truthy(value: str | None) -> bool:
    return str(value if value is not None else "true").strip().lower() not in {"0", "false", "f", "no", "n", "live"}
