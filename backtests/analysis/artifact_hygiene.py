from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactHygieneResult:
    passed: bool
    failures: tuple[str, ...]


REQUIRED_HASH_KEYS = (
    "source_data_fingerprint",
    "score_spec_hash",
    "config_hash",
    "strategy_code_hash",
)


def validate_official_artifact(
    summary: dict[str, Any],
    optimized_config: dict[str, Any],
    final_diagnostics: dict[str, Any] | None = None,
) -> ArtifactHygieneResult:
    failures: list[str] = []
    if summary.get("promotion_status") == "official":
        optimized_metadata = optimized_config if "mutations" in optimized_config else {}
        for key in REQUIRED_HASH_KEYS:
            if not summary.get(key):
                failures.append(f"missing_{key}")
            elif optimized_metadata and optimized_metadata.get(key) != summary.get(key):
                failures.append(f"{key}_mismatch")
        if "mutations" not in optimized_config:
            failures.append("optimized_config_missing_mutations")
        if summary.get("live_parity_fill_timing") != optimized_metadata.get("live_parity_fill_timing"):
            failures.append("fill_timing_mismatch")
        if summary.get("risk_basis") != "mark_to_market":
            failures.append("risk_basis_not_mtm")
        if final_diagnostics:
            for key in ("source_data_fingerprint", "score_spec_hash", "config_hash"):
                if final_diagnostics.get(key) and final_diagnostics.get(key) != summary.get(key):
                    failures.append(f"diagnostics_{key}_mismatch")
    return ArtifactHygieneResult(not failures, tuple(failures))
