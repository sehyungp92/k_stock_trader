"""Versions are part of every replay, audit and scored-result identity."""

AUDIT_VERSION = "execution-ledger-v18"
METRIC_VERSION = "session-equity-trade-groups-v5"
EXECUTION_VERSION = "timestamp-phases-v6"
LEGACY_EXECUTION_VERSION = "legacy-symbol-bars-v1"
CONTRACT_VERSIONS = (AUDIT_VERSION, METRIC_VERSION, EXECUTION_VERSION)


def execution_valid(metrics: dict, *, expected_request=None, expected_mutations=None) -> bool:
    """Parity and a status label are never evidence of execution validity."""
    contract = metrics.get("metric_contract") or {}
    if not isinstance(contract, dict):
        contract = {}
    for value in (metrics, contract):
        if value.get("audit_pass") is False or str(value.get("audit_status", "")).lower() in {
            "fail", "failed", "missing", "not_run", "unverified",
        }:
            return False
    from .integrity import receipt_matches
    return metrics.get("audit_pass", contract.get("audit_pass")) is True and receipt_matches(
        metrics, expected_request=expected_request, expected_mutations=expected_mutations)


def require_execution_valid(metrics: dict) -> None:
    if not execution_valid(metrics):
        raise ValueError("Execution audit failed or required evidence is unavailable")
