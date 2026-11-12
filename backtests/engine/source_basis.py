"""Source-bound price/share and clock conversions used by qualification."""
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math


def target_basis(contract):
    normalization = contract.get("normalization") or {}
    basis = normalization.get("target_basis") or {}
    if (not basis.get("id") or basis.get("price_unit") != "KRW"
            or basis.get("volume_unit") != "shares" or not basis.get("adjustment_as_of")):
        raise ValueError("missing_normalized_basis_contract")
    date.fromisoformat(basis["adjustment_as_of"])
    if not normalization.get("source_basis_id"):
        raise ValueError("missing_source_basis_identity")
    return basis


def conversion(contract, symbol, day, read):
    basis = target_basis(contract)
    normalization = contract["normalization"]
    date.fromisoformat(day)
    for rule in normalization.get("transforms", []):
        if date.fromisoformat(rule["start"]) > date.fromisoformat(rule["end"]):
            raise ValueError("invalid_conversion_scope")
    rules = [r for r in normalization.get("transforms", [])
             if r.get("symbol") in {symbol, "*"} and r["start"] <= day <= r["end"]]
    if len(rules) > 1:
        raise ValueError("ambiguous_basis_conversion")
    if not rules:
        if normalization["source_basis_id"] != basis["id"]:
            raise ValueError("missing_basis_conversion")
        return {"kind": "identity", "price_factor": 1.0, "share_factor": 1.0,
                "timestamp_shift_seconds": 0}
    rule = rules[0]
    if rule.get("kind") not in {"identity", "split", "price_adjustment", "provider_conversion"}:
        raise ValueError("unsupported_basis_conversion")
    factors = [float(rule[k]) for k in ("price_factor", "share_factor")]
    if not all(math.isfinite(v) and v > 0 for v in factors):
        raise ValueError("invalid_basis_factors")
    if rule["kind"] == "split" and not math.isclose(math.prod(factors), 1.0):
        raise ValueError("inconsistent_split_factors")
    shift = rule.get("timestamp_shift_seconds", 0)
    if not isinstance(shift, int):
        raise ValueError("invalid_clock_conversion")
    if rule["kind"] == "identity" and (factors != [1.0, 1.0] or shift or rule.get("price_rounding") or rule.get("volume_rounding")):
        raise ValueError("nonidentity_conversion")
    changed = (factors != [1.0, 1.0] or shift != 0
               or normalization["source_basis_id"] != basis["id"]
               or rule.get("price_rounding") or rule.get("volume_rounding"))
    if changed:
        if not rule.get("evidence_paths"):
            raise ValueError("unsupported_conversion_evidence")
        for path in rule["evidence_paths"]:
            if not read(path):
                raise ValueError("empty_conversion_evidence")
    return rule


def converted_value(value, field, rule):
    factor = rule["share_factor"] if field == "volume" else rule["price_factor"]
    result = Decimal(str(value)) * Decimal(str(factor))
    rounding = rule.get("volume_rounding" if field == "volume" else "price_rounding")
    if rounding:
        modes = {"half_up": ROUND_HALF_UP, "floor": ROUND_FLOOR, "ceiling": ROUND_CEILING}
        quantum = Decimal(str(rounding["quantum"]))
        if not quantum.is_finite() or quantum <= 0 or rounding.get("mode") not in modes:
            raise ValueError("invalid_conversion_rounding")
        result = (result / quantum).to_integral_value(rounding=modes[rounding["mode"]]) * quantum
    if not result.is_finite():
        raise ValueError("nonfinite_converted_value")
    return float(result)


def converted_timestamp(value, rule):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("source_timestamp_timezone_missing")
    return stamp + timedelta(seconds=rule.get("timestamp_shift_seconds", 0))
