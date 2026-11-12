"""Verify immutable observations, never promote a supplied pass/fail label.

Evidence files contain normalized source events, candidate/daily source rows,
vendor basis observations and execution-phase observations. A declared bar model
can qualify historical research conditionally; it never certifies phase liquidity.
"""
from collections import defaultdict
from decimal import InvalidOperation
from datetime import date, datetime, timedelta, time
from dataclasses import asdict
from pathlib import Path
import json
import math

from backtests.auto.shared.cache_keys import source_generation_hash, stable_signature
from backtests.core.input_manifest import InputManifest
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.session_coverage import session_bar_coverage
from .source_basis import target_basis, conversion, converted_value, converted_timestamp


CHECKS = ("input_provenance", "source_completeness", "replay_completeness", "dated_references", "price_volume_basis", "capacity")
SOURCE_SCOPE_VERSIONS=('causal-feature-and-management-v1','causal-feature-and-management-v2','causal-timed-candidates-v1')


def validate_source_scope(scope):
    version=scope.get('version');strategy=scope.get('strategy')
    expected={'olr':('14:30','16:00'),'kalcb':('09:30','15:20')}.get(strategy)
    if version=='causal-timed-candidates-v1':
        if strategy!='kalcb':raise ValueError('unsupported_source_dependency_scope')
        expected=('09:45','15:20')
    if (version not in SOURCE_SCOPE_VERSIONS or expected is None or
            (scope.get('feature_intraday_before'),scope.get('candidate_intraday_before'))!=expected):
        raise ValueError('unsupported_source_dependency_scope')


def verify_qualification(broker, identity, *, capacity_report=None):
    scopes = (identity.get("configuration") or {}).get("source_dependency_scopes")
    if scopes is not None:
        return _verify_shared_scopes(broker, identity, scopes, capacity_report)
    coverage = dict.fromkeys(CHECKS, "unverified")
    details = {}
    receipt = identity.get("input_manifest") or {}
    if not receipt.get("manifest_path"):
        return coverage, {"input_provenance": "missing_manifest"}
    try:
        generation = InputManifest(receipt["manifest_path"])
        if generation.identity != receipt.get("manifest_hash") or generation.payload["source_generation_hash"] != source_generation_hash():
            raise ValueError("manifest_identity_or_source_mismatch")
        path = generation.payload.get("evidence", {}).get("qualification_path")
        if not path:
            return coverage, {"input_provenance": "missing_qualification_evidence"}

        def read(path):
            return json.loads(generation.read(generation.root / path))

        evidence = read(path)
        if evidence.get("version") != "qualification-observations-v2":
            raise ValueError("unsupported_qualification_schema")
        expected = [bar for batch in getattr(broker, "expected_batches", ()) for bar in batch]
        key = lambda row: (row["symbol"], row["timestamp"], row["timeframe"])
        values = lambda row: tuple(row[k] for k in ("open", "high", "low", "close", "volume", "is_completed"))
        indexed, grouped = {}, defaultdict(list)
        for row in source_rows(read, generation, evidence):
            if key(row) in indexed:
                raise ValueError("duplicate_source_event")
            indexed[key(row)] = values(row)
            stamp = datetime.fromisoformat(row["timestamp"])
            if stamp.tzinfo is None:
                raise ValueError("source_timestamp_timezone_missing")
            from strategy_common.clock import ensure_kst
            stamp = ensure_kst(stamp)
            grouped[(row["symbol"], stamp.date(), row["timeframe"])].append(stamp)
        if not expected or any(indexed.get(key(row)) != values(row) for row in expected):
            raise ValueError("source_events_mismatch_or_empty")
        events = expected
        coverage["input_provenance"] = "pass"
        with generation.activate():
            from .retained_source import sparse_dispositions
            sparse = sparse_dispositions(read, evidence)
            calendar = get_trading_calendar()
            missing = []
            try:
                required = required_sessions(identity.get("configuration") or {}, generation, calendar)
            except ValueError as exc:
                required = set()
                missing.append(str(exc))
            scope = (identity.get("configuration") or {}).get("source_dependency_scope") or {}
            for symbol, day, timeframe in sorted(required):
                timestamps = grouped.get((symbol, day, timeframe), [])
                minutes = int(timeframe[:-1]) if timeframe.endswith("m") else 0
                scope = (identity.get("configuration") or {}).get("source_dependency_scope") or {}
                if scope:
                    validate_source_scope(scope)
                    if set(getattr(broker, "qualification_snapshot_scopes", {})) != {scope["strategy"].upper()}:
                        raise ValueError("source_dependency_strategy_mismatch")
                cutoff = scope.get("feature_intraday_before") if scope.get("version") in SOURCE_SCOPE_VERSIONS else None
                if minutes <= 0 or not _covered(timestamps, day, minutes, symbol, sparse, before=cutoff):
                    missing.append(f"{symbol}:{day}:incomplete")
            coverage["source_completeness"] = "unverified" if missing else "pass"
            details["source_completeness"] = missing
            if scope.get("version") == "causal-feature-and-management-v2" and scope.get("strategy") == "olr":
                _verify_feature_fallback(broker, identity, grouped, required)
                details["raw_source_completeness_flags"] = missing
                details["source_completeness"] = "complete_consumed_prefixes; incomplete_symbols_excluded_from_candidates_and_sector_features"
                coverage["source_completeness"] = "pass"
            details["required_sessions"] = len(required)
            details["candidate_sessions"] = sorted({f"{candidate['symbol']}:{snapshot['trade_date']}"
                for snapshot in getattr(broker, "qualification_snapshots", ())
                for candidate in snapshot["candidates"]})
            replay_sparse = {key: frozenset(grouped.get((key[0], key[1], "5m"), ())) for key in sparse}
            _check(coverage, details, "replay_completeness", lambda: _replay_scope(broker, identity, events, required, sparse=replay_sparse, source_stamps=grouped))
            _check(coverage, details, "dated_references", lambda: _references(read, evidence, calendar, identity, broker))
            _check(coverage, details, "price_volume_basis", lambda: _basis(read, evidence, events))
            _check(coverage, details, "capacity", lambda: _capacity(read, evidence, broker, capacity_report=capacity_report))
            # One immutable source bundle can serve independently bound strategy,
            # shared-account and candidate scopes without changing raw identity.
            details["scope_hash"] = stable_signature({"configuration": identity.get("configuration"),
                "candidate_hash": identity.get("qualification_snapshot_hash"),
                "events_hash": stable_signature(events), "policy": execution_policy(broker)})
        details["evidence_hash"] = stable_signature(evidence)
        details["manifest_hash"] = generation.identity
    except (ValueError, KeyError, TypeError, OSError, IndexError, OverflowError, InvalidOperation) as exc:
        details["input_provenance"] = str(exc)
        coverage["input_provenance"] = "failed"
    return coverage, details


def _verify_shared_scopes(broker, identity, scopes, capacity_report):
    """Apply every native source obligation to the same shared execution path.

    Only candidate/management attribution is filtered; market events, cash,
    fills and capacity retain the complete account. Never merge cutoffs by
    taking the less restrictive strategy's clock.
    """
    from copy import copy
    bound = getattr(broker, "qualification_snapshot_scopes", {})
    if not scopes or set(scopes) != set(bound):
        return dict.fromkeys(CHECKS, "unverified"), {"input_provenance": "shared_scope_strategy_mismatch"}
    reports = {}
    for strategy, definition in scopes.items():
        if definition.get("scope", {}).get("strategy", "").upper() != strategy:
            return dict.fromkeys(CHECKS, "unverified"), {"input_provenance": "shared_scope_strategy_mismatch"}
        view = copy(broker)
        view.qualification_snapshot_scopes = {strategy: bound[strategy]}
        view.qualification_snapshots = bound[strategy]
        view.qualification_snapshot_hash = stable_signature(bound[strategy])
        view.qualification_day_dispositions = {strategy: getattr(broker, "qualification_day_dispositions", {}).get(strategy, {})}
        view.journal = [row for row in broker.journal if not row.get("order") or row["order"].get("strategy_id") == strategy]
        config = {k: v for k, v in identity["configuration"].items() if k != "source_dependency_scopes"}
        config.update(source_dependency_scope=definition["scope"],
                      entry_require_complete_prefix=definition["entry_require_complete_prefix"])
        child = {**identity, "configuration": config, "qualification_snapshot_hash": view.qualification_snapshot_hash,
                 "qualification_day_dispositions_hash": stable_signature(view.qualification_day_dispositions)}
        coverage, details = verify_qualification(view, child, capacity_report=capacity_report)
        reports[strategy] = {"coverage": coverage, "details": details}
    combined = {k: "pass" if all(r["coverage"][k] == "pass" for r in reports.values()) else "unverified" for k in CHECKS}
    return combined, {"shared_scopes": reports, "scope_hash": stable_signature(identity["configuration"])}


def source_rows(read, generation, evidence):
    """Common frozen rows, independent of adapter-specific metadata/fingerprints."""
    if evidence.get("market_events"):
        yield from read(evidence["market_events"])
        return
    import pandas as pd
    from strategy_common.market import MarketBar
    from strategy_common.clock import ensure_kst
    for item in evidence["market_files"]:
        if item.get("parser_version") != "completed_ohlcv_parquet_v1":
            raise ValueError("unregistered_market_parser")
        with generation.open_verified(generation.root / item["path"]) as stream:
            frame = pd.read_parquet(stream)
        for row in frame.itertuples(index=False):
            yield MarketBar(symbol=item["symbol"], timestamp=ensure_kst(row.timestamp), timeframe=item["timeframe"],
                open=row.open, high=row.high, low=row.low, close=row.close, volume=row.volume,
                is_completed=bool(getattr(row, "is_completed", True))).to_json_dict()


def _check(coverage, details, name, verify):
    try:
        result = verify()
        coverage[name] = "pass"
        if result is not None:
            details[name] = result
    except (ValueError, KeyError, TypeError, OSError, IndexError, OverflowError, InvalidOperation) as exc:
        details[name] = str(exc)


def required_sessions(config, generation, calendar):
    """Derive obligations from the request, never from rows that happened to load."""
    if config.get("market_inputs"):
        market = config["market_inputs"]
        config = {**config, "start": config.get("start", market["train_start"]),
                  "end": config.get("end", market["train_end"]),
                  "universe": config.get("universe", market["universe"]),
                  "timeframe": config.get("timeframe", market["timeframe"])}
    window = config.get("date_range") or config
    start, end = window.get("start"), window.get("end")
    symbols = config.get("symbols")
    if not symbols and config.get("universe"):
        import yaml
        name = Path(config["universe"])
        candidates = [generation.root / name, generation.root / "config" / name]
        path = next((p for p in candidates if generation.digest(p)), None)
        if path is not None:
            symbols = yaml.safe_load(generation.read(path))["symbols"]
    if not start or not end or not symbols or isinstance(symbols, str):
        raise ValueError("independent_requested_dates_and_universe_required")
    start, end = date.fromisoformat(str(start)[:10]), date.fromisoformat(str(end)[:10])
    if start > end:
        raise ValueError("invalid_requested_window")
    timeframe = str(config.get("timeframe", "5m"))
    required = set()
    day = start
    while day <= end:
        if not calendar.has_coverage(day) or not calendar.has_session_coverage(day):
            raise ValueError("requested_calendar_coverage_missing")
        if calendar.is_trading_day(day):
            required.update((str(symbol).zfill(6), day, timeframe) for symbol in symbols)
        day += timedelta(days=1)
    if not required:
        raise ValueError("empty_requested_session_scope")
    return required


def execution_policy(broker):
    policy = getattr(broker, "capacity_policy", None)
    costs = getattr(broker, "strategy_costs", {})
    return {"execution_clock": getattr(broker, "execution_clock", None),
            "capacity": policy.contract() if policy is not None else None,
            "default_costs": asdict(broker.costs) if hasattr(broker, "costs") else None,
            "costs": {key: asdict(value) for key, value in sorted(costs.items())}}


def _covered(timestamps, day, minutes, symbol, sparse, *, before=None):
    result = session_bar_coverage(timestamps, day, minutes, symbol=symbol)
    cutoff = None
    if before:
        from strategy_common.execution_session import session_cutoff
        cutoff = session_cutoff(day, time.fromisoformat(before), anchor="open")
        missing = [v for v in result.get("missing_intervals", ())
                   if datetime.fromisoformat(v).time().replace(tzinfo=None) < cutoff]
        if not missing and not result.get("observations_during_halt") and not result.get("observations_outside_periodic_auctions") and not result.get("reason"):
            return True
    if result["status"] == "pass":
        return True
    accepted = sparse.get((symbol, day))
    selected = lambda values: {v for v in values if cutoff is None or v.time().replace(tzinfo=None) < cutoff}
    exact_sparse_feed = not isinstance(accepted, (set, frozenset)) or selected(timestamps) == selected(accepted)
    return (bool(timestamps) and accepted and exact_sparse_feed
            and not result.get("observations_during_halt")
            and not result.get("observations_outside_periodic_auctions")
            and result.get("reason") not in {"observations_during_suspension", "calendar_coverage"})


def _verify_feature_fallback(broker, identity, grouped, required):
    from strategy_common.source_quality import POLICY, missing_prefix
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff
    if not identity["configuration"].get("require_complete_feature_prefix"):
        raise ValueError("feature_fallback_not_enabled")
    exclusions = defaultdict(set)
    for symbol, day, timeframe in required:
        available = datetime.combine(day, session_cutoff(day, time(14, 30), anchor="open"), KST)
        stamps = grouped.get((symbol, day, timeframe), ())
        prefix = [stamp for stamp in stamps if stamp < available]
        observed = session_bar_coverage(prefix, day, 5, symbol=symbol)
        if (observed.get("observations_during_halt") or observed.get("observations_outside_periodic_auctions")
                or observed.get("reason") in {"observations_during_suspension", "calendar_coverage"}):
            raise ValueError("feature_prefix_calendar_contradiction")
        if missing_prefix(symbol, day, stamps, available):
            exclusions[day.isoformat()].add(symbol)
    for snapshot in broker.qualification_snapshots:
        metadata = snapshot.get("metadata") or {}
        excluded = exclusions[snapshot["trade_date"]]
        if (metadata.get("feature_prefix_policy") != POLICY
                or set(metadata.get("feature_prefix_exclusions", ())) != excluded
                or excluded & {row["symbol"] for row in snapshot["candidates"]}):
            raise ValueError("feature_fallback_evidence_mismatch")


def _replay_scope(broker, identity, events, required, *, sparse=None, source_stamps=None):
    if not required:
        raise ValueError("requested_replay_scope_missing")
    from strategy_common.clock import ensure_kst
    grouped = defaultdict(list)
    for row in events:
        stamp = ensure_kst(datetime.fromisoformat(row["timestamp"]))
        key = (row["symbol"], stamp.date(), row["timeframe"])
        if key not in required:
            raise ValueError("replay_event_outside_requested_scope")
        grouped[key].append(stamp)
    snapshots = getattr(broker, "qualification_snapshots", None)
    if not snapshots or stable_signature(snapshots) != identity.get("qualification_snapshot_hash"):
        raise ValueError("replay_candidate_scope_unbound")
    requested_days = {day for _, day, _ in required}
    scopes = getattr(broker, "qualification_snapshot_scopes", {"combined": snapshots})
    day_dispositions = getattr(broker, "qualification_day_dispositions", {})
    if day_dispositions and identity.get("qualification_day_dispositions_hash") != stable_signature(day_dispositions):
        raise ValueError("unbound_candidate_day_dispositions")
    for strategy, rows in scopes.items():
        if strategy.startswith("unbound_adapter:"):
            raise ValueError("adapter_strategy_identity_missing")
        observed_days = {date.fromisoformat(row["trade_date"]) for row in rows}
        disposed = {date.fromisoformat(day) for day, reason in day_dispositions.get(strategy, {}).items()
                    if reason in {"builder_verified_empty", "outside_declared_entry_horizon"}}
        if observed_days - requested_days or requested_days - observed_days != disposed:
            raise ValueError(f"candidate_session_coverage_incomplete:{strategy}")
    needed = set()
    for snapshot in snapshots:
        day = date.fromisoformat(snapshot["trade_date"])
        for candidate in snapshot["candidates"]:
            needed.add((candidate["symbol"], day))
    if {symbol for symbol, _ in needed} - {symbol for symbol, _, _ in required}:
        raise ValueError("candidate_outside_requested_universe")
    candidate_sessions = set(needed)
    policy = (identity.get("configuration") or {}).get("source_dependency_scope") or {}
    scoped = policy.get("version") in SOURCE_SCOPE_VERSIONS
    guarded = policy.get("version") in {'causal-feature-and-management-v2','causal-timed-candidates-v1'} and policy.get("strategy") == "kalcb"
    if guarded and not identity["configuration"].get("entry_require_complete_prefix"):
        raise ValueError("entry_prefix_fallback_not_enabled")
    management = _management_cutoffs(broker, requested_days) if scoped else {}
    needed.update(management if scoped else _management_sessions(broker, requested_days))
    for symbol, day, timeframe in sorted(required):
        if (symbol, day) in needed:
            stamps = grouped.get((symbol, day, timeframe), [])
            cutoff = None
            if scoped:
                cutoff = max(policy["candidate_intraday_before"] if (symbol, day) in candidate_sessions else "00:00",
                             management.get((symbol, day), "00:00"))
            if guarded and (symbol, day) in candidate_sessions:
                from strategy_common.clock import KST
                from strategy_common.source_quality import missing_prefix
                from strategy_common.execution_session import session_cutoff
                available = datetime.combine(day, session_cutoff(day, time(15, 20), anchor="open"), KST)
                retained = (source_stamps or {}).get((symbol, day, timeframe), ())
                if {v for v in stamps if v < available} != {v for v in retained if v < available}:
                    raise ValueError(f"candidate_observations_dropped:{symbol}:{day}")
                missing = missing_prefix(symbol, day, stamps, available)
                if missing:
                    first = missing[0]
                    # The core cancels pending flat entries and blocks new risk
                    # as soon as a missing completed interval is observable.
                    for row in broker.journal:
                        order = row.get("order") or {}
                        decision = order.get("metadata", {}).get("decision_available_at")
                        if (row["event"] == "submit" and order.get("side") == "BUY" and order.get("symbol") == symbol
                                and decision and first + timedelta(minutes=5) <= datetime.fromisoformat(decision) < available):
                            raise ValueError("entry_after_missing_prefix")
                    shift = datetime.combine(day, session_cutoff(day, time(9), anchor="open")) - datetime.combine(day, time(9))
                    cutoff = max((first - shift).strftime("%H:%M"), management.get((symbol, day), "00:00"))
            if not _covered(stamps, day, int(timeframe[:-1]), symbol, sparse or {}, before=cutoff):
                raise ValueError(f"management_feed_incomplete:{symbol}:{day}")


def _management_cutoffs(broker, requested_days):
    """Intervals required by actual pending orders and position lifetimes."""
    from strategy_common.clock import ensure_kst, KST
    from strategy_common.execution_session import session_cutoff
    result, orders, positions = {}, {}, {}
    current = datetime.combine(min(requested_days), time(9), tzinfo=KST)
    def include(symbol, first, last):
        for day in requested_days:
            if first.date() <= day <= last.date():
                if day < last.date():
                    before = "16:00"
                else:
                    # _covered translates reference-day clock times to special sessions.
                    reference = session_cutoff(day, time(9), anchor="open")
                    shifted = last + timedelta(minutes=5) - (datetime.combine(day, reference) - datetime.combine(day, time(9)))
                    before = shifted.strftime("%H:%M")
                result[(symbol, day)] = max(result.get((symbol, day), "00:00"), before)
    for row in getattr(broker, "journal", ()):
        kind, order, fill = row["event"], row.get("order") or {}, row.get("fill") or {}
        stamp = fill.get("timestamp") or row.get("timestamp") or order.get("submitted_at")
        if stamp is not None:
            current = max(current, ensure_kst(datetime.fromisoformat(stamp) if isinstance(stamp, str) else stamp))
        oid = order.get("order_id")
        if kind in {"submit", "amend"}:
            orders.setdefault(oid, (order["symbol"], current))
        elif kind in {"cancel", "expire", "reject"} and oid in orders:
            symbol, first = orders.pop(oid)
            include(symbol, first, current)
        elif kind == "fill":
            if oid in orders and int(fill["qty"]) >= int(order.get("qty") or fill["qty"]):
                symbol, first = orders.pop(oid)
                include(symbol, first, current)
            key = (fill["strategy_id"], fill["symbol"])
            quantity, first = positions.get(key, (0, current))
            quantity += int(fill["qty"]) * (1 if fill["side"] == "BUY" else -1)
            if quantity < 0:
                raise ValueError("invalid_management_position_lifecycle")
            if quantity:
                positions[key] = (quantity, first)
            else:
                positions.pop(key, None)
                include(key[1], first, current)
    last = datetime.combine(max(requested_days), time(23, 50), tzinfo=KST)
    for symbol, first in orders.values():
        include(symbol, first, last)
    for (_, symbol), (_, first) in positions.items():
        include(symbol, first, last)
    return result


def _management_sessions(broker, requested_days):
    """Reconstruct actual order/position lifetimes instead of requiring forever feeds.

    Candidate sessions remain separate obligations. Full source/feature coverage
    is still checked against the independently declared research universe.
    """
    from strategy_common.clock import ensure_kst
    needed, orders, positions = set(), {}, {}
    current = min(requested_days)

    def day_of(value):
        stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
        return ensure_kst(stamp).date()

    def include(symbol, first, last):
        needed.update((symbol, day) for day in requested_days if first <= day <= last)

    for row in getattr(broker, "journal", ()):
        kind, order = row["event"], row.get("order") or {}
        fill = row.get("fill") or {}
        stamp = fill.get("timestamp") or row.get("timestamp")
        if kind in {"submit", "amend"}:
            stamp = order.get("metadata", {}).get("activation_at") or order.get("submitted_at") or stamp
        if stamp is not None:
            current = max(current, day_of(stamp))
        oid = order.get("order_id")
        if kind in {"submit", "amend"}:
            orders.setdefault(oid, (order["symbol"], current))
        elif kind in {"cancel", "expire", "reject"}:
            if oid in orders:
                symbol, first = orders.pop(oid)
                include(symbol, first, current)
        elif kind == "fill":
            if oid in orders and int(fill["qty"]) >= int(order.get("qty") or fill["qty"]):
                symbol, first = orders.pop(oid)
                include(symbol, first, current)
            key = (fill["strategy_id"], fill["symbol"])
            quantity, first = positions.get(key, (0, current))
            quantity += int(fill["qty"]) * (1 if fill["side"] == "BUY" else -1)
            if quantity < 0:
                raise ValueError("invalid_management_position_lifecycle")
            positions[key] = (quantity, first)
            if quantity == 0:
                include(key[1], first, current)
                positions.pop(key)
    for symbol, first in orders.values():
        include(symbol, first, max(requested_days))
    for (_, symbol), (_, first) in positions.items():
        include(symbol, first, max(requested_days))
    return needed


def _references(read, evidence, calendar, identity, broker=None):
    refs = evidence["references"]
    if refs.get("unavailable_reason"):
        raise ValueError(refs["unavailable_reason"])
    snapshots = getattr(broker, "qualification_snapshots", None)
    if snapshots is None:
        snapshots = [read(path) for path in refs["snapshots"]]
    # Bind the evidence to the snapshots actually consumed by the adapter.
    if not snapshots or stable_signature(snapshots) != identity.get("qualification_snapshot_hash"):
        raise ValueError("candidate_snapshot_binding_missing_or_mismatched")
    rows = read(refs["daily_rows"])
    documents = _source_contracts(read, refs["source_documents"], "daily_reference_contract_v1")
    market_documents = _source_contracts(read, evidence["basis"]["source_documents"], "market_basis_contract_v1")
    market_bases = {stable_signature(target_basis(c)) for c in market_documents.values()}
    if len(market_bases) != 1:
        raise ValueError("incompatible_market_source_bases")
    daily = {(r["symbol"], r["date"]): r for r in rows}
    if len(daily) != len(rows):
        raise ValueError("duplicate_daily_reference")
    from .retained_source import VERSION, verify_acceptance
    retained = refs.get("representation") == VERSION
    if retained:
        verify_acceptance(read, evidence)
    for snapshot in snapshots:
        day = date.fromisoformat(snapshot["trade_date"])
        prior = calendar.previous_trading_day(day).isoformat()
        for candidate in snapshot["candidates"]:
            if candidate["trade_date"] != day.isoformat() or candidate["metadata"].get("prior_day_date") != prior:
                raise ValueError("wrong_previous_session")
            source = daily[(candidate["symbol"], prior)]
            contract = documents[refs["daily_rows"]] if retained else documents[source["source_path"]]
            if stable_signature(target_basis(contract)) not in market_bases:
                raise ValueError("incompatible_daily_intraday_basis")
            rule = conversion(contract, candidate["symbol"], prior, read)
            if contract.get("price_unit") != "KRW" or contract.get("date_convention") != "KRX_session_date":
                raise ValueError("daily_reference_source_semantics_mismatch")
            if retained:
                for field in ("high", "low", "close"):
                    value = float(candidate[f"prior_day_{field}"])
                    if not math.isfinite(value) or value <= 0 or not math.isclose(value, float(source[field]), rel_tol=1e-10):
                        raise ValueError("prior_reference_value_mismatch")
                continue
            index = source["source_index"]
            if not isinstance(index, int) or index < 0:
                raise ValueError("invalid_daily_source_index")
            raw = read(source["source_path"])[index]
            if (raw["symbol"], raw["date"]) != (candidate["symbol"], prior):
                raise ValueError("daily_reference_source_row_mismatch")
            for field in ("high", "low", "close"):
                value = float(candidate[f"prior_day_{field}"])
                if (not math.isfinite(value) or value <= 0
                    or not math.isclose(float(source[field]), converted_value(raw[field], field, rule), rel_tol=1e-10)
                    or not math.isclose(value, float(source[field]), rel_tol=1e-10)):
                    raise ValueError("prior_reference_value_mismatch")


def _basis(read, evidence, events):
    basis = evidence["basis"]
    if basis.get("unavailable_reason"):
        raise ValueError(basis["unavailable_reason"])
    if not basis["source_documents"]:
        raise ValueError("missing_vendor_basis_sources")
    documents = _source_contracts(read, basis["source_documents"], "market_basis_contract_v1")
    if len({stable_signature(target_basis(c)) for c in documents.values()}) != 1:
        raise ValueError("incompatible_market_source_bases")
    from .retained_source import VERSION, verify_acceptance
    if basis.get("representation") == VERSION:
        report = verify_acceptance(read, evidence)
        affected = {(r["symbol"], r["timestamp"][:10]) for r in events}
        failures = [r for r in report["basis_failures"] if (r["symbol"], r["date"]) in affected]
        if failures:
            raise ValueError("retained_basis_unverified:" + stable_signature(failures))
        for contract in documents.values():
            if (contract.get("volume_unit"), contract.get("volume_kind"), contract.get("price_unit"),
                    contract.get("timezone"), contract.get("timestamp_convention")) != (
                    "shares", "interval", "KRW", "Asia/Seoul", "completed_interval_start"):
                raise ValueError("retained_basis_contract_mismatch")
        return {"scope": report["scope"], "acceptance_hash": stable_signature(report),
                "limitations": report["limitations"]}
    rows = read(basis["observations"])
    indexed = {(r["symbol"], r["timestamp"]): r for r in rows}
    if len(indexed) != len(rows) or not {(e["symbol"], e["timestamp"]) for e in events}.issubset(indexed):
        raise ValueError("incomplete_basis_observations")
    for event in events:
        row = indexed[(event["symbol"], event["timestamp"])]
        contract = documents[row["source_path"]]
        if (contract.get("price_unit"), contract.get("volume_unit"), contract.get("volume_kind")) != ("KRW", "shares", "interval"):
            raise ValueError("basis_source_units_mismatch")
        if contract["timestamp_convention"] not in {"completed_interval_start", "source_label_with_verified_shift"} or contract["timezone"] != "Asia/Seoul":
            raise ValueError("unsupported_source_clock_convention")
        if (row["price_unit"], row["volume_unit"], row["volume_kind"], row["venue"]) != ("KRW", "shares", "interval", "KRX"):
            raise ValueError("unsupported_price_volume_basis")
        rule = conversion(contract, row["symbol"], row["timestamp"][:10], read)
        for field in ("price_factor", "share_factor"):
            if field in row and float(row[field]) != float(rule[field]):
                raise ValueError("basis_factor_contract_mismatch")
        index = row["source_index"]
        if not isinstance(index, int) or index < 0:
            raise ValueError("invalid_basis_source_index")
        source = read(row["source_path"])[index]
        if source["symbol"] != event["symbol"] or converted_timestamp(source["timestamp"], rule) != datetime.fromisoformat(event["timestamp"]):
            raise ValueError("basis_source_event_mismatch")
        for field in ("open", "high", "low", "close", "volume"):
            if not math.isclose(converted_value(source[field], field, rule), float(event[field]), rel_tol=1e-10):
                raise ValueError("basis_conversion_mismatch")


def _capacity(read, evidence, broker, *, capacity_report=None):
    capacity = evidence["capacity"]
    if capacity.get("unavailable_reason"):
        raise ValueError(capacity["unavailable_reason"])
    if capacity.get("kind") == "declared_bar_participation_model":
        return _declared_capacity(read, evidence, broker, capacity_report=capacity_report)
    if capacity.get("kind") not in {"observed_phase_liquidity", "calibrated_participation_model"} or not capacity["source_documents"]:
        raise ValueError("supported_phase_liquidity_evidence_required")
    documents = _source_contracts(read, capacity["source_documents"], "phase_liquidity_contract_v1")
    market_documents = _source_contracts(read, evidence["basis"]["source_documents"], "market_basis_contract_v1")
    bases = {stable_signature(target_basis(d)) for d in market_documents.values()}
    if len(bases) != 1 or any(stable_signature(target_basis(d)) not in bases for d in documents.values()):
        raise ValueError("incompatible_capacity_market_basis")
    policy_contract = execution_policy(broker)
    policies = capacity.get("registered_execution_policies", [])
    registered = [p for p in policies if p.get("policy") == policy_contract]
    if len(registered) != 1:
        raise ValueError("execution_policy_mismatch")
    fraction = float(registered[0]["participation"])
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("invalid_participation")
    policy = getattr(broker, "capacity_policy", None)
    if fraction != (policy.participation if policy is not None else 1.0):
        raise ValueError("participation_policy_mismatch")
    if capacity["kind"] == "calibrated_participation_model":
        return _capacity_model(read, capacity, broker, documents, fraction)
    rows = read(capacity["observations"])
    indexed = {(r["symbol"], r["timestamp"], r["phase"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate_phase_observation")
    demand = defaultdict(int)
    for fill in broker.fills:
        phase = fill.metadata["execution_phase"]
        key = (fill.symbol, fill.timestamp.isoformat(), phase)
        row = indexed[key]
        contract = documents[row["source_path"]]
        if phase not in contract["phases"] or contract["volume_unit"] != "shares":
            raise ValueError("phase_source_semantics_mismatch")
        index = row["source_index"]
        if not isinstance(index, int) or index < 0:
            raise ValueError("invalid_phase_source_index")
        source = read(row["source_path"])[index]
        if any(row[field] != source[field] for field in ("symbol", "timestamp", "phase", "volume_kind", "volume", "price_low", "price_high")):
            raise ValueError("phase_source_row_mismatch")
        if row["volume_kind"] != "phase_executable_shares":
            raise ValueError("bar_volume_is_not_phase_liquidity")
        rule = conversion(contract, row["symbol"], row["timestamp"][:10], read)
        volume = converted_value(row["volume"], "volume", rule)
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("invalid_phase_volume")
        if not converted_value(row["price_low"], "low", rule) <= float(fill.metadata["base_price"]) <= converted_value(row["price_high"], "high", rule):
            raise ValueError("execution_price_outside_phase_observation")
        demand[key] += fill.qty
        if demand[key] > math.floor(volume * fraction):
            raise ValueError("shared_phase_demand_exceeds_registered_limit")


def _declared_capacity(read, evidence, broker, *, capacity_report=None):
    from .capacity import BarParticipationScenario, CAPACITY_VERSION
    from .capacity_audit import audit_capacity
    declaration = evidence["capacity"]
    if (not isinstance(broker.capacity_policy, BarParticipationScenario)
            or declaration.get("model_version") != CAPACITY_VERSION
            or declaration.get("qualification_scope") != "historical_research_model"
            or declaration.get("phase_liquidity_verified") is not False):
        raise ValueError("unsupported_declared_research_model")
    matches = [r for r in declaration.get("registered_execution_policies", [])
        if r.get("policy") == execution_policy(broker)
        and r.get("initial_equity") == broker.initial_equity
        and r.get("buying_power_leverage") == broker.buying_power_leverage]
    if len(matches) != 1:
        raise ValueError("research_model_account_or_policy_mismatch")
    if not getattr(broker, "expected_batches", None):
        raise ValueError("research_model_source_events_missing")
    basis = evidence["basis"]
    if basis.get("unavailable_reason"):
        raise ValueError(basis["unavailable_reason"])
    documents = _source_contracts(read, basis["source_documents"], "market_basis_contract_v1")
    bases = {stable_signature(target_basis(c)) for c in documents.values()}
    if len(bases) != 1 or stable_signature(declaration.get("market_basis")) not in bases:
        raise ValueError("research_model_market_basis_mismatch")
    report = capacity_report if capacity_report is not None else audit_capacity(broker)
    shadows = report.get("selection_shadow_reports")
    if shadows is None:
        shadows = [audit_capacity(b) for b in getattr(broker, "capacity_shadow_brokers", ())]
    if report["status"] != "model_consistent" or report["errors"] or any(r["errors"] or r["status"] != "model_consistent" for r in shadows):
        raise ValueError("research_model_capacity_path_failed")
    return {"basis": "declared_research_model", "qualification_scope": "historical_research_model",
        "model_version": CAPACITY_VERSION, "policy": execution_policy(broker),
        "market_basis": declaration["market_basis"], "initial_equity": broker.initial_equity,
        "capital_observed": report["capital"], "phase_liquidity_verified": False,
        "exact_executable_quantity_observed": False, "paper_live_qualified": False,
        "validation": "independent_causal_budget_ledger_and_realized_volume_checks",
        "account_registration_hash": stable_signature(matches[0])}


def _capacity_model(read, capacity, broker, documents, fraction):
    """Qualify a declared volume proxy against independent earlier phase prints.

    This establishes measured model coverage, not counterfactual queue certainty.
    Source rows must include the parsed predictor and phase prints; those rows and
    their parser/source contracts are frozen just like the observed-data path.
    """
    from .capacity import BarParticipationScenario, CAPACITY_VERSION
    if not isinstance(broker.capacity_policy, BarParticipationScenario):
        raise ValueError("unsupported_capacity_model_policy")
    model = capacity["model"]
    if model.get("algorithm") != CAPACITY_VERSION:
        raise ValueError("unsupported_capacity_model")
    low, high = (float(v) for v in model["capital_range"])
    if not 0 < low <= broker.initial_equity <= high or not all(math.isfinite(v) for v in (low, high)):
        raise ValueError("capacity_model_capital_outside_support")
    maximum = float(model["max_participation_of_phase_prints"])
    tolerance = float(model["max_budget_exceedance_fraction"])
    minimum = model["min_observations_per_phase"]
    if (not 0 < maximum <= 1 or not 0 <= tolerance < 1
            or not isinstance(minimum, int) or minimum < 2):
        raise ValueError("invalid_capacity_model_validation_contract")
    first = min(datetime.fromisoformat(e["timestamp"]) for batch in broker.expected_batches for e in batch)
    fit_end = datetime.fromisoformat(model["fit_end"])
    if fit_end.tzinfo is None or fit_end >= first:
        raise ValueError("capacity_model_fit_leaks_evaluation")
    groups = defaultdict(list)
    seen = set()
    for row in read(capacity["observations"]):
        key = (row["symbol"], row["timestamp"], row["phase"])
        if key in seen:
            raise ValueError("duplicate_model_observation")
        seen.add(key)
        stamp = datetime.fromisoformat(row["timestamp"])
        if stamp.tzinfo is None or stamp >= first:
            raise ValueError("capacity_model_validation_leaks_evaluation")
        contract = documents[row["source_path"]]
        if (row["phase"] not in contract["phases"] or contract["volume_unit"] != "shares"
                or contract.get("predictor") != "previous_contiguous_continuous_5m_else_current_auction"):
            raise ValueError("capacity_model_predictor_semantics_mismatch")
        index = row["source_index"]
        if not isinstance(index, int) or index < 0:
            raise ValueError("invalid_model_source_index")
        source = read(row["source_path"])[index]
        fields = ("symbol", "timestamp", "phase", "volume_kind", "volume", "predictor_volume")
        if any(row[field] != source[field] for field in fields) or row["volume_kind"] != "phase_traded_shares":
            raise ValueError("capacity_model_source_mismatch")
        rule = conversion(contract, row["symbol"], row["timestamp"][:10], read)
        volume, predictor = (converted_value(row[k], "volume", rule) for k in ("volume", "predictor_volume"))
        if not all(math.isfinite(v) and v >= 0 for v in (volume, predictor)):
            raise ValueError("invalid_model_volume")
        groups[(row["symbol"], row["phase"], "fit" if stamp <= fit_end else "validation")].append(
            math.floor(predictor * fraction) > math.floor(volume * maximum))
    phases = {f.metadata["execution_phase"] for f in broker.fills}
    if not phases.issubset(set(model["supported_phases"])):
        raise ValueError("capacity_model_phase_outside_support")
    summary = {}
    for symbol, phase in sorted({(f.symbol, f.metadata["execution_phase"]) for f in broker.fills}):
        for split in ("fit", "validation"):
            values = groups[(symbol, phase, split)]
            if len(values) < minimum or sum(values) / len(values) > tolerance:
                raise ValueError(f"capacity_model_validation_failed:{symbol}:{phase}:{split}")
            summary[f"{symbol}:{phase}:{split}"] = {"observations": len(values), "budget_exceedances": sum(values)}
    return {"basis": "model_conditioned", "algorithm": model["algorithm"],
            "capital_range": [low, high], "max_budget_exceedance_fraction": tolerance,
            "max_participation_of_phase_prints": maximum, "validation": summary,
            "exact_executable_quantity_observed": False}


def _source_contracts(read, paths, kind):
    documents = {}
    for path in paths:
        contract = read(path)
        if not isinstance(contract, dict) or contract.get("kind") != kind:
            raise ValueError("unrelated_source_document")
        if (contract.get("venue") != "KRX" or not contract.get("provider")
                or not contract.get("source_url") or not contract.get("parser_version")
                or not contract.get("raw_paths")):
            raise ValueError("incomplete_source_semantics")
        for raw in contract["raw_paths"]:
            if raw in documents:
                raise ValueError("ambiguous_source_contract")
            documents[raw] = contract
    return documents
