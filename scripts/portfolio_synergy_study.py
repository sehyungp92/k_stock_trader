"""Freeze and replay current KALCB/OLR through the native shared account.

Development window only. This program never changes deployment configuration.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import date
import gzip
from itertools import groupby
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.executed_value import digest, signature, write_json

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = {"baseline": (.01, 3, 5, "research-baseline"),
             "participation": (.005, 3, 5, "participation-stress"),
             "cost": (.01, 13, 15, "cost-stress"),
             "combined": (.005, 13, 15, "combined-stress")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def policy_recipes(equity):
    base = dict(max_gross_notional=2*equity, max_symbol_notional=.65*equity,
        max_sector_notional=equity, notional_cap_mode="equity_scaled",
        notional_cap_reference_equity=equity, funding_mode="native_buying_power",
        same_symbol_policy="aggregate", strategy_priority=["KALCB", "OLR"])
    return {
        "cash_control": {"policy": {**base, "funding_mode": "cash", "same_symbol_policy": "block", "max_gross_notional": equity}},
        "native_block": {"policy": {**base, "same_symbol_policy": "block"}},
        "native_aggregate": {"policy": base},
        "olr_priority": {"policy": {**base, "strategy_priority": ["OLR", "KALCB"]}},
        "candidate_rank": {"policy": base, "decision_rank": True},
        "quality": {"policy": base, "quality": True, "decision_rank": True},
        "diversified": {"policy": {**base, "max_gross_notional": 1.8*equity, "max_symbol_notional": .45*equity, "max_sector_notional": .9*equity}, "quality": True, "decision_rank": True},
        "balanced": {"policy": {**base, "max_gross_notional": 1.5*equity, "max_symbol_notional": .35*equity, "max_sector_notional": .75*equity}, "quality": True, "decision_rank": True},
        "active": {"policy": {**base, "max_symbol_notional": .5*equity, "max_sector_notional": 1.2*equity}, "quality": True, "decision_rank": True},
    }


def prepare(args):
    from scripts.kalcb_executed_value_replay import verify_policy, timed_policy
    from backtests.strategies.olr.runner import attach_overnight_labels_to_snapshots, _aggregate_snapshot_hash
    from strategy_olr.models import OLRDailySnapshot
    from backtests.strategies.olr.replay_cache import _eligible_snapshot_dates
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.engine.sim_broker import BrokerCosts
    from backtests.engine.qualification_evidence import execution_policy
    from backtests.strategies.portfolio_synergy.source_replay import NativeRiskSharedBroker
    parent = read(args.kalcb_manifest)
    payload = {k: v for k, v in parent.items() if k != "manifest_hash"}
    if signature(payload) != parent["manifest_hash"]:
        raise ValueError("Parent manifest changed")
    plan = read(args.kalcb_plan)
    for path, expected in plan["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Changed KALCB dependency: " + path)
    pm, ks = verify_policy(args.kalcb_policies, "interaction_cost_margin")
    timed_policy(args.kalcb_policies, pm, ks)
    dataset_file = next(args.olr_dataset.glob("*.pkl"))
    if digest(dataset_file) != dataset_file.with_suffix(".sha256").read_text():
        raise ValueError("Changed OLR dataset")
    with dataset_file.open("rb") as stream:
        dataset = pickle.load(stream)
    osnaps = {}
    eligible = set(_eligible_snapshot_dates(dataset.trading_dates))
    for path in sorted(args.olr_snapshots.glob("candidate_snapshot_*.json")):
        raw = read(path)
        snap = OLRDailySnapshot.from_json_dict(raw)
        if snap.trade_date not in eligible:
            continue
        if snap.artifact_hash != raw["artifact_hash"] or snap.trade_date in osnaps:
            raise ValueError("Changed or duplicate OLR snapshot")
        osnaps[snap.trade_date] = snap
    osnaps = attach_overnight_labels_to_snapshots(osnaps, dataset.overnight_labels_by_key)
    canonical_olr = REPO / "data/backtests/output/olr/round_2/diagnostics/training"
    expected = read(canonical_olr / "metrics.json")["candidate_snapshot_hash"]
    if _aggregate_snapshot_hash(osnaps) != expected:
        raise ValueError("OLR canonical consumed candidate identity mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "inputs"
    root = bundle / "files"
    root.mkdir(parents=True)
    files = {}

    def bind(source, relative, expected=None, mutable=False):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Duplicate immutable binding: " + relative)
        if mutable:
            shutil.copy2(source, target)
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        actual = digest(target)
        if expected and actual != expected:
            raise ValueError("Changed parent input: " + relative)
        files[relative] = actual

    for name, expected_hash in parent["files"].items():
        code = REPO / name
        replacement = code.is_file() and name.endswith(".py")
        bind(code if replacement else args.kalcb_manifest.parent / "files" / name,
             name, None if replacement else expected_hash, mutable=replacement)
    for directory in ("backtests", "strategy_common", "strategy_kalcb", "strategy_olr", "strategy_pcim", "deployment", "instrumentation", "oms", "oms_client", "kis_core", "scripts"):
        for path in (REPO / directory).rglob("*.py"):
            name = path.relative_to(REPO).as_posix()
            if name not in files:
                bind(path, name, mutable=True)

    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
        files[name] = digest(path)
        return name

    prefix = "research/portfolio_current/"
    kconfig = save(prefix + "kalcb_config.json", read(plan["config"]))
    orequest = save(prefix + "olr_request.json", read(canonical_olr / "request.json"))
    osnapshot_path = save(prefix + "olr_snapshots.json", [s.to_json_dict() for s in osnaps.values()])
    ktrades = read(args.kalcb_trades)
    with gzip.open(canonical_olr / "trades.jsonl.gz", "rt", encoding="utf-8") as stream:
        otrades = [json.loads(line) for line in stream if line.strip()]
    labels = save(prefix + "quality_labels.json", ktrades + otrades)
    database_hash = plan["input_sha256"][plan["database"]]
    database = next((name for name, sha in files.items() if sha == database_hash), None)
    if database is None:
        database = prefix + "market.sqlite"
        bind(plan["database"], database, database_hash)
    # Bind exact new costs/account sizes to the existing declared capacity model.
    evidence = dict(parent["evidence"])
    q = read(root / evidence["qualification_path"])
    for _, (fraction, kslip, oslip, scenario_name) in SCENARIOS.items():
        broker = NativeRiskSharedBroker(args.equity,
            strategy_costs={"KALCB": BrokerCosts(2, 20, kslip), "OLR": BrokerCosts(2, 20, oslip)},
            strategy_leverage={"KALCB": 2, "OLR": 1},
            capacity_policy=BarParticipationScenario(fraction, scenario_name))
        registration = dict(account="shared_current_20261007", initial_equity=args.equity,
            buying_power_leverage=2., policy=execution_policy(broker))
        q["capacity"]["registered_execution_policies"].append(registration)
    evidence["qualification_path"] = save(prefix + "qualification.json", q)
    reference = read(plan["historical_reference"])["metrics"]["audit_receipt"]["identity"]["configuration"]
    configuration = {**reference, "date_range": {"start": pm["evaluation_from"], "end": pm["evaluation_through"]},
        "start": pm["evaluation_from"], "end": pm["evaluation_through"],
        "source_dependency_scopes": {
            "KALCB": {"scope": {"version": "causal-timed-candidates-v1", "strategy": "kalcb", "feature_intraday_before": "09:45", "candidate_intraday_before": "15:20"}, "entry_require_complete_prefix": True},
            "OLR": {"scope": {"version": "causal-feature-and-management-v2", "strategy": "olr", "feature_intraday_before": "14:30", "candidate_intraday_before": "16:00"}, "entry_require_complete_prefix": False}}}
    configuration.pop("source_dependency_scope", None)
    configuration["require_complete_feature_prefix"] = bool(read(canonical_olr / "request.json")["config"]["require_complete_feature_prefix"])
    spec = dict(version="current-native-portfolio-v1", initial_equity=args.equity,
        kalcb_config=kconfig, olr_request=orequest, olr_snapshots=osnapshot_path,
        kalcb_policies="research/kalcb_executed_value/policies", kalcb_policy="interaction_cost_margin",
        database=database, database_hash=database_hash, labels=labels,
        configuration=configuration, candidates=policy_recipes(args.equity),
        initial_mutations={"recipe": "cash_control"},
        canonical_olr_snapshot_hash=expected, parent_manifest_hash=parent["manifest_hash"],
        scope="reused_development_only", risk_preferences={"baseline_dd": 8, "combined_dd": 10},
        limitations=["Selected-trade labels; no labels for never-admitted opportunities", "No independent holdout claim", "Declared capacity model, not observed queue liquidity", "Tax conservative flat 20bps for both sleeves"])
    from scripts.refresh_portfolio_synergy_runtime import bind_calendar_evidence, bind_shared_calendar_contract
    bind_calendar_evidence(REPO, root, files)
    bind_shared_calendar_contract(evidence, root, files)
    spec["configuration"]["market_input_hash"] = evidence["shared_market_inputs"]["market_input_hash"]
    spec_path = save(prefix + "study.json", spec)
    program = "from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())"
    source = subprocess.check_output([sys.executable, "-B", "-c", program], cwd=root, text=True).strip()
    evidence["portfolio_current_study"] = spec_path
    payload = dict(version=parent["version"], source_root=parent["source_root"], source_generation_hash=source, files=files, evidence=evidence)
    write_json(bundle / "manifest.json", {**payload, "manifest_hash": signature(payload)})
    write_json(args.output / "study.json", {"manifest": str((bundle / "manifest.json").resolve()), "source_generation_hash": source,
        "code": str(root.resolve()), "spec_path": spec_path, **spec})
    print(json.dumps({"prepared": str(args.output), "source": source, "files": len(files)}), flush=True)


def replay(args):
    from backtests.core.input_manifest import InputManifest
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.engine.sim_broker import BrokerCosts
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.integrity import audit_broker
    from backtests.analysis.metrics import compute_trade_metrics
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter, _collapse_exit_legs
    from backtests.strategies.olr.runner import OLRReplayAdapter
    from strategy_kalcb.config import KALCBConfig
    from strategy_olr.config import OLRConfig
    from strategy_olr.models import OLRDailySnapshot
    from backtests.strategies.portfolio_synergy.source_replay import NativeRiskSharedBroker
    from backtests.strategies.portfolio_synergy.quality import CausalQualityModel
    from deployment.olr_kalcb.portfolio import PortfolioPolicyConfig
    from scripts.kalcb_executed_value_replay import verify_policy, timed_policy, observation_prefix, equity_records
    from scripts.kalcb_executed_value_study import connect, load_day
    study = read(args.study)
    generation = InputManifest(Path(study["manifest"]))
    if REPO != generation.root.resolve() or source_generation_hash() != study["source_generation_hash"]:
        raise ValueError("Replay must execute frozen registered source")
    frozen_spec = json.loads(generation.read(generation.root / study["spec_path"]))
    if any(study[k] != value for k, value in frozen_spec.items()):
        raise ValueError("Changed study registration")
    def bound(name):
        return json.loads(generation.read(generation.root / study[name]))
    recipe = study["candidates"][args.recipe]
    action_events = bound("corporate_actions")["events"] if study.get("corporate_actions") else []
    action_checks = []
    allocation = recipe.get("allocation") if not args.standalone else None
    native_budget = recipe.get('native_budget')
    policy = PortfolioPolicyConfig(**recipe["policy"])
    if args.absolute_caps:
        policy = replace(policy, notional_cap_mode="absolute")
    if args.no_upward_compounding:
        policy = replace(policy, notional_cap_mode="equity_capped")
    fraction, kslip, oslip, scenario_name = SCENARIOS[args.scenario]
    kcfg = KALCBConfig.from_mapping({**bound("kalcb_config"), "slippage_bps": kslip, "tax_bps_on_sell": 20})
    request = bound("olr_request")
    ocfg = replace(OLRConfig.from_mapping(request["config"], request["mutations"]), slippage_bps=oslip, tax_bps_on_sell=20)
    if recipe.get("olr_capacity"):
        settings = dict(recipe["olr_capacity"])
        if any(not key.startswith("capacity_") for key in settings):
            raise ValueError("OLR capacity study cannot alter signal or cost settings")
        ocfg = replace(ocfg, **settings)
    if recipe.get('olr_execution_stress'):
        settings = dict(recipe['olr_execution_stress'])
        if set(settings) - {'auction_adverse_bps', 'auction_nonfill_rate'}:
            raise ValueError('Unsupported OLR execution perturbation')
        ocfg = replace(ocfg, **settings)
    costs = {"KALCB": BrokerCosts(2, 20, kslip), "OLR": BrokerCosts(2, 20, oslip)}
    if args.canonical_costs:
        if args.standalone != "KALCB" or args.full_audit:
            raise ValueError("Canonical cost control is a standalone KALCB parity check")
        kcfg = replace(kcfg, tax_bps_on_sell=18)
        costs["KALCB"] = BrokerCosts(2, 18, kslip)
    generation.read(generation.root / study["kalcb_policies"] / "policies_manifest.json")
    pm, ks = verify_policy(generation.root / study["kalcb_policies"], study["kalcb_policy"])
    updates = timed_policy(generation.root / study["kalcb_policies"], pm, ks)
    marginal_evidence = None
    if recipe.get('kalcb_marginal_quantity'):
        if recipe['kalcb_marginal_quantity'] != 'supported_next_lower_tier_v1':
            raise ValueError('Unregistered marginal quantity policy')
        from backtests.strategies.portfolio_synergy.targeted_policy import transform_marginal_policy
        ks, updates, marginal_evidence = transform_marginal_policy(
            generation.root / study['kalcb_policies'], study['kalcb_policy'], ks, updates)
    osnaps = {s.trade_date: s for raw in bound("olr_snapshots") if (s := OLRDailySnapshot.from_json_dict(raw)).trade_date in ks}
    missing_olr = set(ks) - set(osnaps)
    if missing_olr != set(sorted(ks)[-2:]):
        raise ValueError("Unexpected OLR entry horizon; require canonical two-session tail")
    model = CausalQualityModel(bound("labels")) if recipe.get("quality") else None
    if args.standalone and native_budget is None:
        policy, model = None, None
    broker_class = NativeRiskSharedBroker
    broker_options = {}
    if native_budget is not None:
        if recipe.get('allocation') or recipe.get('reference_sizing') or recipe.get('quality') or recipe.get('decision_rank'):
            raise ValueError('Native budget recipes require unchanged native allocation and ordering')
        from deployment.olr_kalcb.native_budget import NativeBudgetConfig
        from backtests.strategies.portfolio_synergy.native_budget import NativeBudgetSharedBroker
        broker_class = NativeBudgetSharedBroker
        broker_options = {'native_budget': NativeBudgetConfig(**native_budget)}
    if recipe.get("reference_sizing"):
        if not args.standalone or allocation is not None:
            raise ValueError("Reference sizing is a standalone benchmark only")
        from backtests.strategies.portfolio_synergy.intent_replay import ReferenceSizingBroker
        broker_class = ReferenceSizingBroker
        broker_options = {"reference_sizing": recipe["reference_sizing"]}
    if allocation is not None:
        from deployment.olr_kalcb.intent_allocator import IntentAllocationConfig
        from backtests.strategies.portfolio_synergy.intent_replay import (
            IntentSharedBroker, IntentKALCBReplayAdapter, IntentOLRReplayAdapter)
        broker_class = IntentSharedBroker
        KALCBReplayAdapter, OLRReplayAdapter = IntentKALCBReplayAdapter, IntentOLRReplayAdapter
        allocation_config = IntentAllocationConfig(**allocation)
        broker_options = {"allocation_config": allocation_config,
            "value_model": CausalQualityModel(bound("labels")) if allocation_config.rank_mode == "supported_value" else None}
    broker = broker_class(study["initial_equity"], strategy_costs=costs,
        strategy_leverage={"KALCB": kcfg.intraday_leverage, "OLR": 1},
        portfolio_policy=policy, quality_model=model,
        decision_rank=recipe.get("decision_rank", False) and not args.standalone,
        sizing_equity_cap=study["initial_equity"] if args.no_upward_compounding else 0,
        capacity_policy=BarParticipationScenario(fraction, scenario_name), **broker_options)
    # The declared sector map is fixed for this research generation.
    sectors = {}
    for snap in [*ks.values(), *osnaps.values(), *(s for rows in updates.values() for s in rows)]:
        for candidate in snap.candidates:
            sector = str(candidate.sector or "UNKNOWN").strip().upper()
            if candidate.symbol in sectors and sectors[candidate.symbol] != sector:
                raise ValueError("Dated sector policy required")
            sectors[candidate.symbol] = sector
    broker.policy_sectors = sectors
    adapters = (KALCBReplayAdapter(kcfg, ks, initial_equity=study["initial_equity"], costs=costs["KALCB"],
        timed_snapshots=updates, observation_symbols=tuple(pm["observation_symbols"])), OLRReplayAdapter(ocfg, osnaps))
    if args.standalone:
        adapters = tuple(a for a in adapters if a.strategy_id == args.standalone)
    broker.evaluation_window = study["configuration"]["date_range"]
    broker.qualification_day_dispositions = {"OLR": {str(day): "outside_declared_entry_horizon" for day in missing_olr}}
    args.output.mkdir(parents=True, exist_ok=False)
    if marginal_evidence is not None:
        write_json(args.output/'marginal_quantity_evidence.json', marginal_evidence)
    write_json(args.output / "request.json", {"study_hash": digest(args.study), "arguments": vars(args), "recipe": recipe,
        "policy": asdict(policy) if policy else None, "source_generation_hash": study["source_generation_hash"]})
    connection = connect({"database": str(generation.root / study["database"])})
    # Verifying the whole SQLite file once binds every loaded session.
    if (generation.digest(generation.root / study["database"]) != study["database_hash"]
            or digest(generation.root / study["database"]) != study["database_hash"]):
        raise ValueError("Changed market database")
    capacity_history_receipt = None
    def auction_observations(day):
        from strategy_common.market import MarketBar
        from datetime import datetime
        from strategy_olr.core.capacity import phase
        rows=connection.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? AND o=h AND h=l AND l=c ORDER BY ts,symbol',(str(day),))
        bars=[MarketBar(s,datetime.fromisoformat(t),'5m',*values,source='kis_krx_parquet',
                       source_fingerprint=study['database_hash']) for s,t,*values in rows]
        return [b for b in bars if phase(b)=='auction']
    if ocfg.capacity_broad_auction_history and args.standalone=='OLR':
        from strategy_olr.core.capacity import seed_history
        from datetime import datetime, time
        from strategy_common.clock import KST
        from backtests.auto.shared.cache_keys import stable_signature
        prior=[r[0] for r in connection.execute('SELECT DISTINCT day FROM bars WHERE day<? ORDER BY day DESC LIMIT 20',(str(min(ks)),))]
        seeds=[b for day in reversed(prior) for b in auction_observations(day)]
        for adapter in adapters:
            if adapter.strategy_id=='OLR':seed_history(adapter.state,seeds,ocfg,datetime.combine(min(ks),time(0),tzinfo=KST))
        capacity_history_receipt={'database_hash':study['database_hash'],'seed_rows':len(seeds),
            'seed_hash':stable_signature([b.to_json_dict() for b in seeds]),'seed_days':sorted(prior),
            'asof':str(min(ks)),'subsequent_auction_observations':[]}
    daily, reasons = [], Counter()
    decision_stream = gzip.open(args.output / "native_decisions.jsonl.gz", "wt", encoding="utf-8") if allocation is not None or native_budget is not None else None
    for index, day in enumerate(sorted(ks)):
        if action_events:
            from backtests.engine.corporate_action_guard import verify_zero_exposure
            action_checks.extend(verify_zero_exposure(broker, day, action_events))
        symbols = {c.symbol for s in (ks[day], osnaps.get(day), *updates.get(day, ())) if s is not None for c in s.candidates}
        symbols.update(symbol for _, symbol in broker.positions)
        symbols.update(order.symbol for order in broker.orders)
        symbols.add("000100")  # Real account marks even on cash-only sessions.
        bars = load_day(connection, str(day), study["database_hash"], symbols | set(pm["observation_symbols"]))
        bars = observation_prefix(bars, day, symbols, pm["observation_minutes"])
        if capacity_history_receipt is not None:
            observations=auction_observations(day)
            capacity_history_receipt['subsequent_auction_observations'].append({'day':str(day),'count':len(observations),
                'hash':stable_signature([b.to_json_dict() for b in observations])})
            bars=sorted({(b.symbol,b.timestamp):b for b in [*bars,*observations]}.values(),key=lambda b:(b.timestamp,b.symbol))
        for _, batch in groupby(bars, key=lambda bar: bar.timestamp):
            for event in drive_timestamp_batch(broker, tuple(batch), adapters):
                reasons[event.reason] += 1
                if decision_stream is not None:
                    decision_stream.write(json.dumps(event.to_json_dict(), default=str) + "\n")
        daily.append({"day": str(day), "equity": broker.equity_curve[-1], "cash": broker.cash,
            "positions": {str(key): pos.qty for key, pos in broker.positions.items()}})
        if index % 40 == 0:
            print(json.dumps({"recipe": args.recipe, "scenario": args.scenario, "session": index+1, "equity": broker.equity_curve[-1]}), flush=True)
    connection.close()
    if action_events:
        if len(action_checks) != sum(str(min(ks)) <= e["ex_date"] <= str(max(ks)) for e in action_events):
            raise ValueError("Incomplete corporate-action account exposure audit")
        write_json(args.output / "corporate_action_audit.json", {"pass": True, "events": action_checks,
            "scope": "No entitlement or pending-order adjustment is required in this replay"})
    if capacity_history_receipt is not None:
        write_json(args.output/'capacity_history_inputs.json',capacity_history_receipt)
    if decision_stream is not None:
        decision_stream.close()
    for adapter in adapters:
        adapter._sync_new_fills(broker)
    trades = _collapse_exit_legs(broker.trades)
    metrics = compute_trade_metrics(trades, broker.equity_curve, timestamps=broker.timestamps,
        mark_coverage=broker.mark_coverage, initial_equity=study["initial_equity"])
    identity = {"study_hash": digest(args.study), "recipe": recipe, "scenario": args.scenario,
        "absolute_caps": args.absolute_caps, "no_upward_compounding": args.no_upward_compounding,
        "standalone": args.standalone, "configuration": dict(study["configuration"])}
    if args.standalone:
        identity["configuration"]["source_dependency_scopes"] = {
            args.standalone: study["configuration"]["source_dependency_scopes"][args.standalone]}
    if args.full_audit:
        identity["input_manifest"] = generation.receipt()
    print(json.dumps({"audit_started": args.full_audit}), flush=True)
    audit = audit_broker(broker, identity=identity)
    allocation_audit = None
    if allocation is not None:
        from backtests.strategies.portfolio_synergy.intent_replay import audit_intent_allocations
        allocation_audit = audit_intent_allocations(broker)
        if not allocation_audit["pass"]:
            audit["execution_validity_pass"] = audit["audit_pass"] = False
            audit["audit_status"] = "failed"
    if native_budget is not None:
        allocation_audit = audit['native_budget_audit']
        with gzip.open(args.output / 'native_budget_admissions.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for row in broker.journal:
                if row['event'] == 'native_budget_admission':
                    stream.write(json.dumps(row, default=str) + '\n')
        write_json(args.output / 'fills.json', [asdict(f) for f in broker.fills])
    if allocation_audit is not None:
        write_json(args.output / "allocation_audit.json", allocation_audit)
        with gzip.open(args.output / "allocation_events.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in broker.allocation_events:
                stream.write(json.dumps(row, default=str) + "\n")
    admissions = [row for row in broker.journal if row["event"] == "portfolio_admission"]
    budget_admissions = [row for row in broker.journal if row['event'] == 'native_budget_admission']
    result = dict(recipe=args.recipe, scenario=args.scenario, return_pct=100*(broker.equity_curve[-1]/study["initial_equity"]-1),
        standalone=args.standalone, canonical_costs=args.canonical_costs,
        drawdown_pct=100*metrics["max_drawdown_pct"], trades=len(trades),
        active_days=len({t.entry_fill_time.date() for t in trades}), sessions=len(ks),
        trades_per_21_sessions=len(trades)*21/len(ks),
        strategy_counts=dict(Counter(t.strategy_id for t in trades)),
        terminal_inventory={str(key): pos.qty for key, pos in broker.positions.items()}, terminal_pending=len(broker.orders),
        execution_validity_pass=audit["execution_validity_pass"], audit_status=audit["audit_status"],
        intraday_residual_sessions=audit["intraday_residual_session_count"],
        source_generation_hash=study["source_generation_hash"], metrics=metrics, decision_counts=reasons,
        allocation_audit=allocation_audit,
        policy_reasons=dict(Counter(row["reason"] for row in admissions)),
        unique_blocked_orders=len({r["order"]["order_id"] for r in admissions if r["admitted_qty"] == 0}),
        unique_resized_orders=len({r["order"]["order_id"] for r in admissions if 0 < r["admitted_qty"] < r["offered_qty"]}),
        ranked_order_count=len({r["order"]["order_id"] for r in admissions if r["quality_prediction"] is not None}))
    if native_budget is not None:
        result['native_budget_blocked_orders'] = len({r['order']['order_id'] for r in budget_admissions if r['offered_qty'] > 0 and r['approved_qty'] == 0})
        result['native_budget_resized_orders'] = len({r['order']['order_id'] for r in budget_admissions if 0 < r['approved_qty'] < r['offered_qty']})
        result['native_budget_binding_reasons'] = dict(Counter(reason for r in budget_admissions
            if r['approved_qty'] < r['offered_qty'] for reason in r['reservation']['binding_constraints']))
    if recipe.get('olr_capacity'):
        result['terminal_deferred_intents']=sum(len(a.state.meta.get('capacity_deferred_intents',{})) for a in adapters if a.strategy_id=='OLR')
        result['terminal_rotation_intents']=sum(len(a.state.meta.get('capacity_rotation_pending',{})) for a in adapters if a.strategy_id=='OLR')
        if result['terminal_rotation_intents']:
            raise ValueError('Unresolved native capital rotation at the declared window end')
        if result['terminal_deferred_intents']:
            raise ValueError('Unresolved native capacity intents at the declared window end')
    for name, content in (("result", result), ("execution_audit", audit), ("daily", daily),
                          ("trades", [t.to_json_dict() for t in trades]), ("equity", equity_records(broker.timestamps, broker.equity_curve))):
        write_json(args.output / (name + ".json"), content)
    with gzip.open(args.output / "admissions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in admissions:
            stream.write(json.dumps(row, default=str) + "\n")
    if source_generation_hash() != study["source_generation_hash"]:
        raise ValueError("Source changed during replay")
    write_json(args.output / "completion.json", {"complete": True, "source_generation_hash": study["source_generation_hash"],
        "result_sha256": digest(args.output / "result.json"), "audit_sha256": digest(args.output / "execution_audit.json"),
        'artifacts': {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    print(json.dumps({k:v for k,v in result.items() if k not in {"metrics", "decision_counts"}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("kalcb-manifest", "kalcb-plan", "kalcb-policies", "kalcb-trades", "olr-dataset", "olr-snapshots", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--equity", type=float, default=100_000_000)
    run = sub.add_parser("replay")
    run.add_argument("--study", type=Path, required=True)
    run.add_argument("--recipe", required=True)
    run.add_argument("--scenario", choices=SCENARIOS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--full-audit", action="store_true")
    run.add_argument("--absolute-caps", action="store_true")
    run.add_argument("--standalone", choices=("KALCB", "OLR"))
    run.add_argument("--canonical-costs", action="store_true")
    run.add_argument("--no-upward-compounding", action="store_true")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else replay)(args)


if __name__ == "__main__":
    main()
