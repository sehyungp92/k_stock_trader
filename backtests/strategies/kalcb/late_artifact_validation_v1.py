"""Versioned first-hour receipt adapter; original independent accounting retained.

The account verifier body is copied exactly from qualified_artifacts.validate_case
apart from replacing its old fixed scope assertion with strict policy-bound scope
validation. Neither source receipts nor failed evidence are rewritten.
"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.qualified_artifacts import read, digest, require, close, SOURCE_CHECKS
from backtests.strategies.kalcb.retention_research_io_v1 import PERIODS, POLICY, write, verify

OLD_SCOPE = {"version":"causal-timed-candidates-v1","strategy":"kalcb",
             "feature_intraday_before":"09:45","candidate_intraday_before":"15:20"}
LATE_SCOPE = {**OLD_SCOPE,"version":"causal-timed-candidates60-v2"}


def validate_scope(receipt, profile):
    policy_path=Path(profile['policy']['directory'])/'policies_manifest.json'
    require(digest(policy_path)==profile['policy']['manifest_sha256'], 'Changed policy manifest')
    policy=read(policy_path)
    scope=receipt['identity']['configuration']['source_dependency_scope']
    minutes=policy.get('observation_minutes')
    require(minutes in (45,60), 'Unsupported observation horizon')
    require(scope==(LATE_SCOPE if minutes==60 else OLD_SCOPE), 'Wrong or shortened policy-bound source scope')
    if minutes==60:
        require(policy.get('version')=='kalcb-cumulative-first-hour-v1', 'Wrong first-hour policy contract')
        proof=receipt['qualification'].get('first_hour_population',{})
        require(proof.get('sector_membership_verified') is True, 'Missing first-hour population proof')
        considered=proof.get('considered'); complete=proof.get('complete_consumed_prefixes'); excluded=proof.get('incomplete_excluded')
        require(type(considered) is int and type(complete) is int and isinstance(excluded,list)
                and 0 < complete <= considered and considered==complete+len(excluded), 'Incomplete first-hour coverage')
        require(len({tuple(x) for x in excluded})==len(excluded), 'Duplicate excluded first-hour prefix')
        manifest=read(profile['audit_manifest'])
        binding=manifest['evidence'].get('first_hour_feature_sources',{})
        require(binding.get('version')=='complete-first-hour-population-v1'
                and len(binding.get('original_populations',[]))==2
                and binding.get('forward_from')=='2026-04-01', 'Unbound continuous first-hour population')
        code=Path(profile['audit_manifest']).parent/'files'
        for rel in [binding['features'],*binding['original_populations']]:
            require(rel in manifest['files'] and digest(code/rel)==manifest['files'][rel], 'Changed first-hour population file')
    return scope


def validate_case(directory, profile, scenario):
    p = Path(directory)
    result, audit, done, request = (read(p / name) for name in
                                   ("result.json", "execution_audit.json", "completion.json", "request.json"))
    require(done.get("complete") is True, "Incomplete native replay")
    require(result.get("audit_status") == audit.get("audit_status") == "passed" and
            result.get("execution_validity_pass") is True and audit.get("audit_pass") is True,
            "A failed or unverified source replay cannot be promoted")
    source = profile["execution_source"]["source_generation_hash"]
    require(all(x.get("source_generation_hash") == source for x in (result, done, request)), "Stale replay generation")
    receipt = dict(audit["audit_receipt"])
    receipt_hash = receipt.pop("receipt_hash", None)
    require(receipt_hash == stable_signature(receipt), "Changed execution audit receipt")
    require(receipt["source_generation_hash"] == source and not receipt["errors"], "Invalid execution receipt")
    require(all(receipt["coverage"].get(k) == "pass" for k in SOURCE_CHECKS), "Incomplete source qualification")
    require(not result["residual_sessions"] and not result["terminal_inventory"] and not result["terminal_pending"], "Unresolved native account")
    require(result["scenario"] == scenario and result["policy"] == profile["policy"]["name"], "Wrong scenario or selector")
    require(receipt["identity"]["candidate_policy_manifest_sha256"] == profile["policy"]["manifest_sha256"], "Wrong candidate evidence")
    validate_scope(receipt, profile)
    require(receipt["identity"]["input_manifest"]["manifest_hash"] == read(profile["audit_manifest"])["manifest_hash"], "Wrong immutable source bundle")
    trades, fills, daily, equity = (read(p / name) for name in ("trades.json", "fills.json", "daily.json", "equity.json"))
    initial = float(profile["initial_equity_krw"])
    cash = initial
    positions = defaultdict(int)
    ids = set()
    costs = receipt["qualification"]["capacity"]["policy"]["default_costs"]
    last = None
    for fill in fills:
        q, price, meta = int(fill["qty"]), float(fill["price"]), fill["metadata"]
        at = datetime.fromisoformat(fill["timestamp"])
        require(at.tzinfo is not None and (last is None or at >= last), "Nonchronological executions")
        last = at
        eid = meta["execution_id"]
        require(eid not in ids and q > 0 and price > 0, "Invalid or duplicate execution")
        ids.add(eid)
        buy = fill["side"] == "BUY"
        require(fill["side"] in ("BUY", "SELL"), "Unsupported execution side")
        fee = q * price * (costs["commission_bps"] + (0 if buy else costs["tax_bps_on_sell"])) / 10000
        close(meta["fee"], fee, "Incorrect per-fill costs")
        slip = costs["auction_slippage_bps"] if meta["execution_phase"] == "auction" else costs["slippage_bps"]
        close(price, meta["base_price"] * (1 + (1 if buy else -1) * slip / 10000), "Non-adverse slippage")
        bar = meta["execution_bar"]
        require(bar["low"] - 1e-7 <= meta["base_price"] <= bar["high"] + 1e-7 and bar["volume"] > 0, "Unattainable source price")
        available = meta.get("decision_available_at")
        if available:
            require(at >= datetime.fromisoformat(available), "Execution before decision availability")
        cash += (-1 if buy else 1) * q * price - fee
        positions[fill["symbol"]] += q if buy else -q
        require(positions[fill["symbol"]] >= 0, "Unfunded short or duplicate exit")
    require(all(q == 0 for q in positions.values()), "Terminal position mismatch")
    close(cash, equity[-1][1], "Independent cash reconciliation failed")
    close(sum(t["net_pnl"] for t in trades), cash - initial, "Trades do not reconcile to cash")
    require(len(trades) == result["trades"], "Trade denominator mismatch")
    gains = sum(max(t["net_pnl"], 0) for t in trades)
    losses = -sum(min(t["net_pnl"], 0) for t in trades)
    close(result["metrics"]["profit_factor"], gains / losses, "Gross or R-based headline profit factor")
    peak, dd = initial, 0.0
    for stamp, value in equity:
        require(profile["date_range"]["start"] <= stamp[:10] <= profile["date_range"]["end"], "Out-of-scope marks")
        peak = max(peak, value)
        dd = max(dd, 100 * (1 - value / peak))
    close(result["drawdown_pct"], dd, "Realized-only or stale drawdown")
    close(result["return_pct"], 100 * (cash / initial - 1), "Inflated account return")
    require(len(daily) == result["sessions"] == profile["date_range"]["sessions"], "Omitted cash sessions")
    require(all(not d["inventory"] and not d["pending"] for d in daily), "Residual intraday inventory")
    for key in ("return_pct", "drawdown_pct", "trades", "active_days"):
        close(result[key], profile["metrics"][scenario][key], "Stale selected metrics: " + key)
    return {"result": result, "audit": audit, "trades": trades, "fills": fills, "daily": daily, "equity": equity,
            "validation": {"passed": True, "independent_cash_krw": cash, "fill_count": len(fills),
                           "net_profit_factor": gains / losses, "mtm_drawdown_pct": dd}}


def validate_native(folder,item,scenario,save=True,receipt_path=None):
    folder=Path(folder); result=read(folder/'result.json'); request=read(folder/'request.json')
    verify([item[k] for k in ('manifest','plan','policy')])
    manifest=read(item['manifest']['path']); signature=manifest.pop('manifest_hash')
    require(stable_signature(manifest)==signature,'Changed source manifest payload')
    require(manifest['source_generation_hash']==item['source_generation_hash'],'Wrong frozen source generation')
    require(request['plan']==read(item['plan']['path']),'Replay plan mismatch')
    require(Path(request['arguments']['audit_manifest']).resolve()==Path(item['manifest']['path']).resolve(),'Source manifest mismatch')
    policy=Path(request['arguments']['policies'])
    require(digest(policy/'policies_manifest.json')==item['policy']['sha256'],'Policy identity mismatch')
    pm=read(policy/'policies_manifest.json')
    for rel,sha in pm['files'].items():
        require(Path(rel).name==rel and digest(policy/rel)==sha,'Changed policy member')
    start,end,sessions=PERIODS[item['period']]
    profile={'execution_source':{'source_generation_hash':item['source_generation_hash']},
        'audit_manifest':item['manifest']['path'],'policy':{'name':POLICY,'manifest_sha256':item['policy']['sha256'],'directory':str(policy)},
        'initial_equity_krw':100000000,'date_range':{'start':start,'end':end,'sessions':sessions},'metrics':{scenario:result}}
    checked=validate_case(folder,profile,scenario)
    require(checked['daily'][0]['day']==start and checked['daily'][-1]['day']==end,'Calendar truncated')
    if save:write(receipt_path or folder/'independent_account_verification.json',checked['validation'])
    return {k:result[k] for k in ('return_pct','drawdown_pct','trades','active_days','audit_status')}
