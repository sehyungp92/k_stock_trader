"""Read-only attribution of the completed portfolio challenge; no policy search.

Reconstructs marked sleeve PNL at the actual bar drawdown endpoints and splits
matched-trade PNL changes into symmetric quantity/per-share accounting terms.
These are descriptive identities, never causal counterfactuals or EV labels.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

REPO = Path(__file__).resolve().parents[1]
INITIAL = 25_000_000.0
SLEEVES = ("KALCB", "OLR")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


@lru_cache(maxsize=None)
def load_case(path):
    path = Path(path)
    completion = read(path / "completion.json")
    assert completion["complete"]
    out = {"path": path, "completion": binding(path / "completion.json")}
    for filename in ("trades.json", "daily.json", "equity.json", "fills.json", "request.json"):
        assert digest(path / filename) == completion["artifacts"][filename], filename
        out[filename[:-5]] = read(path / filename)
    assert all(f["timestamp"] <= g["timestamp"] for f, g in zip(out["fills"], out["fills"][1:]))
    assert abs(sum(t["net_pnl"] for t in out["trades"]) - (out["daily"][-1]["equity"] - INITIAL)) < .01
    assert not out["daily"][-1]["positions"]
    return out


VERIFIED_DATABASES = set()


def endpoint_pnl(case, timestamp, expected_nav):
    """Cash flows plus last frozen close at an observed post-batch mark."""
    study_path = Path(case["request"]["arguments"]["study"])
    assert digest(study_path) == case["request"]["study_hash"]
    study = read(study_path)
    db = Path(study["code"]) / study["database"]
    stat = db.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, study["database_hash"])
    if identity not in VERIFIED_DATABASES:
        assert digest(db) == study["database_hash"]
        VERIFIED_DATABASES.add(identity)
    cash, holdings = defaultdict(float), defaultdict(int)
    for fill in case["fills"]:
        if fill["timestamp"] > timestamp:
            break
        sid, sym = fill["strategy_id"], fill["symbol"]
        sign = 1 if fill["side"] == "BUY" else -1
        cash[sid] -= sign * fill["qty"] * fill["price"] + fill["metadata"]["fee"]
        holdings[sid, sym] += sign * fill["qty"]
        assert holdings[sid, sym] >= 0
    value = dict(cash)
    held = []
    con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    try:
        for (sid, sym), qty in sorted(holdings.items()):
            if not qty:
                continue
            row = con.execute("SELECT ts,c FROM bars WHERE symbol=? AND ts<=? AND c>0 ORDER BY ts DESC LIMIT 1",
                              (sym, timestamp.replace(" ", "T"))).fetchone()
            assert row is not None
            value[sid] += qty * row[1]
            held.append({"strategy": sid, "symbol": sym, "qty": qty, "mark": row[1], "mark_time": row[0]})
    finally:
        con.close()
    error = INITIAL + sum(value.values()) - expected_nav
    assert abs(error) < .01, (case["path"], timestamp, error)
    return {"timestamp": timestamp, "nav": expected_nav, "cumulative_pnl": value,
            "holdings": held, "reconciliation_error_krw": error, "database_sha256": study["database_hash"]}


def path_diagnosis(case):
    equity = case["equity"]
    nav = np.asarray([v for _, v in equity])
    dd = 1 - nav / np.maximum.accumulate(nav)
    trough = int(np.argmax(dd))
    peak = int(np.argmax(nav[:trough + 1]))
    endpoints = [endpoint_pnl(case, equity[i][0], float(nav[i])) for i in (peak, trough)]
    contributions = {sid: endpoints[1]["cumulative_pnl"].get(sid, 0) - endpoints[0]["cumulative_pnl"].get(sid, 0)
                     for sid in SLEEVES}
    assert abs(sum(contributions.values()) - (nav[trough] - nav[peak])) < .01
    daily = case["daily"]
    assert all(not qty or ast.literal_eval(key)[0] != "KALCB" for d in daily for key, qty in d["positions"].items())
    kd = defaultdict(float)
    for trade in case["trades"]:
        if trade["strategy_id"] == "KALCB":
            assert trade["entry_fill_time"][:10] == trade["exit_fill_time"][:10]
            kd[trade["exit_fill_time"][:10]] += trade["net_pnl"]
    daily_nav = np.asarray([INITIAL] + [d["equity"] for d in daily])
    k = np.asarray([kd[d["day"]] for d in daily]) / daily_nav[:-1]
    ret = daily_nav[1:] / daily_nav[:-1] - 1
    o = ret - k
    tail = ret <= np.quantile(ret, .1)
    fees_by_sleeve = {sid: sum(f["metadata"]["fee"] for f in case["fills"]
                             if equity[peak][0] < f["timestamp"] <= equity[trough][0] and f["strategy_id"] == sid)
                      for sid in SLEEVES}
    phases = {}
    for label, start, end in (("before_2026", "2025-08-07", "2025-12-31"),
                              ("january_february", "2026-01-01", "2026-02-28"),
                              ("march", "2026-03-01", "2026-03-31")):
        ix = [i for i, d in enumerate(daily) if start <= d["day"] <= end]
        k_pnl = sum(kd[daily[i]["day"]] for i in ix)
        account_pnl = daily_nav[ix[-1] + 1] - daily_nav[ix[0]]
        phases[label] = {"account_pnl_krw": float(account_pnl), "kalcb_pnl_krw": k_pnl,
                         "olr_marked_pnl_krw": float(account_pnl - k_pnl),
                         "net_return_pct": float(account_pnl / daily_nav[ix[0]] * 100)}
    return {"case_completion": case["completion"], "max_bar_drawdown_pct": float(dd[trough] * 100),
            "endpoints": endpoints, "drawdown_pnl_by_sleeve_krw": contributions,
            "drawdown_contribution_pct_of_peak_nav": {sid: v / nav[peak] * 100 for sid, v in contributions.items()},
            "explicit_fees_paid_during_drawdown_krw": sum(fees_by_sleeve.values()),
            "explicit_drawdown_fees_by_sleeve_krw": fees_by_sleeve,
            "calendar_phase_attribution": phases,
            "daily_correlation": float(np.corrcoef(k, o)[0, 1]),
            "daily_worst_decile": {"sessions": int(sum(tail)), "both_negative": int(sum(tail & (k < 0) & (o < 0))),
                                   "mean_return_pct": float(np.mean(ret[tail]) * 100),
                                   "mean_kalcb_contribution_pp": float(np.mean(k[tail]) * 100),
                                   "mean_olr_contribution_pp": float(np.mean(o[tail]) * 100)},
            "scope": "Exact net marked PNL accounting at bar drawdown endpoints. Fees are included in contributions, not an additional loss. Correlation and tails are endogenous realized paths, not causal sleeve-isolation experiments."}


def trade_key(trade):
    return trade["strategy_id"], trade["symbol"], trade["entry_decision_time"]


def matched_decomposition(candidate, reference):
    output = {}
    for sid in SLEEVES:
        c = {trade_key(t): t for t in candidate["trades"] if t["strategy_id"] == sid}
        r = {trade_key(t): t for t in reference["trades"] if t["strategy_id"] == sid}
        assert len(c) == sum(t["strategy_id"] == sid for t in candidate["trades"])
        assert len(r) == sum(t["strategy_id"] == sid for t in reference["trades"])
        matched = c.keys() & r.keys()
        detail = []
        for key in sorted(matched):
            q0, q1 = r[key]["qty"], c[key]["qty"]
            u0, u1 = r[key]["net_pnl"] / q0, c[key]["net_pnl"] / q1
            quantity = (q1 - q0) * (u1 + u0) / 2
            unit_pnl = (u1 - u0) * (q1 + q0) / 2
            delta = c[key]["net_pnl"] - r[key]["net_pnl"]
            assert abs(quantity + unit_pnl - delta) < .01
            detail.append({"symbol": key[1], "entry_decision_time": key[2], "reference_qty": q0, "candidate_qty": q1,
                           "reference_pnl": r[key]["net_pnl"], "candidate_pnl": c[key]["net_pnl"],
                           "quantity_component": quantity, "per_share_pnl_component": unit_pnl, "pnl_delta": delta,
                           "same_first_entry_fill_time": c[key]["entry_fill_time"] == r[key]["entry_fill_time"],
                           "same_final_exit_time": c[key]["exit_fill_time"] == r[key]["exit_fill_time"]})
        new = [{"identity": key, "net_pnl": c[key]["net_pnl"]} for key in sorted(c.keys() - r.keys())]
        missing = [{"identity": key, "net_pnl": r[key]["net_pnl"]} for key in sorted(r.keys() - c.keys())]
        total = sum(t["net_pnl"] for t in c.values()) - sum(t["net_pnl"] for t in r.values())
        quantity = sum(t["quantity_component"] for t in detail)
        unit = sum(t["per_share_pnl_component"] for t in detail)
        entry_set = sum(t["net_pnl"] for t in new) - sum(t["net_pnl"] for t in missing)
        assert abs(total - quantity - unit - entry_set) < .01
        output[sid] = {"matched": len(matched), "pnl_change": total, "quantity_component": quantity,
                       "per_share_pnl_component": unit, "entry_set_component": entry_set,
                       "new": new, "missing": missing, "matched_detail": detail}
    output["scope"] = "Symmetric exact identity: delta(q*u)=delta(q)*mean(u)+delta(u)*mean(q). Per-share PNL contains price, fill mix, exits and costs. Quantity contains integer sizing, wealth feedback and caps. Neither component estimates a causal isolated effect; new/missing PNL is hindsight."
    return output


def run(report_path, output):
    report = read(report_path)
    canonical = REPO / "config/research/portfolio_synergy_selected.json"
    before = binding(canonical)
    assert before == report["canonical_pointer_at_report"]
    candidates = {**report["initial_challenge"]["candidates"], **report["selection"]["candidates"]}
    aggregates = {}
    for name, cases in candidates.items():
        if len(cases) != 8:
            continue
        aggregates[name] = {"baseline_return_pct": cases["baseline"]["return_pct"],
                            "combined_return_pct": cases["combined"]["return_pct"],
                            "minimum_adverse_return_pct": min(v["return_pct"] for s, v in cases.items() if s != "baseline"),
                            "worst_drawdown_pct": max(v["drawdown_pct"] for v in cases.values()),
                            "max_mean_overnight_pct": max(v["overnight_mean"] * 100 for v in cases.values()),
                            "max_overnight_pct": max(v["overnight_max"] * 100 for v in cases.values()),
                            "minimum_entries": min(v["trades"] for v in cases.values()),
                            "failures": {s: [k for k, passed in v["gate"]["tests"].items() if not passed] for s, v in cases.items() if not v["gate"]["pass"]}}
    paths = {}
    for name in ("k075_o075", "k75_o62.5", "k81.25_o62.5", "k87.5_o62.5"):
        for scenario in ("baseline", "combined", "auction_price_combined", "auction_nonfill_combined"):
            case = load_case(candidates[name][scenario]["case"])
            paths[name + "_" + scenario] = path_diagnosis(case)
    decompositions = {}
    for c, r, scenario in (("k75_o62.5", "k075_o075", "baseline"), ("k75_o62.5", "k075_o075", "combined"),
                           ("k81.25_o50", "k75_o50", "combined")):
        decompositions[c + "_minus_" + r + "_" + scenario] = matched_decomposition(
            load_case(candidates[c][scenario]["case"]), load_case(candidates[r][scenario]["case"]))
    costs = {}
    for name in ("k075_o075", "k75_o62.5"):
        case = load_case(candidates[name]["baseline"]["case"])
        buys = sum(f["qty"] * f["price"] for f in case["fills"] if f["side"] == "BUY")
        sells = sum(f["qty"] * f["price"] for f in case["fills"] if f["side"] == "SELL")
        costs[name] = {"baseline_buy_notional": buys, "baseline_sell_notional": sells,
                       "baseline_explicit_fees_and_tax": sum(f["metadata"]["fee"] for f in case["fills"]),
                       "first_order_extra_10bp_each_side_fixed_baseline_fills": .001 * (buys + sells),
                       "full_cost_stress_pnl_difference": (candidates[name]["cost"]["return_pct"] - candidates[name]["baseline"]["return_pct"]) * INITIAL / 100,
                       "scope": "Turnover times 10bp is only a first-order fixed-fill cost estimate; full cost-stress replay also changes fills, sizing, eligibility and compounding. Difference is not pure financing or an independently identified causal residual."}
    selected_case = load_case(candidates["k75_o62.5"]["baseline"]["case"])
    selected_study = read(selected_case["request"]["arguments"]["study"])
    snapshots_path = Path(selected_study["code"]) / selected_study["olr_snapshots"]
    snapshot_months = defaultdict(Counter)
    for snapshot in read(snapshots_path):
        if snapshot["trade_date"] < "2025-08-07":
            continue
        snapshot_months[snapshot["trade_date"][:7]].update({"snapshots": 1,
            "candidates": len(snapshot["candidates"]), "days_with_candidates": int(bool(snapshot["candidates"]))})
    prior_names = ("portfolio_synergy_critical_review_20261009.txt", "olr_capacity_recovery_20261008.txt",
                   "olr_capacity_policy_robustness_review_20261008.txt", "olr_allocation_recovery_20261008.txt",
                   "olr_conditional_recycling_round1_20261008.txt", "kalcb_selected_full_recovery_20261007.txt",
                   "kalcb_post_recovery_next_steps_20261007.txt")
    assert before == binding(canonical)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_report": binding(report_path),
              "generator": binding(Path(__file__)), "canonical_pointer_unchanged": before,
              "starting_capital": INITIAL, "scope": "Read-only post-selection diagnosis; zero new strategy replays, selections or policy changes.",
              "candidate_aggregates": aggregates, "drawdown_attribution": paths, "pnl_decomposition": decompositions,
              "cost_sensitivity": costs, "checks": {"case_files_match_completion_hashes": True,
              "database_hashes_match_frozen_studies": True, "endpoint_and_trade_pnl_reconciliation_tolerance_krw": .01,
              "fills_chronologically_ordered": True, "kalcb_flat_at_every_checked_session_end": True,
              "cases_consumed": load_case.cache_info().currsize},
              "prior_research": [binding(REPO / "docs/research" / name) for name in prior_names],
              "olr_frozen_afternoon_snapshot_availability": {"source": binding(snapshots_path),
                  "by_month": dict(snapshot_months), "scope": "Frozen final afternoon snapshots, not every intraday update. Zero March entries independently verified in saved trades; this is native opportunity availability, not evidence of missing market bars."},
              "primary_method_sources": ["https://web.stanford.edu/~boyd/papers/cvx_portfolio.html",
                                         "https://web.stanford.edu/~boyd/papers/markowitz.html",
                                         "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"]}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(result, output.with_suffix(".png"))
    print(json.dumps({"output": str(output), "checks": result["checks"]}))


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    names = ["k75_o62.5", "k81.25_o62.5", "k87.5_o62.5", "k075_o075"]
    labels = ["Selected\n0.75 / 0.625", "Midpoint\n0.8125 / 0.625", "Higher K\n0.875 / 0.625", "Incumbent\n0.75 / 0.75"]
    k = [-result["drawdown_attribution"][name + "_auction_nonfill_combined"]["drawdown_contribution_pct_of_peak_nav"]["KALCB"] for name in names]
    o = [-result["drawdown_attribution"][name + "_auction_nonfill_combined"]["drawdown_contribution_pct_of_peak_nav"]["OLR"] for name in names]
    x = np.arange(len(names))
    axes[0].bar(x, k, color="#235789", label="KALCB marked loss")
    axes[0].bar(x, o, bottom=k, color="#d49035", label="OLR marked loss")
    for i, (a, b) in enumerate(zip(k, o)):
        axes[0].text(i, a / 2, f"{a:.2f}", color="white", ha="center", va="center")
        axes[0].text(i, a + b / 2, f"{b:.2f}", color="#17212b", ha="center", va="center", fontsize=10)
        axes[0].text(i, a + b + .3, f"{a+b:.2f}%", ha="center", weight="bold")
    axes[0].axhline(12.5, color="#b91c1c", ls="--", lw=1, label="12.5% research DD gate")
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 15.3), ylabel="Loss / account NAV at peak (%)",
                title="KALCB dominates the worst stressed drawdown")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    dec = result["pnl_decomposition"]["k75_o62.5_minus_k075_o075_baseline"]
    metrics = ["quantity_component", "per_share_pnl_component", "entry_set_component", "pnl_change"]
    vals = [sum(dec[s][m] for s in SLEEVES) / 1e6 for m in metrics]
    colors = ["#235789", "#6b7280", "#0d9488", "#b91c1c"]
    axes[1].bar(np.arange(4), vals, color=colors, width=.65)
    for i, v in enumerate(vals):
        axes[1].text(i, v - .13 if v < 0 else v + .1, f"{v:+.3f}", ha="center", va="top" if v < 0 else "bottom", weight="bold")
    axes[1].axhline(0, color="#94a3b8", lw=1)
    axes[1].set(xticks=np.arange(4), xticklabels=["Quantity\ncomponent", "Per-share PNL\ncomponent", "Entry-set\ncomponent", "Total PNL\nchange"],
                ylim=(-4.7, .75), ylabel="Selected minus incumbent net PNL (KRW million)",
                title="Baseline return loss comes from smaller quantities")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Risk repair reduced exposure; return preservation remains unresolved", fontsize=16, y=.98)
    fig.text(.07, .035, "KRW25m; reused development window 7 Aug 2025–31 Mar 2026. Native sizing factors are not fixed portfolio weights.\n"
             "Left: combined costs/participation plus auction nonfills; each path's actual bar peak-to-trough. Right: exact accounting identity, not causal attribution.\n"
             "No new policy replay, selection or independent performance validation was performed.", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .14, 1, .93))
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPO / "docs/research/portfolio_synergy_followup_implementation_20261009.json")
    parser.add_argument("--output", type=Path, default=REPO / "docs/research/portfolio_synergy_risk_return_diagnosis_20261009.json")
    args = parser.parse_args()
    run(args.report, args.output)
