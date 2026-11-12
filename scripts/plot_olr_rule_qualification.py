"""Render accepted, registered OLR sensitivity evidence without selecting rules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render(root):
    accepted = read(root / "final_identity_verification.json")
    if accepted.get("status") != "passed" or accepted.get("scope") != "complete_study":
        raise ValueError("Complete independent acceptance is required before plotting")
    native = read(root / "native_receipt_verification.json")
    if (native.get("status") != "passed"
            or native["structural_acceptance_sha256"] != sha(root / "final_identity_verification.json")
            or len(native["cases"]) != len(accepted["cases"])):
        raise ValueError("Complete native receipt verification is required before plotting")
    manifest = read(root / "study_artifacts.json")
    names = ("qualification_results.json", "qualification_decision.json", "analysis/core_results.json")
    for name in names:
        if sha(root / name) != manifest["files"][name]:
            raise ValueError("Accepted source artifact changed: " + name)
    qualification = read(root / "qualification_results.json")
    control = read(root / "analysis/core_results.json")["cases"]["control"]["nominal"]
    neighbours = qualification["stability"]["control"]
    order = ["weights_x0.5", "weights_x1.5", "low_-25", "low_+25",
             "middle_-25", "middle_+25", "high_-25", "high_+25"]
    if set(neighbours) != set(order):
        raise ValueError("Registered sensitivity set is incomplete")
    labels = ["Two soft weights × 0.5", "Two soft weights × 1.5", "Low-band maximum: 275",
              "Low-band maximum: 325", "Middle interval: 375–475", "Middle interval: 425–525",
              "High-band minimum: 625", "High-band minimum: 675"]
    qualified = {name: m for name, m in neighbours.items() if m.get("audit_pass") is True}
    returns = [100 * m["official_mtm_net_return_pct"] for m in qualified.values()]
    draws = [100 * m["max_drawdown_pct"] for m in qualified.values()]
    summary = {
        "status": "diagnostic_only_no_neighbour_promotion", "qualified_neighbours": len(qualified),
        "registered_neighbours": len(order), "minimum_return_pct": min(returns) if returns else None,
        "median_return_pct": median(returns) if returns else None,
        "maximum_return_pct": max(returns) if returns else None,
        "maximum_drawdown_pct": max(draws) if draws else None,
        "all_neighbours_audit_pass": len(qualified) == len(order),
        "return_fragility_thresholds_satisfied": bool(returns) and min(returns) >= .6 * control["return_pct"]
                                                  and median(returns) >= .9 * control["return_pct"],
        "drawdown_threshold_satisfied": bool(draws) and max(draws) <= 15,
        "scope": "Historical control sensitivity; same registered challenger fragility thresholds shown diagnostically. Independent OOS and paper parity remain pending.",
    }
    summary["local_robustness_checks_pass"] = all(summary[key] for key in (
        "all_neighbours_audit_pass", "return_fragility_thresholds_satisfied", "drawdown_threshold_satisfied"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 7.4), sharey=True,
                                   gridspec_kw={"width_ratios": [2.25, 1]})
    fig.subplots_adjust(left=.245, right=.965, top=.82, bottom=.22, wspace=.23)
    fig.suptitle("OLR: registered local sensitivity", x=.04, ha="left", y=.965, fontsize=21, weight="bold")
    fig.text(.04, .915, "Historical development evidence · Full account replays · No neighbour can replace the frozen finalist",
             fontsize=10, color="#475569")
    colors = ["#2563eb"] * 2 + ["#0f766e"] * 6
    max_return = max([control["return_pct"], *returns])
    for y, name in enumerate(order):
        m = qualified.get(name)
        if m is None:
            left.text(2, y, "Unqualified — economic result excluded", va="center", color="#b91c1c", fontsize=9)
            right.text(.3, y, "Unqualified", va="center", color="#b91c1c", fontsize=9)
            continue
        result, draw = 100 * m["official_mtm_net_return_pct"], 100 * m["max_drawdown_pct"]
        left.barh(y, result, color=colors[y], height=.57)
        direction = 1 if result >= 0 else -1
        left.text(result + direction * .012 * max_return, y, f"{result:.2f}% / {m['total_trades']:.0f} trades",
                  va="center", ha="left" if result >= 0 else "right", fontsize=9,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
        right.barh(y, draw, color="#b91c1c" if draw > 15 else "#94a3b8", height=.57)
        right.text(draw + .2, y, f"{draw:.2f}%", va="center", fontsize=9,
                   bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
    left.set_yticks(range(len(order)), labels)
    left.invert_yaxis()
    left.set_xlim(min([0, *returns]) * 1.5, max_return * 1.47)
    right.set_xlim(0, max([16.8, *draws]) * 1.1)
    left.axvline(control["return_pct"], color="#111827", linestyle="--", linewidth=1.1)
    right.axvline(control["drawdown_pct"], color="#111827", linestyle="--", linewidth=1.1)
    right.axvline(15, color="#b91c1c", linestyle=":", linewidth=1.1)
    left.set_xlabel("Net account return (%)")
    right.set_xlabel("Maximum MTM drawdown (%)")
    for ax in (left, right):
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.text(.245, .14, f"Dashed lines: control {control['return_pct']:.2f}% return, {control['drawdown_pct']:.2f}% drawdown, {control['closed_trades']} trades.", fontsize=10)
    fig.text(.245, .10, "Red dotted line: 15% drawdown threshold. Local return check: median ≥90% and minimum ≥60% of parent return.", fontsize=9)
    fig.text(.245, .06, "This sensitivity test does not correct historical rule-selection bias or establish independent forward alpha.", fontsize=9, color="#475569")
    output = root / "delivery"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "OLR_local_sensitivity.png", dpi=160, facecolor="white")
    fig.savefig(output / "OLR_local_sensitivity.svg", facecolor="white")
    plt.close(fig)
    (output / "sensitivity_review.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sources = {name: sha(root / name) for name in (*names, "final_identity_verification.json",
                                                 "native_receipt_verification.json", "study_artifacts.json")}
    sources["plotter"] = sha(Path(__file__))
    outputs = {p.name: sha(p) for p in output.iterdir() if p.name in {
        "OLR_local_sensitivity.png", "OLR_local_sensitivity.svg", "sensitivity_review.json"}}
    (output / "plot_provenance.json").write_text(json.dumps({"source_sha256": sources, "outputs_sha256": outputs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    render(parser.parse_args().study_root)
