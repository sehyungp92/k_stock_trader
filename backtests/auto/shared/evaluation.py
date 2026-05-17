from __future__ import annotations

from .phase_state import PhaseState
from .types import EndOfRoundArtifacts

EVALUATION_DIMENSIONS = (
    "signal_extraction",
    "signal_discrimination",
    "entry_mechanism",
    "trade_management",
    "exit_mechanism",
)


def build_end_of_round_report(strategy_name: str, state: PhaseState, artifacts: EndOfRoundArtifacts) -> str:
    lines = [
        "=" * 70,
        f"{strategy_name.upper()} END-OF-ROUND EVALUATION",
        "=" * 70,
        "",
        "Phase Progression Summary:",
    ]
    for phase in sorted(state.phase_results):
        result = state.phase_results[phase]
        lines.append(
            f"  Phase {phase}: score {result.get('base_score', 0.0):.4f} -> "
            f"{result.get('final_score', 0.0):.4f} ({len(result.get('kept_features', []))} accepted)"
        )
    lines.extend(["", "Cumulative Mutations Applied:"])
    if state.cumulative_mutations:
        lines.extend(f"  {key}: {value}" for key, value in sorted(state.cumulative_mutations.items()))
    else:
        lines.append("  (none)")
    for dimension in EVALUATION_DIMENSIONS:
        lines.extend(["", dimension.replace("_", " ").title(), artifacts.dimension_reports.get(dimension, "No report provided.")])
    for name, body in artifacts.extra_sections.items():
        if name not in EVALUATION_DIMENSIONS:
            lines.extend(["", name.replace("_", " ").title(), body])
    lines.extend(["", "Overall Verdict", artifacts.overall_verdict or "Round incomplete."])
    return "\n".join(lines)

