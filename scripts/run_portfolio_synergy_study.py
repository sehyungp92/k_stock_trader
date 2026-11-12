"""Run registered allocation candidates through the shared optimization lifecycle."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.strategies.portfolio_synergy.current_phase import CurrentPortfolioPlugin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--phase-output", type=Path)
    args = parser.parse_args()
    plugin = CurrentPortfolioPlugin(args.study)
    PhaseRunner(plugin, args.phase_output or args.study.parent / "phased", round_name="Current KALCB and OLR shared account",
        max_rounds=1, min_delta=.00001, max_retries=0, max_diagnostic_retries=0).run_all_phases()
