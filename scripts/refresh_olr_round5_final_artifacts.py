"""Compatibility entrypoint; use refresh_olr_final_artifacts.py --round N.

Historical commands retain their round-5 default. All implementation lives in
the round-neutral entrypoint; this file performs no reporting or replay itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.refresh_olr_final_artifacts import main as refresh_main


def main(argv=None):
    return refresh_main(argv, default_round=5)


if __name__ == "__main__":
    main()
