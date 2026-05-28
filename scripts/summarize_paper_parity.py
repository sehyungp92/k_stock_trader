from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.olr_kalcb.replay import summarize_paper_parity


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OLR/KALCB paper-session parity reports.")
    parser.add_argument("--root", default="data/paper_live/olr_kalcb")
    args = parser.parse_args()
    summary = summarize_paper_parity(args.root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
