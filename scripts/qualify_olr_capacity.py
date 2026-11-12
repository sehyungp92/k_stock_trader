"""Select the audited capacity finalist through the shared PhaseRunner."""
import argparse
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--screening', type=Path, required=True)
    args = parser.parse_args()
    # Qualification code is separately frozen. Run its evaluator against the
    # original replay packages so source/contract validation remains exact,
    # without rerunning economic paths for a receipt-adapter-only correction.
    repo = Path(__file__).resolve().parents[1]
    study = json.loads((args.root/'study_100000000.json').read_text(encoding='utf-8'))
    folder = args.root/'qualification_code'
    folder.mkdir(exist_ok=False)
    sources = [repo/'backtests/strategies/olr/capacity_qualification.py', Path(__file__)]
    hashes = {}
    for source in sources:
        target = folder/source.name
        shutil.copy2(source, target)
        hashes[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps({'files':hashes,
        'evaluated_source_generation_hash':study['source_generation_hash'],
        'reason':'Native audit identity adapter; evaluated trading source and economics unchanged'},indent=2),encoding='utf-8')
    program = '''import importlib.util, sys
from pathlib import Path
from backtests.auto.shared.phase_runner import PhaseRunner
spec=importlib.util.spec_from_file_location('capacity_qualification_adapter',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path(sys.argv[2]);plugin=module.OLRCapacityQualificationPlugin(root,Path(sys.argv[3]))
PhaseRunner(plugin,root/'phased_qualified',round_name='OLR high-capital qualification',
    max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
'''
    subprocess.run([sys.executable,'-B','-u','-c',program,str(folder/'capacity_qualification.py'),
        str(args.root.resolve()),str(args.screening.resolve())],cwd=study['code'],check=True)
