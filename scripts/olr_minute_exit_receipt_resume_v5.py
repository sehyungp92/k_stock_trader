"""Resume a saved native case with a separately bound receipt-schema reader.

The original frozen harness, qualifier and native outputs remain byte-identical.
The effective qualifier is a saved exact copy with one JSON-path substitution.
Every original harness check remains active, plus the additive repair bindings.
"""
import argparse,importlib.util,sys,traceback
from pathlib import Path
from scripts import olr_minute_exit_clockfix_v5 as c
t,n,s=c.t,c.n,c.s
ORIGINAL_QUALIFIER_SHA='cffbed51147a75dd4bc6625b79b0b5e6c0947da7dce96bf296241c8b6eddb304'
FROZEN_HARNESS_SHA='f101cfeb297995ffebcd03126456bacbddaf641c3f9ee6649056aff3c86cf5ac'
PRESERVATION_SHA='0ba337c7b7860dcdf9c62b7b6729bc39652155b06a5fdf8fdf2728d09757196b'
OLD="clock=native['minute_clock_audit']"
NEW="clock=native['audit_receipt']['minute_clock_audit']"


def corrected_source(original):
    if original.count(OLD)!=1 or NEW in original:
        raise ValueError('Unexpected qualifier schema; refuse nonexact patch')
    return original.replace(OLD,NEW)


def load_module(path):
    spec=importlib.util.spec_from_file_location('scripts.olr_minute_exit_qualify_receipt_v5',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def verify_repair():
    proof=n.read(c.ROOT/'receipt_schema_repair.json')
    assert n.sha(c.OUT/'research_harness.json')==FROZEN_HARNESS_SHA==proof['original_harness_sha256']
    original=t.REPO/'scripts/olr_minute_exit_qualify_v5.py'
    assert n.sha(original)==ORIGINAL_QUALIFIER_SHA==proof['original_qualifier_sha256']
    assert n.sha(c.ROOT/'qualification_failed_attempt_0/preservation.json')==PRESERVATION_SHA==proof['preservation_sha256']
    for name,digest in proof['files'].items():assert n.sha(name)==digest,name
    runtime=Path(proof['runtime_qualifier'])
    assert runtime.read_text(encoding='utf-8')==corrected_source(original.read_text(encoding='utf-8'))
    for name,digest in proof['completed_case_files'].items():assert n.sha(name)==digest,name
    s.forensic.authenticate(Path(proof['completed_case']))
    return proof


def configure():
    c.configure();proof=verify_repair();q=load_module(proof['runtime_qualifier'])
    original_verify=q.verify_harness
    def effective_verify():
        value=original_verify();verify_repair()
        return {**value,'receipt_schema_repair_sha256':n.sha(c.ROOT/'receipt_schema_repair.json')}
    q.verify_harness=effective_verify
    # Report verification must use the same effective audited reader and gates.
    import scripts
    sys.modules['scripts.olr_minute_exit_qualify_v5']=q
    scripts.olr_minute_exit_qualify_v5=q
    return q


def main(phase,chart_reviewed=False):
    q=configure();q.verify_harness();q.check_preflight()
    if phase=='audit_completed':
        candidate=n.read(c.ROOT/'candidate_plan.json')['candidates'][0]
        proof=q.audit(candidate,'train')
        n.write(c.ROOT/'receipt_schema_repair_completed_case_audit.json',dict(utc=s.utc(),
            passed=True,proof=proof,repair_sha256=n.sha(c.ROOT/'receipt_schema_repair.json')))
        print(dict(passed=True,completed_case_reused=True,clock=proof['minute_clock']),flush=True)
    elif phase=='qualify':q.qualify()
    elif phase=='report':
        from scripts import olr_minute_exit_report_v5 as r
        r.main()
    else:
        from scripts import olr_minute_exit_verify_v5 as v
        v.main(chart_reviewed)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('audit_completed','qualify','report','verify'));ap.add_argument('--chart-reviewed',action='store_true');args=ap.parse_args()
    try:main(args.phase,args.chart_reviewed)
    except Exception:
        n.write(c.ROOT/('receipt_'+args.phase+'_failure.json'),dict(utc=s.utc(),traceback=traceback.format_exc()));raise
