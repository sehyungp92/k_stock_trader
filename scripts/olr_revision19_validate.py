"""Native performance proof, including full-audit negative controls."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.olr_native_acceleration import read, write, sha, prepare_replay

ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10')
V = ROOT/'revisions/revision_19_native_overheads'


def validate_case(window, candidate, label):
    registration = ROOT/'registrations'/window/(candidate+'_100000000')/'study.json'
    generation = Path('C:/Users/sehyu/.codex/research/olr_native_r19')/window/'inputs/generation.json'
    destination = V/'native_proofs'/label
    target = destination/'study.json'
    if (destination/'replay').exists():
        raise ValueError('Preserve previous validation output')
    started = time.perf_counter()
    prepare_replay(registration, generation, target)
    prep_seconds = time.perf_counter()-started
    study = read(target)
    started = time.perf_counter()
    with (destination/'native.log').open('w', encoding='utf-8') as log:
        code = subprocess.call([sys.executable,'-B','-u',str(Path(study['code'])/'scripts/portfolio_synergy_study.py'),
            'replay','--study',str(target),'--recipe','candidate','--scenario','baseline','--standalone','OLR',
            '--output',str(destination/'replay'),'--full-audit'],cwd=study['code'],stdout=log,stderr=subprocess.STDOUT)
    elapsed = time.perf_counter()-started
    assert code == 0, (code, str(destination/'native.log'))
    out = destination/'replay'
    old = ROOT/'cases'/candidate/window/'100000000_baseline'
    result, previous = read(out/'result.json'), read(old/'result.json')
    audit, before = read(out/'execution_audit.json'), read(old/'execution_audit.json')
    completion = read(out/'completion.json')
    assert sha(out/'result.json') == completion['result_sha256']
    assert sha(out/'execution_audit.json') == completion['audit_sha256']
    artifacts = {n:sha(out/n) for n in ('result.json','execution_audit.json','daily.json','trades.json','completion.json')}
    equality = {n: read(out/n)==read(old/n) for n in ('daily.json','trades.json')}
    metric_keys = ('return_pct','trades','drawdown_pct','win_rate','sessions','terminal_inventory','terminal_pending',
                   'execution_validity_pass','audit_status')
    equality.update({k:result[k]==previous[k] for k in metric_keys if k in result})
    equality['audit_pass'] = audit['audit_pass']==before['audit_pass']
    for k in ('coverage','errors','daily_equity_coverage'):
        equality['audit_receipt_'+k] = audit['audit_receipt'].get(k)==before['audit_receipt'].get(k)
    equality['capacity_model_audit'] = audit.get('capacity_model_audit')==before.get('capacity_model_audit')
    # Audit signatures and provenance identify the new manifest/code, so they
    # must differ. Actual input events, economic paths and audit decisions must not.
    a, b = audit['audit_receipt'], before['audit_receipt']
    for k in a:
        if k.startswith('source_event'):
            equality[k] = a[k]==b.get(k)
    proof = {'window':window,'candidate':candidate,'label':label,'passed':all(equality.values()),
             'equivalence':equality,'preparation_seconds':prep_seconds,'native_replay_seconds':elapsed,
             'native_audit_status':result['audit_status'],'native_audit_pass':audit['audit_pass'],
             'artifact_sha256':artifacts,'original_result_sha256':sha(old/'result.json'),
             'source_generation_hash':study['source_generation_hash'],
             'original_source_generation_hash':previous['source_generation_hash'],
             'full_audit':True,'strategy_trials_added':0}
    write(destination/'comparison.json', proof)
    print(proof, flush=True)
    assert proof['passed'], 'Native parity failed; preserve all evidence'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('window');p.add_argument('candidate');p.add_argument('label')
    a=p.parse_args();validate_case(a.window,a.candidate,a.label)
