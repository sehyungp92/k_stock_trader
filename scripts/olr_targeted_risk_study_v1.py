"""Bounded native exit-priority / scenario-loss / sixth-slot experiment.

Frozen generations preserve the qualified history contract. No live promotion.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import olr_boundary_restart_study_v1 as b
n, s = b.n, b.s
PARENT = b.ROOT
ROOT = PARENT.parent / 'targeted_risk_v1'
ENGINES = Path.home() / '.codex/research/olr_targeted_risk_v1'
OUT = REPO / 'data/backtests/analysis/olr_targeted_risk_20261010'
REFERENCE = 'c480e063db1f3b716'
PATCHED = ('strategy_olr/core/logic.py', 'strategy_olr/core/allocation.py',
           'strategy_olr/core/loss_budget.py', 'strategy_olr/config.py', 'strategy_olr/research.py')


def register():
    if (ROOT/'registration.json').exists():
        return n.read(ROOT/'candidate_plan.json')
    base = next(c['mutations'] for c in n.read(PARENT/'candidate_plan.json')['candidates'] if c['id']==REFERENCE)
    candidates = []
    for label, priority, budgets, slots in (
        ('guard_control', False, (0,0,0), 5),
        ('auction_priority', True, (0,0,0), 5),
        ('priority_aggressive_budget', True, (.04,.06,.08), 5),
        ('priority_balanced_budget', True, (.03,.045,.06), 5),
        ('aggressive_budget_six_slots', True, (.04,.06,.08), 6),
        ('priority_position_budget', True, (.04,0,0), 5),
        ('priority_sector_account_budget', True, (0,.06,.08), 5)):
        mutations=copy.deepcopy(base)
        mutations.update(capacity_exit_priority=priority, capacity_loss_position_pct=budgets[0],
            capacity_loss_sector_pct=budgets[1], capacity_loss_total_pct=budgets[2], capacity_loss_atr_multiple=1.25)
        mutations['olr.overnight.slot_count']=slots
        candidates.append({'id':'c'+n.sig({'mutations':mutations,'version':'targeted_risk_v1'})[:16],
            'label':label, 'history_policy':'literal_carry', 'mutations':mutations,
            'mutations_hash':n.sig(mutations), 'aliases':[], 'kind':'targeted_risk_v1',
            'paths':[list(p) for p,_,_ in n.mutation_edits(base,mutations)]})
    cp={'baseline':base,'baseline_id':candidates[0]['id'],'candidates':candidates}
    n.write(ROOT/'candidate_plan.json',cp)
    protocol=copy.deepcopy(n.read(PARENT/'protocol.json'))
    protocol['generation_directories']={w:str(ENGINES) for w in ('train','oos')}
    n.write(ROOT/'protocol.json',protocol)
    plan=copy.deepcopy(n.read(PARENT/'mechanism_plan.json'));plan.update(cp)
    n.write(ROOT/'mechanism_plan.json',plan)
    n.write(ROOT/'registration.json', {'registered_utc':s.utc(), 'candidates':candidates,
        'reference':REFERENCE, 'windows':protocol['windows'],
        'hypotheses':[
            'Matching cancellation acknowledgement lets completed-bar risk exits supersede resting close auctions without overlapping sells.',
            'Bound 1.25 ATR plus round-trip costs per position, sector and account; same native ranks, pending buys and top-ups counted.',
            'A single extra native-ranked slot may use released cash to capture more qualifying alpha; no signal thresholds relaxed.'],
        'budget_rationale':'Two round scenario constraints chosen before this replay: aggressive 4/6/8% and tighter 3/4.5/6% of decision NAV. The 1.25 ATR allowance is heuristic, not fitted tail quantiles or a guaranteed loss cap. Budgets target raw-ATR risk of roughly 3.2/4.8/6.4% and 2.4/3.6/4.8% before costs.',
        'selection':{'is_return_loss_max_pp_vs_guard':3., 'is_return_loss_max_pp_vs_original_saved':3.,
            'stress_original_saved_is_loss_max_pp':3., 'is_dd_increase_max_pp':2.,
            'oos_return_must_exceed_guard':True, 'oos_dd_target_reduction_pp':2.,
            'frequency':'Report both trade counts and entry days. Prefer Pareto improvements; do not sacrifice the return gate merely to increase count.',
            'tie_break':'Among qualified variants prefer higher OOS return/drawdown, then IS return, then trade count; report frontier and original saved baseline.',
            'stress':'Repeat selected policy and matching new-source control at 0.5% participation / 15bp slippage; require positive OOS and IS return loss no more than 3pp versus matching stressed guard.'},
        'development_only':True,'independent_validation':False,'promotion':False,
        'maximum_primary_candidates':7})
    return cp


def patch_config(text):
    workspace=(REPO/'strategy_olr/config.py').read_text(encoding='utf-8')
    fields=workspace.split('    capacity_exit_priority:')[1].split('    capacity_defer_minutes:')[0]
    text=text.replace('    capacity_joint_allocation: bool = False\n',
        '    capacity_joint_allocation: bool = False\n    capacity_exit_priority:'+fields,1)
    checks=workspace.split('        if type(self.capacity_exit_priority)')[1].split('        if type(self.capacity_stale_release)')[0]
    return text.replace('        import math\n','        import math\n        if type(self.capacity_exit_priority)'+checks,1)


def prepare():
    register();hashes={}
    prior_receipt=ROOT/'preparation_complete.json'
    records=n.read(prior_receipt).get('changes',[]) if prior_receipt.exists() else []
    for w in ('train','oos'):
        ready=ENGINES/w/'ready.json'
        if ready.exists():
            hashes[w]=n.read(ready)['study']['source_generation_hash'];continue
        parent_study=n.read(PARENT/'runtime'/REFERENCE/w/'baseline/study.json')
        manifest=n.read(parent_study['manifest']);manifest.pop('manifest_hash',None)
        parent_files=Path(parent_study['code']);files=ENGINES/w/'inputs/files'
        print('Preparing isolated research generation',w,flush=True)
        for name, expected in manifest['files'].items():
            source=parent_files/name;target=files/name
            assert n.sha(source)==expected,name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                if source.suffix.lower() in ('.parquet','.sqlite','.db','.jsonl','.gz','.pickle','.joblib','.pkl'):
                    os.link(source,target)
                else:shutil.copy2(source,target)
            # A verified source and its hardlink are the same immutable bytes.
            if not os.path.samefile(source,target):assert n.sha(target)==expected,name
        before={name:manifest['files'].get(name) for name in PATCHED}
        for name in PATCHED:
            target=files/name
            if name=='strategy_olr/config.py':
                target.write_text(patch_config(target.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
            elif name=='strategy_olr/research.py':
                # Only extend the list of execution fields excluded from signal identity.
                value=target.read_text(encoding='utf-8')
                marker='    "capacity_joint_allocation",\n'
                assert value.count(marker)==1
                value=value.replace(marker,marker+''.join('    "'+x+'",\n' for x in (
                    'capacity_exit_priority','capacity_loss_position_pct','capacity_loss_sector_pct',
                    'capacity_loss_total_pct','capacity_loss_atr_multiple')))
                target.write_text(value,encoding='utf-8',newline='\n')
            else:shutil.copy2(REPO/name,target)
            manifest['files'][name]=n.sha(target)
        actual=subprocess.check_output([sys.executable,'-B','-c',
            'from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        hashes[w]=actual;manifest['source_generation_hash']=actual
        study=copy.deepcopy(parent_study)
        for name in ('capacity_checkpoint','capacity_checkpoint_completion','capacity_checkpoint_result'):
            study.pop(name,None)
        study.update(code=str(files),source_generation_hash=actual,
            manifest=str(ENGINES/w/'inputs/manifest.json'),spec_path='research/targeted_risk_v1/control_spec.json')
        study['configuration']['input_manifest_path']=study['manifest']
        n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
        n.verify_market_bindings(manifest)
        assert not n.retained_parser_mismatches(manifest,files)
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        n.write(ready,{'files':str(files),'study':study,'manifest':manifest})
        records.append({'window':w,'parent_source':parent_study['source_generation_hash'],'new_source':actual,
            'before':before,'after':{name:n.sha(files/name) for name in PATCHED},
            'signal_snapshots':{'path':study['olr_snapshots'],'sha256':n.sha(files/study['olr_snapshots'])}})
    plan=n.read(ROOT/'mechanism_plan.json')
    plan['snapshot_acceleration'].update(validated_source_hashes=hashes)
    n.write(ROOT/'mechanism_plan.json',plan)
    n.write(ROOT/'preparation_complete.json',{'utc':s.utc(),'sources':hashes,'changes':records,
        'execution_only_plus_explicit_fingerprint_exclusions':True})


def activate():
    b.ROOT=ROOT;b.ENGINES=ENGINES


def parity(candidate,w):
    old=PARENT/'cases'/REFERENCE/w/'100000000_baseline'
    new=b.output(candidate,w)
    fields=('return_pct','drawdown_pct','trades','sessions')
    assert all(n.read(old/'result.json')[k]==n.read(new/'result.json')[k] for k in fields)
    for name in ('daily.json','equity.json'):
        assert n.read(old/name)==n.read(new/name),name
    a,z=n.read(old/'trades.json'),n.read(new/'trades.json')
    assert len(a)==len(z)
    economic=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time',
              'exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
    assert all({k:x.get(k) for k in economic}=={k:y.get(k) for k in economic} for x,y in zip(a,z))
    n.write(ROOT/f'control_parity_{w}.json',{'passed':True,'parent':str(old),'new':str(new),
        'daily_and_intraday_marks_exact':True,'economic_trade_fields':economic,'trades':len(a),
        'parent_completion':n.sha(old/'completion.json'),'new_completion':n.sha(new/'completion.json')})


def run(labels=None,scenario='baseline',capped=False):
    prepare();activate()
    candidates=n.read(ROOT/'candidate_plan.json')['candidates']
    for c in candidates:
        if labels and c['label'] not in labels:continue
        for w in ('train','oos'):
            b.evaluate(c,w,scenario,capped)
            if c['label']=='guard_control' and scenario=='baseline' and not capped:parity(c,w)
    n.write(ROOT/('batch_'+scenario+('_capped' if capped else '')+'.json'),
        {'completed_utc':s.utc(),'labels':labels or [c['label'] for c in candidates],'scenario':scenario,'capped':capped})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','prepare','run'])
    p.add_argument('--labels',nargs='+');p.add_argument('--scenario',default='baseline');p.add_argument('--capped',action='store_true')
    a=p.parse_args()
    if a.action=='run':run(a.labels,a.scenario,a.capped)
    else:globals()[a.action]()
