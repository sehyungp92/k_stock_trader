"""Bounded paired follow-up to the OLR forensic review; never promotes live.

The fixed grid is registered before outcomes, uses continuous feature history,
and retains full native execution audits. Earlier OOS is development data.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import olr_round1_mechanism_study as mechanism
from scripts import olr_round1_oos_forensic_review as forensic

n = mechanism.native
PARENT = forensic.PARENT
PRIOR = forensic.OUT / 'replays'
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/v1')
OUT = REPO / 'data/backtests/analysis/olr_round1_optimization_20261010'
ENTRY, EXIT = 'olr.trade_plan.entry', 'olr.trade_plan.exit'


def utc():
    return datetime.now(timezone.utc).isoformat()


def register():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'registration.json').exists():
        return n.read(ROOT/'candidate_plan.json')
    old=n.read(PARENT/'mechanism_plan.json'); base=old['baseline']
    catalog={c['id']:c for c in old['candidates']}
    controls=[catalog[cid] for cid in (old['baseline_id'],'cbd14aff8c9557723','c272fe0390df855bd','c1e5c4b5e26e960b8','ce0d97e2f019d24c1')]
    candidates=[]
    def add(label, guard=None, confirmation=None, sector=None):
        m=copy.deepcopy(base)
        if guard is not None:
            m[EXIT].update(failed_followthrough_bars=1, failed_followthrough_close_r=-guard, failed_followthrough_mfe_r=4.)
        if confirmation is not None: m[ENTRY]['min_vwap_ret']=confirmation
        if sector is not None: m['olr.research.min_sector_participation']=sector
        key=n.sig(m)
        candidates.append({'id':'c'+key[:16],'label':label,'kind':'registered_focused_followup',
            'mutations':m,'mutations_hash':key,'aliases':[],
            'paths':[list(p) for p,_,_ in n.mutation_edits(base,m)]})
    add('guard_1p0_neighborhood', guard=1.)
    add('confirmation_minus_10bp', confirmation=-.001)
    for guard in (.8,1.2):
        for confirmation in (-.001,-.002):
            add(f'guard_{guard}_confirmation_{confirmation}',guard=guard,confirmation=confirmation)
    add('guard_1p4_neighborhood',guard=1.4)
    add('guard_1p2_sector_participation_40pct',guard=1.2,sector=.4)
    assert len({c['id'] for c in candidates})==8
    protocol=copy.deepcopy(n.read(PRIOR/'protocol.json'))
    protocol['review_scope']='Registered guard neighborhood, entry-confirmation factorial, and one admission interaction; continuous history; no live promotion.'
    protocol['generation_directories']={w:str(n.generation_folder(PARENT,w).parent) for w in ('train','oos')}
    plan={**old,'candidates':controls+candidates,'new_ids':[c['id'] for c in candidates]}
    registration={'registered_utc':utc(),'program_sha256':n.sha(__file__),
        'parent_evidence_sha256':n.sha(forensic.OUT/'comparisons.json'),
        'candidates':candidates,'reused_control_ids':[c['id'] for c in controls],
        'windows':protocol['windows'],'history_policy':'continuous_is_to_oos_no_feature_reset',
        'acceptance':{'is_return_max_drop_pp':3,'oos_gain_min_pp':2,'positive_oos_return':True,'drawdown_max_increase_pp':2,
            'activity':'Report trade changes; do not override the user return objective solely for fewer trades.',
            'stress':'Finalist versus matched control; same 3pp IS preservation limit and positive OOS objective.'},
        'selection':'Prefer policies improving both returns. Otherwise maximize minimum IS/OOS return gain among eligible policies; tie by OOS return, then lower OOS drawdown.',
        'finalists':'At most two Pareto candidates: IS-preservation winner and highest-OOS-return winner meeting gates. Matched combined stress, then no-upward-compounding for eligible successor.',
        'adaptive_feature_scope':'Separately test causal continuous current-afternoon sector panel using the same retained prior history; retain single-change baseline and selected-guard controls.',
        'independent_validation':False,'promotion':False,'new_configurations':8}
    n.write(ROOT/'protocol.json',protocol)
    n.write(ROOT/'mechanism_plan.json',plan)
    n.write(ROOT/'candidate_plan.json',{'baseline_id':old['baseline_id'],'baseline':base,'candidates':controls+candidates})
    n.write(ROOT/'registration.json',registration)
    n.write(OUT/'study_location.json',{'root':str(ROOT),'registration':str(ROOT/'registration.json'),'created_utc':utc()})
    return n.read(ROOT/'candidate_plan.json')


def run():
    register()
    plan=n.read(ROOT/'mechanism_plan.json')
    new=[c for c in plan['candidates'] if c['id'] in plan['new_ids']]
    mechanism.batch(ROOT,new,workers=2,phase='focused_grid')
    rows=n.results(ROOT)
    for row in rows:
        if row.get('qualified'): forensic.authenticate(Path(row['output']))
    n.write(ROOT/'grid_completion.json',{'completed_utc':utc(),'paired_candidates':8,'native_replays':len(rows),'all_qualified':all(r.get('qualified') for r in rows),'promotion':False})
    print({'completed':len(rows)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run']);args=parser.parse_args()
    register() if args.action=='register' else run()
