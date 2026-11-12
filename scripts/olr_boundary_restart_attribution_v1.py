"""Read-only payoff, quantity and opportunity accounting for qualified replays."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_boundary_restart_results_v1 as r
b=r.b;s=b.s;n=b.n
HELPER=REPO/'data/backtests/analysis/olr_exit_diagnosis_20261010/analyze.py'


def main():
    spec=importlib.util.spec_from_file_location('qualified_olr_exit_accounting',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    plan=n.read(b.ROOT/'candidate_plan.json');bylabel={c['label']:c for c in plan['candidates']}
    out={'asof_utc':s.utc(),'accounting_helper':str(HELPER),'accounting_helper_sha256':n.sha(HELPER),
        'interpretation':'Exact observed-fill accounting. Price/cost component uses baseline quantity; quantity component uses candidate per-share outcome. Neither is a fixed-quantity executable causal counterfactual. Final exit timestamps do not establish identical partial-fill schedules. No-upward-compounding still permits native cash reuse.',
        'comparisons':{}}
    comparisons=[('original_to_'+c['label'],None,c) for c in plan['candidates']]
    for left,right in (
        ('literal_carry_original','literal_carry_guard_1p0'),
        ('literal_carry_guard_1p0','literal_carry_full_panel_guard_1p0'),
        ('literal_carry_full_panel','literal_carry_full_panel_guard_1p0'),
        ('broad_declared_universe_full_panel','broad_declared_universe_full_panel_guard_1p0')):
        if left in bylabel and right in bylabel:comparisons.append((left+'_to_'+right,bylabel[left],bylabel[right]))
    for right in plan['candidates']:
        if not right['mutations'].get(s.EXIT,{}).get('failed_followthrough_bars'):continue
        left=('literal_carry_full_panel' if right['mutations'].get(b.x.PANEL_KEY) else 'literal_carry_original') if right['history_policy']=='literal_carry' else 'broad_declared_universe_full_panel'
        label=left+'_to_'+right['label']
        if left in bylabel and label not in {v[0] for v in comparisons}:comparisons.append((label,bylabel[left],right))
    for label,left,right in comparisons:
        for w in ('train','oos'):
            fp=b.output(left,w) if left else s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline'
            fq=b.output(right,w)
            if not all((p/'assessment.json').exists() and n.read(p/'assessment.json').get('qualified') for p in (fp,fq)):continue
            s.forensic.authenticate(fp);s.forensic.authenticate(fq)
            a,z=n.read(fp/'trades.json'),n.read(fq/'trades.json')
            comparison=helper.compare(a,z)
            comparison['baseline_summary']=helper.summary(a);comparison['candidate_summary']=helper.summary(z)
            comparison['source_completions']={str(p):n.sha(p/'completion.json') for p in (fp,fq)}
            out['comparisons'].setdefault(label,{})[w]=comparison
    n.write(s.OUT/'boundary_restart_exit_and_quantity_attribution.json',out)
    for label,windows in out['comparisons'].items():
        for w,v in windows.items():
            print({'comparison':label,'window':w,'net_delta':v['account_pnl_delta'],'components':v['accounting_components'],
                'matched':v['matched'],'new':v['new'],'missing':v['missing'],'cohorts':v['cohorts']},flush=True)


if __name__=='__main__':main()
