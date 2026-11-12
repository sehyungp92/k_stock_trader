"""Funded allocation comparisons within the shared phase lifecycle; no promotion."""
from pathlib import Path
from backtests.strategies.kalcb.population_qualification_plugin_v1 import PopulationQualificationPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest


class FixedAllocationPlugin(PopulationQualificationPlugin):
    name='kalcb_fixed_allocation_v1'

    def diagnostic(self):
        if digest(self.study)!=self.study_hash:raise ValueError('Registered study changed')
        records=[read(p) for p in sorted((self.root/'cases').glob('*/record.json'))]
        byname={r['recipe']:r for r in records}
        for r in records:
            parent='control' if r['changes']['parent']=='expanding' else 'control_expanding_atr5'
            m=r['metrics'];pm=byname[parent]['metrics']
            r['fixed_budget_local_deltas']={k:m[k]-pm[k] for k in m if k.startswith(('is_','oos_')) and k in pm}
            r['global_compounding_IS_sacrifice_pp']={s:self.spec['global_IS_reference'][s]-m['is_'+s+'_return'] for s in ('baseline','combined')}
            r['IS_reporting_bands']={str(b):all(v<=b for v in r['global_compounding_IS_sacrifice_pp'].values()) for b in (2,5,10)}
        payload={'trials':records,'study':bound(self.study),'no_promotion':True,
            'scope':'Native funded fixed nominal allocation budget, actual cash and drawdown. No replenishment or post-hoc rescaling. Compare policy deltas within each budget regime; actual funding and integer quantities can still change.',
            'final_qualification_review_pending':True}
        write(self.root/'allocation_ledger.json',payload);return payload

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        result=self.diagnostic();output_dir=Path(output_dir)
        write(output_dir/'round_final_full_diagnostics.json',result)
        lines=['KALCB FUNDED FIXED ALLOCATION COMPARISON; REUSED DEVELOPMENT; NO PROMOTION']
        for r in result['trials']:
            lines.append(r['recipe'])
            for p in ('is','oos'):
                for s in ('baseline','combined'):
                    m=r['metrics'];k=p+'_'+s
                    lines.append(f"  {k}: return {m[k+'_return']:.6f}%, MTM DD {m[k+'_dd']:.6f}%, trades {m[k+'_trades']}")
        (output_dir/'round_final_diagnostics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return {'complete':True,'study_complete':False,'promotion_status':'research_only',
            'trial_count':len(result['trials']),'final_metrics':self.compute_final_metrics(state.cumulative_mutations)}
