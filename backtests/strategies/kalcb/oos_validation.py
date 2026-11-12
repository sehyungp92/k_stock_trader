"""Shared-framework evaluation of the one frozen standalone OOS profile."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager
from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import GateCriterion, ScoredCandidate, EndOfRoundArtifacts
from backtests.strategies.common.plugin_base import attach_official_metric_contract
from .qualified_artifacts import read, digest, require, validate_case, diagnostic_result
from .full_diagnostics import analyze_kalcb_result, render_kalcb_diagnostics_report
from backtests.strategies.kalcb.executed_value import write_json
from strategy_kalcb.config import KALCBConfig

SCENARIOS=('baseline','participation','cost','combined')


class KALCBOOSValidationPlugin:
    name='kalcb'
    num_phases=1
    ultimate_targets={}
    capability_level='real_replay'
    requires_full_diagnostics=True

    def __init__(self, root):
        self.root=Path(root).resolve()
        self.preparation=read(self.root/'preparation.json')
        require(self.preparation['passed'], 'Forward preparation failed')
        self.protocol=read(self.root/'protocol.json')
        for key in ('canonical_profile','canonical_config'):
            item=self.protocol[key]
            require(digest(item['path'])==item['sha256'],'Canonical round changed before OOS evaluation')
        self.selected=read(self.protocol['canonical_profile']['path'])
        self.manifest=self.root/'inputs/manifest.json'
        self.code=self.root/'inputs/files'
        self.source_fingerprint=read(self.manifest)['source_generation_hash']
        require(self.source_fingerprint==self.preparation['source_generation_hash'],'Changed OOS source generation')
        self.initial_mutations={'profile':'locked_round1','window':'20260601_20260710'}
        self.config={'initial_equity':100_000_000,'date_range':self.protocol['period'],
            'promotion_status':'oos_evaluation_only_no_promotion','live_parity_fill_timing':'next_5m_open',
            'auction_mode':'non_auction_continuous','capability_level':'real_replay',
            'artifact_promotion_policy':'Preserve canonical round; no OOS tuning or selection',
            'initial_mutations':self.initial_mutations,'selected_candidate':self.selected['policy']['name'],
            'require_full_diagnostics':True,'risk_basis':'native_bar_level_mark_to_market'}
        self.execution_context={**self.config,'source_fingerprint':self.source_fingerprint,
            'replay_mode':'audited_standalone_frozen_model_oos','train_start':'2026-06-01','train_end':'2026-07-10',
            'expected_universe_size':103,'candidate_snapshot_hash':digest(self.policy_dir/'policies_manifest.json'),
            'evaluation_role':'reserved_validation_already_inspected_in_related_portfolio',
            'protocol':str(self.root/'protocol.json'),'protocol_sha256':digest(self.root/'protocol.json')}
        self.bundle=None

    @property
    def policy_dir(self):
        return self.code/'research/kalcb_executed_value/policies'

    def canonicalize_mutations(self, mutations):
        merged={**self.initial_mutations,**mutations}
        require(merged==self.initial_mutations,'OOS evaluation cannot change the selected strategy')
        return merged

    def _run_case(self,scenario):
        target=self.root/'cases'/scenario
        log=self.root/'logs'/f'{scenario}.log'
        if not (target/'completion.json').exists():
            require(not target.exists(),'Incomplete case retained; inspect before attempting a new replay')
            log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'NUMEXPR_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1','PYTHONUTF8':'1',
                 'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(self.code)}
            args=[sys.executable,'-B','-m','scripts.kalcb_executed_value_replay',
                '--plan',str(self.root/'plan.json'),'--policies',str(self.policy_dir),
                '--policy',self.selected['policy']['name'],'--scenario',scenario,
                '--audit-manifest',str(self.manifest),'--output',str(target)]
            with log.open('x',encoding='utf-8') as stream:
                code=subprocess.run(args,cwd=self.code,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            require(code==0,'Native OOS replay failed: '+str(log))
        result=read(target/'result.json')
        require(read(target/'completion.json')['source_generation_hash']==self.source_fingerprint,'Stale completed OOS case')
        require(result['audit_status']=='passed' and result['execution_validity_pass'],'Failed source audit: '+scenario)
        return result

    def _load(self):
        from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
        if self.bundle is None:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=dict(zip(SCENARIOS,pool.map(self._run_case,SCENARIOS)))
        else:
            results={s:read(self.root/'cases'/s/'result.json') for s in SCENARIOS}
        profile={'configuration':self.selected['configuration'],
            'configuration_sha256':self.selected['configuration_sha256'],
            'source_is_profile':self.protocol['canonical_profile'],
            'date_range':self.protocol['period'],
            'june_july_2026_used':True,'initial_equity_krw':100_000_000,
            'execution_source':{'code':str(self.code),'source_generation_hash':self.source_fingerprint},
            'policy':{**self.selected['policy'],'directory':str(self.policy_dir),
                      'manifest_sha256':digest(self.policy_dir/'policies_manifest.json')},
            'audit_manifest':str(self.manifest),'audit_manifest_sha256':digest(self.manifest),
            'result_paths':{s:str(self.root/'cases'/s/'result.json') for s in SCENARIOS},
            'metrics':{s:{k:r[k] for k in ('return_pct','drawdown_pct','trades','active_days','audit_status')} for s,r in results.items()},
            'status':'evaluated_oos_only','live_enabled':False,'evaluation_protocol':str(self.root/'protocol.json')}
        cases={s:validate_case(self.root/'cases'/s,profile,s) for s in SCENARIOS}
        policy,opening=verify_policy(self.policy_dir,profile['policy']['name'])
        updates=timed_policy(self.policy_dir,policy,opening)
        require(len(opening)==29 and str(max(opening))=='2026-07-10','Incomplete OOS calendar')
        self.bundle={'profile':profile,'cases':cases,'policy':policy,'opening':opening,'updates':updates}
        return self.bundle

    def compute_final_metrics(self,mutations):
        from .full_diagnostics import _final_metrics
        self.canonicalize_mutations(mutations)
        bundle=self._load(); result=diagnostic_result(bundle)
        metrics=_final_metrics(result.metrics,result.trades,self.config,result.replay_result.equity_curve,
                               timestamps=result.replay_result.timestamps)
        metrics['oos_scenarios']=bundle['profile']['metrics']
        metrics['oos_source_audit_pass']=True
        metrics['oos_positive_return_all_scenarios']=all(r['return_pct']>0 for r in bundle['profile']['metrics'].values())
        return attach_official_metric_contract(metrics,requires_audit_pass=True,audit_pass=True,
            execution_contract={**self.execution_context,'production_enabled':False,
                'metric_units':'Return/drawdown fractions; scenario tables explicitly percent'})

    def get_phase_spec(self,phase,state):
        return PhaseSpec('Evaluate locked standalone round_1 on the reserved historical window',[],
            lambda m:[GateCriterion('all_four_native_source_audits',1,int(m['oos_source_audit_pass']),m['oos_source_audit_pass'])],
            None,{},PhaseAnalysisPolicy(focus_metrics=['official_mtm_net_return_pct','max_drawdown_pct']),
            max_rounds=1,phase_metric_basis='audited_standalone_oos',
            primary_promotion_metric='official_mtm_net_return_pct',
            official_metric_keys=('official_mtm_net_return_pct','max_drawdown_pct'),promotion_requires_audit_pass=True)

    def create_evaluate_batch(self,phase,cumulative_mutations,**kwargs):
        def evaluate(candidates,current):
            return [ScoredCandidate(c.name,(m:=self.compute_final_metrics({**current,**c.mutations}))['official_mtm_net_return_pct'],metrics=m) for c in candidates]
        return evaluate

    def run_phase_diagnostics(self,phase,state,metrics,greedy_result):
        return json.dumps({'evaluation_only':True,'new_search_trials':0,
            'scenarios':self.bundle['profile']['metrics'],'protocol':self.protocol,
            'interpretation':'Data/execution qualification is separate from profitable performance. No canonical promotion.'},indent=2)

    run_enhanced_diagnostics=run_phase_diagnostics

    def build_end_of_round_artifacts(self,state):
        metrics=self.compute_final_metrics(state.cumulative_mutations)
        report=self.run_phase_diagnostics(1,state,metrics,None)
        return EndOfRoundArtifacts(report,{'standalone_oos':report},'OOS evaluation completed; canonical round retained unchanged')

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        output_dir=Path(output_dir)
        bundle=self._load(); result=diagnostic_result(bundle)
        raw=read(self.selected['configuration'])
        analysis=analyze_kalcb_result(result,config=self.config,mutations=raw)
        analysis.update(round=1,round_name='standalone_oos_evaluation',promotion_status='oos_evaluation_only_no_promotion')
        analysis['live_parity_audit'].update(status='historical_native_execution_only',production_ready=False,
            observation_watch_count=103,observation_window_minutes=45,
            note='Admitted snapshots do not certify live observation/feed capacity.')
        metrics=analysis['final_metrics']
        comparison={'is_period':self.selected['date_range'],'oos_period':self.protocol['period'],
            'is':self.selected['metrics'],'oos':bundle['profile']['metrics'],
            'initial_equity_krw':100_000_000,'protocol':self.protocol,
            'source_audits':{s:c['validation'] for s,c in bundle['cases'].items()},
            'canonical_round_changed':False,'oos_fitting_performed':False,'new_search_trials':0}
        report=render_kalcb_diagnostics_report(analysis)
        report=('STANDALONE KALCB OOS EVALUATION; NO PROMOTION\n'
                'Reserved window: 2026-06-01 through 2026-07-10, all 29 sessions.\n'
                'Frozen pre-OOS model/calibration; not an expanding-refit experiment.\n'
                'This window was already inspected in a related portfolio study.\n\n'+report+
                '\n\nIS/OOS COMPARISON AND REGISTERED CONTRACT\n'+json.dumps(comparison,indent=2)+'\n')
        (output_dir/'round_final_diagnostics.txt').write_text(report,encoding='utf-8')
        write_json(output_dir/'round_final_full_diagnostics.json',analysis)
        write_json(output_dir/'is_oos_comparison.json',comparison)
        write_json(output_dir/'strategy_config.json',raw)
        write_json(output_dir/'oos_profile.json',bundle['profile'])
        write_json(output_dir/'diagnostics_summary.json',{'final_metrics':metrics,'comparison':comparison})
        for key in ('canonical_profile','canonical_config'):
            item=self.protocol[key]; require(digest(item['path'])==item['sha256'],'Canonical round was modified')
        write_json(output_dir/'validation_receipt.json',{'passed':True,'source_audits_passed':list(SCENARIOS),
            'reconciled_fill_count':sum(c['validation']['fill_count'] for c in bundle['cases'].values()),
            'canonical_round_unchanged':True,'report_sha256':digest(output_dir/'round_final_diagnostics.txt'),
            'evaluator_program':{'path':__file__,'sha256':digest(__file__)}})
        return {'strategy':'kalcb','promotion_status':'oos_evaluation_only_no_promotion',
                'selected_candidate':self.selected['policy']['name'],'final_metrics':metrics,
                'scenarios':comparison['oos'],'evaluation_complete':True,'production_ready':False}


def evaluate(root):
    root=Path(root).resolve()
    plugin=KALCBOOSValidationPlugin(root)
    manager=RoundManager('stock','kalcb',base_dir=root/'evaluation')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='standalone_oos_evaluation',round_manager=manager,
        round_num=1,max_rounds=1,max_retries=0,max_diagnostic_retries=0).run_all_phases()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    evaluate(parser.parse_args().root)
