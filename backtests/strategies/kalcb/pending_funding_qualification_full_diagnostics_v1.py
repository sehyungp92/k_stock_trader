"""Truthful V28 four-stress report-only adapter over the existing full diagnostic engine."""
from dataclasses import asdict
from datetime import time
from pathlib import Path
import json,shutil
from backtests.auto.shared.types import EndOfRoundArtifacts
from backtests.strategies.common.plugin_base import attach_official_metric_contract
from strategy_kalcb.config import KALCBConfig
from .exit_joint_full_diagnostics_v2 import ExitJointReportPlugin
from .qualified_artifacts import diagnostic_result,require,close
from .late_artifact_validation_v2 import validate_native,validate_case
from .late_entry_v1 import cumulative_timed_policy
from .retention_research_io_v1 import read,write,bound,verify,digest,PERIODS
from .full_diagnostics import (analyze_kalcb_result,render_kalcb_diagnostics_report,_final_metrics,
    _atomic_write_text,_atomic_write_json,_atomic_write_jsonl,_artifact_paths,_index_payload)

STATUS='audited_pending_funding_four_stress_evidence_only_no_promotion'


def truthful_status(analysis):
    analysis['promotion_status']=STATUS
    analysis['diagnostics_summary']['promotion_status']=STATUS
    return analysis


def authenticated_protocol(output):
    p=read(Path(output)/'protocol.json')
    require(p['version']=='pending-funding-qualification-full-reports-v1','Wrong report protocol')
    verify(p['programs']+p['bindings'])
    require(read(p['assessment_completion']['path'])['complete'] is True,'Assessment incomplete')
    require(p['report_count']==p['native_case_count']==16,'Wrong report count')
    require(set(p['recipes'])=={'control','pending_100'},'Wrong recipes')
    require(all(set(p['profiles'][n])=={'is','oos'} and all(set(s)=={'baseline','participation','cost','combined'} for s in p['profiles'][n].values()) for n in p['recipes']),
        'Missing report cells')
    require(p['fresh_report_count']==8 and p['reused_report_count']==8,'Wrong report reuse count')
    return p


def load_bundle(output,name,period,scenario):
    p=authenticated_protocol(output);entry=p['profiles'][name][period][scenario]
    folder=Path(entry['folder']);item=read(entry['input_profile']['path'])
    verify([item[k] for k in ('manifest','plan','policy')]);validate_native(folder,item,scenario,save=False)
    start,end,sessions=PERIODS[period];plan=read(item['plan']['path']);policy_dir=Path(item['policy']['path']).parent
    from scripts.kalcb_executed_value_replay import verify_policy
    from backtests.auto.shared.cache_keys import stable_signature
    policy,opening=verify_policy(policy_dir,'interaction_cost_margin');updates=cumulative_timed_policy(policy_dir,policy,opening)
    profile={'primary_profile':name,'period':period,'recipe':p['recipes'][name],'status':STATUS,'live_enabled':False,
        'initial_equity_krw':100000000.,'date_range':{'start':start,'end':end,'sessions':sessions},
        'configuration':plan['config'],'audit_manifest':item['manifest']['path'],
        'execution_source':{'source_generation_hash':item['source_generation_hash']},
        'policy':{'name':'interaction_cost_margin','manifest_sha256':item['policy']['sha256'],'directory':str(policy_dir)},
        'result_paths':{scenario:str(folder/'result.json')},'metrics':{scenario:read(folder/'result.json')}}
    lineage={'basis':'Authenticated original parent policy/model/label chain and new opening-only pending-funding source/config. No forecast/refit or new entry action.',
        'input_profile':entry['input_profile'],'assessment':p['assessment'],'native_protocol':p['native_protocol']}
    return {'profile':profile,'cases':{scenario:validate_case(folder,profile,scenario)},'opening':opening,'updates':updates,
        'policy':{**policy,'prepared_sha256':stable_signature(lineage)},'feature_lineage':lineage,
        'manifest':str(Path(output)/'protocol.json'),'manifest_hash':digest(Path(output)/'protocol.json'),'case_entry':entry}


class PendingFundingQualificationReportPlugin(ExitJointReportPlugin):
    """Inherits only locked evaluation and the integrity-only phase specification."""
    def __init__(self,output,name,period,scenario):
        require(scenario in ('baseline','participation','cost','combined'),'Unscored scenario')
        self.output,self.recipe,self.period,self.scenario=Path(output),name,period,scenario
        self.bundle=load_bundle(output,name,period,scenario);self.profile=self.bundle['profile']
        self.acceptance_manifest=Path(self.bundle['manifest']);self.source_fingerprint=self.profile['execution_source']['source_generation_hash']
        cfg=KALCBConfig.from_mapping(read(self.profile['configuration']))
        self.initial_mutations=json.loads(json.dumps(asdict(cfg),default=lambda x:x.isoformat() if isinstance(x,time) else x.value))
        self.config={'initial_equity':100000000.,'date_range':self.profile['date_range'],'promotion_status':STATUS,
            'artifact_promotion_policy':'no_canonical_or_live_promotion','capability_level':'real_replay',
            'risk_basis':'native_bar_level_mark_to_market','selected_candidate':name,'scenario':scenario,
            'initial_mutations':self.initial_mutations,'live_parity_fill_timing':'next_5m_open','auction_mode':'non_auction_continuous',
            'report_protocol':str(self.acceptance_manifest),'required_candidate_policy':self.profile['policy']}
        self.execution_context={**self.config,'source_fingerprint':self.source_fingerprint,
            'feature_manifest_hash':self.bundle['policy']['prepared_sha256'],'feature_manifest_hash_basis':self.bundle['feature_lineage']['basis'],
            'candidate_snapshot_hash':self.profile['policy']['manifest_sha256'],'replay_mode':'audited_native_timed_candidates',
            'expected_universe_size':103,'train_start':self.profile['date_range']['start'],'train_end':self.profile['date_range']['end']}
        self.result=diagnostic_result(self.bundle,scenario)
        metrics=_final_metrics(self.result.metrics,self.result.trades,self.config,self.result.replay_result.equity_curve,
            timestamps=self.result.replay_result.timestamps)
        self.metrics=attach_official_metric_contract(metrics,requires_audit_pass=True,audit_pass=True,
            execution_contract={**self.execution_context,'production_enabled':False,'metric_units':'Return/DD fractions; currency KRW'})

    def run_phase_diagnostics(self,phase,state,metrics,greedy_result):
        return json.dumps({'recipe':self.recipe,'scenario':self.scenario,'study_complete':False,'new_trials':0,
            'scope':'Audited prior100 four-stress pending-funding evidence; funded nominal-allocation qualification remains pending.',
            'checks':{s:c['validation'] for s,c in self.bundle['cases'].items()}})

    run_enhanced_diagnostics=run_phase_diagnostics

    def build_end_of_round_artifacts(self,state):
        self.canonicalize_mutations(state.cumulative_mutations);authenticated_protocol(self.output)
        text=self.run_phase_diagnostics(1,state,self.metrics,None)
        return EndOfRoundArtifacts(text,{'qualification_integrity':text},'Pending-funding four-stress evidence archived; no promotion')

    def write_full_diagnostics(self,state,output_dir,*,round_num=1,round_name=''):
        output_dir=Path(output_dir);analysis=analyze_kalcb_result(self.result,config=self.config,mutations=state.cumulative_mutations)
        native=self.bundle['cases'][self.scenario]
        close(analysis['final_metrics']['net_return_pct']*100,native['result']['return_pct'],'Report return drift')
        close(analysis['final_metrics']['max_drawdown_pct']*100,native['result']['drawdown_pct'],'Report DD drift')
        close(analysis['final_metrics']['profit_factor'],native['validation']['net_profit_factor'],'Report PF drift')
        analysis.update(round=round_num,round_name=round_name);truthful_status(analysis)
        analysis['gate_result']={'status':'source_account_passed_four_stress_evidence_only','production_ready':False,'four_stress_evidence_complete':True,'economic_winner_qualified':False,'funded_comparison_complete':False}
        analysis['live_parity_audit'].update(status='shared_core_only_no_live_qualification',observation_watch_count=103,
            observation_window_minutes=45,active_symbol_count_basis='Parent policy admissions, not live occupancy',
            paper_promotion_blockers=['Joint economic benefit requires full assessment.','No independent validation or funded nominal-allocation comparison.','Paper/live parity remains unqualified.'])
        analysis['diagnostics_summary']['live_parity_audit']=analysis['live_parity_audit']
        p=read(self.output/'protocol.json');a=read(p['assessment']['path']);key='/'.join((self.recipe,self.period,self.scenario));account=a['accounts'][key]
        context={'profile':self.profile,'scenario':self.scenario,'full_assessment':p['assessment'],'feature_lineage':self.bundle['feature_lineage'],
            'mechanism':'Opening only: original learned quantity then remaining actual 2x-equity funding cap including pending buys; parent order and checkpoint sizing unchanged.',
            'actual_intervention_support':account['support'],'profitable_activity':account['activity'],'funding':account['funding'],
            'meaningful_activity_sensitivity':account['meaningful_activity_sensitivity'],
            'comparison':a['comparisons'].get(key),'scope':'All sixteen prior100 parent/cap IS/OOS accounts across baseline, participation, cost and combined. Eight baseline/combined reports authentically reused from V25; eight new full stress reports. Funded comparison pending.',
            'history':'Continuous history, original clocks/embargoes. April-July reused development, not independent validation.',
            'limitations':['Changed filled orders and unique position outcomes are dependent account-path evidence, not isolated causal PnL.',
                'One-share and 0.1/1/5%-equity size sensitivities are reported separately; floors are descriptive, not new selection rules.',
                'Paired5/10session10000repseed20261011 is conditional sensitivity, not independent or multiplicity adjusted.',
                'Actual funding journals and pre-submission cash/inventory are bound; last-mark timestamps are not decision clocks.',
                'Any further mechanism requires a finite evidence-supported design; no automatic grid or promotion.'],
            'production_enabled':False,'canonical_replaced':False,'promotion_status':STATUS}
        analysis['diagnostics_summary']['pending_funding_context']=context
        report=render_kalcb_diagnostics_report(analysis).replace('Full-universe alpha','Non-selected outcome evidence').replace('Intraday Alpha Curve','Intraday realized PnL attribution')
        report+='\n\nAUDITED OPENING PENDING-FUNDING FOUR-STRESS EVIDENCE — NO PROMOTION\n'+json.dumps(context,indent=2,default=str)+'\n'
        _atomic_write_text(output_dir/'round_final_diagnostics.txt',report)
        for filename,data in [('diagnostics_summary.json',analysis['diagnostics_summary']),('candidate_frontier.json',analysis['candidate_frontier']),
            ('live_parity_audit.json',analysis['live_parity_audit']),('round_final_full_diagnostics.json',{k:v for k,v in analysis.items() if k not in ('trades','decisions')}),
            ('qualification_context.json',context)]:_atomic_write_json(data,output_dir/filename)
        _atomic_write_jsonl(output_dir/'trade_events.jsonl',analysis['trade_events'])
        _atomic_write_jsonl(output_dir/'decision_events.jsonl',[d.to_json_dict() for d in self.result.decisions])
        shutil.copy2(self.profile['configuration'],output_dir/'strategy_config.json')
        paths=_artifact_paths(output_dir);paths.update({n:str(output_dir/n) for n in ('round_final_full_diagnostics.json','qualification_context.json','decision_events.jsonl','strategy_config.json','diagnostic_integrity.json')})
        _atomic_write_json(_index_payload(analysis,paths,{'qualification_protocol':str(self.acceptance_manifest)}),output_dir/'full_diagnostics_index.json')
        authenticated_protocol(self.output)
        write(output_dir/'diagnostic_integrity.json',{'passed':True,'native_evidence_unchanged':True,'scenario':self.scenario,
            'report':bound(output_dir/'round_final_diagnostics.txt'),'native_return_pct':native['result']['return_pct'],
            'native_mtm_drawdown_pct':native['result']['drawdown_pct'],'promotion_status':STATUS,'economic_promotion':False})
        return analysis['diagnostics_summary']
