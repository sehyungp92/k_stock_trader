from copy import deepcopy
import json
from pathlib import Path
import pytest
from backtests.strategies.kalcb.round3_contract_v1 import scalar,qualify,selectable_score,verify_acceptance,digest
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.plugin import PhaseSpec,PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment,GateCriterion,ScoredCandidate,PhaseDecision


def cells(ret=10):
    return {f'{p}/{s}':dict(return_pp=ret,mtm_dd_pp=8,native_valid=True)
            for p in ('is','oos') for s in ('baseline','participation','cost','combined')}


def test_invalid_and_stale_evidence_cannot_score_or_select(tmp_path):
    bad=cells(100);bad['is/baseline']['native_valid']=False
    with pytest.raises(ValueError):scalar(bad)
    with pytest.raises(ValueError):selectable_score(dict(native_valid=True,score={'S':10000},qualification={'promote':False}))
    p=tmp_path/'proof.json';p.write_text('{}');receipt=dict(passed=True,files={'proof':dict(path=str(p),sha256=digest(p))})
    assert verify_acceptance(receipt)
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError):verify_acceptance(receipt)
    with pytest.raises(ValueError):verify_acceptance({'passed':False,'files':{}})


def test_c16_ranking_and_full_rejection_are_separate():
    control=cells();good=cells(12)
    allcells={f'{p}/{s}/{c}':deepcopy(good[f'{p}/{s}']) for p in ('is','oos')
              for s in ('baseline','participation','cost','combined') for c in (95000000,100000000,105000000)}
    stats={'5':dict(simultaneous_Bonferroni_CI95_pp=[.5,3.5],planning_MDE_pp=1.)}
    checks={k:True for k in ('mechanism','independent_accounting','source','flat','ablation','neighbourhood','delay','cost','retained_components')}
    assert qualify(control,good,allcells,stats,checks)['promote']
    hidden=deepcopy(allcells);hidden['oos/combined/105000000']['finer_dd_pp']=15.01
    assert not qualify(control,good,hidden,stats,checks)['promote']
    assert not qualify(control,good,allcells,stats,{**checks,'mechanism':False})['promote']
    assert not qualify(control,good,allcells,stats,checks,diagnostic=True)['promote']
    lucky={'5':dict(simultaneous_Bonferroni_CI95_pp=[-1,4],planning_MDE_pp=4.)}
    assert not qualify(control,good,allcells,lucky,checks)['promote']
    assert scalar(good)['S']>scalar(control)['S']
    assert not qualify(control,control,allcells,stats,checks)['promote']


def test_qualification_evidence_seal_includes_conditional_checks(tmp_path):
    from backtests.strategies.kalcb.round3_evidence_v1 import seal,authenticate
    from backtests.strategies.kalcb.round3_native_io_v1 import bound
    main=tmp_path/'main.json';main.write_text('{}')
    delay=tmp_path/'delay.json';delay.write_text('{}')
    receipts=[dict(passed=True,files={'account':bound(p)}) for p in (main,delay)]
    path=tmp_path/'evaluated.json';seal(path,dict(native_valid=True,qualification={'promote':True}),receipts)
    assert authenticate(path)['qualification']['promote']
    delay.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Stale evaluation or qualification'):authenticate(path)


def test_frequency_padding_does_not_change_c16_rank():
    c=cells();padded=deepcopy(c)
    for v in padded.values():v.update(useful_issuer_days=[['x','x']]*10000,lifetimes=1000000)
    assert scalar(c)==scalar(padded)


def test_plugin_withholds_invalid_and_unqualified_candidates():
    from types import SimpleNamespace
    from backtests.strategies.kalcb.round3_plugin_v1 import C17RoundPlugin
    seen=[]
    class Service:
        def name_for_config(self,c):return c.get('name','control')
        def evaluate(self,name,progress=None):
            seen.append(name)
            if name=='control':return dict(native_valid=True,accepted_control=True,score={'S':10})
            return dict(native_valid=name!='invalid',score={'S':10000},qualification={'promote':False,'failures':['synthetic']})
    plugin=C17RoundPlugin.__new__(C17RoundPlugin);plugin.service=Service()
    evaluator=plugin.create_evaluate_batch(2,{})
    result=evaluator([Experiment('__baseline__',{}),Experiment('invalid',{'name':'invalid'}),Experiment('failed',{'name':'failed'})],{})
    assert [r.name for r in result]==['__baseline__'] and seen==['control','invalid','failed']


@pytest.mark.parametrize('candidate_rejected',[False,True])
def test_shared_phase_restores_all_settings_after_failed_high_score(tmp_path,candidate_rejected):
    class Plugin:
        name='kalcb';num_phases=2;ultimate_targets={}
        initial_mutations={'a':1,'route':{'votes':6},'hold':0}
        def canonicalize_mutations(self,m):return {**self.initial_mutations,**m}
        def get_phase_spec(self,phase,state):
            return PhaseSpec('synthetic invalid high score',[Experiment('invalid',{'a':99,'route':{'votes':5},'hold':8})],
                lambda m:[GateCriterion('qualification',1,int(m['qualified']),m['qualified'])],None,{},
                PhaseAnalysisPolicy(decide_action_fn=lambda *a:PhaseDecision('advance','fixed synthetic test')),max_rounds=1)
        def create_evaluate_batch(self,*args,**kwargs):
            def evaluate(candidates,current):
                return [ScoredCandidate(c.name,10000 if c.mutations.get('a')==99 else 1,
                    rejected=candidate_rejected and c.mutations.get('a')==99,reject_reason='failed qualification') for c in candidates]
            return evaluate
        def compute_final_metrics(self,m):return {'qualified':m['a']==1}
        def run_phase_diagnostics(self,*args):return 'Synthetic qualification test only'
    p=Plugin();runner=PhaseRunner(p,tmp_path,max_rounds=1,max_retries=0,max_diagnostic_retries=0)
    state=runner.run_phase(1)
    assert state.cumulative_mutations==p.initial_mutations
    assert json.loads((tmp_path/'phase_state.json').read_text())['cumulative_mutations']==p.initial_mutations
