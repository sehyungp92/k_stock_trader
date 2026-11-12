"""Component backout contracts and separation from optimization runners."""
import ast
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import pytest
from backtests.strategies.kalcb import population_policy_v1 as policy
from backtests.strategies.kalcb import population_policy_primitives_v1 as pp


def row(symbol, q=.25):
    return {'opportunity_id':f'2026-04-01/{symbol}/q{q}',
            'base_opportunity_id':f'2026-04-01/{symbol}',
            'symbol':symbol,'features':{'action_quantity_scale':q}}


def test_removal_needs_supported_nonpositive_exact_action_and_keeps_order():
    a,b,c=row('A'),row('B'),row('C')
    parent=[(a,9),(b,7),(c,6)];origins={r['opportunity_id']:'frozen_core' for r,_ in parent}
    records={r['opportunity_id']:r for r,_ in parent}
    with patch.object(policy.pc,'descriptor',side_effect=lambda r,o,x:r), \
         patch.object(policy.pc,'supported',side_effect=lambda d,p:d['symbol']!='B'), \
         patch.object(policy.pc,'corrected_values',side_effect=lambda r,o,x,p:{'baseline':-1 if r['symbol']!='C' else 2,'combined':1}):
        selected,_=policy.select_backout('removals_only',parent,origins,[],{},records,records,{},None,None)
    assert selected==[(b,7),(c,6)]  # Unsupported negative B is preserved, despite absence from full rerank.
    assert parent==[(a,9),(b,7),(c,6)]


def test_additions_never_substitute_quantity_or_reprice_incumbent():
    a,b,c=row('A'),row('B'),row('C',1)
    parent=[(a,7)];full=[(row('A',.5),99),(b,8),(c,12)]
    origins={a['opportunity_id']:'frozen_core'};fo={r['opportunity_id']:'new_forecast' for r,_ in full}
    records={r['opportunity_id']:r for r,_ in parent+full}
    def admission(items,*args,**kwargs):
        assert kwargs['priority_keys']==[a['opportunity_id']]
        assert items==[(a,7),(b,8)]
        return items,{}
    with patch.object(policy.pc,'descriptor',return_value={}),patch.object(policy.pc,'supported',return_value=True), \
         patch.object(policy,'read',return_value={'mutations':{}}),patch.object(pp,'config_for',return_value=None), \
         patch.object(pp,'admit_actions',side_effect=admission):
        selected,co=policy.select_backout('additions_only',parent,origins,full,fo,records,records,{}, {'seed':'s','q3':'q'},'s')
    assert selected==[(a,7),(b,8)] and co[a['opportunity_id']]=='frozen_core'


def test_displaced_parent_fails_closed():
    a=row('A');key=a['opportunity_id']
    with patch.object(policy,'read',return_value={'mutations':{}}),patch.object(pp,'config_for',return_value=None), \
         patch.object(pp,'admit_actions',return_value=([],{})):
        with pytest.raises(ValueError,match='displaced'):
            policy.select_backout('additions_only',[(a,7)],{key:'frozen_core'},[],{}, {},{}, {},{'seed':'s','q3':'q'},'s')


def test_empty_parent_does_not_fall_back_to_unrestricted_selector():
    with patch.object(policy,'read',return_value={'mutations':{}}),patch.object(pp,'config_for',return_value=None), \
         patch.object(pp,'admit_actions',return_value=([],{})) as admit:
        result=policy.select_backout('additions_only',[],{},[],{}, {},{}, {},{'seed':'s','q3':'q'},'s')
    assert result==([],{}) and admit.call_args.args[0]==[]


def test_no_optimization_runner_imports_in_new_policy_libraries():
    for module in (policy,pp):
        tree=ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        assert not [n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and (n.module or '').startswith('scripts.')]


def test_extracted_algorithms_match_original_asts_after_import_removal():
    root=Path(__file__).resolve().parents[3]
    expected={'scripts/kalcb_action_quantity_ranker.py':['capacity_ok','choose_quantities'],
        'scripts/kalcb_action_value_study.py':['recheck_actions','admit_actions'],
        'scripts/kalcb_checkpoint_fit.py':['update_snapshot','native_admit'],
        'scripts/kalcb_selective_recovery_v2.py':['choose_day','package_day']}
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            return None if (node.module or '').startswith('scripts.') else node
    generated=ast.parse(Path(pp.__file__).read_text(encoding='utf-8'))
    for file,names in expected.items():
        original=ast.parse((root/file).read_text(encoding='utf-8'))
        for name in names:
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in generated.body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(Strip().visit(a))==ast.dump(b)
