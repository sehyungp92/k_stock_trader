import copy
from datetime import date,timedelta
import pytest
from backtests.strategies.kalcb import checkpoint_holding_models_v1 as base
from backtests.strategies.kalcb import checkpoint_context_interaction_v1 as m


def fixture():
    sessions=[(date(2025,5,1)+timedelta(days=i)).isoformat() for i in range(66)];groups=[]
    for i,day in enumerate(sessions):
        for q in (.25,.5,1.):
            x=(-1.,1.)[i%2];context=(-1.,1.)[(i//2)%2]
            row={'day':day,'symbol':'A','opportunity_id':f'{day}/A/q{q:g}',
                 'quantity_scale':q,'eligibility':'eligible','held_quantity':10,'planned_quantity':10,
                 'pending_buy_quantity':0,'checkpoint_at':day+'T09:45:00+09:00',
                 'own_features':dict.fromkeys(m.OWN,1.),'state_features':dict.fromkeys(m.STATE,1.),
                 'context_features':dict.fromkeys(m.CONTEXT,1.),'current_holding_label_valid':True,
                 'current_holding_advantage_bps':100*x*context,'label_available_at':day+'T15:35:00+09:00'}
            row['own_features']['post_entry_return_atr']=x;row['context_features']['peer_market_vwap_breadth_change']=context
            row['state_features']['pending_to_held_quantity']=0.
            groups.append({s:copy.deepcopy(row) for s in m.SCENARIOS})
    old=base.fit(groups,sessions,sessions[61],sessions[-1])
    return groups,sessions,old


def test_exact_inherited_additive_coefficients_reused():
    g,s,old=fixture();p=m.fit(g,s,s[61],s[-1],old)
    for scenario in m.SCENARIOS:
        assert p['models'][scenario]['coefficients']['peer_context']==old['models'][scenario]['coefficients']['peer_context']
        assert len(p['models'][scenario]['coefficients']['context_modulated'])==18


def test_interaction_represents_slope_change_missing_from_additive():
    g,s,old=fixture();p=m.fit(g,s,s[61],s[-1],old);pair=g[61*3]
    a=m.forecast(pair,p,'peer_context');b=m.forecast(pair,p,'context_modulated');y=pair['baseline']['current_holding_advantage_bps']
    assert abs(b['values']['baseline']-y)<abs(a['values']['baseline']-y)


def test_changed_inherited_clock_or_target_is_rejected():
    g,s,old=fixture();old['models']['baseline']['target_hash']='wrong'
    with pytest.raises(ValueError):m.fit(g,s,s[61],s[-1],old)


def test_future_targets_do_not_change_interaction_fit():
    g,s,old=fixture();p=m.fit(g,s,s[61],s[-1],old)
    for pair in g:
        if pair['baseline']['day']>=s[60]:
            for row in pair.values():row['current_holding_advantage_bps']=1e99
    assert m.fit(g,s,s[61],s[-1],old)==p


def test_product_formulas_are_causal_and_not_fitted_thresholds():
    g,_,_=fixture();row=g[0]['baseline'];v=m.feature_vector(row)
    assert v[-2]==row['own_features']['post_entry_return_atr']*row['context_features']['peer_market_vwap_breadth_change']
    assert v[-1]==row['own_features']['post_entry_return_atr']*row['context_features']['peer_sector_vwap_breadth_change']
    row['current_holding_advantage_bps']=1e20;assert m.feature_vector(row)==v


def test_missing_or_nonfinite_input_abstains():
    g,_,_=fixture();r=g[0]['baseline'];r['context_features']['peer_sector_vwap_breadth_change']=None
    assert m.feature_vector(r) is None


@pytest.mark.parametrize('recipe',m.RECIPES)
def test_new_product_domain_applies_to_both_models(recipe):
    g,s,old=fixture();p=m.fit(g,s,s[61],s[-1],old);pair=copy.deepcopy(g[61*3])
    for scenario in m.SCENARIOS:p['models'][scenario]['maximum'][-2]=-.5
    pair['baseline']['own_features']['post_entry_return_atr']=1.
    pair['baseline']['context_features']['peer_market_vwap_breadth_change']=1.
    assert not m.forecast(pair,p,recipe)['supported']
