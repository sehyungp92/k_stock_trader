from scripts.kalcb_action_value_study import admit_actions
from scripts.kalcb_core_preserving_policy import merge_core_choices


def row(key,symbol):
    return {'opportunity_id':key,'base_opportunity_id':symbol,'symbol':symbol}


def test_preserves_known_quantity_without_losing_new_candidates():
    old=row('old/q0.25','old');upsize=row('old/q1','old');new=row('new/q1','new')
    merged=merge_core_choices({'old':(upsize,50),'new':(new,20)},
        [{'opportunity_id':'old','action_id':'old/q0.25','score':3}],{'old/q0.25':old})
    assert merged=={'old':(old,3),'new':(new,20)}


def test_core_slot_priority_preserves_real_scores_and_applies_guards():
    core=row('core','core');new=row('new','new')
    items=[(new,100),(core,2)]
    select=lambda **kw:admit_actions(items,lambda r,rank:True,None,'source',ready=False,budget=1,**kw)[0]
    assert select()==[(new,100)]
    assert select(priority_keys=['core'])==[(core,2)]
    chosen,_=admit_actions(items,lambda r,rank:r['symbol']!='core',None,'source',ready=False,budget=1,priority_keys=['core'])
    assert chosen==[(new,100)]


def test_priority_never_bypasses_nonpositive_value():
    core=row('core','core')
    selected,reasons=admit_actions([(core,-1)],lambda r,rank:True,None,'source',ready=False,priority_keys=['core'])
    assert selected==[] and reasons['core']=='net_value_threshold'
def test_new_quantity_cap_keeps_incumbent_actions_and_never_uses_outcomes():
    from scripts.kalcb_core_preserving_policy import cap_new_actions
    rows=[{'base_opportunity_id':base,'features':{'action_quantity_scale':q}}
          for base in ('core','new') for q in (.25,.5,1.)]
    chosen=cap_new_actions(rows,{'core'},.5)
    assert len(chosen)==5
    assert any(r['base_opportunity_id']=='core' and r['features']['action_quantity_scale']==1 for r in chosen)
    assert not any(r['base_opportunity_id']=='new' and r['features']['action_quantity_scale']==1 for r in chosen)

def test_incumbent_keeps_its_own_clock_when_new_model_rejects_it():
    from scripts.kalcb_core_preserving_policy import selected_forecast_origin
    row={'base_opportunity_id':'2025-10-10/A','opportunity_id':'2025-10-10/A/q1'}
    core={row['base_opportunity_id']:{'action_id':row['opportunity_id'],'valid_from':'2025-10-01'}}
    assert selected_forecast_origin(row,core,{})==('2025-10-01','frozen_core')
    new={'base_opportunity_id':'2025-10-10/B','opportunity_id':'2025-10-10/B/q0.5'}
    assert selected_forecast_origin(new,core,{new['opportunity_id']:{'valid_from':'2025-10-02'}})==('2025-10-02','new_forecast')

