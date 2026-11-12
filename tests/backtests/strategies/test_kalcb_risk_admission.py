from scripts.kalcb_risk_admission import choose

def fixture():
    rows=[];values={}
    for q,base,stress in ((.25,3,3),(.5,9,7),(1,10,2)):
        key='A/q'+str(q)
        rows.append({'opportunity_id':key,'base_opportunity_id':'A',
            'features':{'action_quantity_scale':q},'action':{'quantity':10},
            'feature_available_at':'2025-08-07T09:30:00+09:00',
            'opening_bars':[{'volume':1000} for _ in range(6)]})
        values[key]={'baseline_value_bps':base,'combined_value_bps':stress,'failure_probability':.0001}
    return rows,values

def test_robust_cash_ranking_and_quantity_cap_are_separate():
    rows,values=fixture()
    assert choose(rows,values,objective='baseline',cap=1)[0][0]['features']['action_quantity_scale']==1
    assert choose(rows,values,objective='combined',cap=1)[0][0]['features']['action_quantity_scale']==.5
    assert choose(rows,values,objective='baseline',cap=.5)[0][0]['features']['action_quantity_scale']==.5

def test_no_action_can_bypass_value_failure_or_capacity_gates():
    rows,values=fixture()
    values[rows[0]['opportunity_id']]['failure_probability']=.001
    values[rows[1]['opportunity_id']]['combined_value_bps']=0
    rows[2]['action']['quantity']=1000000
    assert choose(rows,values,objective='baseline',cap=1)==[]
