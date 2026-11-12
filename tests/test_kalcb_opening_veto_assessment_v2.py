"""Exercise complete C10 assessment against actual native evidence interfaces."""
from pathlib import Path
from scripts import kalcb_opening_veto_qualification_v3 as runner
from scripts.kalcb_opening_veto_assessment_v2 import assess


def test_full_assessment_real_standard_and_capital_identity_accounts(tmp_path):
    # Temporary linked evidence is an interface fixture, not a new qualified account.
    protocol=runner.read(runner.PREVIOUS/'protocol.json');rows=[]
    for row in runner.read(runner.PREVIOUS/'completed_cases.json'):
        c=row['case'];protocol['reused'][c['period']+'/'+c['scenario']]=str(runner.PREVIOUS/'cases'/c['id'])
    for s,old in protocol['capital_controls'].items():
        c=dict(id='capital_'+s,scenario=s,initial_equity=protocol['capitals'][s],kind='capital')
        rows.append(dict(case=c,verdict=old['verdict']))
        if old['qualified']:
            for name in ('result.json','daily.json','equity.json','fills.json','trades.json','independent_account_verification.json'):
                runner.link(Path(old['folder'])/name,tmp_path/'cases'/c['id']/name)
    runner.write(tmp_path/'protocol.json',protocol);runner.write(tmp_path/'completed_cases.json',rows)
    assess(tmp_path);result=runner.read(tmp_path/'assessment.json')
    assert len(result['cells'])==8 and result['criteria']['all_standard_valid']
    assert sum(v['matched_qualified'] for v in result['capital'].values())==3
    for s in ('baseline','cost','combined'):
        row=result['capital'][s]
        assert row['net_delta_krw']==0
        assert abs(row['account']['net_pnl_krw']-row['initial_equity']*row['metrics']['return_pct']/100)<.01
    assert not result['capital']['participation']['matched_qualified']
    assert 'net_delta_krw' not in result['capital']['participation']
    assert result['criteria']['economic_objective_achieved'] is False
