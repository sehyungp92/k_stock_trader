from scripts.kalcb_restoration_readiness_v3 import summarise


def row(**kw):
    return dict(base_id='2026-01-01/000001',day='2026-01-01',label_valid=True,prefix_supported=True,
        full_krw=20000,earned_krw=-10000,remaining_krw=30000,filled_notional=2000000,
        same_day_donor=False,quantity_scale=.5,**kw)


def test_invalid_and_missing_prefix_never_become_zero_economic_evidence():
    a=row();b=row();b.update(label_valid=False,full_krw=None,earned_krw=None,remaining_krw=None,prefix_supported=False)
    c=row();c.update(base_id='2026-01-01/000002',prefix_supported=False,earned_krw=None,remaining_krw=None)
    result=summarise([a,b,c])
    assert result['opportunities']==3 and result['invalid']==1 and result['missing_prefix']==2
    assert result['full_lifetime_krw']==40000 and result['remaining_after35_krw']==30000
    assert result['issuer_days']==2 and result['sessions']==1


def test_negative_early_value_does_not_hide_positive_remaining_or_lifetime_value():
    result=summarise([row()])
    assert result['full_lifetime_krw']==result['earned_to35_krw']+result['remaining_after35_krw']
    assert result['profitable_material']==1 and result['donor_overlap_sessions']==0
