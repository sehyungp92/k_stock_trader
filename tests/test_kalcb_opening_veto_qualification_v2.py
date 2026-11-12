"""Protect financial declarations while rebinding a repaired frozen package."""
from copy import deepcopy
from pathlib import Path
from scripts import kalcb_opening_veto_qualification_v2 as repair


def test_plan_rebinding_changes_only_generation_paths(tmp_path):
    reg=repair.read(repair.PREVIOUS/'registration.json')
    for item in reg['capital_profiles'].values():
        old=repair.read(item['plan']['path']);original=deepcopy(old)
        context=tmp_path/'context.json';protocol=tmp_path/'protocol.json'
        repair.write(context,{'fixture':True});repair.write(protocol,{'fixture':True})
        new=repair.rebind_plan(old,context,protocol,repair.bound(protocol))
        assert old==original
        assert {k:v for k,v in new.items() if k not in ('audit_configuration','input_sha256','capital_sensitivity')}=={k:v for k,v in old.items() if k not in ('audit_configuration','input_sha256','capital_sensitivity')}
        assert {k:v for k,v in new['capital_sensitivity'].items() if k!='protocol'}=={k:v for k,v in old['capital_sensitivity'].items() if k!='protocol'}
        assert old['capital_sensitivity']['protocol']['path'] not in new['input_sha256']
        assert old['audit_configuration'] not in new['input_sha256']


def test_missing_helper_is_historical_unchanged_dependency():
    old=repair.old.CAP/'v1/inputs';manifest=repair.read(old/'manifest.json')
    assert repair.digest(old/'files'/repair.HELPER)==manifest['files'][repair.HELPER]
    assert repair.digest(repair.REPO/repair.HELPER)==manifest['files'][repair.HELPER]
    reg=repair.read(repair.PREVIOUS/'registration.json');mp=Path(reg['capital_profiles']['bridge_disabled']['manifest']['path'])
    assert repair.HELPER not in repair.read(mp)['files']
    assert not (mp.parent/'files'/repair.HELPER).exists()
    assert not (repair.PREVIOUS/'cases/bridge_disabled').exists()
