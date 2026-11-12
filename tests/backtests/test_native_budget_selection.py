import json
import gzip
import copy

import pytest

from scripts import portfolio_native_budget_study as study


def setup(tmp_path, monkeypatch, rows):
    def write(name,value): (tmp_path/name).write_text(json.dumps(value))
    jobs=[];results={};recipes={}
    for name,(group,baseline,stress) in rows.items():
        recipes[name]={}
        for scenario,values in zip(study.SCENARIOS,(baseline,stress)):
            jobs.append({'recipe':name,'scenario':scenario,'group':group})
            ret,dd,trades=values
            results[name+'_'+scenario]={'recipe':name,'scenario':scenario,
                'return_pct':ret,'drawdown_pct':dd,'trades':trades}
    monkeypatch.setattr(study,'checked',lambda path:results[path.name])
    write('study.json',{'source_generation_hash':'fixture','candidates':recipes,
        'native_budget_registration':{'selection':study.SELECTION}})
    write('registration.json',{'jobs':jobs,'selection':study.SELECTION})


def cases():
    return {'olr':('standalone',(80,6,165),(50,9,165)),
        'oversized_k':('standalone',(200,10,350),(100,20,350)),
        'a':('search',(95,10,500),(60,12,500)),
        'frequent':('search',(94,10.2,520),(60.6,11.9,520)),
        'unsafe':('search',(180,11.001,530),(90,12,530)),
        'too_few':('search',(160,9,350),(100,10,550))}


def test_risk_and_frequency_precede_return_and_tie_band_prefers_frequency(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,cases());result=study.select(tmp_path)
    assert result['return_floors']==[80,50]
    assert result['standalone_feasible']==['olr']
    assert set(result['feasible'])=={'a','frequent'}
    assert result['winner']=='frequent'


def test_frequency_does_not_overrule_materially_larger_stressed_return(tmp_path,monkeypatch):
    rows=cases();rows['return_leader']=('search',(99,10.4,480),(63,12,480))
    setup(tmp_path,monkeypatch,rows)
    assert study.select(tmp_path)['winner']=='return_leader'


def test_empty_feasible_set_stays_unselected(tmp_path,monkeypatch):
    rows=cases();rows.pop('a');rows.pop('frequent')
    setup(tmp_path,monkeypatch,rows)
    assert study.select(tmp_path)['winner'] is None


def test_post_registration_risk_gate_changes_are_rejected(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,cases())
    path=tmp_path/'registration.json';registration=json.loads(path.read_text())
    registration['selection']['baseline_max_dd_pct']=30
    path.write_text(json.dumps(registration))
    with pytest.raises(ValueError,match='Changed selection registration'):study.select(tmp_path)


def test_single_conflict_session_cannot_promote_a_more_profitable_priority(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,{'olr':('standalone',(80,6,165),(50,9,165)),
        'base':('priority',(100,9,500),(60,11,500)),
        'alternate':('priority',(120,9,500),(80,11,500))})
    path=tmp_path/'registration.json';r=json.loads(path.read_text())
    r.update(retain_parent_on_exact_tie='base',minimum_priority_collision_sessions=10);path.write_text(json.dumps(r))
    for scenario in study.SCENARIOS:
        case=tmp_path/'cases'/('base_'+scenario);case.mkdir(parents=True)
        stamp='2026-01-05T14:30:00+09:00'
        with gzip.open(case/'allocation_events.jsonl.gz','wt') as stream:
            for oid in ('a','b'):stream.write(json.dumps({'event':'native_submission','submitted_order_id':oid,'timestamp':stamp})+'\n')
        with gzip.open(case/'native_budget_admissions.jsonl.gz','wt') as stream:
            stream.write(json.dumps({'order':{'submitted_at':stamp},'ledger':{'allocations':{
                'a':{'strategy_id':'KALCB','qty':4,'requested_qty':4},
                'b':{'strategy_id':'OLR','qty':1,'requested_qty':4}}}})+'\n')
    result=study.select(tmp_path)
    assert result['priority_support']['distinct_potential_conflict_sessions']==1
    assert result['winner']=='base'


def test_exact_priority_tie_preserves_existing_precedence(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,{'olr':('standalone',(80,6,165),(50,9,165)),
        'base':('priority',(100,9,500),(60,11,500)),
        'z_alternate':('priority',(100,9,500),(60,11,500))})
    path=tmp_path/'registration.json';r=json.loads(path.read_text())
    r['retain_parent_on_exact_tie']='base';path.write_text(json.dumps(r))
    assert study.select(tmp_path)['winner']=='base'


def test_exact_risk_cap_tie_uses_registered_tighter_forward_guard(tmp_path, monkeypatch):
    rows = {'olr': ('standalone', (80, 6, 165), (50, 9, 165))}
    for name in ('base', 'risk20', 'risk25', 'risk30'):
        rows[name] = ('risk', (100, 9, 500), (60, 11, 500))
    setup(tmp_path, monkeypatch, rows)
    path = tmp_path / 'study.json'; value = json.loads(path.read_text())
    for name, cap in (('base', 0), ('risk20', .02), ('risk25', .025), ('risk30', .03)):
        value['candidates'][name] = {'native_budget': {'max_kalcb_stop_risk_fraction': cap}}
    path.write_text(json.dumps(value))
    path = tmp_path / 'registration.json'; value = json.loads(path.read_text())
    value['prefer_tighter_risk_cap_on_exact_tie'] = True
    path.write_text(json.dumps(value))
    assert study.select(tmp_path)['winner'] == 'risk20'


def frozen_parent(tmp_path, monkeypatch):
    root = tmp_path / 'parent' / 'inputs' / 'files'
    root.mkdir(parents=True)
    name = 'research/portfolio_native_budget/study.json'
    policy = {'notional_cap_reference_equity': 25_000_000,
              'max_gross_notional': 50_000_000,
              'max_symbol_notional': 25_000_000,
              'max_sector_notional': 50_000_000}
    spec = {'configuration': {'initial_equity': 25_000_000},
            'initial_equity': 25_000_000, 'candidates': {'selected': {'policy': policy}}}
    study.write_json(root / name, spec)
    study.write_json(root / 'qualification.json', {'capacity': {'registered_execution_policies': []}})
    for owned in study.OWNED:
        path = root / owned
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen fixture\n')
    manifest = {'files': {p.relative_to(root).as_posix(): study.digest(p)
                          for p in root.rglob('*') if p.is_file()},
                'evidence': {'qualification_path': 'qualification.json'},
                'source_generation_hash': 'fixture'}
    mp = root.parent / 'manifest.json'
    study.write_json(mp, {**manifest, 'manifest_hash': study.signature(manifest)})
    parent = root.parent.parent / 'study.json'
    study.write_json(parent, {**spec, 'code': str(root), 'manifest': str(mp),
                              'spec_path': name, 'source_generation_hash': 'fixture'})
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *args, **kwargs: 'fixture\n')
    return parent


def test_child_freeze_preserves_ancestor_and_sibling_specification_bytes(tmp_path, monkeypatch):
    parent = frozen_parent(tmp_path, monkeypatch)
    original = study.read(parent)
    frozen = study.Path(original['code']) / original['spec_path']
    original_bytes = frozen.read_bytes()
    recipes = copy.deepcopy(original['candidates'])
    first, second = tmp_path / 'first', tmp_path / 'second'
    study.freeze(parent, first, recipes, [], capital=20_000_000, overlay=False)
    child = study.read(first / 'study.json')
    child_spec = study.Path(child['code']) / child['spec_path']
    child_bytes = child_spec.read_bytes()
    study.freeze(first / 'study.json', second, recipes, [], capital=30_000_000, overlay=False)
    sibling = tmp_path / 'sibling'
    study.freeze(parent, sibling, recipes, [], capital=22_500_000, overlay=False)
    assert frozen.read_bytes() == original_bytes
    assert child_spec.read_bytes() == child_bytes
    assert frozen.stat().st_ino != child_spec.stat().st_ino
    assert study.verified_spec(original)['initial_equity'] == 25_000_000
    assert study.verified_spec(child)['initial_equity'] == 20_000_000
    assert study.verified_spec(study.read(second / 'study.json'))['initial_equity'] == 30_000_000
    assert study.verified_spec(study.read(sibling / 'study.json'))['initial_equity'] == 22_500_000


def test_changed_frozen_spec_is_rejected_before_creating_another_generation(tmp_path, monkeypatch):
    parent = frozen_parent(tmp_path, monkeypatch)
    value = study.read(parent)
    frozen = study.Path(value['code']) / value['spec_path']
    frozen.write_text('{}')
    out = tmp_path / 'child'
    with pytest.raises(ValueError, match='Frozen study specification changed'):
        study.freeze(parent, out, value['candidates'], [], overlay=False)
    assert not out.exists()
