from pathlib import Path
import importlib.util
import json
import pytest

spec=importlib.util.spec_from_file_location('scope_repair',Path(__file__).resolve().parents[2]/'scripts/olr_sector_feature_scope_v10.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def index():
    p=json.loads((m.ROOT/'protocol.json').read_text())
    return json.loads((Path(p['source']['cache'])/'index.json').read_text())


def test_exact_existing_source_scope_and_tail():
    dates=m.scope(index());assert len(dates)==215 and dates[-1]=='2026-03-27'
    assert '2026-03-30' not in dates and '2026-03-31' not in dates


def test_added_or_removed_source_day_rejected():
    raw=index();raw['identity']['base']['context_dates'][-1]='2026-03-31'
    with pytest.raises(AssertionError):m.scope(raw)
    raw=index();del raw['files']['2026-03-27.pickle']
    with pytest.raises(AssertionError):m.scope(raw)


def test_one_exact_literal_change_only():
    raw=m.ORIGINAL.read_text(encoding='utf-8');new=m.corrected(raw)
    assert new.replace(m.NEW,m.OLD,1)==raw
    with pytest.raises(AssertionError):m.corrected(raw+raw)
