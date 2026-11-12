import importlib.util
from pathlib import Path
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / 'backtests/strategies/kalcb/decision_value_contract_v1.py'
SPEC = importlib.util.spec_from_file_location('decision_value_contract_under_test', PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class ContractTests(unittest.TestCase):
    def values(self):
        r = dict(action_id='2026-04-01/X/q0.5', day='2026-04-01', available_at='2026-04-01T09:30:00+09:00',
                 base_valid_from='2026-03-10', quantity_scale=.5, reference_quantity=137,
                 notional_equity=.214131, descriptor={'tier': 'top2'},
                 calibrated_core={'baseline': 8., 'combined': 2.}, calibrated_support={'baseline': 4., 'combined': 1.})
        other = {**r, 'valid_from': '2026-03-10'}
        cells = {s+'/frozen_core/q0.5/self1_other1/top2': {'supported': True, 'offset_bps': -1.}
                 for s in M.SCENARIOS}
        p = dict(family='expanding', valid_from='2026-03-31', valid_through='2026-04-06',
                 observed_through='2026-03-27', latest_label_available_at='2026-03-27T15:35:00+09:00', cohorts=cells)
        return r, other, p

    def test_keeps_score_distinct_from_final_value(self):
        r,o,p=self.values(); v=M.reconcile_value(r,o,'frozen_core',p,8.)
        self.assertEqual((v.selection_score,v.final['baseline'],v.final['combined']),(8.,7.,1.))
        self.assertTrue(v.retained)

    def test_supported_stress_veto(self):
        r,o,p=self.values(); p['cohorts']['combined/frozen_core/q0.5/self1_other1/top2']['offset_bps']=-3
        self.assertFalse(M.reconcile_value(r,o,'frozen_core',p,8.).retained)

    def test_unsupported_identity_cannot_fake_support(self):
        r,o,p=self.values()
        for c in p['cohorts'].values(): c.update(supported=False,offset_bps=0)
        self.assertFalse(M.reconcile_value(r,o,'frozen_core',p,8.).supported)
        next(iter(p['cohorts'].values()))['offset_bps']=-1
        with self.assertRaises(ValueError): M.reconcile_value(r,o,'frozen_core',p,8.)

    def test_future_clock_and_counterpart_rejected(self):
        for field,value in [('valid_from','2026-04-02'),('available_at','2026-04-01T09:35:00+09:00'),('action_id','wrong')]:
            r,o,p=self.values();o[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): M.reconcile_value(r,o,'frozen_core',p,8.)

    def test_mature_label_clock(self):
        r,o,p=self.values();p['latest_label_available_at']='2026-03-31T00:00:00+09:00'
        with self.assertRaises(ValueError): M.reconcile_value(r,o,'frozen_core',p,8.)

    def test_integer_quantity_features(self):
        r,_,_=self.values();r['features']={k:1. for k in M.SCALED_FEATURES}
        r['features'].update(action_notional_equity=.429825,action_quantity_scale=1.)
        f=M.action_features(r,275)
        self.assertAlmostEqual(f['action_quantity_last_volume'],137/275)
        self.assertNotEqual(f['action_quantity_last_volume'],.5)
        with self.assertRaises(ValueError): M.action_features(r,276)

    def test_lifetime_is_joined_only_once_at_selected_quantity(self):
        r,_,_=self.values();r.update(period='oos',selected={'control':{'action_id':r['action_id']}},actual={
            'control/baseline':[dict(position_id='X/lifetime',route='first30_open_anchor',entry_day=r['day'],qty=137,net_pnl_krw=100.)]})
        seen=set(); self.assertEqual(len(M.unique_position_links(r,'control','baseline',seen)),1)
        with self.assertRaises(ValueError): M.unique_position_links(r,'control','baseline',seen)
        r['selected']={};self.assertEqual(M.unique_position_links(r,'control','baseline',seen),[])


if __name__ == '__main__': unittest.main()
