import importlib.util
from pathlib import Path
import sys, unittest
from datetime import date,timedelta

SPEC=importlib.util.spec_from_file_location('funded_calibration_test',Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/funded_decision_calibration_v1.py')
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)

class Tests(unittest.TestCase):
    def data(self):
        sessions=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(100)]
        records=[]
        for i,day in enumerate(sessions[:80]):
            for name in ['A','B']:
                records.append(dict(id=day+'/'+name,day=day,available_at=day+'T09:30:00+09:00',
                    mature_at=day+'T15:35:00+09:00',forecast_valid_from=sessions[0],target_valid=True,
                    prediction=10.+i,isolated=20.+i,funded=-10.,origin='frozen_core',quantity_scale=1.))
        return sessions,records
    def test_future_and_embargo_labels_do_not_change_fit(self):
        s,r=self.data();p=M.fit(r,s[70],s,'funded')
        for x in r:
            if x['day']>=s[69]:x['funded']=1e100
        self.assertEqual(p,M.fit(r,s[70],s,'funded'))
        self.assertEqual(p['mature_sessions'],69)
    def test_support_not_inflated(self):
        s,r=self.data();self.assertFalse(M.fit(r,s[60],s,'funded')['cells'])
        with self.assertRaises(ValueError):M.fit(r+[r[0]],s[70],s,'funded')
    def test_monotone_bounded_slope_and_prior_domain(self):
        s,r=self.data();p=M.fit(r,s[70],s,'funded');c=next(iter(p['cells'].values()))
        self.assertTrue(-1<=c['slope_delta']<=0)
        x={**r[0],'day':s[70],'prediction':1e9}
        self.assertFalse(M.predict(x,p)['supported'])
    def test_same_training_cohort_for_target_contrast(self):
        s,r=self.data();a=M.fit(r,s[70],s,'funded');b=M.fit(r,s[70],s,'isolated')
        self.assertEqual(next(iter(a['cells'].values()))['training_ids'],next(iter(b['cells'].values()))['training_ids'])
    def test_requires_both_support_and_negative_value(self):
        self.assertFalse(M.paired_abstention({'supported':False,'value':-1},{'supported':True,'value':2}))
        self.assertTrue(M.paired_abstention({'supported':True,'value':-1},{'supported':True,'value':2}))
    def test_no_fill_is_retained_and_matures_at_close(self):
        e={'order_id':'o','decision_available_at':'2025-01-01T09:30:00+09:00','action':{'qty':2,'metadata':{'entry_route':'first30_open_anchor','entry_price_ref':100}}}
        t=M.native_target(e,[],[{'order_id':'o','event':'reject'}],'2025-01-01')
        self.assertEqual(t['value'],0);self.assertEqual(t['mature_at'],'2025-01-01T15:35:00+09:00')
        with self.assertRaises(ValueError):M.native_target(e,[],[],'2025-01-01')
    def test_filled_target_uses_exact_planned_notional_and_costs(self):
        e={'order_id':'o','decision_available_at':'2025-01-01T09:30:00+09:00','action':{'qty':2,'metadata':{'entry_route':'first30_open_anchor','entry_price_ref':100}}}
        p={'entry_order_ids':['o'],'position_id':'p','entry_day':'2025-01-01','live_qty':0,'qty':1,'exit_fill_time':'2025-01-01T10:00:00+09:00','net_pnl_krw':-2}
        t=M.native_target(e,[p],[],'2025-01-01');self.assertEqual(t['value'],-100.)
        p['entry_order_ids']=['o','retry']
        with self.assertRaises(ValueError):M.native_target(e,[p],[],'2025-01-01')

if __name__=='__main__':unittest.main()
