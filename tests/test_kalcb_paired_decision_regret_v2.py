import importlib.util,sys,unittest
from pathlib import Path
from datetime import date,timedelta
import numpy as np
S=importlib.util.spec_from_file_location('regret_test',Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/paired_decision_regret_v2.py');M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)

class Tests(unittest.TestCase):
    def records(self):
        dates=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(90)];rs=[]
        for day in dates[:80]:
            for k,value in enumerate([-1.,-1.,-1.,20.]):
                rs.append(dict(id=day+'/'+str(k),base_id=day+'/'+str(k),day=day,available_at=day+'T09:30:00+09:00',
                    mature_at=day+'T15:35:00+09:00',target_valid=True,value=value,x=np.array([1.]),physical=[1,2,3,4],low=.5,high=1.))
        return dates,rs
    def test_rare_winner_cost_changes_majority_decision(self):
        days,rs=self.records();p=M.fit(rs,days[70],days)
        self.assertGreater(p['coefficients'][0],0);self.assertEqual(p['mature_sessions'],69)
        self.assertAlmostEqual(sum(p['weights']),p['issuer_days'])
    def test_objective_gradient(self):
        x=np.array([[1.,2.],[-2.,3.],[4.,1.]]);b=np.array([.2,-.1]);y=np.array([1.,0.,1.]);w=np.array([3.,2.,7.])
        _,g=M.objective(b,x,y,w)
        for k in range(2):
            d=np.zeros(2);d[k]=1e-6
            numeric=(M.objective(b+d,x,y,w)[0]-M.objective(b-d,x,y,w)[0])/2e-6
            self.assertAlmostEqual(g[k],numeric,places=6)
    def test_embargo_future_outcomes_never_used(self):
        days,rs=self.records();a=M.fit(rs,days[70],days)
        for r in rs:
            if r['day']>=days[69]:r['value']=-1e20
        self.assertEqual(a,M.fit(rs,days[70],days))
    def test_unsupported_duplicate_domain(self):
        days,rs=self.records();self.assertFalse(M.fit(rs,days[60],days)['supported'])
        with self.assertRaises(ValueError):M.fit(rs+[rs[0]],days[70],days)
        p=M.fit(rs,days[70],days);r={**rs[0],'day':days[70],'physical':[999,2,3,4]}
        self.assertFalse(M.predict(r,p)['supported'])
    def test_outcome_unit_invariance(self):
        days,rs=self.records();a=M.fit(rs,days[70],days)
        for r in rs:r['value']*=10000
        b=M.fit(rs,days[70],days);np.testing.assert_allclose(a['coefficients'],b['coefficients'],rtol=1e-8,atol=1e-10)
    def test_high_dynamic_basis_converges(self):
        days,rs=self.records()
        for i,r in enumerate(rs):r['x']=np.array([1.,1000.*(i%7-3),float(i%3)])
        p=M.fit(rs,days[70],days)
        self.assertLess(p['optimization']['max_gradient'],1e-4)
if __name__=='__main__':unittest.main()
