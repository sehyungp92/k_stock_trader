import importlib.util,sys,unittest
from pathlib import Path
from datetime import date,timedelta
import numpy as np
S=importlib.util.spec_from_file_location('paired_quantity_under_test',Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/paired_quantity_value_v1.py')
M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
class Tests(unittest.TestCase):
    def test_potential_differences_telescope(self):
        z=np.array([[1.,2.],[3.,4.],[5.,6.]])
        v=M.potential_basis([.25,.5,1.],z)
        np.testing.assert_allclose((v[2]-v[1])+(v[1]-v[0]),v[2]-v[0])
        with self.assertRaises(ValueError):M.potential_basis([.3,.5,1.],z)
    def rows(self):
        s=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(90)];rs=[]
        for i,day in enumerate(s[:80]):
            for name in ('A','B'):
                for j,(low,high) in enumerate(M.PAIRS):
                    rs.append(dict(id=day+'/'+name+str(j),base_id=day+'/'+name,day=day,available_at=day+'T09:30:00+09:00',mature_at=day+'T15:35:00+09:00',
                                   target_valid=True,value=i-30.,x=np.array([high-low,i/80]),physical=[1.,2.,3.,4.],low=low,high=high))
        return s,rs
    def test_count_dependence_and_embargo(self):
        s,r=self.rows();p=M.fit(r,s[70],s)
        self.assertEqual(p['issuer_days'],138);self.assertEqual(p['mature_sessions'],69)
        self.assertAlmostEqual(sum(p['weights']),138)
        for x in r:
            if x['day']>=s[69]:x['value']=1e20
        self.assertEqual(p,M.fit(r,s[70],s))
    def test_unsupported_and_duplicate(self):
        s,r=self.rows();self.assertFalse(M.fit(r,s[60],s)['supported'])
        with self.assertRaises(ValueError):M.fit(r+[r[0]],s[70],s)
    def test_pair_identity_and_integer_alias(self):
        label={s:{'label_valid':True,'label_available_at':'2025-01-01T15:35:00+09:00','net_value_equity_bps':1} for s in ('baseline','combined')}
        a=dict(q=.25,base_id='d/X',day='2025-01-01',available_at='2025-01-01T09:30:00+09:00',source='same',qty=1,notional=1,labels=label,features={k:1. for k in M.PHYSICAL})
        b={**a,'q':.5};self.assertIsNone(M.make_contrast(a,b,[1],[2],'baseline'))
        b.update(qty=2,notional=2,source='different')
        with self.assertRaises(ValueError):M.make_contrast(a,b,[1],[2],'baseline')
    def test_domain_abstention(self):
        s,r=self.rows();p=M.fit(r,s[70],s);x={**r[0],'day':s[70]}
        self.assertTrue(M.predict(x,p,'paired_state')['supported'])
        x['physical']=[99.,2.,3.,4.];self.assertFalse(M.predict(x,p,'paired_state')['supported'])
if __name__=='__main__':unittest.main()
