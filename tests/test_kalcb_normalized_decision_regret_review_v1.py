"""Independent normalization evidence must reject plausible poisoned receipts."""
import copy
import importlib.util
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
M=load('norm_reader_fixture_model',Path('backtests/strategies/kalcb/normalized_decision_regret_v1.py'))
R=load('norm_independent_reader',Path('scripts/kalcb_normalized_decision_regret_review_v1.py'))

class Tests(unittest.TestCase):
    def fixture(self):
        days=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(90)]
        rows=[]
        for day in days[:80]:
            for k,y in enumerate([-2.,4.,-1.,15.]):
                rows.append(dict(id=f'{day}/{k}',base_id=f'{day}/{k}',day=day,
                    available_at=day+'T09:30:00+09:00',mature_at=day+'T15:35:00+09:00',target_valid=True,
                    value=y,x=np.array([1.,float(k-1)]),physical=[1,2,3,4],low=.5,high=1.))
        p=M.fit(rows,days[70],days)
        training=[r for r in rows if r['id'] in set(p['training_ids'])]
        return np.array([r['x'] for r in training]),np.array(p['base_weights']),np.array([r['value'] for r in training]),p,rows,days
    def test_independent_reader_accepts_contract(self):
        x,w,y,p,_,_=self.fixture();r=R.verify_normalization(x,w,y,p)
        self.assertAlmostEqual(r['target_df'],r['zero_score_df'],places=8)
        self.assertLessEqual(r['fitted_local_df'],r['zero_score_df']+1e-8)
    def test_consistently_poisoned_penalty_rejected(self):
        x,w,y,p,_,_=self.fixture();p['penalty']*=2;p['normalization']['penalty']=p['penalty']
        with self.assertRaises((ValueError,AssertionError)):R.verify_normalization(x,w,y,p)
    def test_poisoned_weight_and_value_units_rejected(self):
        x,w,y,p,_,_=self.fixture();p['weights'][0]*=2
        with self.assertRaises((ValueError,AssertionError)):R.verify_normalization(x,w,y,p)
        x,w,y,p,_,_=self.fixture();p['training_absolute_value_scale']*=2
        with self.assertRaises((ValueError,AssertionError)):R.verify_normalization(x,w,y,p)
    def test_poisoned_dimensions_and_df_rejected(self):
        x,w,y,p,_,_=self.fixture();q=copy.deepcopy(p);q['normalization']['training_rows']+=1
        with self.assertRaises((ValueError,AssertionError)):R.verify_normalization(x,w,y,q)
        q=copy.deepcopy(p);q['normalization']['zero_score_df']+=.1
        with self.assertRaises((ValueError,AssertionError)):R.verify_normalization(x,w,y,q)
    def test_future_features_cannot_change_normalization_or_fit(self):
        _,_,_,p,rows,days=self.fixture()
        for r in rows:
            if r['day']>=days[69]:r['x']=np.array([np.nan,np.inf]);r['value']=-1e30
        self.assertEqual(p,M.fit(rows,days[70],days))
    def test_new_and_old_scores_have_separate_contracts(self):
        row={'normalized_regret_prediction':{'score':-2},'regret_prediction':{'score':3},
             'predictions':{'paired_state':{'value':12},'pair_mean':{'value':5}}}
        self.assertEqual([R.decision_score(row,k) for k in ['pair_mean','paired_state','decision_regret','normalized_decision_regret']],[5,12,3,-2])

if __name__=='__main__':unittest.main()
