"""Paired executed quantity increments, learned from mature stock-days only."""
from collections import defaultdict
import numpy as np
from . import action_value as av, continuation_value as cv

CHECKPOINT_COLUMNS=('checkpoint_return','checkpoint_above_or','checkpoint_vwap_distance',
    'checkpoint_volume_ratio','checkpoint_up_share','checkpoint_efficiency',
    'checkpoint_sector_return','checkpoint_sector_breadth')
QUANTITIES=(.25,.5,1.)


class PairedPayoff:
    def __init__(self,columns=cv.COLUMNS):self.columns=tuple(columns)

    def fit(self,x,values,weights):
        from sklearn.linear_model import Ridge
        self.map=cv.StateMap(self.columns).fit(x)
        increments=np.diff(np.column_stack([np.zeros(len(values)),values]),axis=1)
        self.regressor=Ridge(alpha=120).fit(self.map.transform(x),increments,sample_weight=weights)
        return self

    def predict(self,x):return np.cumsum(self.regressor.predict(self.map.transform(x)),axis=1)


def fit_forecast(rows,labels,sessions,first,through,*,columns=cv.COLUMNS,min_sessions=60):
    indices,weights=av.training_indices(rows,labels,sessions,first,av.ModelSpec('paired'),min_sessions=min_sessions)
    train=defaultdict(dict);valid=defaultdict(dict);weight={}
    for i,w in zip(indices,weights):
        r=rows[i];train[r['base_opportunity_id']][r['features']['action_quantity_scale']]=r
        weight[r['base_opportunity_id']]=w
    train={k:v for k,v in train.items() if set(v)==set(QUANTITIES)}
    if len({v[1.]['trade_date'] for v in train.values()})<min_sessions:raise ValueError('Insufficient mature paired sessions')
    for r in rows:
        if first<=r['trade_date']<=through:valid[r['base_opportunity_id']][r['features']['action_quantity_scale']]=r
    if any(set(v)!=set(QUANTITIES) for v in valid.values()):raise ValueError('Missing forecast quantity')
    keys=sorted(train);vkeys=sorted(valid)
    y=np.asarray([[labels[train[k][q]['opportunity_id']]['net_value_equity_bps'] for q in QUANTITIES] for k in keys])
    if not np.isfinite(y).all():raise ValueError('Nonfinite paired target')
    model=PairedPayoff(columns).fit(cv.matrix([train[k][1.] for k in keys],columns),y,[weight[k] for k in keys])
    values=model.predict(cv.matrix([valid[k][1.] for k in vkeys],columns))
    predicted=[(valid[k][q],float(values[i,j])) for i,k in enumerate(vkeys) for j,q in enumerate(QUANTITIES)]
    used=[train[k][q] for k in keys for q in QUANTITIES]
    proof={'name':'paired_increments','window':'expanding','valid_from':first,'valid_through':through,
        'fit_at':first+'T00:00:00+09:00','train_from':min(r['trade_date'] for r in used),
        'train_through':max(r['trade_date'] for r in used),
        'label_available_through':max(labels[r['opportunity_id']]['label_available_at'] for r in used),
        'train_rows':len(used),'train_stock_days':len(keys),'feature_columns':columns,
        'training_keys_hash':av.ev.signature([r['opportunity_id'] for r in used]),
        'training_target_hash':av.ev.signature(y.tolist()),'target':'V(.25), V(.5)-V(.25), V(1)-V(.5); cash=0'}
    return model,predicted,proof
