"""Causal continuation-state representations and quantity-dependent payoff models.

Feature transformations, state cutoffs and estimators are fitted on the training
prefix only. Execution outcomes and symbol identity are never predictor inputs.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import math
import numpy as np

from . import action_value as av

PATH_FEATURES = ('path_efficiency','path_up_share','path_above_vwap_share',
    'path_last15_return','path_first15_return','path_return_acceleration',
    'path_late_volume_share','path_close_vs_early_high','path_drawdown',
    'prior_afternoon_return','prior_close_location','prior_above_vwap',
    'prior_late_volume_share','prior_intraday_return','prior_path_available')
INTERACTIONS = (
    ('daily_return_20d','sector_intraday_breadth'),
    ('daily_return_5d','sector_intraday_breadth'),
    ('first30_ret','first30_rel_volume_log'),
    ('first30_ret','sector_intraday_breadth'),
    ('first30_ret','path_efficiency'),
    ('first30_vwap_ret','path_above_vwap_share'),
    ('daily_return_20d','path_efficiency'),
    ('first30_range_atr','path_return_acceleration'),
    ('stock_sector_daily_ret5_spread','sector_intraday_ret'),
    ('prior_afternoon_return','first30_ret'),
    ('first30_gap','path_last15_return'),
    ('flow_combined_5d','first30_rel_volume_log'),
)
SPLINE_FEATURES = ('first30_ret','first30_vwap_ret','first30_gap','first30_rel_volume_log',
    'first30_range_atr','daily_return_5d','daily_return_20d','daily_return_60d',
    'daily_atr_pct','stock_sector_daily_ret5_spread','sector_intraday_ret',
    'sector_intraday_breadth','path_efficiency','path_return_acceleration',
    'prior_afternoon_return','flow_combined_5d')
COLUMNS = tuple(k for k in av.predictor_columns(True)
    if k not in {'action_avwap_distance','action_is_anchor','action_is_pullback'}) + PATH_FEATURES


def opening_path(bars, available_at):
    cutoff=datetime.fromisoformat(available_at)
    prefix=sorted((b for b in bars if datetime.fromisoformat(b['timestamp'])+timedelta(minutes=5)<=cutoff),
                  key=lambda b:b['timestamp'])
    if not prefix:raise ValueError('No completed opening prefix')
    if any(datetime.fromisoformat(b['timestamp']).date()!=cutoff.date() for b in prefix):
        raise ValueError('Opening prefix crosses sessions')
    closes=np.asarray([b['close'] for b in prefix],dtype=float)
    opens=np.asarray([b['open'] for b in prefix],dtype=float)
    volumes=np.asarray([b['volume'] for b in prefix],dtype=float)
    typical=np.asarray([(b['high']+b['low']+b['close'])/3 for b in prefix])
    vwap=np.cumsum(typical*volumes)/np.maximum(1,np.cumsum(volumes))
    changes=np.diff(np.r_[opens[0],closes]);middle=max(1,len(prefix)//2)
    early=closes[middle-1]/opens[0]-1;late=closes[-1]/closes[middle-1]-1
    return {'path_efficiency':float(changes.sum()/max(1e-12,np.abs(changes).sum())),
        'path_up_share':float(np.mean(closes>=opens)),
        'path_above_vwap_share':float(np.mean(closes>=vwap)),
        'path_last15_return':float(late),'path_first15_return':float(early),
        'path_return_acceleration':float(late-early),
        'path_late_volume_share':float(volumes[middle:].sum()/max(1,volumes.sum())),
        'path_close_vs_early_high':float(closes[-1]/max(b['high'] for b in prefix[:middle])-1),
        'path_drawdown':float(np.min(closes/np.maximum.accumulate(np.r_[opens[0],closes])[1:]-1))}


def prior_path(bars, trade_date):
    # Bars are deliberately restricted again here even if the caller supplied
    # a correct previous-session query. A poisoned current/future row is ignored.
    prior=[b for b in bars if b['timestamp'][:10]<trade_date]
    if not prior:return {k:0.0 if k=='prior_path_available' else None for k in PATH_FEATURES if k.startswith('prior_')}
    last=max(b['timestamp'][:10] for b in prior)
    prior=sorted((b for b in prior if b['timestamp'].startswith(last)),key=lambda b:b['timestamp'])
    afternoon=[b for b in prior if b['timestamp'][11:16]>='14:00']
    if not afternoon or prior[-1]['timestamp'][11:16]<'15:15':return prior_path([],trade_date)
    close=prior[-1]['close'];high=max(b['high'] for b in prior);low=min(b['low'] for b in prior)
    volume=sum(b['volume'] for b in prior)
    vwap=sum((b['high']+b['low']+b['close'])/3*b['volume'] for b in prior)/max(1,volume)
    return {'prior_afternoon_return':close/afternoon[0]['open']-1,
        'prior_close_location':(close-low)/max(1e-9,high-low),'prior_above_vwap':close/max(1e-9,vwap)-1,
        'prior_late_volume_share':sum(b['volume'] for b in afternoon)/max(1,volume),
        'prior_intraday_return':close/prior[0]['open']-1,'prior_path_available':1.0}


def matrix(rows, columns=COLUMNS):
    result=np.empty((len(rows),len(columns)),dtype=float)
    for i,row in enumerate(rows):
        for j,key in enumerate(columns):
            value=row['features'].get(key)
            result[i,j]=float(value) if value is not None else np.nan
    return result


class StateMap:
    """Saturated marginal shapes and bounded, declared economic interactions."""
    def __init__(self, columns=COLUMNS, interactions=INTERACTIONS):
        self.columns=tuple(columns)
        self.interactions=tuple(interactions)

    def fit(self,x):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, SplineTransformer
        self.impute=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        filled=self.impute.transform(x)
        self.scale=StandardScaler().fit(filled)
        columns=getattr(self,'columns',COLUMNS)
        self.indices=[columns.index(k) for k in SPLINE_FEATURES if np.ptp(filled[:,columns.index(k)])>1e-10]
        self.spline=SplineTransformer(n_knots=4,degree=2,knots='quantile',extrapolation='constant',include_bias=False)
        if self.indices:self.spline.fit(filled[:,self.indices])
        return self

    def transform(self,x):
        filled=self.impute.transform(x);z=np.clip(self.scale.transform(filled),-5,5)
        columns=getattr(self,'columns',COLUMNS)
        inter=np.column_stack([z[:,columns.index(a)]*z[:,columns.index(b)] for a,b in getattr(self,'interactions',INTERACTIONS)])
        blocks=[z,inter]
        if self.indices:blocks.append(self.spline.transform(filled[:,self.indices]))
        return np.column_stack(blocks)


class StatePayoff:
    def __init__(self,kind='interaction',columns=COLUMNS,interactions=INTERACTIONS):
        if kind not in {'interaction','mixture'}:raise ValueError('Undeclared continuation model')
        self.kind=kind
        self.columns=tuple(columns)
        self.interactions=tuple(interactions)

    def fit(self,x,y,weights):
        from sklearn.linear_model import Ridge,LogisticRegression
        self.features=StateMap(self.columns,self.interactions).fit(x);z=self.features.transform(x)
        q=x[:,self.columns.index('action_quantity_scale')]
        if not np.isfinite(q).all() or (q<=0).any() or (q>1).any():raise ValueError('Invalid training quantity')
        target=np.asarray(y)/q;w=np.asarray(weights)*q*q
        if self.kind=='interaction':
            self.regressor=Ridge(alpha=120).fit(z,target,sample_weight=w)
            return self
        full=target[q==1];positive=full[full>0]
        self.tail_cutoff=float(np.quantile(positive,.8)) if len(positive)>=20 else max(1,float(np.quantile(target,.9)))
        states=np.where(target<=0,0,np.where(target<=self.tail_cutoff,1,2))
        self.classes=np.unique(states);self.classifier=None
        if len(self.classes)>1:
            self.classifier=LogisticRegression(C=.03,max_iter=1500,random_state=7).fit(z,states,sample_weight=w)
        self.payoffs={};self.means={}
        for state in self.classes:
            use=states==state;self.means[int(state)]=float(np.average(target[use],weights=w[use]))
            self.payoffs[int(state)]=Ridge(alpha=240).fit(z[use],target[use],sample_weight=w[use]) if use.sum()>=40 else None
        return self

    def predict(self,x):
        q=x[:,getattr(self,'columns',COLUMNS).index('action_quantity_scale')]
        if not np.isfinite(q).all() or (q<0).any() or (q>1).any():raise ValueError('Invalid forecast quantity')
        z=self.features.transform(x)
        if self.kind=='interaction':return q*self.regressor.predict(z)
        probabilities=self.classifier.predict_proba(z) if self.classifier is not None else np.ones((len(z),1))
        value=np.zeros(len(z))
        for i,state in enumerate(self.classes):
            model=self.payoffs[int(state)]
            payoff=model.predict(z) if model is not None else np.full(len(z),self.means[int(state)])
            payoff=np.minimum(payoff,0) if state==0 else np.clip(payoff,0,self.tail_cutoff) if state==1 else np.maximum(payoff,self.tail_cutoff)
            value+=probabilities[:,i]*payoff
        return q*value


def fit_forecast(rows,labels,sessions,first,through,kind='interaction',window='expanding',
                 *,columns=COLUMNS,interactions=INTERACTIONS):
    spec=av.ModelSpec(kind,window=window)
    indices,weights=av.training_indices(rows,labels,sessions,first,spec)
    train=[rows[i] for i in indices];valid=[r for r in rows if first<=r['trade_date']<=through]
    y=np.asarray([labels[r['opportunity_id']]['net_value_equity_bps'] for r in train])
    if not np.isfinite(y).all():raise ValueError('Nonfinite executed target')
    model=StatePayoff(kind,columns,interactions).fit(matrix(train,columns),y,weights)
    proof={'name':kind,'window':window,'valid_from':first,'valid_through':through,'fit_at':first+'T00:00:00+09:00',
        'train_from':min(r['trade_date'] for r in train),'train_through':max(r['trade_date'] for r in train),
        'label_available_through':max(labels[r['opportunity_id']]['label_available_at'] for r in train),
        'train_rows':len(train),'train_stock_days':len({r['base_opportunity_id'] for r in train}),
        'training_keys_hash':av.ev.signature([r['opportunity_id'] for r in train]),
        'training_target_hash':av.ev.signature(y.tolist()),'feature_columns':columns}
    return model,list(zip(valid,map(float,model.predict(matrix(valid,columns))))),proof
