"""Small, predeclared action-value learners and session-aware training windows."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import numpy as np

from . import executed_value as ev

VERSION = "kalcb-action-value-v1"
ACTION_FEATURES = (
    "action_signal_cpr", "action_avwap_distance", "action_quality_votes",
    "action_quality_margin", "action_risk_fraction", "action_notional_equity",
    "action_quantity_last_volume", "action_quantity_opening_volume",
    "action_low_volume_exit_slices", "action_momentum_score", "action_is_anchor",
    "action_is_pullback", "action_quantity_scale",
)
FEATURES = tuple(ev.FEATURES) + ACTION_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    learner: str = "ridge"
    window: str = "expanding"
    rich: bool = False
    conditional: bool = True


# Declare before inspecting the new results. No symbol/sector-specific searches.
SPECS = (
    ModelSpec("unconditional_ridge", conditional=False),
    ModelSpec("ready_ridge"),
    ModelSpec("ready_hgb", "hgb"),
    ModelSpec("ready_hurdle", "hurdle"),
    ModelSpec("ready_hgb_rich", "hgb", rich=True),
    ModelSpec("ready_hgb_rolling120", "hgb", "rolling120"),
    ModelSpec("ready_hgb_decay60", "hgb", "decay60"),
)


def predictor_columns(rich=False):
    if not rich:
        return FEATURES
    from scripts.kalcb_r_capture_optimizer import FEATURE_GROUPS
    return tuple(dict.fromkeys((*FEATURES, *FEATURE_GROUPS["all_context"])))


def matrix(rows, columns):
    allowed = set(predictor_columns(True))
    if set(columns) - allowed:
        raise ValueError("Predictor is outside the causal allowlist")
    return np.asarray([[float(r["features"][k]) if r["features"].get(k) is not None else np.nan
                        for k in columns] for r in rows], dtype=float)


def training_indices(rows, labels, sessions, first, spec, *, min_sessions=60):
    allowed = sessions[:max(0, sessions.index(first) - 1)]
    if spec.window == "rolling120":
        allowed = allowed[-120:]
    elif spec.window not in {"expanding", "decay60"}:
        raise ValueError("Undeclared training window")
    allowed_set = set(allowed)
    cutoff = datetime.fromisoformat(first + "T00:00:00+09:00")
    indices = []
    for i, row in enumerate(rows):
        label = labels.get(row["opportunity_id"])
        if not label or not label["label_valid"] or not label.get("label_available_at"):
            continue
        if row["trade_date"] not in allowed_set or datetime.fromisoformat(label["label_available_at"]) >= cutoff:
            continue
        if spec.conditional and not row["action"]["ready"]:
            continue
        indices.append(i)
    if len({rows[i]["trade_date"] for i in indices}) < min_sessions:
        raise ValueError("Insufficient mature action training sessions")
    age = {d: len(allowed)-1-j for j, d in enumerate(allowed)}
    weights = np.asarray([2**(-age[rows[i]["trade_date"]]/60) if spec.window == "decay60" else 1.0
                          for i in indices])
    # Common weight scale keeps regularisation comparable across window policies.
    weights *= len(weights) / weights.sum()
    return indices, weights


def learner(kind, count):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "ridge":
        estimator = Ridge(alpha=12)
    elif kind == "hgb":
        estimator = HistGradientBoostingRegressor(max_iter=120, learning_rate=.04,
            max_leaf_nodes=7, max_depth=3, min_samples_leaf=max(30, min(100, count//20)),
            l2_regularization=10, early_stopping=False, random_state=7)
    else:
        raise ValueError("Undeclared learner")
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("model", estimator)])


class HurdleModel:
    """P(any fill) times expected complete net cash payoff given a fill."""
    def fit(self, x, y, filled, weights):
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        self.probability = float(np.average(filled, weights=weights))
        self.fill_model = None
        if len(set(filled)) > 1:
            self.fill_model = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()), ("model", LogisticRegression(C=.1, max_iter=1000, random_state=7))])
            self.fill_model.fit(x, filled, model__sample_weight=weights)
        self.payoff = None
        if filled.any():
            self.payoff = learner("ridge", int(filled.sum()))
            self.payoff.fit(x[filled], y[filled], model__sample_weight=weights[filled])
        return self

    def predict(self, x):
        p = self.fill_model.predict_proba(x)[:, 1] if self.fill_model is not None else np.full(len(x), self.probability)
        return p * self.payoff.predict(x) if self.payoff is not None else np.zeros(len(x))


class ScaledRidgeModel:
    """Quantity-aware value q*g(context,q), with squared-error weights in cash units.

    The per-size payoff may depend on quantity, so partial fills and liquidation
    costs need not scale linearly. Quantity zero is always worth zero. Training
    still uses separately executed outcomes at each nonzero quantity.
    """
    def __init__(self, quantity_column):
        self.quantity_column = quantity_column

    def fit(self, x, y, weights):
        q = x[:, self.quantity_column]
        if not np.isfinite(q).all() or (q <= 0).any() or (q > 1).any():
            raise ValueError('Scaled ridge requires a valid nonzero quantity')
        self.payoff = learner('ridge', len(y))
        self.payoff.fit(x, y/q, model__sample_weight=weights*q*q)
        return self

    def predict(self, x):
        return x[:, self.quantity_column]*self.payoff.predict(x)


class FeasibilityModel:
    """Forecast a physical-fill/deadline failure, including invalid cash labels."""
    def fit(self, x, unsafe):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # A failure-free prefix does not establish a zero probability of failure.
        self.constant = float((unsafe.sum()+1)/(len(unsafe)+100))
        self.model = None
        if len(set(unsafe)) > 1:
            self.model = Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                ('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=2000,random_state=7))])
            self.model.fit(x,unsafe)
        return self

    def predict(self, x):
        return self.model.predict_proba(x)[:,1] if self.model is not None else np.full(len(x),self.constant)


def fit_feasibility(rows, outcomes, sessions, first, through, *, min_sessions=60):
    """Fit only failures already observable before this fold's information cutoff."""
    allowed=set(sessions[:max(0,sessions.index(first)-1)])
    cutoff=datetime.fromisoformat(first+'T00:00:00+09:00')
    train=[r for r in rows if r['trade_date'] in allowed and r['opportunity_id'] in outcomes
           and datetime.fromisoformat(outcomes[r['opportunity_id']]['risk_available_at'])<cutoff]
    if len({r['trade_date'] for r in train})<min_sessions:
        raise ValueError('Insufficient mature feasibility training sessions')
    valid=[r for r in rows if first<=r['trade_date']<=through]
    y=np.asarray([outcomes[r['opportunity_id']]['unsafe'] for r in train],dtype=bool)
    model=FeasibilityModel().fit(matrix(train,FEATURES),y)
    values=model.predict(matrix(valid,FEATURES))
    proof={'name':'execution_failure_probability','valid_from':first,'valid_through':through,
        'fit_at':cutoff.isoformat(),'train_from':min(r['trade_date'] for r in train),
        'train_through':max(r['trade_date'] for r in train),
        'label_available_through':max(outcomes[r['opportunity_id']]['risk_available_at'] for r in train),
        'train_rows':len(train),'unsafe_rows':int(y.sum()),'feature_columns':FEATURES,
        'training_keys_hash':ev.signature([r['opportunity_id'] for r in train]),
        'training_target_hash':ev.signature(y.tolist()),
        'target':'physical-fill failure, missed liquidation deadline or unresolved inventory in either scenario',
        'selection':'past-only expanding feasibility fit; invalid cash outcomes retained as failure observations'}
    return model,list(zip(valid,map(float,values))),proof


def fit_forecast(rows, labels, sessions, first, through, spec, *, min_sessions=60):
    indices, weights = training_indices(rows, labels, sessions, first, spec, min_sessions=min_sessions)
    train = [rows[i] for i in indices]
    valid = [r for r in rows if first <= r["trade_date"] <= through]
    columns = predictor_columns(spec.rich)
    # The unconditional control exactly retains the previous 20 predictors.
    if not spec.conditional:
        columns = tuple(ev.FEATURES)
    x, xv = matrix(train, columns), matrix(valid, columns)
    y = np.asarray([labels[r["opportunity_id"]]["net_value_equity_bps"] for r in train])
    if not np.isfinite(y).all():
        raise ValueError("Invalid executed value")
    if spec.learner == "scaled_ridge":
        model = ScaledRidgeModel(columns.index('action_quantity_scale')).fit(x, y, weights)
    elif spec.learner == "hurdle":
        filled = np.asarray([labels[r["opportunity_id"]]["buy_qty"] > 0 for r in train])
        model = HurdleModel().fit(x, y, filled, weights)
    else:
        model = learner(spec.learner, len(train))
        model.fit(x, y, model__sample_weight=weights)
    predictions = model.predict(xv)
    proof = {"name": spec.name, "valid_from": first, "valid_through": through,
        "fit_at": first + "T00:00:00+09:00", "train_from": min(r["trade_date"] for r in train),
        "train_through": max(r["trade_date"] for r in train),
        "label_available_through": max(labels[r["opportunity_id"]]["label_available_at"] for r in train),
        "train_rows": len(train), "train_sessions": len({r["trade_date"] for r in train}),
        "training_keys_hash": ev.signature([r["opportunity_id"] for r in train]),
        "training_target_hash": ev.signature(y.tolist()), "training_weights_hash": ev.signature(weights.tolist()),
        "feature_columns": columns, "window": spec.window, "conditional": spec.conditional,
        "selection": "predeclared_development_comparison"}
    return model, list(zip(valid, map(float, predictions))), proof


def continuation_score(row):
    """Untuned OLR-inspired opening control; no afternoon thresholds transferred."""
    f = row["features"]
    return (f["first30_ret"] + .25*f["first30_vwap_ret"] + .05*max(f["first30_gap"], 0)
        + .001*min(math.expm1(f["first30_rel_volume_log"]), 5)
        + .002*f["first30_close_location"]
        + .25*(f["first30_ret"]-f["sector_intraday_ret"]))
