#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage0/core_model_lgbm.py
Core ML (LightGBM) utilities.
Only model code here: fit + predict_proba.
"""

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb


DEFAULT_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=31,
    min_data_in_leaf=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMClassifier:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["random_state"] = int(seed)

    model = lgb.LGBMClassifier(**p)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]