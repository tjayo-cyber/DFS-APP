import numpy as np
from sklearn.linear_model import LogisticRegression

def train_meta_model(rf, gb, X, y):
    rf_pred = rf.predict_proba(X)[:, 1]
    gb_pred = gb.predict_proba(X)[:, 1]
    X_meta = np.column_stack((rf_pred, gb_pred))
    meta = LogisticRegression()
    meta.fit(X_meta, y)
    return meta