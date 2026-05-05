import numpy as np

def predict(rf, gb, meta, scaler, input_data):
    data = scaler.transform(input_data)
    rf_p = rf.predict_proba(data)[:, 1]
    gb_p = gb.predict_proba(data)[:, 1]
    meta_input = np.column_stack((rf_p, gb_p))
    return meta.predict_proba(meta_input)[:, 1]