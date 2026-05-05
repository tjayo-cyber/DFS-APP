from fastapi import FastAPI
import joblib
import numpy as np
from src.models.predict import predict

app = FastAPI()

rf = joblib.load('models/rf_model.pkl')
gb = joblib.load('models/gb_model.pkl')
meta = joblib.load('models/meta_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.post('/predict')
def get_prediction(data: list):
    input_array = np.array(data).reshape(1, -1)
    risk = predict(rf, gb, meta, scaler, input_array)[0]
    return {'risk_score': float(risk)}