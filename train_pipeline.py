import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from src.data.preprocessing import preprocess
from src.models.train_base_models import train_base_models
from src.models.stacking import train_meta_model

# Example dataset path
df = pd.read_csv('data/raw/dataset.csv')
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, scaler = preprocess(X_train, X_test)

rf, gb = train_base_models(X_train, y_train)
meta = train_meta_model(rf, gb, X_train, y_train)

joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(gb, 'models/gb_model.pkl')
joblib.dump(meta, 'models/meta_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')