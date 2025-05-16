import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset dari folder lokal
X_train = np.load('dataset_preprocessing/x_train.npy')
y_train = np.load('dataset_preprocessing/y_train.npy')
X_val = np.load('dataset_preprocessing/x_val.npy')
y_val = np.load('dataset_preprocessing/y_val.npy')

# Training dan Logging
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Simpan dan log model
    joblib.dump(model, "model_rf.pkl", compress=3)
    mlflow.log_artifact("model_rf.pkl", artifact_path="model")

print("Training selesai.")