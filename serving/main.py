from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os

app = FastAPI()

# ✅ Define model path inside the container
model_path = "/app/models/stock_model.pkl"

# ✅ Load model at startup
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# ✅ Health check route
@app.get("/")
def root():
    return {"message": "Stock Prediction API is running"}

# ✅ Prediction endpoint
@app.post("/predict")
def predict(ma_3: float, pct_change_1d: float, volume: float):
    try:
        X = np.array([[ma_3, pct_change_1d, volume]])
        prediction = model.predict(X)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
