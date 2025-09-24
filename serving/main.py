from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
import mlflow
import mlflow.sklearn

app = FastAPI(title="Stock Price Prediction API", version="1.0.0")

# ✅ Try to load model from MLflow first, fallback to local file
model = None
model_version = "latest"

def load_model():
    global model, model_version
    
    try:
        # Try loading from MLflow Model Registry
        mlflow.set_tracking_uri("http://mlflow:5000")
        model_uri = f"models:/StockPricePredictor/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded from MLflow Registry (version: {model_version})")
        return True
    except Exception as e:
        print(f"⚠️ MLflow model loading failed: {e}")
        
        # Fallback to local file
        model_path = "/app/models/stock_model.pkl"
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print("✅ Model loaded from local file")
                return True
            except Exception as e:
                print(f"❌ Local model loading failed: {e}")
                return False
        else:
            print(f"❌ Model file not found at {model_path}")
            return False

# Load model at startup
if not load_model():
    raise RuntimeError("❌ Failed to load model from both MLflow and local file")

# ✅ Health check route
@app.get("/")
def root():
    return {
        "message": "Stock Prediction API is running",
        "model_version": model_version,
        "mlflow_uri": "http://localhost:5000"
    }

# ✅ Model info endpoint
@app.get("/model/info")
def model_info():
    return {
        "model_type": str(type(model).__name__),
        "model_version": model_version,
        "features": ["ma_3", "pct_change_1d", "volume"]
    }

# ✅ Prediction endpoint
@app.post("/predict")
def predict(ma_3: float, pct_change_1d: float, volume: float):
    try:
        X = np.array([[ma_3, pct_change_1d, volume]])
        prediction = model.predict(X)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
