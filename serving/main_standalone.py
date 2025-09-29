from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

app = FastAPI(title="Stock Price Prediction API", version="1.0.0")

# Create a simple model if none exists
def create_sample_model():
    """Create a simple sample model for demonstration"""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: ma_3, pct_change_1d, volume
    X = np.random.randn(n_samples, 3)
    # Simple target: linear combination with noise
    y = 2 * X[:, 0] + 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model

# Load or create model
model = None
model_version = "demo"

def load_model():
    global model, model_version
    
    # Check if model file exists locally
    local_model_path = "stock_model.pkl"
    if os.path.exists(local_model_path):
        try:
            model = joblib.load(local_model_path)
            print("✅ Model loaded from local file")
            return True
        except Exception as e:
            print(f"⚠️ Local model loading failed: {e}")
    
    # Create sample model
    try:
        model = create_sample_model()
        print("✅ Sample model created successfully")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

# Load model at startup
if not load_model():
    raise RuntimeError("❌ Failed to load or create model")

# Health check route
@app.get("/")
def root():
    return {
        "message": "Stock Prediction API is running",
        "model_version": model_version,
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }

# Model info endpoint
@app.get("/model/info")
def model_info():
    return {
        "model_type": str(type(model).__name__),
        "model_version": model_version,
        "features": ["ma_3", "pct_change_1d", "volume"],
        "description": "Stock price prediction model"
    }

# Prediction endpoint
@app.post("/predict")
def predict(ma_3: float, pct_change_1d: float, volume: float):
    try:
        X = np.array([[ma_3, pct_change_1d, volume]])
        prediction = model.predict(X)[0]
        return {
            "prediction": float(prediction),
            "features": {
                "ma_3": ma_3,
                "pct_change_1d": pct_change_1d,
                "volume": volume
            },
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(predictions: list):
    try:
        results = []
        for item in predictions:
            X = np.array([[item["ma_3"], item["pct_change_1d"], item["volume"]]])
            prediction = model.predict(X)[0]
            results.append({
                "prediction": float(prediction),
                "features": item
            })
        return {
            "predictions": results,
            "count": len(results),
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)