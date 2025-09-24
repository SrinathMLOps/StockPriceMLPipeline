#!/usr/bin/env python3
# Simple MLflow training script

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn
import os

def train_stock_model():
    """Train stock price prediction model with MLflow tracking"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("stock_price_prediction")
    
    print("🚀 Starting MLflow experiment...")
    
    with mlflow.start_run():
        # Load data
        try:
            df = pd.read_csv('dags/data/stock_raw.csv')
            print(f"📊 Loaded {len(df)} records")
        except FileNotFoundError:
            print("❌ Stock data not found. Creating sample data...")
            # Create sample data
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=1000, freq='H')
            df = pd.DataFrame({
                'datetime': dates,
                'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
                'volume': np.random.randint(1000, 10000, 1000)
            })
        
        # Feature engineering
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['ma_3'] = df['close'].rolling(window=3).mean()
        df['pct_change_1d'] = df['close'].pct_change()
        df = df.dropna()
        
        # Prepare features and target
        X = df[['ma_3', 'pct_change_1d', 'volume']]
        y = df['close']
        
        print(f"🔧 Features: {list(X.columns)}")
        print(f"📈 Target: stock close price")
        print(f"📊 Dataset shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        
        # Log metrics
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        
        # Log feature coefficients
        for i, feature in enumerate(X.columns):
            mlflow.log_metric(f"coef_{feature}", model.coef_[i])
        mlflow.log_metric("intercept", model.intercept_)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "stock_model",
            registered_model_name="StockPricePredictor"
        )
        
        # Print results
        print(f"✅ Train R²: {train_r2:.4f}")
        print(f"✅ Test R²: {test_r2:.4f}")
        print(f"✅ Train MSE: {train_mse:.4f}")
        print(f"✅ Test MSE: {test_mse:.4f}")
        print(f"✅ Train MAE: {train_mae:.4f}")
        print(f"✅ Test MAE: {test_mae:.4f}")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, 'models/stock_model.pkl')
        print("📦 Model saved to models/stock_model.pkl")
        print("🔬 Experiment logged to MLflow")
        
        return model

if __name__ == "__main__":
    # Install required packages
    import subprocess
    import sys
    
    packages = ['pandas', 'scikit-learn', 'mlflow==2.8.1', 'joblib', 'numpy']
    for package in packages:
        try:
            __import__(package.split('==')[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    train_stock_model()