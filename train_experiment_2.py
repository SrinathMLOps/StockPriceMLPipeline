#!/usr/bin/env python3
# Experiment 2: Different model parameters

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn

def train_random_forest():
    """Train Random Forest model with MLflow tracking"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("stock_price_prediction")
    
    with mlflow.start_run():
        # Create sample data
        np.random.seed(123)  # Different seed
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        df = pd.DataFrame({
            'datetime': dates,
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.8),  # More volatility
            'volume': np.random.randint(500, 15000, 1000)  # Different volume range
        })
        
        # Feature engineering
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['ma_3'] = df['close'].rolling(window=3).mean()
        df['pct_change_1d'] = df['close'].pct_change()
        df = df.dropna()
        
        X = df[['ma_3', 'pct_change_1d', 'volume']]
        y = df['close']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=123  # Different test size
        )
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=123)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 123)
        
        # Log metrics
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "stock_model",
            registered_model_name="StockPricePredictor"
        )
        
        print(f"🌲 Random Forest Results:")
        print(f"✅ Train R²: {train_r2:.4f}")
        print(f"✅ Test R²: {test_r2:.4f}")
        print(f"✅ Train MSE: {train_mse:.4f}")
        print(f"✅ Test MSE: {test_mse:.4f}")

if __name__ == "__main__":
    train_random_forest()