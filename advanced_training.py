#!/usr/bin/env python3
# Advanced training with hyperparameter tuning

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def hyperparameter_tuning():
    """Run hyperparameter tuning experiments"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("stock_price_hyperparameter_tuning")
    
    # Generate more complex sample data
    np.random.seed(42)
    n_samples = 2000
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='h')
    
    # Create realistic stock data with trends and seasonality
    trend = np.linspace(100, 150, n_samples)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily pattern
    noise = np.random.normal(0, 2, n_samples)
    
    df = pd.DataFrame({
        'datetime': dates,
        'close': trend + seasonality + noise,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Feature engineering
    df['ma_3'] = df['close'].rolling(window=3).mean()
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['pct_change_1d'] = df['close'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=3).mean()
    df = df.dropna()
    
    X = df[['ma_3', 'ma_7', 'pct_change_1d', 'volume', 'volume_ma']]
    y = df['close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model configurations to test
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
    }
    
    best_model = None
    best_score = float('-inf')
    
    for model_name, config in models.items():
        print(f"🔧 Tuning {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_hyperparameter_tuning"):
            # Grid search
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Best model predictions
            y_pred_train = grid_search.predict(X_train)
            y_pred_test = grid_search.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            # Log parameters
            mlflow.log_param("model_type", model_name)
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Log metrics
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("cv_score", grid_search.best_score_)
            
            # Log model
            mlflow.sklearn.log_model(
                grid_search.best_estimator_,
                "model",
                registered_model_name="StockPricePredictor"
            )
            
            print(f"✅ {model_name} - Test R²: {test_r2:.4f}, CV Score: {grid_search.best_score_:.4f}")
            print(f"   Best params: {grid_search.best_params_}")
            
            if test_r2 > best_score:
                best_score = test_r2
                best_model = model_name
    
    print(f"\n🏆 Best Model: {best_model} (R² = {best_score:.4f})")

if __name__ == "__main__":
    hyperparameter_tuning()