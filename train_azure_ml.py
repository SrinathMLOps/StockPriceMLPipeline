#!/usr/bin/env python3
"""
Azure ML Training Script for Stock Price Prediction
Converts local MLflow training to Azure ML format
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn
from azureml.core import Run

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='random_forest', 
                       choices=['linear_regression', 'random_forest', 'ridge'])
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    return parser.parse_args()

def create_sample_data():
    """Create sample stock data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic stock features
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Moving averages
    ma_3 = np.random.normal(100, 20, n_samples)
    ma_7 = ma_3 + np.random.normal(0, 5, n_samples)
    ma_21 = ma_7 + np.random.normal(0, 3, n_samples)
    
    # Percentage changes
    pct_change_1d = np.random.normal(0.001, 0.02, n_samples)
    pct_change_5d = np.random.normal(0.005, 0.05, n_samples)
    
    # Volume
    volume = np.random.lognormal(15, 1, n_samples)
    
    # Target: next day's closing price
    close_price = ma_3 + np.random.normal(0, 2, n_samples)
    
    df = pd.DataFrame({
        'date': dates,
        'ma_3': ma_3,
        'ma_7': ma_7,
        'ma_21': ma_21,
        'pct_change_1d': pct_change_1d,
        'pct_change_5d': pct_change_5d,
        'volume': volume,
        'close_price': close_price
    })
    
    return df

def train_model(algorithm, args, X_train, X_test, y_train, y_test, run):
    """Train model based on algorithm choice"""
    
    if algorithm == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Log hyperparameters
        run.log("n_estimators", args.n_estimators)
        run.log("max_depth", args.max_depth)
        
    elif algorithm == 'ridge':
        model = Ridge(alpha=args.alpha)
        model.fit(X_train, y_train)
        
        # Log hyperparameters
        run.log("alpha", args.alpha)
        
    else:  # linear_regression
        model = LinearRegression()
        model.fit(X_train, y_train)
    
    return model

def main():
    """Main training function"""
    args = parse_args()
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Enable MLflow tracking in Azure ML
    mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
    mlflow.set_experiment(run.experiment.name)
    
    with mlflow.start_run():
        print("🚀 Starting Azure ML training job...")
        print(f"Algorithm: {args.algorithm}")
        
        # Create sample data
        df = create_sample_data()
        print(f"📊 Created dataset with {len(df)} samples")
        
        # Prepare features and target
        feature_columns = ['ma_3', 'pct_change_1d', 'volume']
        X = df[feature_columns]
        y = df['close_price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"🔧 Training {args.algorithm} model...")
        print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train model
        model = train_model(args.algorithm, args, X_train, X_test, y_train, y_test, run)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Log metrics to Azure ML
        run.log("train_r2", train_r2)
        run.log("test_r2", test_r2)
        run.log("train_mse", train_mse)
        run.log("test_mse", test_mse)
        run.log("train_mae", train_mae)
        run.log("test_mae", test_mae)
        run.log("algorithm", args.algorithm)
        run.log("n_features", len(feature_columns))
        
        # Log metrics to MLflow
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("n_features", len(feature_columns))
        
        # Save model locally
        os.makedirs("outputs", exist_ok=True)
        model_path = "outputs/model.pkl"
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Register model in Azure ML
        model_azure = run.register_model(
            model_name="stock-price-predictor",
            model_path="outputs/model.pkl",
            description=f"Stock price prediction model using {args.algorithm}",
            tags={
                "algorithm": args.algorithm,
                "test_r2": f"{test_r2:.4f}",
                "framework": "scikit-learn",
                "features": ",".join(feature_columns)
            },
            properties={
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_mse": test_mse
            }
        )
        
        print("✅ Training completed successfully!")
        print(f"📊 Model Performance:")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Test MSE: {test_mse:.4f}")
        print(f"   Test MAE: {test_mae:.4f}")
        print(f"📦 Model registered: {model_azure.name} (Version {model_azure.version})")
        
        # Return metrics for hyperparameter tuning
        return {
            "test_r2": test_r2,
            "test_mse": test_mse,
            "model_version": model_azure.version
        }

if __name__ == "__main__":
    results = main()
    print(f"🎯 Final Results: {results}")