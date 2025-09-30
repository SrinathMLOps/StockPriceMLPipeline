# ☁️ Azure Machine Learning Studio Setup Guide

## 🎯 Overview

Transform your local MLOps pipeline into an enterprise-grade Azure ML solution, similar to what you saw in the Azure ML Studio interface.

## 🏗️ Azure ML Architecture for Stock Price Prediction

```
┌─────────────────────────────────────────────────────────────────┐
│                    AZURE ML WORKSPACE                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   DATA ASSETS   │   EXPERIMENTS   │      DEPLOYMENT             │
│                 │                 │                             │
│ • Datasets      │ • Training Jobs │ • Real-time Endpoints       │
│ • Datastores    │ • Pipelines     │ • Batch Endpoints           │
│ • Data Drift    │ • AutoML        │ • Model Registry            │
│ • Monitoring    │ • Experiments   │ • A/B Testing               │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 📋 Step-by-Step Setup

### **Step 1: Create Azure ML Workspace**

#### **Option A: Azure Portal**
1. **Go to**: https://portal.azure.com
2. **Create Resource** → Search "Machine Learning"
3. **Configure**:
   - **Workspace Name**: `stock-price-mlops`
   - **Resource Group**: `rg-mlops-demo`
   - **Region**: `East US` or your preferred region
   - **Storage Account**: Auto-create
   - **Key Vault**: Auto-create
   - **Application Insights**: Auto-create

#### **Option B: Azure CLI**
```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login to Azure
az login

# Create resource group
az group create --name rg-mlops-demo --location eastus

# Create ML workspace
az ml workspace create --name stock-price-mlops --resource-group rg-mlops-demo
```

### **Step 2: Setup Development Environment**

#### **Install Azure ML SDK**
```bash
pip install azureml-sdk
pip install azureml-core
pip install azureml-train-core
pip install azureml-pipeline
```

#### **Create Azure ML Configuration**
```python
# azure_ml_config.py
from azureml.core import Workspace, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.model import Model
import os

# Connect to workspace
ws = Workspace.from_config()  # Uses config.json
print(f"Workspace: {ws.name}, Resource Group: {ws.resource_group}")

# Or connect directly
# ws = Workspace(subscription_id="your-subscription-id",
#                resource_group="rg-mlops-demo", 
#                workspace_name="stock-price-mlops")
```

### **Step 3: Convert Your Training Script for Azure ML**

<function_calls>
<invoke name="fsWrite">
<parameter name="path">azure_train_model.pyin/env python3
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
from azureml.core import Run, Dataset, Model
from azureml.core.model import Model as AzureModel

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--model-name', type=str, default='stock-price-predictor')
    parser.add_argument('--experiment-name', type=str, default='stock-price-prediction')
    parser.add_argument('--algorithm', type=str, default='random_forest', 
                       choices=['linear_regression', 'random_forest', 'ridge'])
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

def train_model(algorithm, X_train, X_test, y_train, y_test, run):
    """Train model based on algorithm choice"""
    
    if algorithm == 'random_forest':
        # Hyperparameter tuning for Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 5]
        }
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        
        # Log hyperparameters
        for param, value in grid_search.best_params_.items():
            run.log(f"best_{param}", value)
            
    elif algorithm == 'ridge':
        # Hyperparameter tuning for Ridge
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        model = Ridge()
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        
        # Log hyperparameters
        run.log("best_alpha", grid_search.best_params_['alpha'])
        
    else:  # linear_regression
        model = LinearRegression()
        model.fit(X_train, y_train)
    
    return model

def main():
    """Main training function"""
    args = parse_args()
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Start MLflow tracking
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        print("🚀 Starting Azure ML training job...")
        
        # Load or create data
        if args.data_path and os.path.exists(args.data_path):
            df = pd.read_csv(args.data_path)
            print(f"📊 Loaded data from {args.data_path}")
        else:
            df = create_sample_data()
            print("📊 Created sample data")
        
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
        model = train_model(args.algorithm, X_train, X_test, y_train, y_test, run)
        
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
        
        # Log metrics to MLflow
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("n_features", len(feature_columns))
        
        # Save model locally
        model_path = "outputs/model.pkl"
        os.makedirs("outputs", exist_ok=True)
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Register model in Azure ML
        run.upload_file("model.pkl", model_path)
        model = run.register_model(
            model_name=args.model_name,
            model_path="model.pkl",
            description=f"Stock price prediction model using {args.algorithm}",
            tags={
                "algorithm": args.algorithm,
                "test_r2": test_r2,
                "framework": "scikit-learn"
            }
        )
        
        print("✅ Training completed successfully!")
        print(f"📊 Model Performance:")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Test MSE: {test_mse:.4f}")
        print(f"📦 Model registered: {model.name} (Version {model.version})")

if __name__ == "__main__":
    main()