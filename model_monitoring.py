#!/usr/bin/env python3
# Model monitoring and performance tracking

import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
import requests
import time

def monitor_model_performance():
    """Monitor model predictions and log performance metrics"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Test scenarios for monitoring
    test_scenarios = [
        {"name": "Normal Market", "ma_3": 150.0, "pct_change_1d": 0.01, "volume": 1000000},
        {"name": "Bull Market", "ma_3": 200.0, "pct_change_1d": 0.05, "volume": 2000000},
        {"name": "Bear Market", "ma_3": 100.0, "pct_change_1d": -0.03, "volume": 500000},
        {"name": "High Volatility", "ma_3": 175.0, "pct_change_1d": 0.08, "volume": 5000000},
        {"name": "Low Volume", "ma_3": 160.0, "pct_change_1d": 0.02, "volume": 100000}
    ]
    
    print("🔍 Starting Model Monitoring...")
    
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("stock_price_prediction").experiment_id):
        mlflow.set_tag("monitoring", "performance_test")
        
        predictions = []
        
        for scenario in test_scenarios:
            try:
                # Make prediction via API
                url = f"http://localhost:8000/predict"
                params = {
                    "ma_3": scenario["ma_3"],
                    "pct_change_1d": scenario["pct_change_1d"],
                    "volume": scenario["volume"]
                }
                
                response = requests.post(url, params=params)
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    predictions.append({
                        "scenario": scenario["name"],
                        "prediction": prediction,
                        **scenario
                    })
                    
                    # Log individual prediction
                    mlflow.log_metric(f"prediction_{scenario['name'].lower().replace(' ', '_')}", prediction)
                    
                    print(f"✅ {scenario['name']}: ${prediction:.2f}")
                else:
                    print(f"❌ {scenario['name']}: API Error")
                    
            except Exception as e:
                print(f"❌ {scenario['name']}: {e}")
        
        # Log summary metrics
        if predictions:
            pred_values = [p["prediction"] for p in predictions]
            mlflow.log_metric("avg_prediction", np.mean(pred_values))
            mlflow.log_metric("prediction_std", np.std(pred_values))
            mlflow.log_metric("min_prediction", np.min(pred_values))
            mlflow.log_metric("max_prediction", np.max(pred_values))
            
            print(f"\n📊 Summary:")
            print(f"Average Prediction: ${np.mean(pred_values):.2f}")
            print(f"Prediction Range: ${np.min(pred_values):.2f} - ${np.max(pred_values):.2f}")
            print(f"Standard Deviation: ${np.std(pred_values):.2f}")

if __name__ == "__main__":
    monitor_model_performance()