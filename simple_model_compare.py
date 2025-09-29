#!/usr/bin/env python3
# Simple Model Comparison Script

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def simple_model_comparison():
    """Simple model comparison without complex analysis"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    print("🔍 Simple Model Comparison")
    print("=" * 50)
    
    # Get the main experiment
    experiment = client.get_experiment_by_name("stock_price_hyperparameter_tuning")
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        print(f"📊 Experiment: {experiment.name}")
        print(f"   Total Runs: {len(runs)}")
        
        models = []
        for run in runs:
            model_data = {
                'Run Name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                'Model Type': run.data.params.get('model_type', 'Unknown'),
                'Test R²': run.data.metrics.get('test_r2', 0),
                'Test MSE': run.data.metrics.get('test_mse', 0),
                'CV Score': run.data.metrics.get('cv_score', 0),
                'Status': run.info.status
            }
            models.append(model_data)
        
        # Create DataFrame and sort by Test R²
        df = pd.DataFrame(models)
        df = df.sort_values('Test R²', ascending=False)
        
        print("\n📋 Model Comparison Table:")
        print(df.to_string(index=False))
        
        # Best model
        if not df.empty:
            best = df.iloc[0]
            print(f"\n🏆 Winner: {best['Model Type']}")
            print(f"   Test R²: {best['Test R²']:.4f}")
            print(f"   Test MSE: {best['Test MSE']:.4f}")
            print(f"   CV Score: {best['CV Score']:.4f}")
    
    # Model Registry Summary
    print(f"\n🏪 Model Registry Summary:")
    registered_models = client.search_registered_models()
    
    for model in registered_models:
        versions = client.search_model_versions(f"name='{model.name}'")
        print(f"   📦 {model.name}: {len(versions)} versions")
        
        # Get latest version performance
        if versions:
            latest = versions[0]  # Most recent version
            if latest.run_id:
                run = client.get_run(latest.run_id)
                if 'test_r2' in run.data.metrics:
                    print(f"      Latest R²: {run.data.metrics['test_r2']:.4f}")

if __name__ == "__main__":
    simple_model_comparison()