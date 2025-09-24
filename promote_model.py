#!/usr/bin/env python3
# Model promotion script

import mlflow
from mlflow.tracking import MlflowClient

def promote_best_model():
    """Promote the best performing model to Production stage"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    # Get all versions of the model
    model_name = "StockPricePredictor"
    versions = client.get_latest_versions(model_name, stages=["None"])
    
    print(f"📊 Found {len(versions)} model versions:")
    
    best_version = None
    best_r2 = 0
    
    for version in versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        test_r2 = run.data.metrics.get("test_r2", 0)
        
        print(f"Version {version.version}: R² = {test_r2:.4f}")
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_version = version
    
    if best_version:
        # Promote to Production
        client.transition_model_version_stage(
            name=model_name,
            version=best_version.version,
            stage="Production"
        )
        print(f"🚀 Promoted Version {best_version.version} to Production (R² = {best_r2:.4f})")
    else:
        print("❌ No model versions found")

if __name__ == "__main__":
    promote_best_model()