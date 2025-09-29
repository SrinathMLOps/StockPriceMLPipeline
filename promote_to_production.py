#!/usr/bin/env python3
# Model Promotion to Production Script

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def promote_best_model_to_production():
    """Promote the best performing model to production stage"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    print("🚀 Model Promotion to Production")
    print("=" * 50)
    
    # Get the StockPricePredictor model
    model_name = "StockPricePredictor"
    
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"❌ No versions found for model '{model_name}'")
            return
        
        print(f"📦 Found {len(versions)} versions of {model_name}")
        
        # Analyze each version to find the best one
        best_version = None
        best_score = 0
        
        print("\n📊 Analyzing Model Versions:")
        for version in versions:
            print(f"\n   Version {version.version}:")
            print(f"   ├── Stage: {version.current_stage}")
            print(f"   ├── Status: {version.status}")
            
            # Get run metrics for this version
            if version.run_id:
                try:
                    run = client.get_run(version.run_id)
                    test_r2 = run.data.metrics.get('test_r2', 0)
                    model_type = run.data.params.get('model_type', 'Unknown')
                    
                    print(f"   ├── Model Type: {model_type}")
                    print(f"   └── Test R²: {test_r2:.4f}")
                    
                    # Track the best version
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_version = version
                        
                except Exception as e:
                    print(f"   └── Error getting run data: {e}")
        
        if best_version:
            print(f"\n🏆 Best Model Identified:")
            print(f"   Version: {best_version.version}")
            print(f"   Score: {best_score:.4f}")
            print(f"   Current Stage: {best_version.current_stage}")
            
            # Check if already in production
            if best_version.current_stage == "Production":
                print(f"   ✅ Model is already in Production stage!")
                return best_version
            
            # Promote to production
            print(f"\n🔄 Promoting Version {best_version.version} to Production...")
            
            # First, archive any existing production models
            for version in versions:
                if version.current_stage == "Production" and version.version != best_version.version:
                    print(f"   📦 Archiving existing production version {version.version}")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
            
            # Promote the best model to production
            client.transition_model_version_stage(
                name=model_name,
                version=best_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            # Add description/annotation
            client.update_model_version(
                name=model_name,
                version=best_version.version,
                description=f"Promoted to Production on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Best R² Score: {best_score:.4f}"
            )
            
            print(f"   ✅ Successfully promoted Version {best_version.version} to Production!")
            
            return best_version
            
        else:
            print("❌ No suitable model found for promotion")
            return None
            
    except Exception as e:
        print(f"❌ Error during promotion: {e}")
        return None

def verify_production_deployment():
    """Verify the production model deployment"""
    
    client = MlflowClient()
    model_name = "StockPricePredictor"
    
    print(f"\n🔍 Verifying Production Deployment")
    print("=" * 40)
    
    try:
        # Get production versions
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if production_versions:
            for version in production_versions:
                print(f"✅ Production Model Found:")
                print(f"   Model: {version.name}")
                print(f"   Version: {version.version}")
                print(f"   Stage: {version.current_stage}")
                print(f"   Status: {version.status}")
                print(f"   Description: {version.description}")
                
                # Test loading the model
                try:
                    model_uri = f"models:/{model_name}/Production"
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"   ✅ Model successfully loaded from registry")
                    print(f"   Model Type: {type(model).__name__}")
                    
                    # Test prediction
                    import numpy as np
                    test_input = np.array([[100.5, 0.02, 5000]])
                    prediction = model.predict(test_input)[0]
                    print(f"   ✅ Test prediction: {prediction:.2f}")
                    
                except Exception as e:
                    print(f"   ❌ Error loading model: {e}")
        else:
            print("❌ No production models found")
            
    except Exception as e:
        print(f"❌ Error verifying deployment: {e}")

def update_api_to_use_production_model():
    """Update the API configuration to use the production model"""
    
    print(f"\n🔧 API Configuration Update")
    print("=" * 35)
    
    print("📝 To update your API to use the production model:")
    print("   1. Modify serving/main.py or serving/main_standalone.py")
    print("   2. Change model loading to use production stage:")
    print("   ")
    print("   # Replace this line:")
    print("   model_uri = f\"models:/StockPricePredictor/{model_version}\"")
    print("   ")
    print("   # With this:")
    print("   model_uri = \"models:/StockPricePredictor/Production\"")
    print("   ")
    print("   3. Restart your API server")
    print("   4. Test the updated API")

def main():
    """Main promotion workflow"""
    
    # Promote best model to production
    promoted_model = promote_best_model_to_production()
    
    if promoted_model:
        # Verify the deployment
        verify_production_deployment()
        
        # Provide API update instructions
        update_api_to_use_production_model()
        
        print(f"\n🎉 Model Promotion Complete!")
        print(f"   ✅ Version {promoted_model.version} is now in Production")
        print(f"   ✅ Ready for production deployment")
        print(f"   ✅ Access via: models:/StockPricePredictor/Production")
        
    else:
        print(f"\n❌ Model promotion failed")

if __name__ == "__main__":
    main()