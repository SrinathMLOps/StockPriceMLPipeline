#!/usr/bin/env python3
# Simple Training with Artifacts (no plotting dependencies)

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_training_with_artifacts():
    """Train model with comprehensive artifact logging"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("training_with_artifacts")
    
    print("🚀 Starting Training with Artifacts")
    print("=" * 50)
    
    with mlflow.start_run(run_name=f"artifacts_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # 1. Generate sample data
        print("📊 Generating sample data...")
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        trend = np.linspace(100, 150, n_samples)
        seasonality = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        noise = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame({
            'datetime': dates,
            'close': trend + seasonality + noise,
            'volume': np.random.lognormal(8, 0.5, n_samples)
        })
        
        # Feature engineering
        df['ma_3'] = df['close'].rolling(window=3).mean()
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['pct_change_1d'] = df['close'].pct_change()
        df['volatility'] = df['pct_change_1d'].rolling(window=7).std()
        df['volume_ma'] = df['volume'].rolling(window=3).mean()
        df = df.dropna()
        
        # 2. Create artifacts directory
        os.makedirs("artifacts", exist_ok=True)
        
        # 3. Log dataset as artifact
        print("💾 Logging dataset artifacts...")
        dataset_path = "artifacts/training_dataset.csv"
        df.to_csv(dataset_path, index=False)
        mlflow.log_artifact(dataset_path, "data")
        
        # Dataset summary
        summary_path = "artifacts/dataset_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Dataset Summary Report\n")
            f.write("=" * 30 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")
            f.write(f"Features: {list(df.columns)}\n")
            f.write(f"Missing values: {df.isnull().sum().sum()}\n\n")
            f.write("Statistical Summary:\n")
            f.write("-" * 20 + "\n")
            for col in df.select_dtypes(include=[np.number]).columns:
                f.write(f"{col}:\n")
                f.write(f"  Mean: {df[col].mean():.4f}\n")
                f.write(f"  Std:  {df[col].std():.4f}\n")
                f.write(f"  Min:  {df[col].min():.4f}\n")
                f.write(f"  Max:  {df[col].max():.4f}\n\n")
        
        mlflow.log_artifact(summary_path, "data")
        
        # 4. Prepare features
        feature_cols = ['ma_3', 'ma_7', 'pct_change_1d', 'volatility', 'volume_ma']
        X = df[feature_cols]
        y = df['close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 5. Train models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Ridge': Ridge(alpha=0.1)
        }
        
        model_results = {}
        
        for model_name, model in models.items():
            print(f"🔧 Training {model_name}...")
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            model_results[model_name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'predictions': y_pred_test
            }
            
            # Log metrics
            mlflow.log_metric(f"{model_name}_train_r2", train_r2)
            mlflow.log_metric(f"{model_name}_test_r2", test_r2)
            mlflow.log_metric(f"{model_name}_test_mse", test_mse)
            
            print(f"   ✅ {model_name} - Test R²: {test_r2:.4f}")
        
        # 6. Create model comparison report
        print("📋 Creating model comparison report...")
        
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
        best_model = model_results[best_model_name]['model']
        best_r2 = model_results[best_model_name]['test_r2']
        
        # Model comparison report
        report_path = "artifacts/model_comparison_report.json"
        report = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "dataset_size": len(df),
                "features": feature_cols,
                "test_size": 0.2,
                "random_state": 42
            },
            "models": {},
            "best_model": {
                "name": best_model_name,
                "test_r2": float(best_r2),
                "recommendation": "Production Ready" if best_r2 > 0.95 else "Needs Improvement"
            }
        }
        
        for name, results in model_results.items():
            report["models"][name] = {
                "train_r2": float(results['train_r2']),
                "test_r2": float(results['test_r2']),
                "train_mse": float(results['train_mse']),
                "test_mse": float(results['test_mse'])
            }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        mlflow.log_artifact(report_path, "reports")
        
        # 7. Save predictions
        predictions_path = "artifacts/test_predictions.csv"
        predictions_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': model_results[best_model_name]['predictions'],
            'residual': y_test.values - model_results[best_model_name]['predictions']
        })
        predictions_df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path, "predictions")
        
        # 8. Save model configuration
        config_path = "artifacts/model_config.json"
        config = {
            "model_type": best_model_name,
            "features": feature_cols,
            "hyperparameters": {
                "RandomForest": {"n_estimators": 50, "random_state": 42},
                "Ridge": {"alpha": 0.1}
            },
            "performance": {
                "train_r2": float(model_results[best_model_name]['train_r2']),
                "test_r2": float(model_results[best_model_name]['test_r2']),
                "test_mse": float(model_results[best_model_name]['test_mse'])
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        mlflow.log_artifact(config_path, "config")
        
        # 9. Save model file
        model_path = "artifacts/trained_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path, "models")
        
        # 10. Log model to MLflow registry
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name="StockPricePredictor"
        )
        
        # 11. Log parameters
        mlflow.log_param("best_model_type", best_model_name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("feature_list", ",".join(feature_cols))
        
        # 12. Log final metrics
        mlflow.log_metric("final_test_r2", best_r2)
        mlflow.log_metric("final_test_mse", model_results[best_model_name]['test_mse'])
        
        # 13. Create README for artifacts
        readme_path = "artifacts/README.md"
        with open(readme_path, 'w') as f:
            f.write("# MLflow Artifacts Documentation\n\n")
            f.write(f"**Experiment**: Training with Artifacts\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Best Model**: {best_model_name}\n")
            f.write(f"**Performance**: {best_r2:.4f} R²\n\n")
            f.write("## Artifacts Structure\n\n")
            f.write("```\n")
            f.write("├── data/\n")
            f.write("│   ├── training_dataset.csv     # Complete training dataset\n")
            f.write("│   └── dataset_summary.txt      # Statistical summary\n")
            f.write("├── reports/\n")
            f.write("│   └── model_comparison_report.json  # Model comparison results\n")
            f.write("├── predictions/\n")
            f.write("│   └── test_predictions.csv     # Test set predictions\n")
            f.write("├── config/\n")
            f.write("│   └── model_config.json        # Model configuration\n")
            f.write("├── models/\n")
            f.write("│   └── trained_model.pkl        # Serialized model\n")
            f.write("└── model/                       # MLflow model format\n")
            f.write("```\n\n")
            f.write("## Usage\n\n")
            f.write("1. **View Experiment**: http://localhost:5000\n")
            f.write("2. **Load Model**: `mlflow.sklearn.load_model('runs:/<run_id>/model')`\n")
            f.write("3. **Download Artifacts**: Use MLflow UI or API\n")
        
        mlflow.log_artifact(readme_path, "documentation")
        
        # Clean up
        import shutil
        if os.path.exists("artifacts"):
            shutil.rmtree("artifacts")
        
        print(f"\n✅ Training with Artifacts Complete!")
        print(f"   🏆 Best Model: {best_model_name}")
        print(f"   📊 Test R²: {best_r2:.4f}")
        print(f"   📁 Artifacts Categories:")
        print(f"      📄 data/ - Dataset and summaries")
        print(f"      📊 reports/ - Model comparison reports")
        print(f"      🔮 predictions/ - Test predictions")
        print(f"      ⚙️  config/ - Model configuration")
        print(f"      🤖 models/ - Trained model files")
        print(f"      📚 documentation/ - README and docs")
        print(f"   🌐 View at: http://localhost:5000")
        
        return best_model_name, best_r2

def list_mlflow_artifacts():
    """List all artifacts in the latest run"""
    
    print("\n🔍 Checking MLflow Artifacts")
    print("=" * 40)
    
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    
    experiment = client.get_experiment_by_name("training_with_artifacts")
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
        
        if runs:
            run = runs[0]
            print(f"📊 Run ID: {run.info.run_id}")
            print(f"📅 Start Time: {datetime.fromtimestamp(run.info.start_time / 1000)}")
            
            artifacts = client.list_artifacts(run.info.run_id)
            
            print(f"\n📁 Artifacts Structure:")
            for artifact in artifacts:
                if artifact.is_dir:
                    print(f"📂 {artifact.path}/")
                    # List contents of directory
                    sub_artifacts = client.list_artifacts(run.info.run_id, artifact.path)
                    for sub_artifact in sub_artifacts:
                        size_mb = sub_artifact.file_size / (1024 * 1024) if sub_artifact.file_size else 0
                        print(f"   📄 {sub_artifact.path} ({size_mb:.2f} MB)")
                else:
                    size_mb = artifact.file_size / (1024 * 1024) if artifact.file_size else 0
                    print(f"📄 {artifact.path} ({size_mb:.2f} MB)")
            
            print(f"\n💡 To download artifacts:")
            print(f"   1. Visit: http://localhost:5000")
            print(f"   2. Navigate to experiment: 'training_with_artifacts'")
            print(f"   3. Click on run: {run.info.run_id[:8]}...")
            print(f"   4. Browse and download from 'Artifacts' tab")
            
        else:
            print("❌ No runs found")
    else:
        print("❌ Experiment not found")

if __name__ == "__main__":
    # Run training with artifacts
    best_model, score = create_training_with_artifacts()
    
    # List the artifacts created
    list_mlflow_artifacts()
    
    print(f"\n🎉 Complete! Best Model: {best_model} (R² = {score:.4f})")
    print(f"🌐 View all artifacts at: http://localhost:5000")