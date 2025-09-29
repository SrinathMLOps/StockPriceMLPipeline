#!/usr/bin/env python3
# Enhanced Training with Comprehensive Artifact Logging

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_training_with_artifacts():
    """Train model with comprehensive artifact logging"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("comprehensive_training_with_artifacts")
    
    print("🚀 Starting Comprehensive Training with Artifacts")
    print("=" * 60)
    
    with mlflow.start_run(run_name=f"comprehensive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # 1. Generate and log sample data
        print("📊 Generating sample data...")
        np.random.seed(42)
        n_samples = 2000
        
        # Create realistic stock data
        dates = pd.date_range('2022-01-01', periods=n_samples, freq='H')
        trend = np.linspace(100, 150, n_samples)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        noise = np.random.normal(0, 2, n_samples)
        
        df = pd.DataFrame({
            'datetime': dates,
            'close': trend + seasonality + noise,
            'volume': np.random.lognormal(10, 1, n_samples)
        })
        
        # Feature engineering
        df['ma_3'] = df['close'].rolling(window=3).mean()
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()
        df['pct_change_1d'] = df['close'].pct_change()
        df['pct_change_7d'] = df['close'].pct_change(periods=7)
        df['volatility'] = df['pct_change_1d'].rolling(window=7).std()
        df['volume_ma'] = df['volume'].rolling(window=7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df = df.dropna()
        
        # 2. Log dataset as artifact
        print("💾 Logging dataset artifacts...")
        os.makedirs("temp_artifacts", exist_ok=True)
        
        # Save dataset
        dataset_path = "temp_artifacts/training_dataset.csv"
        df.to_csv(dataset_path, index=False)
        mlflow.log_artifact(dataset_path, "data")
        
        # Save dataset summary
        summary_path = "temp_artifacts/dataset_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Dataset Summary\n")
            f.write("=" * 20 + "\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")
            f.write(f"Features: {list(df.columns)}\n")
            f.write(f"Missing values: {df.isnull().sum().sum()}\n")
            f.write("\nStatistical Summary:\n")
            f.write(df.describe().to_string())
        mlflow.log_artifact(summary_path, "data")
        
        # 3. Prepare features and target
        feature_cols = ['ma_3', 'ma_7', 'ma_21', 'pct_change_1d', 'pct_change_7d', 
                       'volatility', 'volume', 'volume_ratio']
        X = df[feature_cols]
        y = df['close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Train multiple models and compare
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=0.1)
        }
        
        model_results = {}
        
        for model_name, model in models.items():
            print(f"🔧 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            model_results[model_name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'y_pred_test': y_pred_test
            }
            
            # Log model-specific metrics
            mlflow.log_metric(f"{model_name}_train_r2", train_r2)
            mlflow.log_metric(f"{model_name}_test_r2", test_r2)
            mlflow.log_metric(f"{model_name}_train_mse", train_mse)
            mlflow.log_metric(f"{model_name}_test_mse", test_mse)
            
            print(f"   ✅ {model_name} - Test R²: {test_r2:.4f}")
        
        # 5. Create and log visualizations
        print("📊 Creating visualization artifacts...")
        
        # Model comparison plot
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: R² Comparison
        plt.subplot(2, 2, 1)
        model_names = list(model_results.keys())
        train_r2_scores = [model_results[name]['train_r2'] for name in model_names]
        test_r2_scores = [model_results[name]['test_r2'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, train_r2_scores, width, label='Train R²', alpha=0.8)
        plt.bar(x + width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Prediction vs Actual (best model)
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
        best_results = model_results[best_model_name]
        
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, best_results['y_pred_test'], alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{best_model_name} - Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Residuals plot
        plt.subplot(2, 2, 3)
        residuals = y_test - best_results['y_pred_test']
        plt.scatter(best_results['y_pred_test'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{best_model_name} - Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Feature importance (if available)
        plt.subplot(2, 2, 4)
        if hasattr(best_results['model'], 'feature_importances_'):
            importances = best_results['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title(f'{best_model_name} - Feature Importance')
            plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45)
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        plt.tight_layout()
        
        # Save and log the plot
        plot_path = "temp_artifacts/model_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path, "plots")
        plt.close()
        
        # 6. Create and log model comparison report
        print("📋 Creating model comparison report...")
        report_path = "temp_artifacts/model_comparison_report.json"
        
        comparison_report = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "dataset_size": len(df),
                "features": feature_cols,
                "test_size": 0.2
            },
            "model_results": {}
        }
        
        for model_name, results in model_results.items():
            comparison_report["model_results"][model_name] = {
                "train_r2": float(results['train_r2']),
                "test_r2": float(results['test_r2']),
                "train_mse": float(results['train_mse']),
                "test_mse": float(results['test_mse']),
                "train_mae": float(results['train_mae']),
                "test_mae": float(results['test_mae'])
            }
        
        # Determine best model
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
        comparison_report["best_model"] = {
            "name": best_model_name,
            "test_r2": float(model_results[best_model_name]['test_r2']),
            "recommendation": "Deploy to production" if model_results[best_model_name]['test_r2'] > 0.95 else "Needs improvement"
        }
        
        with open(report_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        mlflow.log_artifact(report_path, "reports")
        
        # 7. Log the best model
        best_model = model_results[best_model_name]['model']
        
        # Save model locally first
        model_path = "temp_artifacts/best_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path, "models")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="StockPricePredictor"
        )
        
        # 8. Log parameters and final metrics
        mlflow.log_param("best_model_type", best_model_name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("test_size", 0.2)
        
        # Log best model metrics
        best_results = model_results[best_model_name]
        mlflow.log_metric("final_train_r2", best_results['train_r2'])
        mlflow.log_metric("final_test_r2", best_results['test_r2'])
        mlflow.log_metric("final_test_mse", best_results['test_mse'])
        mlflow.log_metric("final_test_mae", best_results['test_mae'])
        
        # 9. Log configuration file
        config_path = "temp_artifacts/training_config.json"
        config = {
            "model_parameters": {
                "RandomForest": {"n_estimators": 100, "random_state": 42},
                "Ridge": {"alpha": 0.1}
            },
            "data_parameters": {
                "n_samples": n_samples,
                "test_size": 0.2,
                "random_state": 42
            },
            "feature_engineering": {
                "moving_averages": [3, 7, 21],
                "percentage_changes": [1, 7],
                "volatility_window": 7,
                "volume_ratio": True
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        mlflow.log_artifact(config_path, "config")
        
        # 10. Clean up temporary files
        import shutil
        if os.path.exists("temp_artifacts"):
            shutil.rmtree("temp_artifacts")
        
        print(f"\n✅ Training Complete!")
        print(f"   🏆 Best Model: {best_model_name}")
        print(f"   📊 Test R²: {best_results['test_r2']:.4f}")
        print(f"   📁 Artifacts logged: data, plots, reports, models, config")
        print(f"   🔗 View in MLflow UI: http://localhost:5000")
        
        return best_model_name, best_results['test_r2']

def check_artifacts_in_mlflow():
    """Check what artifacts are available in MLflow"""
    
    print("\n🔍 Checking MLflow Artifacts")
    print("=" * 40)
    
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    
    # Get recent experiment
    experiment = client.get_experiment_by_name("comprehensive_training_with_artifacts")
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
        
        if runs:
            latest_run = runs[0]
            print(f"📊 Latest Run: {latest_run.info.run_id}")
            
            # List artifacts
            artifacts = client.list_artifacts(latest_run.info.run_id)
            
            print(f"📁 Artifacts found:")
            for artifact in artifacts:
                print(f"   📄 {artifact.path} ({artifact.file_size} bytes)")
                
                # If it's a directory, list contents
                if artifact.is_dir:
                    sub_artifacts = client.list_artifacts(latest_run.info.run_id, artifact.path)
                    for sub_artifact in sub_artifacts:
                        print(f"      └── {sub_artifact.path}")
        else:
            print("❌ No runs found in experiment")
    else:
        print("❌ Experiment not found")

if __name__ == "__main__":
    # Run comprehensive training with artifacts
    best_model, score = create_comprehensive_training_with_artifacts()
    
    # Check what artifacts were created
    check_artifacts_in_mlflow()
    
    print(f"\n🎉 Comprehensive Training with Artifacts Complete!")
    print(f"   🏆 Best Model: {best_model} (R² = {score:.4f})")
    print(f"   📁 All artifacts logged to MLflow")
    print(f"   🌐 View at: http://localhost:5000")