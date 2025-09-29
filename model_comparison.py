#!/usr/bin/env python3
# Model Comparison and Analysis Script

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """Compare and analyze MLflow models"""
    
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
    def get_all_experiments(self):
        """Get all experiments"""
        experiments = self.client.search_experiments()
        return experiments
    
    def get_experiment_runs(self, experiment_name):
        """Get all runs from an experiment"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])
            return runs
        return []
    
    def compare_experiments(self):
        """Compare all experiments and their runs"""
        print("🔍 MLflow Model Comparison Report")
        print("=" * 60)
        
        experiments = self.get_all_experiments()
        
        all_runs_data = []
        
        for exp in experiments:
            print(f"\n📊 Experiment: {exp.name}")
            print(f"   ID: {exp.experiment_id}")
            
            runs = self.client.search_runs(experiment_ids=[exp.experiment_id])
            print(f"   Total Runs: {len(runs)}")
            
            for run in runs:
                run_data = {
                    'experiment_name': exp.name,
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'status': run.info.status,
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                    'end_time': datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None
                }
                
                # Add metrics
                for metric_name, metric_value in run.data.metrics.items():
                    run_data[metric_name] = metric_value
                
                # Add parameters
                for param_name, param_value in run.data.params.items():
                    run_data[f"param_{param_name}"] = param_value
                
                all_runs_data.append(run_data)
                
                print(f"   └── Run: {run_data['run_name']}")
                print(f"       Status: {run_data['status']}")
                if 'test_r2' in run.data.metrics:
                    print(f"       Test R²: {run.data.metrics['test_r2']:.4f}")
                if 'test_mse' in run.data.metrics:
                    print(f"       Test MSE: {run.data.metrics['test_mse']:.4f}")
        
        return pd.DataFrame(all_runs_data)
    
    def create_comparison_table(self, df):
        """Create a detailed comparison table"""
        print("\n📋 Detailed Model Comparison Table")
        print("=" * 80)
        
        # Select key columns for comparison
        comparison_cols = ['experiment_name', 'run_name', 'param_model_type', 
                          'test_r2', 'train_r2', 'test_mse', 'train_mse']
        
        # Filter columns that exist
        available_cols = [col for col in comparison_cols if col in df.columns]
        
        if available_cols:
            comparison_df = df[available_cols].copy()
            
            # Sort by test_r2 if available
            if 'test_r2' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('test_r2', ascending=False)
            
            print(comparison_df.to_string(index=False))
            
            # Find best model
            if 'test_r2' in comparison_df.columns:
                best_model = comparison_df.iloc[0]
                print(f"\n🏆 Best Model:")
                print(f"   Experiment: {best_model['experiment_name']}")
                print(f"   Run: {best_model['run_name']}")
                print(f"   Model Type: {best_model.get('param_model_type', 'Unknown')}")
                print(f"   Test R²: {best_model['test_r2']:.4f}")
                
                return best_model
        
        return None
    
    def compare_model_registry(self):
        """Compare models in the model registry"""
        print("\n🏪 Model Registry Comparison")
        print("=" * 50)
        
        registered_models = self.client.search_registered_models()
        
        for model in registered_models:
            print(f"\n📦 Model: {model.name}")
            print(f"   Description: {model.description or 'No description'}")
            
            # Get all versions
            versions = self.client.search_model_versions(f"name='{model.name}'")
            
            print(f"   Total Versions: {len(versions)}")
            
            for version in versions:
                print(f"   └── Version {version.version}")
                print(f"       Stage: {version.current_stage}")
                print(f"       Status: {version.status}")
                
                # Get run metrics for this version
                if version.run_id:
                    run = self.client.get_run(version.run_id)
                    if 'test_r2' in run.data.metrics:
                        print(f"       Test R²: {run.data.metrics['test_r2']:.4f}")
                    if 'param_model_type' in run.data.params:
                        print(f"       Model Type: {run.data.params['param_model_type']}")
    
    def performance_analysis(self, df):
        """Analyze model performance trends"""
        print("\n📈 Performance Analysis")
        print("=" * 40)
        
        if 'test_r2' in df.columns:
            print(f"📊 R² Score Statistics:")
            print(f"   Mean: {df['test_r2'].mean():.4f}")
            print(f"   Std:  {df['test_r2'].std():.4f}")
            print(f"   Min:  {df['test_r2'].min():.4f}")
            print(f"   Max:  {df['test_r2'].max():.4f}")
        
        if 'test_mse' in df.columns:
            print(f"\n📊 MSE Statistics:")
            print(f"   Mean: {df['test_mse'].mean():.4f}")
            print(f"   Std:  {df['test_mse'].std():.4f}")
            print(f"   Min:  {df['test_mse'].min():.4f}")
            print(f"   Max:  {df['test_mse'].max():.4f}")
        
        # Model type performance
        if 'param_model_type' in df.columns and 'test_r2' in df.columns:
            print(f"\n📊 Performance by Model Type:")
            model_performance = df.groupby('param_model_type')['test_r2'].agg(['mean', 'std', 'count'])
            print(model_performance.to_string())
    
    def generate_recommendations(self, df):
        """Generate model selection recommendations"""
        print("\n💡 Model Selection Recommendations")
        print("=" * 50)
        
        if 'test_r2' in df.columns and 'param_model_type' in df.columns:
            # Best overall model
            best_idx = df['test_r2'].idxmax()
            best_model = df.loc[best_idx]
            
            print(f"🥇 Best Overall Model:")
            print(f"   Type: {best_model.get('param_model_type', 'Unknown')}")
            print(f"   R²: {best_model['test_r2']:.4f}")
            print(f"   Run: {best_model['run_name']}")
            
            # Most consistent model type
            model_stats = df.groupby('param_model_type')['test_r2'].agg(['mean', 'std'])
            most_consistent = model_stats.loc[model_stats['std'].idxmin()]
            
            print(f"\n🎯 Most Consistent Model Type:")
            print(f"   Type: {model_stats['std'].idxmin()}")
            print(f"   Mean R²: {most_consistent['mean']:.4f}")
            print(f"   Std Dev: {most_consistent['std']:.4f}")
            
            # Production recommendation
            print(f"\n🚀 Production Recommendation:")
            if best_model['test_r2'] > 0.99:
                print(f"   ✅ Deploy {best_model.get('param_model_type', 'Unknown')} model")
                print(f"   ✅ Excellent performance (R² > 0.99)")
            elif best_model['test_r2'] > 0.95:
                print(f"   ⚠️  Consider {best_model.get('param_model_type', 'Unknown')} model")
                print(f"   ⚠️  Good performance (R² > 0.95)")
            else:
                print(f"   ❌ Need more training (R² < 0.95)")
                print(f"   ❌ Consider feature engineering or more data")

def main():
    """Main comparison workflow"""
    comparator = ModelComparator()
    
    # Get all experiment data
    df = comparator.compare_experiments()
    
    if not df.empty:
        # Create comparison table
        best_model = comparator.create_comparison_table(df)
        
        # Compare model registry
        comparator.compare_model_registry()
        
        # Performance analysis
        comparator.performance_analysis(df)
        
        # Generate recommendations
        comparator.generate_recommendations(df)
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"model_comparison_report_{timestamp}.csv"
        df.to_csv(report_file, index=False)
        print(f"\n💾 Detailed report saved to: {report_file}")
        
    else:
        print("❌ No experiment data found. Make sure MLflow is running and experiments exist.")

if __name__ == "__main__":
    main()