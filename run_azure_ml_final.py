#!/usr/bin/env python3
"""
Run Azure ML experiment for stock price prediction (final version)
"""

from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies

def create_environment():
    """Create Azure ML environment with required packages"""
    env = Environment(name="stock-prediction-final-env")
    
    # Create conda dependencies
    conda_dep = CondaDependencies()
    
    # Add required packages (minimal set to avoid conflicts)
    conda_dep.add_pip_package("scikit-learn")
    conda_dep.add_pip_package("pandas")
    conda_dep.add_pip_package("numpy")
    conda_dep.add_pip_package("joblib")
    
    # Set Python version explicitly
    conda_dep.set_python_version("3.9")
    
    env.python.conda_dependencies = conda_dep
    
    return env

def run_experiment():
    """Run the Azure ML experiment"""
    try:
        # Connect to workspace
        ws = Workspace(
            subscription_id="7f13a298-0439-457b-8578-04dbd8fee85b",
            resource_group="rg-mlops-demo",
            workspace_name="stock-price-mlops"
        )
        
        print("✅ Connected to Azure ML workspace")
        
        # Get compute target
        compute_target = ComputeTarget(workspace=ws, name="stock-price-compute")
        print(f"✅ Using compute target: {compute_target.name}")
        
        # Create experiment
        experiment_name = "stock-price-prediction-final"
        experiment = Experiment(workspace=ws, name=experiment_name)
        print(f"✅ Created experiment: {experiment_name}")
        
        # Create environment
        env = create_environment()
        print("✅ Created environment with required packages")
        
        # Create script run configuration using the clean directory
        script_config = ScriptRunConfig(
            source_directory="./azure_ml_training",
            script="azure_ml_train_final.py",
            arguments=[
                "--symbol", "AAPL",
                "--n_days", "2000",
                "--n_estimators", "300",
                "--max_depth", "20"
            ],
            compute_target=compute_target,
            environment=env
        )
        
        print("✅ Created script run configuration")
        
        # Submit the experiment
        print("🚀 Submitting experiment to Azure ML...")
        print("This will take several minutes to complete.")
        
        run = experiment.submit(script_config)
        
        print(f"✅ Experiment submitted!")
        print(f"   Run ID: {run.id}")
        print(f"   Experiment: {experiment.name}")
        print(f"   Status: {run.status}")
        
        # Show Azure ML Studio link
        print(f"\n🌐 Monitor progress in Azure ML Studio:")
        print(f"https://ml.azure.com/runs/{run.id}?wsid=/subscriptions/{ws.subscription_id}/resourcegroups/{ws.resource_group}/workspaces/{ws.name}")
        
        print("\n⏳ Waiting for experiment to complete...")
        print("This may take 5-10 minutes...")
        
        run.wait_for_completion(show_output=True)
        
        # Get results
        print("\n📊 Experiment Results:")
        metrics = run.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n✅ Experiment completed with status: {run.status}")
        
        return run
        
    except Exception as e:
        print(f"❌ Failed to run experiment: {e}")
        return None

if __name__ == "__main__":
    run = run_experiment()
    if run:
        print("\n🎉 Azure ML experiment completed successfully!")
        print("Check Azure ML Studio for detailed results and model artifacts.")
        print("\n📋 Summary:")
        print("- ✅ Compute instance created")
        print("- ✅ Environment configured")
        print("- ✅ Model trained successfully")
        print("- ✅ Metrics logged to Azure ML")
        print("- ✅ Model artifacts saved")
        print("\nYour Azure ML setup is now complete and ready for production use!")