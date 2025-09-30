# ☁️ Azure Machine Learning Studio Setup Guide

## 🎯 Overview

This guide shows how to migrate your MLflow-based stock price prediction project to Azure Machine Learning Studio, creating an enterprise-grade MLOps pipeline similar to what you saw in the screenshot.

## 🏗️ Azure ML Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AZURE ML WORKSPACE                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   COMPUTE       │   EXPERIMENTS   │      MODEL REGISTRY         │
│                 │                 │                             │
│ • Compute       │ • Training      │ • Model Versions            │
│   Instances     │   Jobs          │ • Model Deployment          │
│ • Compute       │ • Pipelines     │ • Model Monitoring          │
│   Clusters      │ • AutoML        │ • A/B Testing               │
│ • Inference     │ • Notebooks     │ • Model Endpoints           │
│   Clusters      │ • Datasets      │ • Real-time Scoring         │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 🚀 Step 1: Azure ML Workspace Setup

### **Prerequisites:**
- Azure subscription
- Azure CLI installed
- Python 3.8+

### **Create Workspace:**
```bash
# Install Azure ML SDK
pip install azureml-sdk azureml-widgets

# Login to Azure
az login

# Create resource group
az group create --name rg-stockprice-ml --location eastus

# Create Azure ML workspace
az ml workspace create --name ws-stockprice-ml --resource-group rg-stockprice-ml
```

### **Python Setup:**
```python
from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import RunConfiguration

# Connect to workspace
ws = Workspace.from_config()  # or Workspace.get()
print(f"Workspace: {ws.name}, Location: {ws.location}")
```

## 🖥️ Step 2: Compute Resources

### **Create Compute Cluster:**
```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "stock-ml-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print(f"Found existing cluster: {cluster_name}")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D3_V2",
        min_nodes=0,
        max_nodes=4,
        idle_seconds_before_scaledown=300
    )
    
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)
```

## 📊 Step 3: Convert Your Training Script

### **Azure ML Training Script (`train_azure.py`):**
```python
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import mlflow
from azureml.core import Run

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='random_forest')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    args = parser.parse_args()
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Enable MLflow tracking
    mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
    mlflow.set_experiment(run.experiment.name)
    
    with mlflow.start_run():
        print("🚀 Starting Azure ML training...")
        
        # Create sample data (replace with your data loading)
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'ma_3': np.random.normal(100, 20, n_samples),
            'pct_change_1d': np.random.normal(0.001, 0.02, n_samples),
            'volume': np.random.lognormal(15, 1, n_samples)
        })
        
        y = X['ma_3'] + np.random.normal(0, 2, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Log metrics
        run.log("test_r2", r2)
        run.log("test_mse", mse)
        run.log("n_estimators", args.n_estimators)
        run.log("max_depth", args.max_depth)
        
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_param("algorithm", args.algorithm)
        
        # Save model
        os.makedirs("outputs", exist_ok=True)
        model_path = "outputs/model.pkl"
        joblib.dump(model, model_path)
        
        # Register model
        model = run.register_model(
            model_name="stock-price-predictor",
            model_path="outputs/model.pkl",
            description="Stock price prediction model",
            tags={"algorithm": args.algorithm, "r2_score": r2}
        )
        
        print(f"✅ Model registered: {model.name} v{model.version}")
        print(f"📊 Test R²: {r2:.4f}")

if __name__ == "__main__":
    main()
```

## 🧪 Step 4: Create Experiment

### **Submit Training Job:**
```python
from azureml.core import ScriptRunConfig, Environment, Experiment
from azureml.core.conda_dependencies import CondaDependencies

# Create environment
env = Environment(name="stock-ml-env")
conda_deps = CondaDependencies()
conda_deps.add_pip_package("scikit-learn")
conda_deps.add_pip_package("pandas")
conda_deps.add_pip_package("numpy")
conda_deps.add_pip_package("mlflow")
conda_deps.add_pip_package("azureml-mlflow")
env.python.conda_dependencies = conda_deps

# Create experiment
experiment = Experiment(workspace=ws, name="stock-price-prediction")

# Configure run
config = ScriptRunConfig(
    source_directory=".",
    script="train_azure.py",
    arguments=[
        "--algorithm", "random_forest",
        "--n-estimators", 200,
        "--max-depth", 15
    ],
    compute_target=compute_target,
    environment=env
)

# Submit run
run = experiment.submit(config)
print(f"Run submitted: {run.get_portal_url()}")

# Wait for completion
run.wait_for_completion(show_output=True)
```

## 📊 Step 5: Hyperparameter Tuning

### **Azure ML HyperDrive:**
```python
from azureml.train.hyperdrive import RandomParameterSampling, HyperDriveConfig
from azureml.train.hyperdrive import choice, uniform
from azureml.train.hyperdrive import PrimaryMetricGoal

# Define parameter space
param_sampling = RandomParameterSampling({
    "--n-estimators": choice(100, 200, 300),
    "--max-depth": choice(10, 15, 20, None)
})

# Configure HyperDrive
hyperdrive_config = HyperDriveConfig(
    run_config=config,
    hyperparameter_sampling=param_sampling,
    primary_metric_name="test_r2",
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=12,
    max_concurrent_runs=4
)

# Submit hyperparameter tuning
hyperdrive_run = experiment.submit(hyperdrive_config)
print(f"HyperDrive run: {hyperdrive_run.get_portal_url()}")
```

## 🚀 Step 6: Model Deployment

### **Real-time Endpoint:**
```python
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice

# Get best model
model = Model(ws, name="stock-price-predictor")

# Create scoring script
scoring_script = """
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("stock-price-predictor")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        prediction = model.predict(np.array(data["data"]))
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
"""

with open("score.py", "w") as f:
    f.write(scoring_script)

# Configure inference
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# Deploy to ACI
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    description="Stock price prediction endpoint"
)

service = Model.deploy(
    workspace=ws,
    name="stock-price-endpoint",
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(f"Endpoint URL: {service.scoring_uri}")
```

## 📊 Step 7: Monitoring & MLflow Integration

### **Enable MLflow Tracking:**
```python
import mlflow
from azureml.core import Workspace

# Set MLflow tracking to Azure ML
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# Your existing MLflow code works!
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.99)
    mlflow.sklearn.log_model(model, "model")
```

## 🎯 Step 8: Create Pipeline

### **Azure ML Pipeline:**
```python
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

# Create pipeline steps
train_step = PythonScriptStep(
    name="train_model",
    script_name="train_azure.py",
    compute_target=compute_target,
    source_directory=".",
    runconfig=run_config
)

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[train_step])

# Publish pipeline
published_pipeline = pipeline.publish(
    name="Stock Price ML Pipeline",
    description="End-to-end stock price prediction pipeline"
)
```

## 📱 Step 9: Azure ML Studio UI

### **What You'll See (Like Your Screenshot):**

1. **Experiments Tab**: All your training runs
2. **Models Tab**: Registered model versions
3. **Endpoints Tab**: Deployed models
4. **Pipelines Tab**: ML workflows
5. **Datasets Tab**: Training data
6. **Compute Tab**: Compute resources

### **Key Features:**
- **Experiment tracking** (like MLflow)
- **Model registry** with versioning
- **Real-time endpoints** for predictions
- **Batch inference** for large datasets
- **Model monitoring** and drift detection
- **AutoML** for automated model selection

## 🔗 Integration with Your Current Project

### **Migration Strategy:**
1. **Keep your local MLflow setup** for development
2. **Add Azure ML for production** workloads
3. **Use Azure ML pipelines** for automated training
4. **Deploy models** to Azure endpoints
5. **Monitor performance** in Azure ML Studio

### **Benefits of Azure ML:**
- **Enterprise security** and compliance
- **Scalable compute** resources
- **Built-in MLOps** capabilities
- **Integration** with Azure services
- **Cost management** and optimization

## 🎉 Final Result

You'll have a setup similar to your screenshot with:
- **Professional ML workspace**
- **Experiment tracking** and comparison
- **Model registry** with versions
- **Real-time endpoints** for predictions
- **Enterprise-grade MLOps** pipeline

This elevates your project from local MLflow to enterprise Azure ML! 🚀