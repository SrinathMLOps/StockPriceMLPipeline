#!/usr/bin/env python3
"""
Complete Azure ML Implementation for Stock Price Prediction
Creates the same professional setup as shown in Azure ML Studio
"""

import os
import json
from azureml.core import Workspace, Experiment, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.hyperdrive import RandomParameterSampling, HyperDriveConfig
from azureml.train.hyperdrive import choice, uniform, PrimaryMetricGoal
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
import pandas as pd
import numpy as np

class StockPriceAzureML:
    def __init__(self, subscription_id=None, resource_group=None, workspace_name=None):
        """Initialize Azure ML workspace"""
        self.subscription_id = subscription_id
        self.resource_group = resource_group or "rg-stockprice-ml"
        self.workspace_name = workspace_name or "ws-stockprice-ml"
        self.ws = None
        self.compute_target = None
        self.environment = None
        
    def setup_workspace(self):
        """Create or connect to Azure ML workspace"""
        try:
            # Try to load existing workspace
            self.ws = Workspace.get(
                name=self.workspace_name,
                subscription_id=self.subscription_id,
                resource_group=self.resource_group
            )
            print(f"✅ Connected to existing workspace: {self.ws.name}")
        except Exception as e:
            print(f"❌ Workspace not found: {e}")
            print("💡 Please create workspace first using Azure CLI:")
            print(f"   az ml workspace create --name {self.workspace_name} --resource-group {self.resource_group}")
            return False
        
        return True
    
    def create_compute_cluster(self, cluster_name="stock-ml-cluster"):
        """Create compute cluster for training"""
        try:
            self.compute_target = ComputeTarget(workspace=self.ws, name=cluster_name)
            print(f"✅ Found existing compute cluster: {cluster_name}")
        except ComputeTargetException:
            print(f"🔧 Creating new compute cluster: {cluster_name}")
            
            compute_config = AmlCompute.provisioning_configuration(
                vm_size="STANDARD_D3_V2",
                min_nodes=0,
                max_nodes=4,
                idle_seconds_before_scaledown=300,
                description="Stock price ML compute cluster"
            )
            
            self.compute_target = ComputeTarget.create(self.ws, cluster_name, compute_config)
            self.compute_target.wait_for_completion(show_output=True)
            print(f"✅ Compute cluster created: {cluster_name}")
    
    def create_environment(self):
        """Create ML environment with dependencies"""
        self.environment = Environment(name="stock-ml-env")
        
        # Define conda dependencies
        conda_deps = CondaDependencies()
        conda_deps.add_pip_package("scikit-learn==1.3.0")
        conda_deps.add_pip_package("pandas==2.0.3")
        conda_deps.add_pip_package("numpy==1.24.3")
        conda_deps.add_pip_package("mlflow==2.5.0")
        conda_deps.add_pip_package("azureml-mlflow==1.51.0")
        conda_deps.add_pip_package("joblib==1.3.1")
        
        self.environment.python.conda_dependencies = conda_deps
        self.environment.docker.enabled = True
        
        print("✅ Environment created with ML dependencies")
    
    def create_dataset(self):
        """Create and register dataset"""
        # Generate sample data
        np.random.seed(42)
        n_samples = 5000
        
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        
        # Realistic stock features
        data = {
            'date': dates,
            'symbol': ['AAPL'] * n_samples,
            'ma_3': np.random.normal(150, 30, n_samples),
            'ma_7': np.random.normal(150, 25, n_samples),
            'ma_21': np.random.normal(150, 20, n_samples),
            'pct_change_1d': np.random.normal(0.001, 0.025, n_samples),
            'pct_change_5d': np.random.normal(0.005, 0.06, n_samples),
            'volume': np.random.lognormal(16, 1.2, n_samples),
            'volatility': np.random.uniform(0.1, 0.8, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'close_price': np.random.normal(150, 25, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save dataset
        os.makedirs("data", exist_ok=True)
        dataset_path = "data/stock_data.csv"
        df.to_csv(dataset_path, index=False)
        
        # Register dataset in Azure ML
        datastore = self.ws.get_default_datastore()
        dataset = Dataset.Tabular.from_delimited_files(path=dataset_path)
        dataset = dataset.register(
            workspace=self.ws,
            name="stock-price-dataset",
            description="Stock price prediction training data",
            tags={"format": "CSV", "samples": n_samples}
        )
        
        print(f"✅ Dataset registered: {dataset.name} ({n_samples} samples)")
        return dataset
    
    def create_training_script(self):
        """Create Azure ML training script"""
        training_script = '''
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn
from azureml.core import Run, Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='random_forest')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--dataset-name', type=str, default='stock-price-dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get Azure ML run context
    run = Run.get_context()
    ws = run.experiment.workspace
    
    # Enable MLflow tracking
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_experiment(run.experiment.name)
    
    with mlflow.start_run():
        print(f"🚀 Starting Azure ML training: {args.algorithm}")
        
        # Load dataset
        dataset = Dataset.get_by_name(ws, name=args.dataset_name)
        df = dataset.to_pandas_dataframe()
        print(f"📊 Loaded dataset: {len(df)} samples")
        
        # Prepare features
        feature_columns = ['ma_3', 'ma_7', 'pct_change_1d', 'volume', 'volatility', 'rsi']
        X = df[feature_columns]
        y = df['close_price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if args.algorithm == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth if args.max_depth > 0 else None,
                random_state=42
            )
        elif args.algorithm == 'ridge':
            model = Ridge(alpha=args.alpha)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Log metrics
        run.log("train_r2", train_r2)
        run.log("test_r2", test_r2)
        run.log("train_mse", train_mse)
        run.log("test_mse", test_mse)
        run.log("algorithm", args.algorithm)
        
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("n_features", len(feature_columns))
        
        # Save model
        os.makedirs("outputs", exist_ok=True)
        model_path = "outputs/model.pkl"
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Register model
        model_azure = run.register_model(
            model_name="stock-price-predictor",
            model_path="outputs/model.pkl",
            description=f"Stock price prediction using {args.algorithm}",
            tags={
                "algorithm": args.algorithm,
                "test_r2": f"{test_r2:.4f}",
                "framework": "scikit-learn"
            }
        )
        
        print(f"✅ Model Performance:")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Model: {model_azure.name} v{model_azure.version}")

if __name__ == "__main__":
    main()
'''
        
        with open("train_azure_stock.py", "w") as f:
            f.write(training_script)
        
        print("✅ Training script created: train_azure_stock.py")
    
    def run_single_experiment(self):
        """Run a single training experiment"""
        experiment = Experiment(workspace=self.ws, name="stock-price-prediction")
        
        config = ScriptRunConfig(
            source_directory=".",
            script="train_azure_stock.py",
            arguments=[
                "--algorithm", "random_forest",
                "--n-estimators", 200,
                "--max-depth", 15
            ],
            compute_target=self.compute_target,
            environment=self.environment
        )
        
        run = experiment.submit(config)
        print(f"🚀 Experiment submitted: {run.get_portal_url()}")
        
        return run
    
    def run_hyperparameter_tuning(self):
        """Run hyperparameter tuning with HyperDrive"""
        experiment = Experiment(workspace=self.ws, name="stock-price-hyperparameter-tuning")
        
        # Base configuration
        config = ScriptRunConfig(
            source_directory=".",
            script="train_azure_stock.py",
            compute_target=self.compute_target,
            environment=self.environment
        )
        
        # Parameter space
        param_sampling = RandomParameterSampling({
            "--algorithm": choice("random_forest", "ridge", "linear_regression"),
            "--n-estimators": choice(100, 200, 300),
            "--max-depth": choice(10, 15, 20),
            "--alpha": uniform(0.1, 10.0)
        })
        
        # HyperDrive configuration
        hyperdrive_config = HyperDriveConfig(
            run_config=config,
            hyperparameter_sampling=param_sampling,
            primary_metric_name="test_r2",
            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
            max_total_runs=20,
            max_concurrent_runs=4
        )
        
        # Submit hyperparameter tuning
        hyperdrive_run = experiment.submit(hyperdrive_config)
        print(f"🔧 HyperDrive submitted: {hyperdrive_run.get_portal_url()}")
        
        return hyperdrive_run
    
    def create_scoring_script(self):
        """Create scoring script for model deployment"""
        scoring_script = '''
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("stock-price-predictor")
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(data["data"])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Return results
        return {
            "predictions": prediction.tolist(),
            "model_version": "latest",
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }
'''
        
        with open("score.py", "w") as f:
            f.write(scoring_script)
        
        print("✅ Scoring script created: score.py")
    
    def deploy_model(self):
        """Deploy model as web service"""
        # Get latest model
        model = Model(self.ws, name="stock-price-predictor")
        
        # Create inference configuration
        inference_config = InferenceConfig(
            entry_script="score.py",
            environment=self.environment
        )
        
        # Configure deployment
        aci_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=2,
            description="Stock price prediction endpoint",
            tags={"model": "stock-price-predictor", "framework": "scikit-learn"}
        )
        
        # Deploy service
        service = Model.deploy(
            workspace=self.ws,
            name="stock-price-endpoint",
            models=[model],
            inference_config=inference_config,
            deployment_config=aci_config,
            overwrite=True
        )
        
        service.wait_for_deployment(show_output=True)
        print(f"✅ Model deployed: {service.scoring_uri}")
        
        return service
    
    def test_endpoint(self, service):
        """Test the deployed endpoint"""
        # Sample test data
        test_data = {
            "data": [
                {
                    "ma_3": 150.0,
                    "ma_7": 148.0,
                    "pct_change_1d": 0.02,
                    "volume": 1000000,
                    "volatility": 0.3,
                    "rsi": 65.0
                }
            ]
        }
        
        # Make prediction
        response = service.run(json.dumps(test_data))
        print(f"🧪 Test prediction: {response}")
        
        return response

def main():
    """Main implementation workflow"""
    print("☁️ Azure ML Stock Price Prediction Setup")
    print("=" * 50)
    
    # Initialize Azure ML
    azure_ml = StockPriceAzureML()
    
    # Step 1: Setup workspace
    if not azure_ml.setup_workspace():
        return
    
    # Step 2: Create compute resources
    azure_ml.create_compute_cluster()
    
    # Step 3: Create environment
    azure_ml.create_environment()
    
    # Step 4: Create and register dataset
    dataset = azure_ml.create_dataset()
    
    # Step 5: Create training script
    azure_ml.create_training_script()
    
    # Step 6: Run single experiment
    print("\n🧪 Running single experiment...")
    single_run = azure_ml.run_single_experiment()
    
    # Step 7: Run hyperparameter tuning
    print("\n🔧 Running hyperparameter tuning...")
    hyperdrive_run = azure_ml.run_hyperparameter_tuning()
    
    # Step 8: Create scoring script
    azure_ml.create_scoring_script()
    
    print("\n✅ Azure ML Setup Complete!")
    print(f"🌐 Workspace: {azure_ml.ws.name}")
    print(f"💻 Compute: {azure_ml.compute_target.name}")
    print(f"📊 Dataset: {dataset.name}")
    print(f"🧪 Single Run: {single_run.get_portal_url()}")
    print(f"🔧 HyperDrive: {hyperdrive_run.get_portal_url()}")
    
    print("\n🎯 Next Steps:")
    print("1. Monitor experiments in Azure ML Studio")
    print("2. Compare model performance")
    print("3. Register best model")
    print("4. Deploy to endpoint")
    print("5. Test real-time predictions")

if __name__ == "__main__":
    main()