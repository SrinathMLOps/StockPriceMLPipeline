#!/usr/bin/env python3
"""
Quick Setup Script for Azure ML Stock Price Prediction
Run this to set up your Azure ML workspace like the screenshot
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        "azureml-sdk[notebooks,automl]==1.51.0",
        "azureml-mlflow==1.51.0",
        "azureml-widgets==1.51.0"
    ]
    
    print("📦 Installing Azure ML SDK...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def create_config_file():
    """Create Azure ML configuration template"""
    config_template = {
        "subscription_id": "<your-subscription-id>",
        "resource_group": "rg-stockprice-ml",
        "workspace_name": "ws-stockprice-ml",
        "location": "eastus"
    }
    
    import json
    with open("azure_ml_config.json", "w") as f:
        json.dump(config_template, f, indent=2)
    
    print("✅ Created azure_ml_config.json template")

def create_azure_cli_commands():
    """Create Azure CLI setup commands"""
    commands = """
# Azure ML Setup Commands
# Run these commands in Azure CLI or Cloud Shell

# 1. Login to Azure
az login

# 2. Set subscription (replace with your subscription ID)
az account set --subscription "<your-subscription-id>"

# 3. Create resource group
az group create --name rg-stockprice-ml --location eastus

# 4. Create Azure ML workspace
az ml workspace create \\
    --name ws-stockprice-ml \\
    --resource-group rg-stockprice-ml \\
    --location eastus \\
    --description "Stock Price Prediction MLOps Workspace"

# 5. Set default workspace
az configure --defaults group=rg-stockprice-ml workspace=ws-stockprice-ml

# 6. Verify setup
az ml workspace show --name ws-stockprice-ml
"""
    
    with open("azure_setup_commands.sh", "w") as f:
        f.write(commands)
    
    print("✅ Created azure_setup_commands.sh")

def create_notebook():
    """Create Jupyter notebook for Azure ML"""
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 Stock Price Prediction with Azure ML\\n",
    "\\n",
    "This notebook demonstrates the complete Azure ML workflow for stock price prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\\n",
    "from azureml.core import Workspace, Experiment, Environment\\n",
    "from azureml.core.compute import ComputeTarget, AmlCompute\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "\\n",
    "print(\\"Azure ML SDK version:\\", azureml.core.VERSION)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Connect to workspace\\n",
    "ws = Workspace.from_config()\\n",
    "print(f\\"Workspace: {ws.name}\\")\\n",
    "print(f\\"Location: {ws.location}\\")\\n",
    "print(f\\"Resource Group: {ws.resource_group}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the complete setup\\n",
    "exec(open('azure_ml_implementation.py').read())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎯 Results\\n",
    "\\n",
    "After running the setup, you'll have:\\n",
    "- ✅ Azure ML Workspace\\n",
    "- ✅ Compute Cluster\\n",
    "- ✅ Training Environment\\n",
    "- ✅ Dataset Registered\\n",
    "- ✅ Experiments Running\\n",
    "- ✅ Model Registry\\n",
    "\\n",
    "Visit Azure ML Studio to see the interface like in the screenshot!"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open("azure_ml_stock_prediction.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("✅ Created azure_ml_stock_prediction.ipynb")

def create_readme():
    """Create README for Azure ML setup"""
    readme_content = """# ☁️ Azure ML Stock Price Prediction

## 🎯 Overview

This implementation creates the same professional Azure ML setup as shown in the screenshot, featuring:

- **Workspace**: Enterprise ML workspace
- **Experiments**: Training job tracking
- **Models**: Versioned model registry
- **Endpoints**: Real-time prediction APIs
- **Compute**: Scalable training clusters
- **Datasets**: Managed data assets

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
python setup_azure_ml.py
```

### Step 2: Setup Azure Resources
```bash
# Run Azure CLI commands
bash azure_setup_commands.sh
```

### Step 3: Configure Workspace
```bash
# Edit azure_ml_config.json with your details
{
  "subscription_id": "your-subscription-id",
  "resource_group": "rg-stockprice-ml",
  "workspace_name": "ws-stockprice-ml"
}
```

### Step 4: Run Implementation
```bash
python azure_ml_implementation.py
```

## 📊 What You'll Get

### Azure ML Studio Interface
- **Experiments Tab**: All training runs with metrics
- **Models Tab**: Registered model versions
- **Endpoints Tab**: Deployed prediction APIs
- **Compute Tab**: Training clusters
- **Datasets Tab**: Training data assets

### Professional Features
- **MLflow Integration**: Experiment tracking
- **HyperDrive**: Automated hyperparameter tuning
- **Model Registry**: Version control and deployment
- **Real-time Endpoints**: Production APIs
- **Monitoring**: Performance tracking

## 🎯 Expected Results

After setup, you'll have:
- ✅ **Professional ML workspace** (like screenshot)
- ✅ **Multiple experiments** running
- ✅ **Model comparison** and selection
- ✅ **Production deployment** ready
- ✅ **Enterprise-grade MLOps** pipeline

## 🔗 Access Your Workspace

Visit: https://ml.azure.com
- Select your workspace: `ws-stockprice-ml`
- View experiments and models
- Monitor training progress
- Deploy to production

This creates the exact same professional interface shown in your screenshot! 🚀
"""
    
    with open("AZURE_ML_README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created AZURE_ML_README.md")

def main():
    """Main setup function"""
    print("☁️ Azure ML Stock Price Prediction Setup")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Create configuration files
    create_config_file()
    create_azure_cli_commands()
    create_notebook()
    create_readme()
    
    print("\n✅ Setup Complete!")
    print("\n🎯 Next Steps:")
    print("1. Edit azure_ml_config.json with your Azure details")
    print("2. Run: bash azure_setup_commands.sh")
    print("3. Run: python azure_ml_implementation.py")
    print("4. Visit: https://ml.azure.com")
    print("\n🚀 You'll have the same professional interface as the screenshot!")

if __name__ == "__main__":
    main()