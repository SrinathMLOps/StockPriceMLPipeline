# Azure ML Setup Commands
# Run these commands in Azure CLI or Cloud Shell

# 1. Login to Azure
az login

# 2. Set subscription (replace with your subscription ID)
az account set --subscription "<your-subscription-id>"

# 3. Create resource group
az group create --name rg-stockprice-ml --location eastus

# 4. Create Azure ML workspace
az ml workspace create \
    --name ws-stockprice-ml \
    --resource-group rg-stockprice-ml \
    --location eastus \
    --description "Stock Price Prediction MLOps Workspace"

# 5. Set default workspace
az configure --defaults group=rg-stockprice-ml workspace=ws-stockprice-ml

# 6. Verify setup
az ml workspace show --name ws-stockprice-ml