#!/usr/bin/env python3
"""
Test Azure ML connection and workspace access
"""

from azureml.core import Workspace

def test_azure_ml_connection():
    """Test connection to Azure ML workspace"""
    try:
        # Connect to workspace using default authentication
        # This will use Azure CLI authentication automatically
        ws = Workspace(
            subscription_id="7f13a298-0439-457b-8578-04dbd8fee85b",
            resource_group="rg-mlops-demo",
            workspace_name="stock-price-mlops"
        )
        
        print("✅ Successfully connected to Azure ML workspace!")
        print(f"Workspace name: {ws.name}")
        print(f"Resource group: {ws.resource_group}")
        print(f"Location: {ws.location}")
        print(f"Subscription ID: {ws.subscription_id}")
        
        # Test getting compute targets
        compute_targets = ws.compute_targets
        print(f"Available compute targets: {len(compute_targets)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Azure ML workspace: {e}")
        return False

if __name__ == "__main__":
    test_azure_ml_connection()