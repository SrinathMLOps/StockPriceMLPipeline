#!/usr/bin/env python3
"""
Create Azure ML compute instance for training
"""

from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

def create_compute_instance():
    """Create a compute instance in Azure ML"""
    try:
        # Connect to workspace
        ws = Workspace(
            subscription_id="7f13a298-0439-457b-8578-04dbd8fee85b",
            resource_group="rg-mlops-demo",
            workspace_name="stock-price-mlops"
        )
        
        print("✅ Connected to Azure ML workspace")
        
        # Compute instance configuration
        compute_name = "stock-price-compute"
        
        # Check if compute instance already exists
        try:
            compute_target = ComputeTarget(workspace=ws, name=compute_name)
            print(f"✅ Found existing compute instance: {compute_name}")
            print(f"   Status: {compute_target.provisioning_state}")
            return compute_target
        except ComputeTargetException:
            print(f"Creating new compute instance: {compute_name}")
        
        # Create compute configuration
        # Using Standard_DS3_v2 which is good for free tier
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="Standard_DS3_v2",  # 4 cores, 14GB RAM - good for ML workloads
            min_nodes=0,                # Scale down to 0 when not in use
            max_nodes=1,                # Maximum 1 node for cost control
            idle_seconds_before_scaledown=300,  # 5 minutes idle before scale down
            vm_priority="dedicated"     # Use dedicated VMs
        )
        
        # Create the compute instance
        compute_target = ComputeTarget.create(
            ws, 
            compute_name, 
            compute_config
        )
        
        # Wait for completion
        print("⏳ Creating compute instance... This may take 5-10 minutes")
        compute_target.wait_for_completion(show_output=True)
        
        print("✅ Compute instance created successfully!")
        print(f"   Name: {compute_target.name}")
        print(f"   VM Size: {compute_target.vm_size}")
        print(f"   Status: {compute_target.provisioning_state}")
        
        return compute_target
        
    except Exception as e:
        print(f"❌ Failed to create compute instance: {e}")
        return None

if __name__ == "__main__":
    compute_target = create_compute_instance()
    if compute_target:
        print("\n🎉 Ready to run Azure ML experiments!")