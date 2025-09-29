#!/usr/bin/env python3
"""
Simple GitHub Showcase Creator for MLflow
"""

import mlflow
import json
from mlflow.tracking import MlflowClient
from datetime import datetime

def create_simple_showcase():
    """Create a simple GitHub showcase"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    print("Creating GitHub Showcase...")
    
    # Get experiments
    experiments = client.search_experiments()
    
    # Get all runs
    all_runs = []
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            run_info = {
                "experiment": exp.name,
                "run_id": run.info.run_id[:8],
                "status": run.info.status,
                "r2_score": run.data.metrics.get("test_r2", 0),
                "model_type": run.data.params.get("model_type", "Unknown")
            }
            all_runs.append(run_info)
    
    # Sort by R2 score
    all_runs.sort(key=lambda x: x["r2_score"], reverse=True)
    
    # Create simple markdown
    markdown = """# MLflow Experiments Showcase

## Project Overview
Stock Price Prediction MLOps Pipeline with MLflow tracking and model registry.

## Experiments Summary

"""
    
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        markdown += f"### {exp.name}\n"
        markdown += f"- Total Runs: {len(runs)}\n"
        markdown += f"- Status: Active\n\n"
    
    markdown += "## Model Performance Results\n\n"
    markdown += "| Experiment | Model Type | R2 Score | Status |\n"
    markdown += "|------------|------------|----------|--------|\n"
    
    for run in all_runs:
        markdown += f"| {run['experiment']} | {run['model_type']} | {run['r2_score']:.4f} | {run['status']} |\n"
    
    markdown += f"""
## Best Model Performance
- **Highest R2 Score**: {all_runs[0]['r2_score']:.4f}
- **Model Type**: {all_runs[0]['model_type']}
- **Experiment**: {all_runs[0]['experiment']}

## How to Run
1. Start MLflow: `docker compose -f docker-compose-simple.yml up -d`
2. Train models: `python train_model_simple.py`
3. View dashboard: `http://localhost:5000`

## Screenshots
*Add screenshots of your MLflow dashboard here*

1. Experiments page showing all runs
2. Model comparison metrics
3. Best performing model details
"""
    
    # Save with UTF-8 encoding
    with open("GITHUB_SHOWCASE.md", 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print("✅ GitHub showcase created: GITHUB_SHOWCASE.md")
    print(f"📊 Found {len(experiments)} experiments with {len(all_runs)} total runs")
    print(f"🏆 Best model: {all_runs[0]['model_type']} with {all_runs[0]['r2_score']:.4f} R2 score")

if __name__ == "__main__":
    create_simple_showcase()