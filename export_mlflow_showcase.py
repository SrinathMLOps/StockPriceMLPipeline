#!/usr/bin/env python3
"""
MLflow Showcase Export Script
Exports MLflow experiments and models data for GitHub showcase
"""

import mlflow
import json
import pandas as pd
from mlflow.tracking import MlflowClient
from datetime import datetime
import os

def export_mlflow_data():
    """Export MLflow experiments and models to JSON for showcase"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    print("🚀 Exporting MLflow Data for GitHub Showcase")
    print("=" * 50)
    
    export_data = {
        "project_name": "Stock Price MLOps Pipeline",
        "export_date": datetime.now().isoformat(),
        "summary": {
            "total_experiments": 0,
            "total_runs": 0,
            "total_models": 0,
            "best_accuracy": 0
        },
        "experiments": [],
        "models": [],
        "runs": []
    }
    
    try:
        # Get all experiments
        experiments = client.search_experiments()
        export_data["summary"]["total_experiments"] = len(experiments)
        
        print(f"📊 Found {len(experiments)} experiments")
        
        for exp in experiments:
            print(f"   Processing experiment: {exp.name}")
            
            exp_data = {
                "id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": exp.creation_time,
                "runs_count": 0,
                "best_run": None
            }
            
            # Get runs for this experiment
            runs = client.search_runs(exp.experiment_id)
            exp_data["runs_count"] = len(runs)
            export_data["summary"]["total_runs"] += len(runs)
            
            best_r2 = 0
            best_run = None
            
            for run in runs:
                run_data = {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "experiment_id": exp.experiment_id,
                    "experiment_name": exp.name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                
                # Track best run
                test_r2 = run.data.metrics.get("test_r2", 0)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_run = run_data
                    
                if test_r2 > export_data["summary"]["best_accuracy"]:
                    export_data["summary"]["best_accuracy"] = test_r2
                
                export_data["runs"].append(run_data)
            
            exp_data["best_run"] = best_run
            export_data["experiments"].append(exp_data)
        
        # Get registered models
        models = client.search_registered_models()
        export_data["summary"]["total_models"] = len(models)
        
        print(f"📦 Found {len(models)} registered models")
        
        for model in models:
            print(f"   Processing model: {model.name}")
            
            model_data = {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "versions": [],
                "production_version": None,
                "latest_version": None
            }
            
            # Get model versions
            versions = client.search_model_versions(f"name='{model.name}'")
            
            for version in versions:
                # Get run data for this version
                run_data = None
                if version.run_id:
                    try:
                        run = client.get_run(version.run_id)
                        run_data = {
                            "metrics": run.data.metrics,
                            "params": run.data.params
                        }
                    except:
                        pass
                
                version_data = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "description": version.description,
                    "run_id": version.run_id,
                    "run_data": run_data
                }
                
                # Track production and latest versions
                if version.current_stage == "Production":
                    model_data["production_version"] = version_data
                
                model_data["versions"].append(version_data)
            
            # Sort versions by version number
            model_data["versions"].sort(key=lambda x: int(x["version"]), reverse=True)
            model_data["latest_version"] = model_data["versions"][0] if model_data["versions"] else None
            
            export_data["models"].append(model_data)
        
        # Save to JSON file
        output_file = "mlflow_showcase_data.json"
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n✅ Export completed successfully!")
        print(f"📁 Data saved to: {output_file}")
        
        # Print summary
        print(f"\n📊 Export Summary:")
        print(f"   Experiments: {export_data['summary']['total_experiments']}")
        print(f"   Total Runs: {export_data['summary']['total_runs']}")
        print(f"   Models: {export_data['summary']['total_models']}")
        print(f"   Best Accuracy: {export_data['summary']['best_accuracy']:.4f}")
        
        return export_data
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        return None

def create_markdown_showcase(export_data):
    """Create a markdown showcase from exported data"""
    
    if not export_data:
        print("❌ No data to create showcase")
        return
    
    markdown_content = f"""# 🚀 MLflow Showcase - {export_data['project_name']}

*Generated on: {export_data['export_date'][:19]}*

## 📊 Project Summary

- **Total Experiments**: {export_data['summary']['total_experiments']}
- **Total Runs**: {export_data['summary']['total_runs']}
- **Registered Models**: {export_data['summary']['total_models']}
- **Best Model Accuracy**: {export_data['summary']['best_accuracy']:.4f} (R² Score)

## 🧪 Experiments

"""
    
    for exp in export_data['experiments']:
        markdown_content += f"""### {exp['name']}
- **Runs**: {exp['runs_count']}
- **Status**: {exp['lifecycle_stage']}
"""
        
        if exp['best_run']:
            best_run = exp['best_run']
            markdown_content += f"""- **Best Run**: {best_run['metrics'].get('test_r2', 'N/A')} R² Score
- **Model Type**: {best_run['params'].get('model_type', 'N/A')}

"""
    
    markdown_content += """## 📦 Model Registry

"""
    
    for model in export_data['models']:
        markdown_content += f"""### {model['name']}

| Version | Stage | Status | R² Score | Model Type |
|---------|-------|--------|----------|------------|
"""
        
        for version in model['versions']:
            stage = version['stage'] or 'None'
            r2_score = 'N/A'
            model_type = 'N/A'
            
            if version['run_data']:
                r2_score = version['run_data']['metrics'].get('test_r2', 'N/A')
                model_type = version['run_data']['params'].get('model_type', 'N/A')
                
                if isinstance(r2_score, float):
                    r2_score = f"{r2_score:.4f}"
            
            # Highlight production version
            version_str = f"**{version['version']}**" if stage == "Production" else version['version']
            stage_str = f"**{stage}**" if stage == "Production" else stage
            
            markdown_content += f"| {version_str} | {stage_str} | {version['status']} | {r2_score} | {model_type} |\n"
        
        markdown_content += "\n"
    
    # Add production model details
    for model in export_data['models']:
        if model['production_version']:
            prod_version = model['production_version']
            markdown_content += f"""## 🏆 Production Model: {model['name']} v{prod_version['version']}

- **Stage**: Production
- **Status**: {prod_version['status']}
"""
            if prod_version['run_data']:
                metrics = prod_version['run_data']['metrics']
                params = prod_version['run_data']['params']
                
                markdown_content += f"""- **R² Score**: {metrics.get('test_r2', 'N/A')}
- **Model Type**: {params.get('model_type', 'N/A')}
- **Features**: {params.get('features', 'N/A')}
"""
    
    markdown_content += """
## 🚀 How to Reproduce

1. **Start MLflow Services**:
   ```bash
   docker compose -f docker-compose-simple.yml up -d
   ```

2. **Train Models**:
   ```bash
   python train_model_simple.py
   python advanced_training.py
   python train_experiment_2.py
   ```

3. **Promote Best Model**:
   ```bash
   python promote_to_production.py
   ```

4. **Access Dashboard**:
   ```bash
   # Open http://localhost:5000
   ```

---
*This showcase demonstrates enterprise-level MLOps practices with complete model lifecycle management.*
"""
    
    # Save markdown file
    with open("MLFLOW_SHOWCASE_GENERATED.md", 'w') as f:
        f.write(markdown_content)
    
    print(f"📝 Markdown showcase created: MLFLOW_SHOWCASE_GENERATED.md")

def main():
    """Main function to export MLflow data and create showcase"""
    
    print("🎯 MLflow Showcase Export Tool")
    print("=" * 40)
    
    # Export data
    export_data = export_mlflow_data()
    
    if export_data:
        # Create markdown showcase
        create_markdown_showcase(export_data)
        
        print(f"\n🎉 Showcase export completed!")
        print(f"📁 Files created:")
        print(f"   - mlflow_showcase_data.json")
        print(f"   - MLFLOW_SHOWCASE_GENERATED.md")
        print(f"\n💡 Next steps:")
        print(f"   1. Take screenshots of your MLflow dashboard")
        print(f"   2. Add screenshots to the markdown file")
        print(f"   3. Commit to GitHub repository")
        print(f"   4. Consider deploying to cloud for live demo")

if __name__ == "__main__":
    main()