# 🌐 Cloud Deployment Options for MLflow Dashboard

## 🚀 Option 1: Heroku Deployment (Free Tier Available)

### Step 1: Create Heroku App
```bash
# Install Heroku CLI
# Create new app
heroku create your-mlflow-app

# Set environment variables
heroku config:set MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
heroku config:set MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
```

### Step 2: Create Procfile
```bash
# Procfile
web: mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Step 3: Deploy
```bash
git add .
git commit -m "Deploy MLflow to Heroku"
git push heroku main
```

## 🌩️ Option 2: AWS EC2 Deployment

### Step 1: Launch EC2 Instance
```bash
# Launch Ubuntu instance
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose

# Clone your repo
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### Step 2: Configure Security Group
- Open port 5000 for MLflow
- Open port 8001 for API

### Step 3: Run MLflow
```bash
# Start services
docker-compose -f docker-compose-simple.yml up -d

# Access via public IP
http://your-ec2-public-ip:5000
```

## 🐙 Option 3: GitHub Pages + Static Export

### Step 1: Export MLflow Data
```python
# export_mlflow_data.py
import mlflow
import json
import pandas as pd
from mlflow.tracking import MlflowClient

def export_experiments_to_json():
    client = MlflowClient()
    
    # Get all experiments
    experiments = client.search_experiments()
    
    export_data = {
        "experiments": [],
        "models": [],
        "runs": []
    }
    
    for exp in experiments:
        exp_data = {
            "id": exp.experiment_id,
            "name": exp.name,
            "lifecycle_stage": exp.lifecycle_stage
        }
        export_data["experiments"].append(exp_data)
        
        # Get runs for this experiment
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "experiment_id": exp.experiment_id,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
            export_data["runs"].append(run_data)
    
    # Get registered models
    models = client.search_registered_models()
    for model in models:
        model_data = {
            "name": model.name,
            "description": model.description,
            "versions": []
        }
        
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in versions:
            version_data = {
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "run_id": version.run_id
            }
            model_data["versions"].append(version_data)
        
        export_data["models"].append(model_data)
    
    # Save to JSON
    with open('mlflow_export.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("✅ MLflow data exported to mlflow_export.json")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    export_experiments_to_json()
```

### Step 2: Create Static HTML Dashboard
```html
<!-- mlflow_dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>MLflow Dashboard - Stock Price Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .experiment { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
        .model { background: #f5f5f5; margin: 10px 0; padding: 15px; }
        .production { background: #e8f5e8; border-left: 4px solid #4caf50; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>🚀 MLflow Dashboard - Stock Price Prediction</h1>
    
    <h2>📊 Experiments</h2>
    <div id="experiments"></div>
    
    <h2>📦 Model Registry</h2>
    <div id="models"></div>
    
    <script>
        // Load and display MLflow data
        fetch('mlflow_export.json')
            .then(response => response.json())
            .then(data => {
                displayExperiments(data.experiments, data.runs);
                displayModels(data.models, data.runs);
            });
        
        function displayExperiments(experiments, runs) {
            const container = document.getElementById('experiments');
            experiments.forEach(exp => {
                const expRuns = runs.filter(r => r.experiment_id === exp.id);
                const html = `
                    <div class="experiment">
                        <h3>${exp.name}</h3>
                        <p>Runs: ${expRuns.length}</p>
                        <table>
                            <tr><th>Run ID</th><th>Status</th><th>R² Score</th><th>Model Type</th></tr>
                            ${expRuns.map(run => `
                                <tr>
                                    <td>${run.run_id.substring(0, 8)}...</td>
                                    <td>${run.status}</td>
                                    <td>${run.metrics.test_r2 || 'N/A'}</td>
                                    <td>${run.params.model_type || 'N/A'}</td>
                                </tr>
                            `).join('')}
                        </table>
                    </div>
                `;
                container.innerHTML += html;
            });
        }
        
        function displayModels(models, runs) {
            const container = document.getElementById('models');
            models.forEach(model => {
                const html = `
                    <div class="model">
                        <h3>📦 ${model.name}</h3>
                        <table>
                            <tr><th>Version</th><th>Stage</th><th>Status</th><th>R² Score</th></tr>
                            ${model.versions.map(version => {
                                const run = runs.find(r => r.run_id === version.run_id);
                                const isProduction = version.stage === 'Production';
                                return `
                                    <tr class="${isProduction ? 'production' : ''}">
                                        <td>${version.version}</td>
                                        <td>${version.stage || 'None'}</td>
                                        <td>${version.status}</td>
                                        <td>${run?.metrics?.test_r2 || 'N/A'}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </table>
                    </div>
                `;
                container.innerHTML += html;
            });
        }
    </script>
</body>
</html>
```

## 🎥 Option 4: Video Demo + Screenshots

### Create a comprehensive showcase:

1. **Record Screen Demo**
   - Navigate through MLflow dashboard
   - Show experiments and models
   - Demonstrate model promotion
   - Test API endpoints

2. **Take Screenshots**
   - Experiments page
   - Model registry
   - Production model details
   - Performance metrics

3. **Create GIF Animations**
   - Model training process
   - Dashboard navigation
   - API testing

## 📋 Recommended Approach for GitHub

### For Portfolio/Resume:
1. ✅ **Screenshots** in README.md
2. ✅ **MLFLOW_SHOWCASE.md** (detailed documentation)
3. ✅ **Video demo** (upload to YouTube/Loom)
4. ✅ **Static HTML export** for GitHub Pages

### For Live Demo:
1. 🌩️ **AWS EC2** deployment (most professional)
2. 🚀 **Heroku** deployment (easiest)
3. 📱 **Streamlit Cloud** (if you create Streamlit dashboard)

## 🎯 Next Steps

1. **Take screenshots** of your current MLflow dashboard
2. **Export MLflow data** using the script above
3. **Create static HTML** version for GitHub Pages
4. **Deploy to cloud** for live demo (optional)

Would you like me to help you with any of these options?