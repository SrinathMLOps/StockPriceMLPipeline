#!/usr/bin/env python3
# Production deployment and monitoring script

import mlflow
import requests
import json
import os
from mlflow.tracking import MlflowClient
from datetime import datetime

class ProductionDeployer:
    def __init__(self):
        self.mlflow_uri = "http://localhost:5000"
        self.api_base = "http://localhost:8000"
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient()
    
    def get_best_model(self):
        """Find the best performing model across all experiments"""
        print("🔍 Finding best model across all experiments...")
        
        experiments = self.client.search_experiments()
        best_model = None
        best_score = 0
        
        for exp in experiments:
            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.test_r2 DESC"],
                max_results=5
            )
            
            for run in runs:
                test_r2 = run.data.metrics.get("test_r2", 0)
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = {
                        "run_id": run.info.run_id,
                        "experiment_name": exp.name,
                        "test_r2": test_r2,
                        "model_type": run.data.params.get("model_type", "Unknown")
                    }
        
        if best_model:
            print(f"🏆 Best Model Found:")
            print(f"   Experiment: {best_model['experiment_name']}")
            print(f"   Model Type: {best_model['model_type']}")
            print(f"   Test R²: {best_model['test_r2']:.4f}")
            print(f"   Run ID: {best_model['run_id']}")
        
        return best_model
    
    def promote_to_production(self, run_id):
        """Promote model to production stage"""
        print(f"\n🚀 Promoting model to production...")
        
        try:
            # Register model if not already registered
            model_uri = f"runs:/{run_id}/model"
            model_name = "StockPricePredictor"
            
            # Create new version
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            
            print(f"✅ Model version {model_version.version} promoted to Production")
            return model_version.version
            
        except Exception as e:
            print(f"❌ Promotion failed: {e}")
            return None
    
    def create_deployment_config(self):
        """Create deployment configuration"""
        config = {
            "deployment": {
                "timestamp": datetime.now().isoformat(),
                "mlflow_uri": self.mlflow_uri,
                "api_endpoint": self.api_base,
                "model_name": "StockPricePredictor",
                "stage": "Production"
            },
            "monitoring": {
                "health_check_interval": 300,  # 5 minutes
                "performance_threshold": 0.05,  # 50ms
                "accuracy_threshold": 0.95
            },
            "scaling": {
                "min_replicas": 1,
                "max_replicas": 5,
                "cpu_threshold": 70,
                "memory_threshold": 80
            }
        }
        
        with open("deployment_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Deployment configuration saved to deployment_config.json")
        return config
    
    def create_docker_compose_prod(self):
        """Create production Docker Compose file"""
        compose_content = """version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_db:/var/lib/postgresql/data
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U airflow"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:latest
    restart: always
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow:/mlflow
      - ./models:/models
    working_dir: /mlflow
    restart: always
    command: >
      bash -c "pip install mlflow==2.8.1 && 
               mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  fastapi:
    build: ./serving
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    restart: always
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - fastapi
    restart: always

volumes:
  postgres_db:
"""
        
        with open("docker-compose.prod.yml", "w") as f:
            f.write(compose_content)
        
        print("✅ Production Docker Compose file created: docker-compose.prod.yml")
    
    def create_nginx_config(self):
        """Create Nginx configuration for load balancing"""
        nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream fastapi_backend {
        server fastapi:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://fastapi_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }
    }
}"""
        
        with open("nginx.conf", "w") as f:
            f.write(nginx_config)
        
        print("✅ Nginx configuration created: nginx.conf")
    
    def deploy(self):
        """Complete deployment process"""
        print("🚀 Starting Production Deployment Process")
        print("=" * 50)
        
        # Find best model
        best_model = self.get_best_model()
        if not best_model:
            print("❌ No suitable model found for deployment")
            return False
        
        # Promote to production
        version = self.promote_to_production(best_model["run_id"])
        if not version:
            print("❌ Model promotion failed")
            return False
        
        # Create deployment artifacts
        self.create_deployment_config()
        self.create_docker_compose_prod()
        self.create_nginx_config()
        
        print("\n✅ Production deployment artifacts created!")
        print("\nNext steps:")
        print("1. Review deployment_config.json")
        print("2. Run: docker-compose -f docker-compose.prod.yml up -d")
        print("3. Access via: http://localhost (Nginx load balancer)")
        print("4. Monitor via: http://localhost:5000 (MLflow)")
        
        return True

if __name__ == "__main__":
    deployer = ProductionDeployer()
    deployer.deploy()