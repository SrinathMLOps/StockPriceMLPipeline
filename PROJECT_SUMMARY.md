# 🚀 Stock Price ML Pipeline - Complete MLOps Project

## 📊 Project Overview
A production-ready end-to-end machine learning pipeline for stock price prediction, demonstrating modern MLOps practices with experiment tracking, model versioning, automated deployment, and monitoring.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Feature Eng.   │───▶│   ML Training   │
│  (Stock Data)   │    │   (Pandas)      │    │  (Scikit-learn) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│   Production    │◀───│   MLflow        │
│   (Testing)     │    │   (FastAPI)     │    │  (Tracking)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Core ML Stack
- **Python 3.11** - Programming language
- **Scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data processing
- **MLflow 2.8.1** - Experiment tracking & model registry

### Infrastructure & Deployment
- **Docker & Docker Compose** - Containerization
- **FastAPI** - Model serving API
- **PostgreSQL** - Database for Airflow
- **Redis** - Caching and message broker
- **Nginx** - Load balancer (production)

### Monitoring & Testing
- **Custom Testing Suite** - API and performance testing
- **Streamlit** - Dashboard (optional)
- **Automated Model Promotion** - Production deployment

## 📈 Model Performance

| Model | Algorithm | Test R² | Status |
|-------|-----------|---------|--------|
| Version 1 | Linear Regression | 0.9928 | Archived |
| Version 2 | Random Forest | 0.9982 | Archived |
| Version 3 | Random Forest (Tuned) | 0.9941 | Archived |
| Version 4 | Ridge Regression | 0.9958 | Archived |
| Version 5 | Random Forest | 0.9982 | **Production** |

## 🚀 Features Implemented

### ✅ MLOps Best Practices
- [x] Experiment tracking with MLflow
- [x] Model versioning and registry
- [x] Automated model promotion
- [x] Containerized deployment
- [x] API-based model serving
- [x] Comprehensive testing suite
- [x] Performance monitoring
- [x] Production deployment configuration

### ✅ API Endpoints
- `GET /` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Stock price prediction

### ✅ Model Features
- **ma_3**: 3-period moving average
- **pct_change_1d**: 1-day percent change
- **volume**: Trading volume

## 📊 Performance Metrics

### API Performance
- **Average Response Time**: 0.019s
- **Success Rate**: 100%
- **Throughput**: ~50 requests/second

### Model Accuracy
- **Best Test R²**: 0.9982
- **Cross-validation Score**: 0.9946
- **Feature Importance**: Volume > MA > Percent Change

## 🔧 How to Run

### Development Environment
```bash
# Start all services
docker compose up --build

# Train new models
python train_model_simple.py
python advanced_training.py

# Run tests
python test_pipeline.py

# Monitor performance
python model_monitoring.py
```

### Production Deployment
```bash
# Deploy to production
python deploy_production.py

# Start production services
docker-compose -f docker-compose.prod.yml up -d
```

## 🌐 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8000 | Model predictions |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Production | http://localhost | Load-balanced production API |

## 📁 Project Structure

```
StockPriceMLPipeline/
├── dags/                          # Airflow DAGs
│   └── data/stock_raw.csv        # Training data
├── serving/                       # FastAPI application
│   ├── main.py                   # API server
│   ├── Dockerfile               # Container config
│   └── requirements.txt         # Dependencies
├── models/                        # Trained models
│   └── stock_model.pkl          # Current model
├── mlflow/                        # MLflow artifacts
├── scripts/                       # Training scripts
│   ├── train_model_simple.py    # Basic training
│   ├── advanced_training.py     # Hyperparameter tuning
│   ├── model_monitoring.py      # Performance monitoring
│   └── test_pipeline.py         # Testing suite
├── deployment/                    # Production configs
│   ├── docker-compose.prod.yml  # Production compose
│   ├── nginx.conf               # Load balancer config
│   └── deployment_config.json   # Deployment settings
└── README.md                     # Documentation
```

## 🎯 Key Achievements

1. **Complete MLOps Pipeline**: From data ingestion to production deployment
2. **Experiment Tracking**: 2 experiments with 5 model versions
3. **High Performance**: 99.82% R² accuracy with sub-20ms response times
4. **Production Ready**: Load balancing, health checks, monitoring
5. **Automated Testing**: 100% test coverage with performance benchmarks
6. **Model Governance**: Automated promotion based on performance metrics

## 🔮 Future Enhancements

- [ ] Real-time data streaming with Apache Kafka
- [ ] A/B testing framework for model comparison
- [ ] Advanced monitoring with Prometheus & Grafana
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Data drift detection
- [ ] Model explainability with SHAP

## 📜 License
MIT License - Feel free to use this project as a template for your own MLOps implementations.

---

**🚀 This project demonstrates production-ready MLOps practices suitable for enterprise environments.**