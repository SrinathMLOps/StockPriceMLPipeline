# 🚀 MLflow Model Registry & Experiments Showcase

## 📊 Project Overview
This project demonstrates a complete MLOps pipeline with MLflow tracking, model registry, and production deployment for stock price prediction.

## 🏆 Key Achievements
- **5 Model Versions** in production registry
- **99.82% Best Accuracy** (Random Forest)
- **2 Experiments** with hyperparameter tuning
- **Production Model** deployment ready

## 📈 Experiments Summary

### Experiment 1: stock_price_prediction
- **3 Runs** with different model types
- **Best Performance**: Random Forest (R² = 0.9982)
- **Models Tested**: Linear Regression, Random Forest

### Experiment 2: stock_price_hyperparameter_tuning  
- **2 Runs** with advanced tuning
- **Ridge Regression**: R² = 0.9958
- **Random Forest**: R² = 0.9941

## 📦 Model Registry

### StockPricePredictor Model Versions:

| Version | Model Type | R² Score | Stage | Status |
|---------|------------|----------|-------|--------|
| **5** | **RandomForestRegressor** | **0.9982** | **Production** | **READY** |
| 4 | Ridge | 0.9958 | None | READY |
| 3 | RandomForest | 0.9941 | None | READY |
| 2 | LinearRegression | 0.9928 | None | READY |
| 1 | LinearRegression | 0.9928 | None | READY |

## 🎯 Production Model Details

**Active Production Model**: Version 5
- **Algorithm**: Random Forest Regressor
- **Accuracy**: 99.82% (R² Score)
- **Features**: ma_3, pct_change_1d, volume
- **Status**: Production Ready
- **API Endpoint**: `models:/StockPricePredictor/Production`

## 🔧 How to Reproduce

### 1. Start MLflow Services
```bash
docker compose -f docker-compose-simple.yml up -d
```

### 2. Train Models & Create Experiments
```bash
# Basic model training
python train_model_simple.py

# Advanced hyperparameter tuning
python advanced_training.py

# Random Forest experiment
python train_experiment_2.py

# Generate model comparison
python model_comparison.py
```

### 3. Promote Best Model to Production
```bash
python promote_to_production.py
```

### 4. Access MLflow Dashboard
```bash
# Open in browser
http://localhost:5000
```

## 📊 Performance Metrics

### Model Comparison Results:
- **Mean R² Score**: 0.9948
- **Best Model**: RandomForestRegressor (99.82%)
- **Most Consistent**: LinearRegression (0% std dev)
- **Production Recommendation**: ✅ Deploy RandomForest

### API Performance:
- **Response Time**: < 100ms
- **Test Prediction**: 101.38 (sample)
- **Model Loading**: ✅ Successful
- **Health Check**: ✅ Passing

## 🚀 Live Demo Commands

```bash
# Health check
curl http://localhost:8001/health

# Make prediction
curl -X POST "http://localhost:8001/predict?ma_3=100&pct_change_1d=0.01&volume=5000"

# View MLflow UI
start http://localhost:5000
```

## 📁 Project Structure

```
├── models/                 # Model training scripts
├── serving/               # API serving code
├── mlruns/               # MLflow experiment data
├── docker-compose.yml    # MLflow services
├── promote_to_production.py  # Model promotion
└── model_comparison.py   # Performance analysis
```

## 🎥 Screenshots

*Add screenshots of:*
1. MLflow Experiments page showing both experiments
2. Model Registry with 5 versions
3. Production model details (Version 5)
4. Model comparison metrics
5. API testing results

## 🔗 Quick Access Links

When running locally:
- **MLflow Dashboard**: http://localhost:5000
- **Experiments**: http://localhost:5000/#/experiments  
- **Model Registry**: http://localhost:5000/#/models
- **Production Model**: http://localhost:5000/#/models/StockPricePredictor/versions/5
- **API Docs**: http://localhost:8001/docs

---

*This showcase demonstrates enterprise-level MLOps practices with complete model lifecycle management, from experimentation to production deployment.*