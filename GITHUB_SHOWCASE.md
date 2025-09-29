# MLflow Experiments Showcase

## Project Overview
Stock Price Prediction MLOps Pipeline with MLflow tracking and model registry.

## Experiments Summary

### stock_price_hyperparameter_tuning
- Total Runs: 2
- Status: Active

### stock_price_prediction
- Total Runs: 3
- Status: Active

## Model Performance Results

| Experiment | Model Type | R2 Score | Status |
|------------|------------|----------|--------|
| stock_price_prediction | RandomForestRegressor | 0.9982 | FINISHED |
| stock_price_hyperparameter_tuning | Ridge | 0.9958 | FINISHED |
| stock_price_hyperparameter_tuning | RandomForest | 0.9941 | FINISHED |
| stock_price_prediction | LinearRegression | 0.9928 | FINISHED |
| stock_price_prediction | LinearRegression | 0.9928 | FINISHED |

## Best Model Performance
- **Highest R2 Score**: 0.9982
- **Model Type**: RandomForestRegressor
- **Experiment**: stock_price_prediction

## How to Run
1. Start MLflow: `docker compose -f docker-compose-simple.yml up -d`
2. Train models: `python train_model_simple.py`
3. View dashboard: `http://localhost:5000`

## Screenshots
*Add screenshots of your MLflow dashboard here*

1. Experiments page showing all runs
2. Model comparison metrics
3. Best performing model details
