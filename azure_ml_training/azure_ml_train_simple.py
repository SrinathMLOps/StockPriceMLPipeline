#!/usr/bin/env python3
"""
Azure ML training script for stock price prediction (simplified version)
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from azureml.core import Run

def generate_sample_stock_data(n_days=1000):
    """Generate sample stock data for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    # Generate realistic stock price data
    initial_price = 150.0
    prices = [initial_price]
    
    for i in range(1, n_days):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% daily trend, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.normal(1000000, 200000)) for _ in range(n_days)]
    })
    
    # Ensure High >= Close >= Low and High >= Open >= Low
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    return df.set_index('Date')

def create_features(df):
    """Create technical indicators and features"""
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Price ratios
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Volatility
    df['Volatility'] = df['Price_Change'].rolling(window=10).std()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Target variable (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def train_model(args):
    """Train the stock price prediction model"""
    
    # Get the Azure ML run context
    run = Run.get_context()
    
    print("Starting Azure ML training...")
    print(f"Stock symbol: {args.symbol}")
    print(f"Number of days: {args.n_days}")
    
    # Generate sample stock data
    print("Generating sample stock data...")
    df = generate_sample_stock_data(args.n_days)
    
    print(f"Generated {len(df)} days of data")
    
    # Create features
    print("Creating features...")
    df = create_features(df)
    
    # Prepare features and target
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'MA_20',
                      'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio', 
                      'Volatility', 'Volume_Ratio']
    
    X = df[feature_columns]
    y = df['Target']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Start MLflow tracking
    mlflow.start_run()
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = 100 * (1 - rmse / np.mean(y_test))
    
    print(f"Model Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Log metrics to Azure ML
    run.log("mse", mse)
    run.log("rmse", rmse)
    run.log("r2_score", r2)
    run.log("accuracy", accuracy)
    run.log("n_estimators", args.n_estimators)
    run.log("max_depth", args.max_depth)
    run.log("symbol", args.symbol)
    run.log("training_samples", len(X_train))
    
    # Log metrics to MLflow
    mlflow.log_param("symbol", args.symbol)
    mlflow.log_param("n_days", args.n_days)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("accuracy", accuracy)
    
    # Save the model
    print("Saving model...")
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/stock_price_model.pkl"
    joblib.dump(model, model_path)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")
    
    # Register model in Azure ML
    run.upload_file("model.pkl", model_path)
    
    print("✅ Training completed successfully!")
    print(f"Model saved to: {model_path}")
    
    mlflow.end_run()
    
    return model, accuracy

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train stock price prediction model in Azure ML")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--n_days", type=int, default=1000, help="Number of days of data")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth")
    
    args = parser.parse_args()
    
    try:
        model, accuracy = train_model(args)
        print(f"\n🎉 Training successful! Model accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()