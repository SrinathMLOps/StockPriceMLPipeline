#!/usr/bin/env python3
# Multi-Asset Extension - Quick Start Implementation

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from typing import List, Dict
import requests
import asyncio
import aiohttp

class MultiAssetPredictor:
    """Extended predictor for multiple assets"""
    
    def __init__(self):
        self.assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        self.models = {}
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("multi_asset_prediction")
    
    async def fetch_asset_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single asset"""
        # Placeholder for real API integration
        # In real implementation, use Alpha Vantage, TwelveData, etc.
        
        # Generate sample data for demo
        dates = pd.date_range('2023-01-01', periods=1000, freq='h')
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        base_price = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300, 'TSLA': 800, 'AMZN': 3000}
        
        df = pd.DataFrame({
            'datetime': dates,
            'symbol': symbol,
            'close': base_price[symbol] + np.cumsum(np.random.randn(1000) * 2),
            'volume': np.random.randint(1000000, 10000000, 1000)
        })
        
        return df
    
    async def fetch_all_assets(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all assets concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_asset_data(symbol) for symbol in self.assets]
            results = await asyncio.gather(*tasks)
            
        return dict(zip(self.assets, results))
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for each asset"""
        df = df.copy()
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Technical indicators
        df['ma_3'] = df['close'].rolling(window=3).mean()
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()
        
        # Price changes
        df['pct_change_1d'] = df['close'].pct_change()
        df['pct_change_7d'] = df['close'].pct_change(periods=7)
        
        # Volatility
        df['volatility'] = df['pct_change_1d'].rolling(window=7).std()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df.dropna()
    
    def train_asset_model(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Train model for a specific asset"""
        
        with mlflow.start_run(run_name=f"{symbol}_model_training"):
            # Feature engineering
            df_features = self.engineer_features(df)
            
            # Prepare features and target
            feature_cols = ['ma_3', 'ma_7', 'ma_21', 'pct_change_1d', 'pct_change_7d', 
                           'volatility', 'volume', 'volume_ratio']
            
            X = df_features[feature_cols]
            y = df_features['close']
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Calculate metrics
            train_score = model.score(X, y)
            
            # Log parameters and metrics
            mlflow.log_param("symbol", symbol)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_samples", len(X))
            
            mlflow.log_metric("train_r2", train_score)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                f"{symbol}_model",
                registered_model_name=f"StockPredictor_{symbol}"
            )
            
            # Store model
            self.models[symbol] = model
            
            print(f"✅ {symbol} model trained - R²: {train_score:.4f}")
            
            return {
                "symbol": symbol,
                "model": model,
                "r2_score": train_score,
                "features": feature_cols
            }
    
    def train_all_models(self) -> Dict[str, Dict]:
        """Train models for all assets"""
        print("🚀 Starting multi-asset model training...")
        
        # Fetch data for all assets
        asset_data = asyncio.run(self.fetch_all_assets())
        
        # Train models
        results = {}
        for symbol, df in asset_data.items():
            results[symbol] = self.train_asset_model(symbol, df)
        
        print(f"✅ Trained {len(results)} asset models")
        return results
    
    def predict_portfolio(self, features_dict: Dict[str, Dict]) -> Dict[str, float]:
        """Make predictions for entire portfolio"""
        predictions = {}
        
        for symbol, features in features_dict.items():
            if symbol in self.models:
                # Convert features to array
                feature_array = np.array([[
                    features['ma_3'], features['ma_7'], features['ma_21'],
                    features['pct_change_1d'], features['pct_change_7d'],
                    features['volatility'], features['volume'], features['volume_ratio']
                ]])
                
                prediction = self.models[symbol].predict(feature_array)[0]
                predictions[symbol] = float(prediction)
            else:
                print(f"⚠️ No model found for {symbol}")
        
        return predictions
    
    def calculate_portfolio_metrics(self, predictions: Dict[str, float]) -> Dict:
        """Calculate portfolio-level metrics"""
        values = list(predictions.values())
        
        return {
            "total_portfolio_value": sum(values),
            "average_prediction": np.mean(values),
            "portfolio_volatility": np.std(values),
            "min_prediction": min(values),
            "max_prediction": max(values),
            "asset_count": len(predictions)
        }

def demo_multi_asset_system():
    """Demonstrate the multi-asset system"""
    print("🎯 Multi-Asset Stock Prediction System Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MultiAssetPredictor()
    
    # Train models for all assets
    training_results = predictor.train_all_models()
    
    # Demo predictions
    sample_features = {
        'AAPL': {
            'ma_3': 150.0, 'ma_7': 148.0, 'ma_21': 145.0,
            'pct_change_1d': 0.02, 'pct_change_7d': 0.05,
            'volatility': 0.15, 'volume': 50000000, 'volume_ratio': 1.2
        },
        'GOOGL': {
            'ma_3': 2500.0, 'ma_7': 2480.0, 'ma_21': 2450.0,
            'pct_change_1d': 0.01, 'pct_change_7d': 0.03,
            'volatility': 0.12, 'volume': 20000000, 'volume_ratio': 0.9
        },
        'MSFT': {
            'ma_3': 300.0, 'ma_7': 298.0, 'ma_21': 295.0,
            'pct_change_1d': 0.015, 'pct_change_7d': 0.04,
            'volatility': 0.10, 'volume': 30000000, 'volume_ratio': 1.1
        }
    }
    
    # Make portfolio predictions
    predictions = predictor.predict_portfolio(sample_features)
    
    print("\n📊 Portfolio Predictions:")
    for symbol, prediction in predictions.items():
        print(f"   {symbol}: ${prediction:.2f}")
    
    # Calculate portfolio metrics
    portfolio_metrics = predictor.calculate_portfolio_metrics(predictions)
    
    print("\n📈 Portfolio Metrics:")
    for metric, value in portfolio_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: ${value:.2f}")
        else:
            print(f"   {metric}: {value}")
    
    print("\n✅ Multi-asset system demonstration complete!")

if __name__ == "__main__":
    demo_multi_asset_system()