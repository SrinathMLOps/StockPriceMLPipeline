#!/usr/bin/env python3
# Enhanced FastAPI Server with Real-Time Portfolio Features

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import redis
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime
import uvicorn
from realtime_portfolio_system import RealTimeDataStreamer, RealTimeFeatureEngine, PortfolioOptimizer

# Pydantic models for API
class PredictionRequest(BaseModel):
    symbol: str
    ma_3: float
    pct_change_1d: float
    volume: float

class PortfolioPredictionRequest(BaseModel):
    assets: List[str]
    features: Dict[str, Dict[str, float]]

class TradingSignal(BaseModel):
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    current_price: float
    predicted_price: float
    timestamp: datetime

class PortfolioMetrics(BaseModel):
    total_value: float
    total_return_pct: float
    sharpe_ratio: float
    volatility: float
    positions: Dict[str, Dict]

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Stock Price ML API",
    description="Real-time portfolio management with ML predictions",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
feature_engine = RealTimeFeatureEngine()
portfolio_optimizer = PortfolioOptimizer()
active_connections: List[WebSocket] = []

# Load models at startup
models = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models and start background tasks"""
    global models
    
    mlflow.set_tracking_uri("http://localhost:5000")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    for symbol in symbols:
        try:
            model_name = f"StockPredictor_{symbol}"
            model_uri = f"models:/{model_name}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            models[symbol] = model
            print(f"✅ Loaded model for {symbol}")
        except Exception as e:
            print(f"⚠️ Could not load model for {symbol}: {e}")
    
    # Start background data streaming
    streamer = RealTimeDataStreamer(symbols)
    asyncio.create_task(streamer.stream_data())
    print("🌊 Background data streaming started")

# Original prediction endpoint (enhanced)
@app.post("/predict")
async def predict_single_stock(request: PredictionRequest):
    """Enhanced single stock prediction with real-time data"""
    
    if request.symbol not in models:
        raise HTTPException(status_code=404, detail=f"Model not found for {request.symbol}")
    
    try:
        # Use provided features or get real-time features
        if all([request.ma_3, request.pct_change_1d, request.volume]):
            features = [request.ma_3, request.pct_change_1d, request.volume]
        else:
            # Get real-time features
            rt_features = feature_engine.get_real_time_features(request.symbol)
            if rt_features:
                features = [rt_features['ma_3'], rt_features['pct_change_1d'], rt_features['current_price']]
            else:
                raise HTTPException(status_code=400, detail="Insufficient data for real-time features")
        
        # Make prediction
        prediction = models[request.symbol].predict([features])[0]
        
        # Get current market data for comparison
        market_data_key = f"market_data:{request.symbol}"
        market_data_json = redis_client.get(market_data_key)
        
        response = {
            "symbol": request.symbol,
            "prediction": float(prediction),
            "timestamp": datetime.now().isoformat()
        }
        
        if market_data_json:
            market_data = json.loads(market_data_json)
            response.update({
                "current_price": market_data['price'],
                "price_change_pct": market_data['change_pct'],
                "volume": market_data['volume']
            })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/portfolio")
async def predict_portfolio(request: PortfolioPredictionRequest):
    """Portfolio-wide predictions with optimization"""
    
    predictions = {}
    
    for symbol in request.assets:
        if symbol in models and symbol in request.features:
            try:
                features_dict = request.features[symbol]
                features = [
                    features_dict.get('ma_3', 0),
                    features_dict.get('pct_change_1d', 0),
                    features_dict.get('volume', 0)
                ]
                
                prediction = models[symbol].predict([features])[0]
                predictions[symbol] = float(prediction)
                
            except Exception as e:
                print(f"Error predicting {symbol}: {e}")
    
    # Calculate portfolio metrics
    if predictions:
        portfolio_value = sum(predictions.values())
        avg_prediction = np.mean(list(predictions.values()))
        volatility = np.std(list(predictions.values()))
        
        return {
            "predictions": predictions,
            "portfolio_metrics": {
                "total_predicted_value": portfolio_value,
                "average_prediction": avg_prediction,
                "prediction_volatility": volatility,
                "asset_count": len(predictions)
            },
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=400, detail="No valid predictions generated")

@app.get("/market/realtime/{symbol}")
async def get_realtime_market_data(symbol: str):
    """Get real-time market data for a symbol"""
    
    market_data_key = f"market_data:{symbol}"
    market_data_json = redis_client.get(market_data_key)
    
    if not market_data_json:
        raise HTTPException(status_code=404, detail=f"No real-time data available for {symbol}")
    
    market_data = json.loads(market_data_json)
    
    # Get technical indicators
    features = feature_engine.get_real_time_features(symbol)
    
    response = {
        "symbol": symbol,
        "market_data": market_data,
        "technical_indicators": features,
        "timestamp": datetime.now().isoformat()
    }
    
    return response

@app.get("/trading/signals")
async def get_trading_signals():
    """Get trading signals for all available assets"""
    
    signals = []
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    for symbol in symbols:
        if symbol in models:
            # Get real-time features
            features = feature_engine.get_real_time_features(symbol)
            market_data_key = f"market_data:{symbol}"
            market_data_json = redis_client.get(market_data_key)
            
            if features and market_data_json:
                market_data = json.loads(market_data_json)
                current_price = market_data['price']
                
                # Make prediction
                try:
                    feature_array = [features['ma_3'], features['pct_change_1d'], features['current_price']]
                    prediction = models[symbol].predict([feature_array])[0]
                    
                    # Generate signal
                    price_diff_pct = (prediction - current_price) / current_price * 100
                    
                    if price_diff_pct > 1.0:
                        signal = "BUY"
                        confidence = min(price_diff_pct / 5.0, 1.0)  # Max confidence at 5% difference
                    elif price_diff_pct < -1.0:
                        signal = "SELL"
                        confidence = min(abs(price_diff_pct) / 5.0, 1.0)
                    else:
                        signal = "HOLD"
                        confidence = 1.0 - abs(price_diff_pct) / 1.0
                    
                    signals.append({
                        "symbol": symbol,
                        "signal": signal,
                        "confidence": round(confidence, 3),
                        "current_price": current_price,
                        "predicted_price": round(prediction, 2),
                        "price_change_pct": round(price_diff_pct, 2),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    print(f"Error generating signal for {symbol}: {e}")
    
    return {"signals": signals, "timestamp": datetime.now().isoformat()}

@app.get("/portfolio/optimize")
async def optimize_portfolio(target_return: Optional[float] = None):
    """Optimize portfolio allocation using Modern Portfolio Theory"""
    
    # Get historical returns (simulated for demo)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    # Generate sample returns data
    np.random.seed(42)
    returns_data = {}
    
    for symbol in symbols:
        # Simulate daily returns
        returns_data[symbol] = np.random.normal(0.001, 0.02, 252)  # 252 trading days
    
    import pandas as pd
    returns_df = pd.DataFrame(returns_data)
    
    # Optimize portfolio
    optimization_result = portfolio_optimizer.optimize_portfolio(returns_df, target_return)
    
    if optimization_result['success']:
        return {
            "optimal_weights": optimization_result['weights'],
            "expected_metrics": optimization_result['metrics'],
            "optimization_successful": True,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=400, detail="Portfolio optimization failed")

@app.websocket("/ws/realtime")
async def websocket_realtime_data(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Get real-time data for all symbols
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            realtime_data = {}
            
            for symbol in symbols:
                market_data_key = f"market_data:{symbol}"
                market_data_json = redis_client.get(market_data_key)
                
                if market_data_json:
                    market_data = json.loads(market_data_json)
                    
                    # Get prediction if model available
                    if symbol in models:
                        features = feature_engine.get_real_time_features(symbol)
                        if features:
                            try:
                                feature_array = [features['ma_3'], features['pct_change_1d'], features['current_price']]
                                prediction = models[symbol].predict([feature_array])[0]
                                market_data['prediction'] = round(prediction, 2)
                            except:
                                market_data['prediction'] = None
                    
                    realtime_data[symbol] = market_data
            
            # Send data to client
            await websocket.send_text(json.dumps({
                "type": "market_update",
                "data": realtime_data,
                "timestamp": datetime.now().isoformat()
            }))
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("WebSocket client disconnected")

@app.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    
    # Check Redis connection
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    # Check model availability
    model_status = {symbol: "loaded" for symbol in models.keys()}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_status,
            "models": model_status,
            "active_websockets": len(active_connections)
        },
        "features": [
            "single_stock_prediction",
            "portfolio_prediction", 
            "real_time_market_data",
            "trading_signals",
            "portfolio_optimization",
            "websocket_streaming"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_fastapi_server:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )