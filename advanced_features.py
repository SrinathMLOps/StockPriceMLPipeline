#!/usr/bin/env python3
# Advanced Features: Real-time streaming and A/B testing

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import mlflow
import requests
from dataclasses import dataclass
import websockets
import threading
import queue

@dataclass
class StockData:
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    ma_3: float
    pct_change_1d: float

class RealTimeDataStreamer:
    """Real-time stock data streaming simulator"""
    
    def __init__(self):
        self.is_streaming = False
        self.subscribers = []
        self.data_queue = queue.Queue()
        
    def generate_stock_data(self, symbol: str = "AAPL") -> StockData:
        """Generate realistic stock data"""
        now = datetime.now()
        
        # Simulate realistic stock price movement
        base_price = 150.0
        volatility = 0.02
        trend = 0.001
        
        # Add some randomness with trend
        price_change = np.random.normal(trend, volatility)
        new_price = base_price * (1 + price_change)
        
        # Generate volume (log-normal distribution)
        volume = int(np.random.lognormal(13, 1))
        
        # Calculate moving average (simplified)
        ma_3 = new_price * (1 + np.random.normal(0, 0.001))
        
        # Calculate percent change
        pct_change_1d = price_change
        
        return StockData(
            timestamp=now,
            symbol=symbol,
            price=new_price,
            volume=volume,
            ma_3=ma_3,
            pct_change_1d=pct_change_1d
        )
    
    async def stream_data(self, websocket, path):
        """WebSocket handler for streaming data"""
        self.subscribers.append(websocket)
        print(f"📡 New subscriber connected. Total: {len(self.subscribers)}")
        
        try:
            while self.is_streaming:
                # Generate new data point
                data = self.generate_stock_data()
                
                # Send to all subscribers
                message = {
                    "timestamp": data.timestamp.isoformat(),
                    "symbol": data.symbol,
                    "price": data.price,
                    "volume": data.volume,
                    "ma_3": data.ma_3,
                    "pct_change_1d": data.pct_change_1d
                }
                
                # Send to WebSocket subscribers
                await websocket.send(json.dumps(message))
                
                # Add to queue for processing
                self.data_queue.put(data)
                
                await asyncio.sleep(1)  # Stream every second
                
        except websockets.exceptions.ConnectionClosed:
            print("📡 Subscriber disconnected")
        finally:
            if websocket in self.subscribers:
                self.subscribers.remove(websocket)
    
    def start_streaming(self, host="localhost", port=8765):
        """Start the WebSocket streaming server"""
        self.is_streaming = True
        print(f"🚀 Starting real-time data stream on ws://{host}:{port}")
        
        start_server = websockets.serve(self.stream_data, host, port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    
    def stop_streaming(self):
        """Stop the streaming"""
        self.is_streaming = False

class ABTestingFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self):
        self.models = {}
        self.test_results = {}
        self.traffic_split = {}
        
    def register_model(self, model_name: str, model_version: str, traffic_percentage: float):
        """Register a model for A/B testing"""
        self.models[model_name] = {
            "version": model_version,
            "traffic_percentage": traffic_percentage,
            "predictions": [],
            "response_times": [],
            "errors": 0
        }
        print(f"✅ Registered model {model_name} v{model_version} with {traffic_percentage}% traffic")
    
    def route_request(self, features: Dict) -> Dict:
        """Route request to appropriate model based on traffic split"""
        # Determine which model to use
        rand = random.random() * 100
        cumulative = 0
        selected_model = None
        
        for model_name, config in self.models.items():
            cumulative += config["traffic_percentage"]
            if rand <= cumulative:
                selected_model = model_name
                break
        
        if not selected_model:
            selected_model = list(self.models.keys())[0]  # Fallback
        
        # Make prediction
        start_time = time.time()
        try:
            prediction = self.make_prediction(selected_model, features)
            response_time = time.time() - start_time
            
            # Record metrics
            self.models[selected_model]["predictions"].append(prediction)
            self.models[selected_model]["response_times"].append(response_time)
            
            return {
                "model_used": selected_model,
                "prediction": prediction,
                "response_time": response_time
            }
            
        except Exception as e:
            self.models[selected_model]["errors"] += 1
            return {
                "model_used": selected_model,
                "error": str(e),
                "response_time": time.time() - start_time
            }
    
    def make_prediction(self, model_name: str, features: Dict) -> float:
        """Make prediction using specified model"""
        # In real implementation, this would call different model endpoints
        # For demo, we'll simulate different models with slight variations
        
        base_prediction = -4305.93  # Base prediction
        
        if model_name == "linear_regression":
            return base_prediction * (1 + random.uniform(-0.05, 0.05))
        elif model_name == "random_forest":
            return base_prediction * (1 + random.uniform(-0.03, 0.03))
        elif model_name == "ridge_regression":
            return base_prediction * (1 + random.uniform(-0.04, 0.04))
        else:
            return base_prediction
    
    def get_test_results(self) -> Dict:
        """Get A/B test results and statistics"""
        results = {}
        
        for model_name, config in self.models.items():
            if config["predictions"]:
                predictions = config["predictions"]
                response_times = config["response_times"]
                
                results[model_name] = {
                    "version": config["version"],
                    "traffic_percentage": config["traffic_percentage"],
                    "total_requests": len(predictions),
                    "avg_prediction": np.mean(predictions),
                    "prediction_std": np.std(predictions),
                    "avg_response_time": np.mean(response_times),
                    "p95_response_time": np.percentile(response_times, 95),
                    "error_rate": config["errors"] / (len(predictions) + config["errors"]) * 100,
                    "errors": config["errors"]
                }
        
        return results
    
    def run_ab_test(self, duration_minutes: int = 5, requests_per_minute: int = 60):
        """Run A/B test simulation"""
        print(f"🧪 Starting A/B test for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        request_interval = 60 / requests_per_minute
        
        while time.time() < end_time:
            # Generate test features
            features = {
                "ma_3": random.uniform(100, 200),
                "pct_change_1d": random.uniform(-0.05, 0.05),
                "volume": random.randint(500000, 5000000)
            }
            
            # Route request
            result = self.route_request(features)
            
            time.sleep(request_interval)
        
        print("✅ A/B test completed!")
        return self.get_test_results()

class ModelPerformanceMonitor:
    """Advanced model performance monitoring"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        
    def collect_metrics(self, model_name: str, prediction: float, actual: float = None, 
                       response_time: float = None, features: Dict = None):
        """Collect performance metrics"""
        timestamp = datetime.now()
        
        metrics = {
            "timestamp": timestamp,
            "model_name": model_name,
            "prediction": prediction,
            "actual": actual,
            "response_time": response_time,
            "features": features
        }
        
        if actual is not None:
            metrics["error"] = abs(prediction - actual)
            metrics["squared_error"] = (prediction - actual) ** 2
        
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self.check_alerts(metrics)
    
    def check_alerts(self, metrics: Dict):
        """Check for performance alerts"""
        # Response time alert
        if metrics.get("response_time", 0) > 0.1:  # 100ms threshold
            self.alerts.append({
                "timestamp": metrics["timestamp"],
                "type": "HIGH_LATENCY",
                "message": f"High response time: {metrics['response_time']:.3f}s",
                "severity": "WARNING"
            })
        
        # Prediction range alert (detect anomalies)
        recent_predictions = [m["prediction"] for m in self.metrics_history[-100:]]
        if len(recent_predictions) > 10:
            mean_pred = np.mean(recent_predictions)
            std_pred = np.std(recent_predictions)
            
            if abs(metrics["prediction"] - mean_pred) > 3 * std_pred:
                self.alerts.append({
                    "timestamp": metrics["timestamp"],
                    "type": "PREDICTION_ANOMALY",
                    "message": f"Unusual prediction: {metrics['prediction']:.2f}",
                    "severity": "WARNING"
                })
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m["timestamp"] > cutoff_time]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        predictions = [m["prediction"] for m in recent_metrics]
        response_times = [m["response_time"] for m in recent_metrics if m["response_time"]]
        errors = [m["error"] for m in recent_metrics if m.get("error")]
        
        summary = {
            "period_hours": hours,
            "total_requests": len(recent_metrics),
            "avg_prediction": np.mean(predictions),
            "prediction_std": np.std(predictions),
            "min_prediction": np.min(predictions),
            "max_prediction": np.max(predictions)
        }
        
        if response_times:
            summary.update({
                "avg_response_time": np.mean(response_times),
                "p95_response_time": np.percentile(response_times, 95),
                "p99_response_time": np.percentile(response_times, 99)
            })
        
        if errors:
            summary.update({
                "avg_error": np.mean(errors),
                "rmse": np.sqrt(np.mean([e**2 for e in errors]))
            })
        
        summary["recent_alerts"] = len([a for a in self.alerts if a["timestamp"] > cutoff_time])
        
        return summary

def demo_advanced_features():
    """Demonstrate all advanced features"""
    print("🚀 Advanced Features Demo")
    print("=" * 50)
    
    # 1. A/B Testing Demo
    print("\n🧪 A/B Testing Framework Demo")
    ab_tester = ABTestingFramework()
    
    # Register models for testing
    ab_tester.register_model("linear_regression", "v1", 40)
    ab_tester.register_model("random_forest", "v2", 40)
    ab_tester.register_model("ridge_regression", "v4", 20)
    
    # Run short test
    results = ab_tester.run_ab_test(duration_minutes=1, requests_per_minute=30)
    
    print("\n📊 A/B Test Results:")
    for model, stats in results.items():
        print(f"\n{model}:")
        print(f"  Requests: {stats['total_requests']}")
        print(f"  Avg Response Time: {stats['avg_response_time']:.3f}s")
        print(f"  Error Rate: {stats['error_rate']:.1f}%")
    
    # 2. Performance Monitoring Demo
    print("\n📈 Performance Monitoring Demo")
    monitor = ModelPerformanceMonitor()
    
    # Simulate some metrics
    for i in range(50):
        monitor.collect_metrics(
            model_name="random_forest",
            prediction=random.uniform(-5000, -1000),
            actual=random.uniform(-5000, -1000),
            response_time=random.uniform(0.01, 0.05),
            features={"ma_3": 150, "pct_change_1d": 0.02, "volume": 1000000}
        )
    
    summary = monitor.get_performance_summary(hours=1)
    print(f"\nPerformance Summary:")
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Avg Response Time: {summary.get('avg_response_time', 0):.3f}s")
    print(f"  RMSE: {summary.get('rmse', 0):.2f}")
    print(f"  Recent Alerts: {summary['recent_alerts']}")
    
    # 3. Real-time Streaming Info
    print("\n📡 Real-time Streaming Setup")
    print("To start real-time streaming:")
    print("1. Run: python -c 'from advanced_features import RealTimeDataStreamer; streamer = RealTimeDataStreamer(); streamer.start_streaming()'")
    print("2. Connect WebSocket client to: ws://localhost:8765")
    print("3. Receive real-time stock data updates")
    
    print("\n✅ Advanced features demo completed!")

if __name__ == "__main__":
    demo_advanced_features()