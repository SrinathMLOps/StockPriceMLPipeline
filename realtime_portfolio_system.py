#!/usr/bin/env python3
# Real-Time Portfolio Management System with Streaming Analytics

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import redis
import threading
import time
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketData:
    """Real-time market data structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    change_pct: float = 0.0

@dataclass
class PortfolioPosition:
    """Portfolio position structure"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float

class RealTimeDataStreamer:
    """Simulates real-time market data streaming"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.is_streaming = False
        self.base_prices = {
            'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 
            'TSLA': 800.0, 'AMZN': 3000.0, 'NVDA': 400.0
        }
        
    async def generate_market_data(self, symbol: str) -> MarketData:
        """Generate realistic market data"""
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Simulate price movement with some volatility
        price_change = np.random.normal(0, base_price * 0.002)  # 0.2% volatility
        new_price = base_price + price_change
        
        # Update base price for next iteration
        self.base_prices[symbol] = new_price
        
        return MarketData(
            symbol=symbol,
            price=round(new_price, 2),
            volume=np.random.randint(100000, 1000000),
            timestamp=datetime.now(),
            bid=round(new_price - 0.01, 2),
            ask=round(new_price + 0.01, 2),
            change_pct=round(price_change / base_price * 100, 4)
        )
    
    async def stream_data(self):
        """Stream market data to Redis"""
        print("🌊 Starting real-time data streaming...")
        self.is_streaming = True
        
        while self.is_streaming:
            for symbol in self.symbols:
                market_data = await self.generate_market_data(symbol)
                
                # Store in Redis with expiration
                data_key = f"market_data:{symbol}"
                data_json = json.dumps({
                    'symbol': market_data.symbol,
                    'price': market_data.price,
                    'volume': market_data.volume,
                    'timestamp': market_data.timestamp.isoformat(),
                    'bid': market_data.bid,
                    'ask': market_data.ask,
                    'change_pct': market_data.change_pct
                })
                
                self.redis_client.setex(data_key, 60, data_json)  # Expire in 60 seconds
                
                # Store historical data for features
                history_key = f"price_history:{symbol}"
                self.redis_client.lpush(history_key, market_data.price)
                self.redis_client.ltrim(history_key, 0, 99)  # Keep last 100 prices
                
            await asyncio.sleep(1)  # Update every second
    
    def stop_streaming(self):
        """Stop the data stream"""
        self.is_streaming = False
        print("🛑 Data streaming stopped")

class RealTimeFeatureEngine:
    """Real-time feature engineering"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.scaler = StandardScaler()
    
    def get_real_time_features(self, symbol: str) -> Optional[Dict]:
        """Calculate features from real-time data"""
        try:
            # Get price history
            history_key = f"price_history:{symbol}"
            price_history = self.redis_client.lrange(history_key, 0, -1)
            
            if len(price_history) < 21:  # Need at least 21 data points
                return None
            
            prices = np.array([float(p) for p in price_history])
            
            # Calculate technical indicators
            features = {
                'ma_3': np.mean(prices[:3]),
                'ma_7': np.mean(prices[:7]),
                'ma_21': np.mean(prices[:21]),
                'pct_change_1': (prices[0] - prices[1]) / prices[1] if len(prices) > 1 else 0,
                'pct_change_7': (prices[0] - prices[7]) / prices[7] if len(prices) > 7 else 0,
                'volatility': np.std(prices[:21]),
                'rsi': self.calculate_rsi(prices[:14]),
                'bollinger_upper': np.mean(prices[:20]) + 2 * np.std(prices[:20]),
                'bollinger_lower': np.mean(prices[:20]) - 2 * np.std(prices[:20]),
                'current_price': prices[0]
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Error calculating features for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

class PortfolioOptimizer:
    """Modern Portfolio Theory optimization"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: np.ndarray, 
                                  cov_matrix: np.ndarray) -> Dict:
        """Calculate portfolio risk and return metrics"""
        portfolio_return = np.sum(weights * returns.mean() * 252)  # Annualized
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_portfolio(self, returns: pd.DataFrame, target_return: float = None) -> Dict:
        """Optimize portfolio using Modern Portfolio Theory"""
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if target_return:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(x * expected_returns) - target_return
            })
        
        # Bounds: no short selling (weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            metrics = self.calculate_portfolio_metrics(optimal_weights, expected_returns, cov_matrix)
            
            return {
                'weights': dict(zip(returns.columns, optimal_weights)),
                'metrics': metrics,
                'success': True
            }
        else:
            return {'success': False, 'message': 'Optimization failed'}

class RealTimePortfolioManager:
    """Complete real-time portfolio management system"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        
        # Initialize components
        self.streamer = RealTimeDataStreamer(symbols)
        self.feature_engine = RealTimeFeatureEngine()
        self.optimizer = PortfolioOptimizer()
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Load models
        self.models = {}
        self.load_models()
        
        # Performance tracking
        self.performance_history = []
        
    def load_models(self):
        """Load trained models for each asset"""
        mlflow.set_tracking_uri("http://localhost:5000")
        
        for symbol in self.symbols:
            try:
                model_name = f"StockPredictor_{symbol}"
                model_version = "latest"
                model_uri = f"models:/{model_name}/{model_version}"
                
                model = mlflow.sklearn.load_model(model_uri)
                self.models[symbol] = model
                print(f"✅ Loaded model for {symbol}")
                
            except Exception as e:
                print(f"⚠️ Could not load model for {symbol}: {e}")
    
    async def get_real_time_predictions(self) -> Dict[str, float]:
        """Get real-time predictions for all assets"""
        predictions = {}
        
        for symbol in self.symbols:
            if symbol in self.models:
                features = self.feature_engine.get_real_time_features(symbol)
                
                if features:
                    # Prepare features for model
                    feature_array = np.array([[
                        features['ma_3'], features['ma_7'], features['ma_21'],
                        features['pct_change_1'], features['pct_change_7'],
                        features['volatility'], features['rsi'], features['current_price']
                    ]])
                    
                    try:
                        prediction = self.models[symbol].predict(feature_array)[0]
                        predictions[symbol] = float(prediction)
                    except Exception as e:
                        print(f"❌ Prediction error for {symbol}: {e}")
        
        return predictions
    
    def calculate_portfolio_value(self) -> Dict:
        """Calculate current portfolio value and metrics"""
        total_value = 0
        positions_value = {}
        
        for symbol in self.symbols:
            # Get current market data
            data_key = f"market_data:{symbol}"
            market_data_json = self.redis_client.get(data_key)
            
            if market_data_json:
                market_data = json.loads(market_data_json)
                current_price = market_data['price']
                
                if symbol in self.positions:
                    position = self.positions[symbol]
                    market_value = position['quantity'] * current_price
                    total_value += market_value
                    
                    positions_value[symbol] = {
                        'quantity': position['quantity'],
                        'avg_cost': position['avg_cost'],
                        'current_price': current_price,
                        'market_value': market_value,
                        'unrealized_pnl': market_value - (position['quantity'] * position['avg_cost']),
                        'weight': 0  # Will be calculated after total_value
                    }
        
        # Calculate weights
        for symbol in positions_value:
            positions_value[symbol]['weight'] = positions_value[symbol]['market_value'] / total_value if total_value > 0 else 0
        
        return {
            'total_value': total_value,
            'cash': self.current_capital,
            'total_portfolio_value': total_value + self.current_capital,
            'positions': positions_value,
            'total_pnl': (total_value + self.current_capital) - self.initial_capital,
            'total_return_pct': ((total_value + self.current_capital) / self.initial_capital - 1) * 100
        }
    
    async def run_real_time_analysis(self, duration_minutes: int = 5):
        """Run real-time portfolio analysis"""
        print(f"🚀 Starting real-time portfolio analysis for {duration_minutes} minutes...")
        
        # Start data streaming in background
        streaming_task = asyncio.create_task(self.streamer.stream_data())
        
        # Wait for initial data
        await asyncio.sleep(5)
        
        start_time = time.time()
        iteration = 0
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                iteration += 1
                print(f"\n📊 Analysis Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Get real-time predictions
                predictions = await self.get_real_time_predictions()
                
                if predictions:
                    print("🔮 Real-time Predictions:")
                    for symbol, prediction in predictions.items():
                        # Get current price for comparison
                        data_key = f"market_data:{symbol}"
                        market_data_json = self.redis_client.get(data_key)
                        
                        if market_data_json:
                            market_data = json.loads(market_data_json)
                            current_price = market_data['price']
                            change_pct = market_data['change_pct']
                            
                            signal = "🟢 BUY" if prediction > current_price * 1.01 else "🔴 SELL" if prediction < current_price * 0.99 else "🟡 HOLD"
                            
                            print(f"   {symbol}: ${current_price:.2f} → ${prediction:.2f} ({change_pct:+.2f}%) {signal}")
                
                # Calculate portfolio metrics
                portfolio_metrics = self.calculate_portfolio_value()
                print(f"\n💼 Portfolio Value: ${portfolio_metrics['total_portfolio_value']:,.2f}")
                print(f"   Total P&L: ${portfolio_metrics['total_pnl']:+,.2f} ({portfolio_metrics['total_return_pct']:+.2f}%)")
                
                # Store performance data
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'portfolio_value': portfolio_metrics['total_portfolio_value'],
                    'pnl': portfolio_metrics['total_pnl'],
                    'predictions': predictions
                })
                
                await asyncio.sleep(10)  # Analysis every 10 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️ Analysis stopped by user")
        
        finally:
            # Stop streaming
            self.streamer.stop_streaming()
            streaming_task.cancel()
            
            # Generate final report
            self.generate_performance_report()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.performance_history:
            print("❌ No performance data to report")
            return
        
        print("\n" + "="*60)
        print("📈 REAL-TIME PORTFOLIO PERFORMANCE REPORT")
        print("="*60)
        
        # Performance metrics
        initial_value = self.performance_history[0]['portfolio_value']
        final_value = self.performance_history[-1]['portfolio_value']
        total_return = (final_value / initial_value - 1) * 100
        
        values = [record['portfolio_value'] for record in self.performance_history]
        max_value = max(values)
        min_value = min(values)
        volatility = np.std(values) / np.mean(values) * 100
        
        print(f"📊 Performance Summary:")
        print(f"   Initial Value: ${initial_value:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Max Value: ${max_value:,.2f}")
        print(f"   Min Value: ${min_value:,.2f}")
        print(f"   Volatility: {volatility:.2f}%")
        print(f"   Analysis Duration: {len(self.performance_history)} data points")
        
        # Best and worst predictions
        all_predictions = {}
        for record in self.performance_history:
            for symbol, prediction in record['predictions'].items():
                if symbol not in all_predictions:
                    all_predictions[symbol] = []
                all_predictions[symbol].append(prediction)
        
        print(f"\n🎯 Prediction Summary:")
        for symbol, predictions in all_predictions.items():
            avg_prediction = np.mean(predictions)
            prediction_volatility = np.std(predictions)
            print(f"   {symbol}: Avg ${avg_prediction:.2f} (±${prediction_volatility:.2f})")

async def demo_realtime_portfolio_system():
    """Demonstrate the complete real-time portfolio system"""
    print("🎯 Real-Time Portfolio Management System Demo")
    print("=" * 60)
    
    # Initialize system
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    portfolio_manager = RealTimePortfolioManager(symbols, initial_capital=100000)
    
    # Run real-time analysis
    await portfolio_manager.run_real_time_analysis(duration_minutes=2)  # 2-minute demo
    
    print("\n✅ Real-time portfolio system demonstration complete!")

if __name__ == "__main__":
    # Check if Redis is available
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connection successful")
    except:
        print("❌ Redis not available. Please start Redis server or use Docker:")
        print("   docker run -d -p 6379:6379 redis:latest")
        exit(1)
    
    # Run the demo
    asyncio.run(demo_realtime_portfolio_system())