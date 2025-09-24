#!/usr/bin/env python3
# Test Enhanced FastAPI Server with Real-Time Features

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime

class EnhancedAPITester:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.websocket_url = "ws://localhost:8001/ws/realtime"
    
    def test_health_check(self):
        """Test enhanced health check endpoint"""
        print("🔍 Testing Enhanced Health Check...")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Status: {data['status']}")
                print(f"✅ Redis Status: {data['services']['redis']}")
                print(f"✅ Models Loaded: {len(data['services']['models'])}")
                print(f"✅ Active WebSockets: {data['services']['active_websockets']}")
                print(f"✅ Available Features: {len(data['features'])}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_single_prediction(self):
        """Test enhanced single stock prediction"""
        print("\n🔮 Testing Enhanced Single Stock Prediction...")
        
        try:
            payload = {
                "symbol": "AAPL",
                "ma_3": 150.0,
                "pct_change_1d": 0.02,
                "volume": 1000000
            }
            
            response = requests.post(f"{self.base_url}/predict", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {data['symbol']} Prediction: ${data['prediction']:.2f}")
                
                if 'current_price' in data:
                    print(f"   Current Price: ${data['current_price']:.2f}")
                    print(f"   Price Change: {data['price_change_pct']:+.2f}%")
                    print(f"   Volume: {data['volume']:,}")
                
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
    
    def test_portfolio_prediction(self):
        """Test portfolio-wide predictions"""
        print("\n📊 Testing Portfolio Predictions...")
        
        try:
            payload = {
                "assets": ["AAPL", "GOOGL", "MSFT"],
                "features": {
                    "AAPL": {"ma_3": 150.0, "pct_change_1d": 0.02, "volume": 1000000},
                    "GOOGL": {"ma_3": 2500.0, "pct_change_1d": 0.01, "volume": 500000},
                    "MSFT": {"ma_3": 300.0, "pct_change_1d": 0.015, "volume": 800000}
                }
            }
            
            response = requests.post(f"{self.base_url}/predict/portfolio", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Portfolio Predictions:")
                
                for symbol, prediction in data['predictions'].items():
                    print(f"   {symbol}: ${prediction:.2f}")
                
                metrics = data['portfolio_metrics']
                print(f"\n📈 Portfolio Metrics:")
                print(f"   Total Value: ${metrics['total_predicted_value']:,.2f}")
                print(f"   Average: ${metrics['average_prediction']:.2f}")
                print(f"   Volatility: ${metrics['prediction_volatility']:.2f}")
                print(f"   Assets: {metrics['asset_count']}")
                
                return True
            else:
                print(f"❌ Portfolio prediction failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Portfolio prediction error: {e}")
            return False
    
    def test_realtime_market_data(self):
        """Test real-time market data endpoint"""
        print("\n📡 Testing Real-Time Market Data...")
        
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        for symbol in symbols:
            try:
                response = requests.get(f"{self.base_url}/market/realtime/{symbol}")
                
                if response.status_code == 200:
                    data = response.json()
                    market_data = data['market_data']
                    
                    print(f"✅ {symbol} Real-Time Data:")
                    print(f"   Price: ${market_data['price']:.2f}")
                    print(f"   Change: {market_data['change_pct']:+.2f}%")
                    print(f"   Volume: {market_data['volume']:,}")
                    print(f"   Bid/Ask: ${market_data['bid']:.2f}/${market_data['ask']:.2f}")
                    
                    if data['technical_indicators']:
                        indicators = data['technical_indicators']
                        print(f"   MA(3): ${indicators.get('ma_3', 0):.2f}")
                        print(f"   RSI: {indicators.get('rsi', 0):.1f}")
                        print(f"   Volatility: {indicators.get('volatility', 0):.4f}")
                    
                elif response.status_code == 404:
                    print(f"⚠️ No real-time data available for {symbol}")
                else:
                    print(f"❌ Market data failed for {symbol}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Market data error for {symbol}: {e}")
        
        return True
    
    def test_trading_signals(self):
        """Test trading signals endpoint"""
        print("\n🎯 Testing Trading Signals...")
        
        try:
            response = requests.get(f"{self.base_url}/trading/signals")
            
            if response.status_code == 200:
                data = response.json()
                signals = data['signals']
                
                print("✅ Trading Signals:")
                for signal in signals:
                    symbol = signal['symbol']
                    action = signal['signal']
                    confidence = signal['confidence']
                    current = signal['current_price']
                    predicted = signal['predicted_price']
                    change = signal['price_change_pct']
                    
                    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
                    
                    print(f"   {emoji} {symbol}: {action} (Confidence: {confidence:.1%})")
                    print(f"      Current: ${current:.2f} → Predicted: ${predicted:.2f} ({change:+.1f}%)")
                
                return True
            else:
                print(f"❌ Trading signals failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Trading signals error: {e}")
            return False
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization endpoint"""
        print("\n⚖️ Testing Portfolio Optimization...")
        
        try:
            response = requests.get(f"{self.base_url}/portfolio/optimize")
            
            if response.status_code == 200:
                data = response.json()
                
                if data['optimization_successful']:
                    weights = data['optimal_weights']
                    metrics = data['expected_metrics']
                    
                    print("✅ Optimal Portfolio Allocation:")
                    for symbol, weight in weights.items():
                        print(f"   {symbol}: {weight:.1%}")
                    
                    print(f"\n📊 Expected Performance:")
                    print(f"   Annual Return: {metrics['expected_return']:.1%}")
                    print(f"   Volatility: {metrics['volatility']:.1%}")
                    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    
                    return True
                else:
                    print("❌ Portfolio optimization failed")
                    return False
                    
            else:
                print(f"❌ Optimization request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Portfolio optimization error: {e}")
            return False
    
    async def test_websocket_stream(self, duration_seconds=10):
        """Test WebSocket real-time streaming"""
        print(f"\n🌊 Testing WebSocket Real-Time Stream ({duration_seconds}s)...")
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                print("✅ WebSocket connected")
                
                start_time = time.time()
                message_count = 0
                
                while time.time() - start_time < duration_seconds:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        
                        if data['type'] == 'market_update':
                            message_count += 1
                            market_data = data['data']
                            
                            print(f"📡 Update #{message_count} - {datetime.now().strftime('%H:%M:%S')}")
                            
                            # Show data for first few assets
                            for i, (symbol, info) in enumerate(list(market_data.items())[:3]):
                                price = info['price']
                                change = info['change_pct']
                                prediction = info.get('prediction', 'N/A')
                                
                                pred_str = f"→ ${prediction:.2f}" if prediction != 'N/A' else ""
                                print(f"   {symbol}: ${price:.2f} ({change:+.2f}%) {pred_str}")
                            
                            if len(market_data) > 3:
                                print(f"   ... and {len(market_data) - 3} more assets")
                        
                    except asyncio.TimeoutError:
                        print("⚠️ WebSocket timeout - no data received")
                        break
                
                print(f"✅ WebSocket test completed - {message_count} messages received")
                return True
                
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Enhanced FastAPI Server Test Suite")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Single Prediction", self.test_single_prediction),
            ("Portfolio Prediction", self.test_portfolio_prediction),
            ("Real-Time Market Data", self.test_realtime_market_data),
            ("Trading Signals", self.test_trading_signals),
            ("Portfolio Optimization", self.test_portfolio_optimization),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                results[test_name] = False
        
        # Test WebSocket separately (async)
        try:
            websocket_result = asyncio.run(self.test_websocket_stream(10))
            results["WebSocket Streaming"] = websocket_result
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            results["WebSocket Streaming"] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if passed_test:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Enhanced API is fully functional!")
        else:
            print("⚠️ Some tests failed. Check the logs above for details.")
        
        return results

if __name__ == "__main__":
    # Wait a moment for server to fully start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    tester = EnhancedAPITester()
    tester.run_all_tests()