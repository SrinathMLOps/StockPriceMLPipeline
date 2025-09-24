#!/usr/bin/env python3
# Comprehensive testing suite for ML Pipeline

import requests
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

class MLPipelineTester:
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.mlflow_base = "http://localhost:5000"
        self.test_results = []
    
    def test_api_health(self):
        """Test API health and availability"""
        print("🔍 Testing API Health...")
        
        try:
            response = requests.get(f"{self.api_base}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Health: {data['message']}")
                print(f"✅ Model Version: {data['model_version']}")
                return True
            else:
                print(f"❌ API Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Connection Failed: {e}")
            return False
    
    def test_model_info(self):
        """Test model information endpoint"""
        print("\n🔍 Testing Model Info...")
        
        try:
            response = requests.get(f"{self.api_base}/model/info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model Type: {data['model_type']}")
                print(f"✅ Features: {data['features']}")
                return True
            else:
                print(f"❌ Model Info Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model Info Error: {e}")
            return False
    
    def test_predictions(self):
        """Test prediction endpoint with various scenarios"""
        print("\n🔍 Testing Predictions...")
        
        test_cases = [
            {"name": "Normal", "ma_3": 150.0, "pct_change_1d": 0.01, "volume": 1000000},
            {"name": "Bull Market", "ma_3": 200.0, "pct_change_1d": 0.05, "volume": 2000000},
            {"name": "Bear Market", "ma_3": 100.0, "pct_change_1d": -0.03, "volume": 500000},
            {"name": "High Volume", "ma_3": 175.0, "pct_change_1d": 0.02, "volume": 5000000},
            {"name": "Low Volume", "ma_3": 160.0, "pct_change_1d": 0.01, "volume": 100000}
        ]
        
        predictions = []
        
        for case in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base}/predict",
                    params={
                        "ma_3": case["ma_3"],
                        "pct_change_1d": case["pct_change_1d"],
                        "volume": case["volume"]
                    },
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    predictions.append(prediction)
                    print(f"✅ {case['name']}: ${prediction:.2f} ({response_time:.3f}s)")
                else:
                    print(f"❌ {case['name']}: Failed ({response.status_code})")
                    
            except Exception as e:
                print(f"❌ {case['name']}: Error - {e}")
        
        if predictions:
            print(f"\n📊 Prediction Summary:")
            print(f"   Average: ${np.mean(predictions):.2f}")
            print(f"   Range: ${np.min(predictions):.2f} - ${np.max(predictions):.2f}")
            print(f"   Std Dev: ${np.std(predictions):.2f}")
            return True
        
        return False
    
    def test_performance(self):
        """Test API performance under load"""
        print("\n🔍 Testing Performance...")
        
        test_params = {"ma_3": 150.0, "pct_change_1d": 0.01, "volume": 1000000}
        response_times = []
        
        for i in range(10):
            try:
                start_time = time.time()
                response = requests.post(f"{self.api_base}/predict", params=test_params, timeout=5)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    response_times.append(response_time)
                    print(f"✅ Request {i+1}: {response_time:.3f}s")
                else:
                    print(f"❌ Request {i+1}: Failed")
                    
            except Exception as e:
                print(f"❌ Request {i+1}: Error - {e}")
        
        if response_times:
            print(f"\n📊 Performance Summary:")
            print(f"   Average Response Time: {np.mean(response_times):.3f}s")
            print(f"   Min Response Time: {np.min(response_times):.3f}s")
            print(f"   Max Response Time: {np.max(response_times):.3f}s")
            print(f"   Success Rate: {len(response_times)}/10 ({len(response_times)*10}%)")
            return True
        
        return False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🔍 Testing Edge Cases...")
        
        edge_cases = [
            {"name": "Zero Values", "ma_3": 0, "pct_change_1d": 0, "volume": 0},
            {"name": "Negative MA", "ma_3": -100, "pct_change_1d": 0.01, "volume": 1000000},
            {"name": "Large Change", "ma_3": 150, "pct_change_1d": 0.5, "volume": 1000000},
            {"name": "Extreme Volume", "ma_3": 150, "pct_change_1d": 0.01, "volume": 100000000}
        ]
        
        for case in edge_cases:
            try:
                response = requests.post(f"{self.api_base}/predict", params=case, timeout=5)
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    print(f"✅ {case['name']}: ${prediction:.2f}")
                else:
                    print(f"⚠️ {case['name']}: Status {response.status_code}")
            except Exception as e:
                print(f"❌ {case['name']}: Error - {e}")
        
        return True
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting ML Pipeline Test Suite")
        print("=" * 50)
        
        tests = [
            ("API Health", self.test_api_health),
            ("Model Info", self.test_model_info),
            ("Predictions", self.test_predictions),
            ("Performance", self.test_performance),
            ("Edge Cases", self.test_edge_cases)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} Test Failed: {e}")
                results[test_name] = False
        
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        return results

if __name__ == "__main__":
    tester = MLPipelineTester()
    tester.run_all_tests()