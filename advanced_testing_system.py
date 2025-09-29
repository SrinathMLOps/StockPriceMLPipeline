#!/usr/bin/env python3
# Advanced Testing System with Load Testing and Chaos Engineering

import asyncio
import aiohttp
import time
import random
import statistics
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import concurrent.futures
import threading
import psutil
import requests
import numpy as np
from dataclasses import dataclass
import logging
import subprocess
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestResult:
    """Results from load testing"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    test_duration: float

@dataclass
class ChaosTestResult:
    """Results from chaos engineering tests"""
    test_name: str
    test_type: str
    duration: float
    success: bool
    recovery_time: float
    impact_metrics: Dict[str, float]
    message: str

class LoadTester:
    """Advanced load testing system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def create_session(self):
        """Create aiohttp session"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def make_request(self, endpoint: str, method: str = "GET", 
                          data: Dict = None) -> Tuple[bool, float, str]:
        """Make a single HTTP request"""
        start_time = time.time()
        
        try:
            if method.upper() == "POST":
                if endpoint == "/predict":
                    # Generate random prediction request
                    params = {
                        "ma_3": random.uniform(100, 300),
                        "pct_change_1d": random.uniform(-0.1, 0.1),
                        "volume": random.randint(100000, 10000000)
                    }
                    async with self.session.post(f"{self.base_url}{endpoint}", params=params) as response:
                        await response.text()
                        success = response.status == 200
                else:
                    async with self.session.post(f"{self.base_url}{endpoint}", json=data) as response:
                        await response.text()
                        success = response.status == 200
            else:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    await response.text()
                    success = response.status == 200
            
            response_time = time.time() - start_time
            return success, response_time, ""
            
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, str(e)
    
    async def run_load_test(self, endpoint: str, concurrent_users: int = 10, 
                           duration_seconds: int = 60, method: str = "GET") -> LoadTestResult:
        """Run load test with specified parameters"""
        
        print(f"🚀 Starting load test: {concurrent_users} users, {duration_seconds}s duration")
        print(f"   Target: {method} {self.base_url}{endpoint}")
        
        await self.create_session()
        
        # Track results
        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        async def worker():
            """Worker function for load testing"""
            while time.time() < end_time:
                success, response_time, error = await self.make_request(endpoint, method)
                results.append({
                    'success': success,
                    'response_time': response_time,
                    'timestamp': time.time(),
                    'error': error
                })
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        # Start concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(concurrent_users)]
        
        # Wait for completion
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.close_session()
        
        # Calculate metrics
        if not results:
            return LoadTestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 100, duration_seconds)
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        response_times = [r['response_time'] for r in results]
        
        total_requests = len(results)
        successful_requests = len(successful)
        failed_requests = len(failed)
        
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_response_time = sorted_times[int(0.95 * len(sorted_times))]
        p99_response_time = sorted_times[int(0.99 * len(sorted_times))]
        
        requests_per_second = total_requests / duration_seconds
        error_rate = (failed_requests / total_requests) * 100
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            test_duration=duration_seconds
        )
    
    def print_load_test_results(self, result: LoadTestResult):
        """Print formatted load test results"""
        
        print("\n" + "="*60)
        print("📊 LOAD TEST RESULTS")
        print("="*60)
        
        print(f"📈 Request Statistics:")
        print(f"   Total Requests: {result.total_requests:,}")
        print(f"   Successful: {result.successful_requests:,} ({(result.successful_requests/result.total_requests)*100:.1f}%)")
        print(f"   Failed: {result.failed_requests:,} ({result.error_rate:.1f}%)")
        print(f"   Requests/Second: {result.requests_per_second:.1f}")
        
        print(f"\n⏱️ Response Time Statistics:")
        print(f"   Average: {result.average_response_time*1000:.1f}ms")
        print(f"   Min: {result.min_response_time*1000:.1f}ms")
        print(f"   Max: {result.max_response_time*1000:.1f}ms")
        print(f"   95th Percentile: {result.p95_response_time*1000:.1f}ms")
        print(f"   99th Percentile: {result.p99_response_time*1000:.1f}ms")
        
        # Performance assessment
        if result.error_rate < 1 and result.p95_response_time < 0.1:
            print(f"\n✅ Performance: EXCELLENT")
        elif result.error_rate < 5 and result.p95_response_time < 0.5:
            print(f"\n🟡 Performance: GOOD")
        else:
            print(f"\n🔴 Performance: NEEDS IMPROVEMENT")

class ChaosEngineer:
    """Chaos engineering testing system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'network_connections': len(psutil.net_connections())
        }
    
    def check_service_health(self) -> bool:
        """Check if the service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def chaos_test_high_load(self, duration: int = 30) -> ChaosTestResult:
        """Test system behavior under extreme load"""
        
        print(f"🌪️ Chaos Test: High Load Attack ({duration}s)")
        
        start_time = time.time()
        initial_metrics = self.get_system_metrics()
        initial_health = self.check_service_health()
        
        if not initial_health:
            return ChaosTestResult(
                test_name="high_load",
                test_type="load",
                duration=0,
                success=False,
                recovery_time=0,
                impact_metrics={},
                message="Service was not healthy before test"
            )
        
        # Create extreme load
        load_tester = LoadTester(self.base_url)
        
        # Run high-intensity load test
        result = await load_tester.run_load_test(
            endpoint="/predict",
            concurrent_users=50,  # High concurrency
            duration_seconds=duration,
            method="POST"
        )
        
        # Check recovery
        recovery_start = time.time()
        await asyncio.sleep(5)  # Wait for system to stabilize
        
        recovered = False
        recovery_time = 0
        
        for i in range(12):  # Check for 1 minute
            if self.check_service_health():
                recovered = True
                recovery_time = time.time() - recovery_start
                break
            await asyncio.sleep(5)
        
        final_metrics = self.get_system_metrics()
        
        # Calculate impact
        impact_metrics = {
            'cpu_increase': final_metrics['cpu_percent'] - initial_metrics['cpu_percent'],
            'memory_increase': final_metrics['memory_percent'] - initial_metrics['memory_percent'],
            'error_rate': result.error_rate,
            'avg_response_time': result.average_response_time * 1000  # Convert to ms
        }
        
        success = recovered and result.error_rate < 10  # Less than 10% error rate
        
        message = f"Load test completed. Error rate: {result.error_rate:.1f}%, Recovery: {'Yes' if recovered else 'No'}"
        
        return ChaosTestResult(
            test_name="high_load",
            test_type="load",
            duration=time.time() - start_time,
            success=success,
            recovery_time=recovery_time,
            impact_metrics=impact_metrics,
            message=message
        )
    
    async def chaos_test_memory_pressure(self, duration: int = 20) -> ChaosTestResult:
        """Test system behavior under memory pressure"""
        
        print(f"🧠 Chaos Test: Memory Pressure ({duration}s)")
        
        start_time = time.time()
        initial_metrics = self.get_system_metrics()
        initial_health = self.check_service_health()
        
        if not initial_health:
            return ChaosTestResult(
                test_name="memory_pressure",
                test_type="resource",
                duration=0,
                success=False,
                recovery_time=0,
                impact_metrics={},
                message="Service was not healthy before test"
            )
        
        # Create memory pressure by allocating large arrays
        memory_hogs = []
        
        try:
            # Allocate memory in chunks
            for i in range(5):
                # Allocate 100MB chunks
                memory_hog = np.random.rand(100 * 1024 * 1024 // 8)  # 100MB
                memory_hogs.append(memory_hog)
                await asyncio.sleep(2)
                
                # Check if service is still responsive
                if not self.check_service_health():
                    break
            
            # Keep memory pressure for duration
            await asyncio.sleep(duration)
            
        finally:
            # Release memory
            memory_hogs.clear()
            
        # Check recovery
        recovery_start = time.time()
        await asyncio.sleep(2)
        
        recovered = False
        recovery_time = 0
        
        for i in range(6):  # Check for 30 seconds
            if self.check_service_health():
                recovered = True
                recovery_time = time.time() - recovery_start
                break
            await asyncio.sleep(5)
        
        final_metrics = self.get_system_metrics()
        
        impact_metrics = {
            'memory_peak': final_metrics['memory_percent'],
            'memory_increase': final_metrics['memory_percent'] - initial_metrics['memory_percent']
        }
        
        success = recovered
        message = f"Memory pressure test completed. Peak memory: {final_metrics['memory_percent']:.1f}%, Recovery: {'Yes' if recovered else 'No'}"
        
        return ChaosTestResult(
            test_name="memory_pressure",
            test_type="resource",
            duration=time.time() - start_time,
            success=success,
            recovery_time=recovery_time,
            impact_metrics=impact_metrics,
            message=message
        )
    
    async def chaos_test_network_latency(self, duration: int = 30) -> ChaosTestResult:
        """Test system behavior with network latency simulation"""
        
        print(f"🌐 Chaos Test: Network Latency Simulation ({duration}s)")
        
        start_time = time.time()
        initial_health = self.check_service_health()
        
        if not initial_health:
            return ChaosTestResult(
                test_name="network_latency",
                test_type="network",
                duration=0,
                success=False,
                recovery_time=0,
                impact_metrics={},
                message="Service was not healthy before test"
            )
        
        # Simulate network latency by adding delays to requests
        response_times = []
        
        async def delayed_request():
            """Make request with simulated network delay"""
            # Add random delay (50-200ms)
            delay = random.uniform(0.05, 0.2)
            await asyncio.sleep(delay)
            
            start = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as response:
                        await response.text()
                        success = response.status == 200
            except:
                success = False
            
            response_time = time.time() - start
            response_times.append(response_time)
            return success
        
        # Run requests with simulated latency
        tasks = []
        end_time = start_time + duration
        
        while time.time() < end_time:
            task = asyncio.create_task(delayed_request())
            tasks.append(task)
            await asyncio.sleep(1)  # One request per second
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check recovery
        recovery_start = time.time()
        await asyncio.sleep(2)
        
        recovered = self.check_service_health()
        recovery_time = time.time() - recovery_start if recovered else 0
        
        # Calculate metrics
        successful_requests = sum(1 for r in results if r is True)
        total_requests = len(results)
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        impact_metrics = {
            'success_rate': success_rate,
            'avg_response_time': avg_response_time * 1000,  # Convert to ms
            'total_requests': total_requests
        }
        
        success = recovered and success_rate > 80
        message = f"Network latency test completed. Success rate: {success_rate:.1f}%, Avg response: {avg_response_time*1000:.1f}ms"
        
        return ChaosTestResult(
            test_name="network_latency",
            test_type="network",
            duration=time.time() - start_time,
            success=success,
            recovery_time=recovery_time,
            impact_metrics=impact_metrics,
            message=message
        )
    
    def print_chaos_test_results(self, result: ChaosTestResult):
        """Print formatted chaos test results"""
        
        status_emoji = "✅" if result.success else "❌"
        
        print(f"\n{status_emoji} Chaos Test: {result.test_name}")
        print(f"   Type: {result.test_type}")
        print(f"   Duration: {result.duration:.1f}s")
        print(f"   Recovery Time: {result.recovery_time:.1f}s")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        
        if result.impact_metrics:
            print(f"   Impact Metrics:")
            for metric, value in result.impact_metrics.items():
                print(f"     {metric}: {value:.2f}")

class AdvancedTestingSuite:
    """Complete advanced testing suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.load_tester = LoadTester(base_url)
        self.chaos_engineer = ChaosEngineer(base_url)
        
    async def run_comprehensive_tests(self):
        """Run complete testing suite"""
        
        print("🚀 ADVANCED TESTING SUITE")
        print("=" * 60)
        print(f"Target System: {self.base_url}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            'load_tests': [],
            'chaos_tests': [],
            'start_time': datetime.now(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
                'python_version': sys.version
            }
        }
        
        # 1. Basic Load Tests
        print("\n🔥 PHASE 1: LOAD TESTING")
        print("-" * 40)
        
        load_test_scenarios = [
            {"name": "Light Load", "users": 5, "duration": 30, "endpoint": "/health"},
            {"name": "Medium Load", "users": 15, "duration": 45, "endpoint": "/predict"},
            {"name": "Heavy Load", "users": 30, "duration": 60, "endpoint": "/predict"},
        ]
        
        for scenario in load_test_scenarios:
            print(f"\n📊 Running {scenario['name']}...")
            result = await self.load_tester.run_load_test(
                endpoint=scenario['endpoint'],
                concurrent_users=scenario['users'],
                duration_seconds=scenario['duration'],
                method="POST" if scenario['endpoint'] == "/predict" else "GET"
            )
            
            self.load_tester.print_load_test_results(result)
            results['load_tests'].append({
                'scenario': scenario['name'],
                'result': result
            })
        
        # 2. Chaos Engineering Tests
        print("\n🌪️ PHASE 2: CHAOS ENGINEERING")
        print("-" * 40)
        
        chaos_tests = [
            self.chaos_engineer.chaos_test_high_load(30),
            self.chaos_engineer.chaos_test_memory_pressure(20),
            self.chaos_engineer.chaos_test_network_latency(25)
        ]
        
        for chaos_test in chaos_tests:
            result = await chaos_test
            self.chaos_engineer.print_chaos_test_results(result)
            results['chaos_tests'].append(result)
            
            # Wait between chaos tests
            await asyncio.sleep(10)
        
        # 3. Generate Summary Report
        self.generate_test_report(results)
        
        return results
    
    def generate_test_report(self, results: Dict):
        """Generate comprehensive test report"""
        
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # System Information
        print(f"🖥️ System Information:")
        print(f"   CPU Cores: {results['system_info']['cpu_count']}")
        print(f"   Memory: {results['system_info']['memory_total']}GB")
        print(f"   Python: {results['system_info']['python_version'].split()[0]}")
        
        # Load Test Summary
        print(f"\n📊 Load Test Summary:")
        total_requests = sum(lt['result'].total_requests for lt in results['load_tests'])
        total_errors = sum(lt['result'].failed_requests for lt in results['load_tests'])
        avg_rps = statistics.mean([lt['result'].requests_per_second for lt in results['load_tests']])
        avg_response_time = statistics.mean([lt['result'].average_response_time for lt in results['load_tests']])
        
        print(f"   Total Requests: {total_requests:,}")
        print(f"   Total Errors: {total_errors:,} ({(total_errors/total_requests)*100:.1f}%)")
        print(f"   Average RPS: {avg_rps:.1f}")
        print(f"   Average Response Time: {avg_response_time*1000:.1f}ms")
        
        # Chaos Test Summary
        print(f"\n🌪️ Chaos Test Summary:")
        successful_chaos_tests = sum(1 for ct in results['chaos_tests'] if ct.success)
        total_chaos_tests = len(results['chaos_tests'])
        
        print(f"   Tests Passed: {successful_chaos_tests}/{total_chaos_tests}")
        print(f"   Success Rate: {(successful_chaos_tests/total_chaos_tests)*100:.1f}%")
        
        # Overall Assessment
        print(f"\n🎯 Overall Assessment:")
        
        overall_error_rate = (total_errors / total_requests) * 100 if total_requests > 0 else 0
        chaos_success_rate = (successful_chaos_tests / total_chaos_tests) * 100
        
        if overall_error_rate < 1 and avg_response_time < 0.1 and chaos_success_rate > 80:
            grade = "A+ (EXCELLENT)"
            emoji = "🏆"
        elif overall_error_rate < 5 and avg_response_time < 0.5 and chaos_success_rate > 60:
            grade = "B+ (GOOD)"
            emoji = "✅"
        elif overall_error_rate < 10 and avg_response_time < 1.0 and chaos_success_rate > 40:
            grade = "C+ (ACCEPTABLE)"
            emoji = "🟡"
        else:
            grade = "D (NEEDS IMPROVEMENT)"
            emoji = "🔴"
        
        print(f"   {emoji} System Grade: {grade}")
        print(f"   Error Rate: {overall_error_rate:.1f}%")
        print(f"   Avg Response: {avg_response_time*1000:.1f}ms")
        print(f"   Resilience: {chaos_success_rate:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if overall_error_rate > 5:
            print("   - Investigate and fix error sources")
        if avg_response_time > 0.5:
            print("   - Optimize response time performance")
        if chaos_success_rate < 80:
            print("   - Improve system resilience and recovery")
        if overall_error_rate < 1 and avg_response_time < 0.1 and chaos_success_rate > 90:
            print("   - System is performing excellently!")
        
        # Save report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to JSON-serializable format
        json_results = {
            'start_time': results['start_time'].isoformat(),
            'system_info': results['system_info'],
            'load_tests': [
                {
                    'scenario': lt['scenario'],
                    'total_requests': lt['result'].total_requests,
                    'successful_requests': lt['result'].successful_requests,
                    'failed_requests': lt['result'].failed_requests,
                    'average_response_time': lt['result'].average_response_time,
                    'requests_per_second': lt['result'].requests_per_second,
                    'error_rate': lt['result'].error_rate
                }
                for lt in results['load_tests']
            ],
            'chaos_tests': [
                {
                    'test_name': ct.test_name,
                    'test_type': ct.test_type,
                    'duration': ct.duration,
                    'success': ct.success,
                    'recovery_time': ct.recovery_time,
                    'impact_metrics': ct.impact_metrics,
                    'message': ct.message
                }
                for ct in results['chaos_tests']
            ],
            'summary': {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'overall_error_rate': overall_error_rate,
                'avg_response_time_ms': avg_response_time * 1000,
                'chaos_success_rate': chaos_success_rate,
                'grade': grade
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Report saved to: {report_file}")

async def main():
    """Main function to run advanced testing suite"""
    
    # Check if service is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Service is not healthy. Please start your MLOps pipeline first.")
            print("   Run: docker compose up -d")
            return
    except:
        print("❌ Cannot connect to service at http://localhost:8000")
        print("   Please start your MLOps pipeline first:")
        print("   Run: docker compose up -d")
        return
    
    # Run comprehensive testing
    testing_suite = AdvancedTestingSuite()
    await testing_suite.run_comprehensive_tests()

if __name__ == "__main__":
    print("🧪 Advanced Testing System for MLOps Pipeline")
    print("=" * 60)
    
    # Install required packages
    try:
        import aiohttp
        import psutil
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "psutil"])
        import aiohttp
        import psutil
    
    # Run the testing suite
    asyncio.run(main())