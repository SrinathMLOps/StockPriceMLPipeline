#!/usr/bin/env python3
# Model Drift Detection System

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import redis
import json
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import logging
from dataclasses import dataclass
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Data class for drift alerts"""
    timestamp: datetime
    model_name: str
    drift_type: str  # 'data', 'concept', 'performance'
    severity: str    # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    message: str

class ModelDriftDetector:
    """Advanced model drift detection system"""
    
    def __init__(self):
        self.mlflow_uri = "http://localhost:5000"
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient()
        
        # Drift detection thresholds
        self.thresholds = {
            'performance_drift': {
                'r2_drop': 0.05,      # 5% drop in R² score
                'mse_increase': 0.20,  # 20% increase in MSE
                'accuracy_drop': 0.03  # 3% drop in accuracy
            },
            'data_drift': {
                'ks_test': 0.05,       # KS test p-value threshold
                'psi_threshold': 0.2,   # Population Stability Index
                'feature_drift': 0.1    # Feature distribution change
            },
            'concept_drift': {
                'prediction_shift': 0.15,  # 15% shift in predictions
                'residual_pattern': 0.1     # Change in residual patterns
            }
        }
        
        # Alert history
        self.alerts = []
        
    def calculate_population_stability_index(self, baseline: np.ndarray, 
                                           current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI) for data drift detection"""
        
        # Create bins based on baseline data
        bin_edges = np.histogram_bin_edges(baseline, bins=bins)
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges, density=True)
        current_dist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        baseline_dist = baseline_dist + epsilon
        current_dist = current_dist + epsilon
        
        # Calculate PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
        
        return psi
    
    def detect_data_drift(self, baseline_data: Dict[str, np.ndarray], 
                         current_data: Dict[str, np.ndarray]) -> List[DriftAlert]:
        """Detect data drift using statistical tests"""
        
        alerts = []
        
        for feature_name in baseline_data.keys():
            if feature_name not in current_data:
                continue
                
            baseline_feature = baseline_data[feature_name]
            current_feature = current_data[feature_name]
            
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(baseline_feature, current_feature)
            
            if ks_p_value < self.thresholds['data_drift']['ks_test']:
                severity = 'high' if ks_p_value < 0.01 else 'medium'
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    model_name='multi_asset',
                    drift_type='data',
                    severity=severity,
                    metric_name=f'{feature_name}_ks_test',
                    current_value=ks_p_value,
                    baseline_value=1.0,
                    threshold=self.thresholds['data_drift']['ks_test'],
                    message=f"Data drift detected in {feature_name}: KS p-value = {ks_p_value:.4f}"
                ))
            
            # Population Stability Index
            try:
                psi = self.calculate_population_stability_index(baseline_feature, current_feature)
                
                if psi > self.thresholds['data_drift']['psi_threshold']:
                    severity = 'critical' if psi > 0.5 else 'high' if psi > 0.3 else 'medium'
                    alerts.append(DriftAlert(
                        timestamp=datetime.now(),
                        model_name='multi_asset',
                        drift_type='data',
                        severity=severity,
                        metric_name=f'{feature_name}_psi',
                        current_value=psi,
                        baseline_value=0.0,
                        threshold=self.thresholds['data_drift']['psi_threshold'],
                        message=f"Data drift detected in {feature_name}: PSI = {psi:.4f}"
                    ))
            except Exception as e:
                logger.warning(f"Could not calculate PSI for {feature_name}: {e}")
        
        return alerts
    
    def detect_performance_drift(self, model_name: str, 
                               baseline_metrics: Dict[str, float],
                               current_metrics: Dict[str, float]) -> List[DriftAlert]:
        """Detect performance drift by comparing model metrics"""
        
        alerts = []
        
        # Check R² score drift
        if 'r2_score' in baseline_metrics and 'r2_score' in current_metrics:
            baseline_r2 = baseline_metrics['r2_score']
            current_r2 = current_metrics['r2_score']
            r2_drop = baseline_r2 - current_r2
            
            if r2_drop > self.thresholds['performance_drift']['r2_drop']:
                severity = 'critical' if r2_drop > 0.1 else 'high'
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    model_name=model_name,
                    drift_type='performance',
                    severity=severity,
                    metric_name='r2_score',
                    current_value=current_r2,
                    baseline_value=baseline_r2,
                    threshold=self.thresholds['performance_drift']['r2_drop'],
                    message=f"Performance drift: R² dropped by {r2_drop:.4f} ({r2_drop/baseline_r2*100:.1f}%)"
                ))
        
        # Check MSE increase
        if 'mse' in baseline_metrics and 'mse' in current_metrics:
            baseline_mse = baseline_metrics['mse']
            current_mse = current_metrics['mse']
            mse_increase = (current_mse - baseline_mse) / baseline_mse
            
            if mse_increase > self.thresholds['performance_drift']['mse_increase']:
                severity = 'critical' if mse_increase > 0.5 else 'high'
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    model_name=model_name,
                    drift_type='performance',
                    severity=severity,
                    metric_name='mse',
                    current_value=current_mse,
                    baseline_value=baseline_mse,
                    threshold=self.thresholds['performance_drift']['mse_increase'],
                    message=f"Performance drift: MSE increased by {mse_increase*100:.1f}%"
                ))
        
        return alerts
    
    def detect_concept_drift(self, baseline_predictions: np.ndarray,
                           current_predictions: np.ndarray,
                           baseline_residuals: np.ndarray,
                           current_residuals: np.ndarray) -> List[DriftAlert]:
        """Detect concept drift by analyzing prediction patterns"""
        
        alerts = []
        
        # Check prediction distribution shift
        ks_stat, ks_p_value = stats.ks_2samp(baseline_predictions, current_predictions)
        
        if ks_p_value < 0.05:  # Significant shift in predictions
            pred_shift = abs(np.mean(current_predictions) - np.mean(baseline_predictions)) / np.std(baseline_predictions)
            
            if pred_shift > self.thresholds['concept_drift']['prediction_shift']:
                severity = 'high' if pred_shift > 0.3 else 'medium'
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    model_name='multi_asset',
                    drift_type='concept',
                    severity=severity,
                    metric_name='prediction_shift',
                    current_value=pred_shift,
                    baseline_value=0.0,
                    threshold=self.thresholds['concept_drift']['prediction_shift'],
                    message=f"Concept drift: Prediction distribution shifted by {pred_shift:.4f} standard deviations"
                ))
        
        # Check residual pattern changes
        baseline_residual_std = np.std(baseline_residuals)
        current_residual_std = np.std(current_residuals)
        residual_change = abs(current_residual_std - baseline_residual_std) / baseline_residual_std
        
        if residual_change > self.thresholds['concept_drift']['residual_pattern']:
            severity = 'high' if residual_change > 0.2 else 'medium'
            alerts.append(DriftAlert(
                timestamp=datetime.now(),
                model_name='multi_asset',
                drift_type='concept',
                severity=severity,
                metric_name='residual_pattern',
                current_value=residual_change,
                baseline_value=0.0,
                threshold=self.thresholds['concept_drift']['residual_pattern'],
                message=f"Concept drift: Residual pattern changed by {residual_change*100:.1f}%"
            ))
        
        return alerts
    
    def get_baseline_data(self, model_name: str, days_back: int = 7) -> Dict:
        """Get baseline data for drift comparison"""
        
        # In a real implementation, this would fetch historical data
        # For demo, we'll generate baseline data
        np.random.seed(42)
        
        baseline_data = {
            'features': {
                'ma_3': np.random.normal(150, 10, 1000),
                'pct_change_1d': np.random.normal(0.01, 0.02, 1000),
                'volume': np.random.lognormal(13, 1, 1000),
                'volatility': np.random.gamma(2, 0.01, 1000)
            },
            'predictions': np.random.normal(150, 5, 1000),
            'residuals': np.random.normal(0, 2, 1000),
            'metrics': {
                'r2_score': 0.9982,
                'mse': 0.15,
                'mae': 0.12
            }
        }
        
        return baseline_data
    
    def get_current_data(self, model_name: str) -> Dict:
        """Get current data for drift comparison"""
        
        # Simulate current data with some drift
        np.random.seed(int(time.time()) % 1000)  # Different seed for current data
        
        # Simulate some drift by shifting distributions
        drift_factor = 0.1  # 10% drift
        
        current_data = {
            'features': {
                'ma_3': np.random.normal(150 * (1 + drift_factor), 10, 500),
                'pct_change_1d': np.random.normal(0.01 * (1 + drift_factor), 0.02, 500),
                'volume': np.random.lognormal(13 * (1 + drift_factor), 1, 500),
                'volatility': np.random.gamma(2, 0.01 * (1 + drift_factor), 500)
            },
            'predictions': np.random.normal(150 * (1 + drift_factor), 5, 500),
            'residuals': np.random.normal(0, 2 * (1 + drift_factor), 500),
            'metrics': {
                'r2_score': 0.9982 * (1 - drift_factor * 0.5),  # Slight performance drop
                'mse': 0.15 * (1 + drift_factor * 2),           # MSE increase
                'mae': 0.12 * (1 + drift_factor)
            }
        }
        
        return current_data
    
    def run_drift_detection(self, model_name: str = 'multi_asset') -> List[DriftAlert]:
        """Run complete drift detection analysis"""
        
        logger.info(f"Starting drift detection for model: {model_name}")
        
        # Get baseline and current data
        baseline_data = self.get_baseline_data(model_name)
        current_data = self.get_current_data(model_name)
        
        all_alerts = []
        
        # Detect data drift
        data_drift_alerts = self.detect_data_drift(
            baseline_data['features'], 
            current_data['features']
        )
        all_alerts.extend(data_drift_alerts)
        
        # Detect performance drift
        performance_drift_alerts = self.detect_performance_drift(
            model_name,
            baseline_data['metrics'],
            current_data['metrics']
        )
        all_alerts.extend(performance_drift_alerts)
        
        # Detect concept drift
        concept_drift_alerts = self.detect_concept_drift(
            baseline_data['predictions'],
            current_data['predictions'],
            baseline_data['residuals'],
            current_data['residuals']
        )
        all_alerts.extend(concept_drift_alerts)
        
        # Store alerts
        self.alerts.extend(all_alerts)
        
        # Log alerts to Redis for real-time monitoring
        for alert in all_alerts:
            alert_data = {
                'timestamp': alert.timestamp.isoformat(),
                'model_name': alert.model_name,
                'drift_type': alert.drift_type,
                'severity': alert.severity,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'baseline_value': alert.baseline_value,
                'threshold': alert.threshold,
                'message': alert.message
            }
            
            # Store in Redis with expiration
            alert_key = f"drift_alert:{alert.model_name}:{alert.timestamp.isoformat()}"
            self.redis_client.setex(alert_key, 86400, json.dumps(alert_data))  # 24 hours
        
        logger.info(f"Drift detection completed. Found {len(all_alerts)} alerts")
        
        return all_alerts
    
    def get_drift_summary(self) -> Dict:
        """Get summary of drift detection results"""
        
        if not self.alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_type': {},
                'latest_alerts': []
            }
        
        # Count by severity
        severity_counts = {}
        for alert in self.alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for alert in self.alerts:
            type_counts[alert.drift_type] = type_counts.get(alert.drift_type, 0) + 1
        
        # Get latest alerts
        latest_alerts = sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:5]
        
        return {
            'total_alerts': len(self.alerts),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'latest_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'model_name': alert.model_name,
                    'drift_type': alert.drift_type,
                    'severity': alert.severity,
                    'message': alert.message
                }
                for alert in latest_alerts
            ]
        }
    
    def generate_drift_report(self) -> str:
        """Generate comprehensive drift detection report"""
        
        summary = self.get_drift_summary()
        
        report = f"""
# 🚨 Model Drift Detection Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary
- **Total Alerts**: {summary['total_alerts']}
- **Critical Issues**: {summary['by_severity'].get('critical', 0)}
- **High Priority**: {summary['by_severity'].get('high', 0)}
- **Medium Priority**: {summary['by_severity'].get('medium', 0)}

## 🔍 Drift Types Detected
- **Data Drift**: {summary['by_type'].get('data', 0)} alerts
- **Performance Drift**: {summary['by_type'].get('performance', 0)} alerts  
- **Concept Drift**: {summary['by_type'].get('concept', 0)} alerts

## 🚨 Latest Alerts
"""
        
        for alert in summary['latest_alerts']:
            severity_emoji = {
                'critical': '🔴',
                'high': '🟠', 
                'medium': '🟡',
                'low': '🟢'
            }.get(alert['severity'], '⚪')
            
            report += f"""
### {severity_emoji} {alert['drift_type'].title()} Drift - {alert['severity'].title()}
**Model**: {alert['model_name']}  
**Time**: {alert['timestamp']}  
**Message**: {alert['message']}
"""
        
        return report

async def continuous_drift_monitoring(interval_minutes: int = 30):
    """Run continuous drift monitoring"""
    
    detector = ModelDriftDetector()
    
    print(f"🔍 Starting continuous drift monitoring (every {interval_minutes} minutes)")
    
    while True:
        try:
            # Run drift detection
            alerts = detector.run_drift_detection()
            
            # Print summary
            if alerts:
                print(f"\n⚠️ {len(alerts)} drift alerts detected at {datetime.now()}")
                for alert in alerts:
                    severity_emoji = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡', 
                        'low': '🟢'
                    }.get(alert.severity, '⚪')
                    print(f"  {severity_emoji} {alert.message}")
            else:
                print(f"✅ No drift detected at {datetime.now()}")
            
            # Wait for next check
            await asyncio.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n🛑 Drift monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error in drift monitoring: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry

def demo_drift_detection():
    """Demonstrate drift detection capabilities"""
    
    print("🚀 Model Drift Detection System Demo")
    print("=" * 50)
    
    # Initialize detector
    detector = ModelDriftDetector()
    
    # Run drift detection
    alerts = detector.run_drift_detection()
    
    # Display results
    if alerts:
        print(f"\n⚠️ Found {len(alerts)} drift alerts:")
        
        for alert in alerts:
            severity_emoji = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(alert.severity, '⚪')
            
            print(f"\n{severity_emoji} {alert.drift_type.upper()} DRIFT - {alert.severity.upper()}")
            print(f"   Model: {alert.model_name}")
            print(f"   Metric: {alert.metric_name}")
            print(f"   Current: {alert.current_value:.4f}")
            print(f"   Baseline: {alert.baseline_value:.4f}")
            print(f"   Threshold: {alert.threshold:.4f}")
            print(f"   Message: {alert.message}")
    else:
        print("✅ No drift detected - models are performing within expected parameters")
    
    # Generate report
    report = detector.generate_drift_report()
    print("\n" + "=" * 50)
    print("📋 DRIFT DETECTION REPORT")
    print("=" * 50)
    print(report)
    
    # Get summary
    summary = detector.get_drift_summary()
    print(f"\n📊 Summary: {summary['total_alerts']} total alerts")
    
    return detector

if __name__ == "__main__":
    # Run demo
    detector = demo_drift_detection()
    
    print("\n🔄 Options:")
    print("1. Run continuous monitoring: python -c \"import asyncio; from model_drift_detection import continuous_drift_monitoring; asyncio.run(continuous_drift_monitoring(5))\"")
    print("2. Integration with existing system: Import ModelDriftDetector class")
    print("3. View alerts in Redis: Check keys matching 'drift_alert:*'")