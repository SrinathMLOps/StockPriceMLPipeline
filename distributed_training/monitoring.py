"""
Performance monitoring and dashboard components for distributed training.
"""

import time
import psutil
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
from queue import Queue

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

from .config import TrainingMetrics


@dataclass
class GPUMetrics:
    """GPU utilization and memory metrics."""
    gpu_id: int
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_draw_w: float

    @property
    def memory_utilization_percent(self) -> float:
        """Calculate memory utilization percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


@dataclass
class NetworkMetrics:
    """Network communication metrics."""
    bytes_sent: int
    bytes_received: int
    packets_sent: int
    packets_received: int
    bandwidth_mbps: float
    latency_ms: float


@dataclass
class NodeMetrics:
    """Complete node performance metrics."""
    node_id: int
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    gpu_metrics: List[GPUMetrics]
    network_metrics: NetworkMetrics
    training_throughput: float


class NodeMonitorInterface(ABC):
    """Abstract interface for node monitoring."""

    @abstractmethod
    def collect_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect GPU metrics from all available GPUs."""
        pass

    @abstractmethod
    def collect_network_metrics(self) -> NetworkMetrics:
        """Collect network communication metrics."""
        pass

    @abstractmethod
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics (CPU, memory)."""
        pass


class NodeMonitor:
    """
    Real-time node performance monitoring.
    
    Collects GPU, network, and system metrics for distributed training nodes.
    """

    def __init__(self, node_id: int, logger: Optional[logging.Logger] = None):
        """
        Initialize node monitor.
        
        Args:
            node_id: Unique identifier for this node
            logger: Optional logger instance
        """
        self.node_id = node_id
        self.logger = logger or logging.getLogger(__name__)
        self.is_monitoring = False
        self.metrics_queue = Queue()
        self.monitor_thread = None
        
        # Initialize NVIDIA ML if available
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.logger.info(f"Initialized NVIDIA ML with {self.gpu_count} GPUs")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVIDIA ML: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
            self.logger.warning("NVIDIA ML not available, GPU monitoring disabled")

    def collect_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect metrics from all available GPUs."""
        gpu_metrics = []
        
        if not NVIDIA_ML_AVAILABLE or self.gpu_count == 0:
            return gpu_metrics
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Get power draw
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0.0
                
                metrics = GPUMetrics(
                    gpu_id=i,
                    utilization_percent=float(util.gpu),
                    memory_used_mb=float(mem_info.used) / (1024 * 1024),
                    memory_total_mb=float(mem_info.total) / (1024 * 1024),
                    temperature_c=float(temp),
                    power_draw_w=power
                )
                gpu_metrics.append(metrics)
                
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {e}")
        
        return gpu_metrics

    def collect_network_metrics(self) -> NetworkMetrics:
        """Collect network communication metrics."""
        try:
            net_io = psutil.net_io_counters()
            
            # Simple bandwidth estimation (would need historical data for accuracy)
            bandwidth_mbps = 0.0  # Placeholder
            latency_ms = 0.0      # Placeholder
            
            return NetworkMetrics(
                bytes_sent=net_io.bytes_sent,
                bytes_received=net_io.bytes_recv,
                packets_sent=net_io.packets_sent,
                packets_received=net_io.packets_recv,
                bandwidth_mbps=bandwidth_mbps,
                latency_ms=latency_ms
            )
        except Exception as e:
            self.logger.error(f"Error collecting network metrics: {e}")
            return NetworkMetrics(0, 0, 0, 0, 0.0, 0.0)

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_utilization": cpu_percent,
                "memory_utilization": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {"cpu_utilization": 0.0, "memory_utilization": 0.0}

    def collect_all_metrics(self) -> NodeMetrics:
        """Collect all node metrics."""
        gpu_metrics = self.collect_gpu_metrics()
        network_metrics = self.collect_network_metrics()
        system_metrics = self.collect_system_metrics()
        
        return NodeMetrics(
            node_id=self.node_id,
            timestamp=datetime.now(),
            cpu_utilization=system_metrics.get("cpu_utilization", 0.0),
            memory_utilization=system_metrics.get("memory_utilization", 0.0),
            gpu_metrics=gpu_metrics,
            network_metrics=network_metrics,
            training_throughput=0.0  # To be updated by training loop
        )

    def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """Start continuous monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"Started monitoring for node {self.node_id}")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info(f"Stopped monitoring for node {self.node_id}")

    def _monitoring_loop(self, interval_seconds: float) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self.collect_all_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)

    def get_latest_metrics(self) -> Optional[NodeMetrics]:
        """Get the most recent metrics."""
        try:
            return self.metrics_queue.get_nowait()
        except:
            return None


class PerformanceDashboard:
    """
    Performance dashboard for aggregating and visualizing metrics.
    
    Collects metrics from all nodes and provides analysis and recommendations.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize performance dashboard."""
        self.logger = logger or logging.getLogger(__name__)
        self.node_monitors: Dict[int, NodeMonitor] = {}
        self.training_metrics: List[TrainingMetrics] = []
        self.node_metrics: List[NodeMetrics] = []

    def add_node_monitor(self, node_monitor: NodeMonitor) -> None:
        """Add a node monitor to the dashboard."""
        self.node_monitors[node_monitor.node_id] = node_monitor
        self.logger.info(f"Added node monitor for node {node_monitor.node_id}")

    def collect_all_node_metrics(self) -> List[NodeMetrics]:
        """Collect metrics from all monitored nodes."""
        all_metrics = []
        
        for node_id, monitor in self.node_monitors.items():
            metrics = monitor.get_latest_metrics()
            if metrics:
                all_metrics.append(metrics)
                self.node_metrics.append(metrics)
        
        return all_metrics

    def add_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Add training metrics to the dashboard."""
        self.training_metrics.append(metrics)

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the entire cluster."""
        if not self.node_metrics:
            return {}
        
        recent_metrics = self.node_metrics[-len(self.node_monitors):]
        
        # Calculate averages
        avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
        
        # GPU metrics
        all_gpu_metrics = []
        for node_metric in recent_metrics:
            all_gpu_metrics.extend(node_metric.gpu_metrics)
        
        avg_gpu_util = 0.0
        avg_gpu_memory = 0.0
        if all_gpu_metrics:
            avg_gpu_util = sum(g.utilization_percent for g in all_gpu_metrics) / len(all_gpu_metrics)
            avg_gpu_memory = sum(g.memory_utilization_percent for g in all_gpu_metrics) / len(all_gpu_metrics)
        
        return {
            "total_nodes": len(self.node_monitors),
            "avg_cpu_utilization": avg_cpu,
            "avg_memory_utilization": avg_memory,
            "avg_gpu_utilization": avg_gpu_util,
            "avg_gpu_memory_utilization": avg_gpu_memory,
            "total_gpus": len(all_gpu_metrics),
            "last_updated": datetime.now().isoformat()
        }

    def get_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        summary = self.get_cluster_summary()
        
        if not summary:
            return recommendations
        
        # GPU utilization recommendations
        if summary.get("avg_gpu_utilization", 0) < 70:
            recommendations.append(
                "GPU utilization is below 70%. Consider increasing batch size or "
                "reducing data loading bottlenecks."
            )
        
        # Memory utilization recommendations
        if summary.get("avg_gpu_memory_utilization", 0) < 50:
            recommendations.append(
                "GPU memory utilization is low. Consider increasing model size or "
                "batch size to better utilize available memory."
            )
        
        # CPU utilization recommendations
        if summary.get("avg_cpu_utilization", 0) > 90:
            recommendations.append(
                "CPU utilization is high. Consider optimizing data preprocessing "
                "or increasing the number of data loading workers."
            )
        
        return recommendations

    def start_all_monitoring(self, interval_seconds: float = 5.0) -> None:
        """Start monitoring on all registered nodes."""
        for monitor in self.node_monitors.values():
            monitor.start_monitoring(interval_seconds)
        self.logger.info("Started monitoring on all nodes")

    def stop_all_monitoring(self) -> None:
        """Stop monitoring on all registered nodes."""
        for monitor in self.node_monitors.values():
            monitor.stop_monitoring()
        self.logger.info("Stopped monitoring on all nodes")