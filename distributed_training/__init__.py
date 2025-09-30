"""
Distributed GPU Training System for Stock Price Prediction

This package provides enterprise-grade distributed GPU training capabilities
using Azure ML and PyTorch DistributedDataParallel (DDP).
"""

__version__ = "1.0.0"

from .config import (
    ClusterConfig,
    DistributedTrainingConfig,
    TrainingMetrics,
    CostMetrics
)
from .cluster_manager import GPUClusterManager
from .distributed_trainer import DistributedTrainer
from .monitoring import NodeMonitor, PerformanceDashboard

__all__ = [
    "ClusterConfig",
    "DistributedTrainingConfig",
    "TrainingMetrics",
    "CostMetrics",
    "GPUClusterManager",
    "DistributedTrainer",
    "NodeMonitor",
    "PerformanceDashboard"
]