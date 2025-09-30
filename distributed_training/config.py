"""
Configuration data models and validation for distributed training system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from enum import Enum
import json
from pathlib import Path


class GPUType(Enum):
    """Supported GPU types for Azure ML compute clusters."""
    STANDARD_NC6S_V3 = "Standard_NC6s_v3"  # Tesla V100 16GB
    STANDARD_NC12S_V3 = "Standard_NC12s_v3"  # 2x Tesla V100 16GB
    STANDARD_NC24S_V3 = "Standard_NC24s_v3"  # 4x Tesla V100 16GB
    STANDARD_ND40RS_V2 = "Standard_ND40rs_v2"  # 8x Tesla V100 32GB


class DistributedBackend(Enum):
    """Supported distributed training backends."""
    NCCL = "nccl"  # NVIDIA Collective Communications Library for GPU
    GLOO = "gloo"  # Facebook's collective communications library
    MPI = "mpi"   # Message Passing Interface


@dataclass
class ClusterConfig:
    """Configuration for GPU compute cluster management."""
    cluster_name: str
    node_count: int
    gpu_type: GPUType
    auto_scale_enabled: bool = True
    min_nodes: int = 1
    max_nodes: int = 10
    idle_timeout_minutes: int = 10
    location: str = "eastus"

    def __post_init__(self):
        """Validate cluster configuration."""
        if self.node_count < 1:
            raise ValueError("node_count must be at least 1")
        if self.min_nodes < 1:
            raise ValueError("min_nodes must be at least 1")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be >= min_nodes")
        if self.node_count > self.max_nodes:
            raise ValueError("node_count cannot exceed max_nodes")
        if self.idle_timeout_minutes < 1:
            raise ValueError("idle_timeout_minutes must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_name": self.cluster_name,
            "node_count": self.node_count,
            "gpu_type": self.gpu_type.value,
            "auto_scale_enabled": self.auto_scale_enabled,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "idle_timeout_minutes": self.idle_timeout_minutes,
            "location": self.location
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterConfig':
        """Create from dictionary."""
        data = data.copy()
        data["gpu_type"] = GPUType(data["gpu_type"])
        return cls(**data)


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training execution."""
    world_size: int
    backend: DistributedBackend = DistributedBackend.NCCL
    master_addr: str = "localhost"
    master_port: int = 29500
    batch_size_per_gpu: int = 32
    gradient_accumulation_steps: int = 1
    checkpoint_frequency: int = 100
    max_epochs: int = 10
    learning_rate: float = 0.001

    def __post_init__(self):
        """Validate distributed training configuration."""
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        if self.master_port < 1024 or self.master_port > 65535:
            raise ValueError("master_port must be between 1024 and 65535")
        if self.batch_size_per_gpu < 1:
            raise ValueError("batch_size_per_gpu must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.checkpoint_frequency < 1:
            raise ValueError("checkpoint_frequency must be at least 1")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size across all GPUs."""
        return (self.batch_size_per_gpu * self.world_size *
                self.gradient_accumulation_steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "world_size": self.world_size,
            "backend": self.backend.value,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "batch_size_per_gpu": self.batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "checkpoint_frequency": self.checkpoint_frequency,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTrainingConfig':
        """Create from dictionary."""
        data = data.copy()
        data["backend"] = DistributedBackend(data["backend"])
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Metrics collected during distributed training."""
    node_id: int
    epoch: int
    step: int
    loss: float
    gpu_utilization: float
    gpu_memory_used: float
    throughput_samples_per_sec: float
    communication_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate training metrics."""
        if self.node_id < 0:
            raise ValueError("node_id must be non-negative")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.gpu_utilization < 0 or self.gpu_utilization > 100:
            raise ValueError("gpu_utilization must be between 0 and 100")
        if self.gpu_memory_used < 0:
            raise ValueError("gpu_memory_used must be non-negative")
        if self.throughput_samples_per_sec < 0:
            raise ValueError("throughput_samples_per_sec must be non-negative")
        if self.communication_time_ms < 0:
            raise ValueError("communication_time_ms must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used": self.gpu_memory_used,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "communication_time_ms": self.communication_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CostMetrics:
    """Cost tracking metrics for distributed training."""
    cluster_name: str
    total_cost_usd: float
    cost_per_node_hour: float
    training_duration_hours: float
    cost_per_epoch: float
    estimated_total_cost: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate cost metrics."""
        if self.total_cost_usd < 0:
            raise ValueError("total_cost_usd must be non-negative")
        if self.cost_per_node_hour < 0:
            raise ValueError("cost_per_node_hour must be non-negative")
        if self.training_duration_hours < 0:
            raise ValueError("training_duration_hours must be non-negative")
        if self.cost_per_epoch < 0:
            raise ValueError("cost_per_epoch must be non-negative")
        if self.estimated_total_cost < 0:
            raise ValueError("estimated_total_cost must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_name": self.cluster_name,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_node_hour": self.cost_per_node_hour,
            "training_duration_hours": self.training_duration_hours,
            "cost_per_epoch": self.cost_per_epoch,
            "estimated_total_cost": self.estimated_total_cost,
            "timestamp": self.timestamp.isoformat()
        }


class ConfigManager:
    """Utility class for managing configuration files."""

    @staticmethod
    def save_config(config: Any, filepath: Path) -> None:
        """Save configuration to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    @staticmethod
    def load_cluster_config(filepath: Path) -> ClusterConfig:
        """Load cluster configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ClusterConfig.from_dict(data)

    @staticmethod
    def load_training_config(filepath: Path) -> DistributedTrainingConfig:
        """Load training configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return DistributedTrainingConfig.from_dict(data)