"""
GPU cluster management interfaces and base classes for Azure ML compute clusters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

from .config import ClusterConfig, CostMetrics


class ClusterStatus(Enum):
    """Cluster status enumeration."""
    CREATING = "creating"
    RUNNING = "running"
    SCALING = "scaling"
    IDLE = "idle"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ClusterHealth:
    """Cluster health status information."""
    cluster_name: str
    status: ClusterStatus
    active_nodes: int
    total_nodes: int
    failed_nodes: int
    gpu_utilization_avg: float
    last_activity: datetime
    error_message: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        """Check if cluster is in healthy state."""
        return (
            self.status in [ClusterStatus.RUNNING, ClusterStatus.IDLE] and
            self.failed_nodes == 0 and
            self.error_message is None
        )


@dataclass
class NodeInfo:
    """Information about individual cluster nodes."""
    node_id: str
    node_rank: int
    gpu_count: int
    gpu_type: str
    status: str
    gpu_utilization: float
    memory_utilization: float
    last_heartbeat: datetime


class ClusterManagerInterface(ABC):
    """Abstract interface for GPU cluster management."""

    @abstractmethod
    def provision_cluster(self, config: ClusterConfig) -> str:
        """
        Provision a new GPU cluster.

        Args:
            config: Cluster configuration

        Returns:
            cluster_id: Unique identifier for the provisioned cluster

        Raises:
            ClusterProvisioningError: If cluster provisioning fails
        """
        pass

    @abstractmethod
    def scale_cluster(self, cluster_id: str, target_nodes: int) -> bool:
        """
        Scale cluster to target number of nodes.

        Args:
            cluster_id: Cluster identifier
            target_nodes: Target number of nodes

        Returns:
            success: True if scaling initiated successfully
        """
        pass

    @abstractmethod
    def get_cluster_health(self, cluster_id: str) -> ClusterHealth:
        """
        Get current cluster health status.

        Args:
            cluster_id: Cluster identifier

        Returns:
            health: Current cluster health information
        """
        pass

    @abstractmethod
    def list_nodes(self, cluster_id: str) -> List[NodeInfo]:
        """
        List all nodes in the cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            nodes: List of node information
        """
        pass

    @abstractmethod
    def cleanup_cluster(self, cluster_id: str) -> bool:
        """
        Clean up and delete cluster resources.

        Args:
            cluster_id: Cluster identifier

        Returns:
            success: True if cleanup initiated successfully
        """
        pass


class ResourceOptimizerInterface(ABC):
    """Abstract interface for resource optimization."""

    @abstractmethod
    def analyze_utilization(self, cluster_id: str) -> Dict[str, Any]:
        """
        Analyze cluster resource utilization.

        Args:
            cluster_id: Cluster identifier

        Returns:
            analysis: Resource utilization analysis
        """
        pass

    @abstractmethod
    def recommend_optimizations(self, cluster_id: str) -> List[str]:
        """
        Generate optimization recommendations.

        Args:
            cluster_id: Cluster identifier

        Returns:
            recommendations: List of optimization suggestions
        """
        pass

    @abstractmethod
    def estimate_cost(self, config: ClusterConfig,
                      duration_hours: float) -> float:
        """
        Estimate training cost for given configuration.

        Args:
            config: Cluster configuration
            duration_hours: Expected training duration

        Returns:
            estimated_cost: Estimated cost in USD
        """
        pass


class CostTrackerInterface(ABC):
    """Abstract interface for cost tracking."""

    @abstractmethod
    def start_tracking(self, cluster_id: str) -> None:
        """
        Start cost tracking for a cluster.

        Args:
            cluster_id: Cluster identifier
        """
        pass

    @abstractmethod
    def stop_tracking(self, cluster_id: str) -> CostMetrics:
        """
        Stop cost tracking and return final metrics.

        Args:
            cluster_id: Cluster identifier

        Returns:
            cost_metrics: Final cost metrics
        """r.info(f
        pass

    @abstractmethod
    def get_current_cost(self, cluster_id: str) -> float:
        """
        Get current accumulated cost.

        Args:
            cluster_id: Cluster identifier

        Returns:
            current_cost: Current cost in USD
        """
        pass


class GPUClusterManager:
    """
    Main GPU cluster manager that coordinates cluster lifecycle.

    This class serves as the primary interface for managing GPU clusters,
    integrating cluster provisioning, resource optimization, and cost tracking.
    """

    def __init__(
        self,
        cluster_manager: ClusterManagerInterface,
        resource_optimizer: ResourceOptimizerInterface,
        cost_tracker: CostTrackerInterface,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize GPU cluster manager.

        Args:
            cluster_manager: Cluster management implementation
            resource_optimizer: Resource optimization implementation
            cost_tracker: Cost tracking implementation
            logger: Optional logger instance
        """
        self.cluster_manager = cluster_manager
        self.resource_optimizer = resource_optimizer
        self.cost_tracker = cost_tracker
        self.logger = logger or logging.getLogger(__name__)
        self._active_clusters: Dict[str, ClusterConfig] = {}

    def create_cluster(self, config: ClusterConfig) -> str:
        """
        Create and provision a new GPU cluster.

        Args:
            config: Cluster configuration

        Returns:
            cluster_id: Unique identifier for the created cluster
        """
        self.logger.info(f"Creating cluster: {config.cluster_name}")

        # Estimate cost before provisioning
        estimated_cost = self.resource_optimizer.estimate_cost(config, 1.0)
        self.logger.info(f"Estimated cost per hour: ${estimated_cost:.2f}")

        # Provision cluster
        cluster_id = self.cluster_manager.provision_cluster(config)
        self._active_clusters[cluster_id] = config

        # Start cost tracking
        self.cost_tracker.start_tracking(cluster_id)

        self.logger.info(f"Cluster created successfully: {cluster_id}")
        return cluster_id

    def monitor_cluster(self, cluster_id: str) -> ClusterHealth:
        """
        Monitor cluster health and performance.

        Args:
            cluster_id: Cluster identifier

        Returns:
            health: Current cluster health status
        """
        health = self.cluster_manager.get_cluster_health(cluster_id)

        if not health.is_healthy:
            self.logger.warning(
                f"Cluster {cluster_id} is unhealthy: {health.error_message}")

        # Check for optimization opportunities
        if health.gpu_utilization_avg < 70:
            recommendations = self.resource_optimizer.recommend_optimizations(
                cluster_id)
            if recommendations:
                self.logger.info(
                    f"Optimization recommendations for {cluster_id}: "
                    f"{recommendations}")

        return health

    def scale_cluster(self, cluster_id: str, target_nodes: int) -> bool:
        """
        Scale cluster to target number of nodes.

        Args:
            cluster_id: Cluster identifier
            target_nodes: Target number of nodes

        Returns:
            success: True if scaling initiated successfully
        """
        if cluster_id not in self._active_clusters:
            raise ValueError(f"Cluster {cluster_id} not found in active clusters")

        config = self._active_clusters[cluster_id]
        if target_nodes < config.min_nodes or target_nodes > config.max_nodes:
            raise ValueError(
                f"Target nodes {target_nodes} outside allowed range "
                f"[{config.min_nodes}, {config.max_nodes}]")

        self.logger.info(f"Scaling cluster {cluster_id} to {target_nodes} nodes")
        return self.cluster_manager.scale_cluster(cluster_id, target_nodes)

    def cleanup_cluster(self, cluster_id: str) -> CostMetrics:
        """
        Clean up cluster and return final cost metrics.

        Args:
            cluster_id: Cluster identifier

        Returns:
            cost_metrics: Final cost metrics
        """
        self.logger.info(f"Cleaning up cluster: {cluster_id}")

        # Stop cost tracking and get final metrics
        cost_metrics = self.cost_tracker.stop_tracking(cluster_id)

        # Clean up cluster resources
        success = self.cluster_manager.cleanup_cluster(cluster_id)
        if not success:
            self.logger.error(f"Failed to cleanup cluster {cluster_id}")

        # Remove from active clusters
        if cluster_id in self._active_clusters:
            del self._active_clusters[cluster_id]

        self.logger.info(
            f"Cluster cleanup completed. Total cost: "
            f"${cost_metrics.total_cost_usd:.2f}")
        return cost_metrics

    def get_active_clusters(self) -> Dict[str, ClusterConfig]:
        """Get all active cluster configurations."""
        return self._active_clusters.copy()


class ClusterProvisioningError(Exception):
    """Exception raised when cluster provisioning fails."""
    pass


class ClusterScalingError(Exception):
    """Exception raised when cluster scaling fails."""
    pass


class ClusterCleanupError(Exception):
    """Exception raised when cluster cleanup fails."""
    pass