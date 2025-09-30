"""
Base classes and interfaces for distributed training coordination.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .config import DistributedTrainingConfig, TrainingMetrics


@dataclass
class DistributedState:
    """State information for distributed training process."""
    rank: int
    local_rank: int
    world_size: int
    is_master: bool
    device: torch.device
    backend: str
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1


class DistributedTrainerInterface(ABC):
    """Abstract interface for distributed training coordination."""
    
    @abstractmethod
    def initialize_distributed(self, config: DistributedTrainingConfig) -> DistributedState:
        """
        Initialize distributed training environment.
        
        Args:
            config: Distributed training configuration
            
        Returns:
            state: Distributed training state information
        """
        pass
    
    @abstractmethod
    def setup_model(self, model: torch.nn.Module, state: DistributedState) -> torch.nn.Module:
        """
        Setup model for distributed training.
        
        Args:
            model: PyTorch model to distribute
            state: Distributed training state
            
        Returns:
            distributed_model: Model wrapped for distributed training
        """
        pass
    
    @abstractmethod
    def setup_dataloader(
        self, 
        dataset: torch.utils.data.Dataset, 
        config: DistributedTrainingConfig,
        state: DistributedState
    ) -> DataLoader:
        """
        Setup data loader for distributed training.
        
        Args:
            dataset: Training dataset
            config: Training configuration
            state: Distributed training state
            
        Returns:
            dataloader: Distributed data loader
        """
        pass
    
    @abstractmethod
    def train_step(
        self,
        model: torch.nn.Module,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        state: DistributedState
    ) -> Dict[str, float]:
        """
        Execute single training step with gradient synchronization.
        
        Args:
            model: Distributed model
            batch: Training batch
            optimizer: Optimizer
            criterion: Loss function
            state: Distributed training state
            
        Returns:
            metrics: Training step metrics
        """
        pass
    
    @abstractmethod
    def synchronize_metrics(self, metrics: Dict[str, float], state: DistributedState) -> Dict[str, float]:
        """
        Synchronize metrics across all nodes.
        
        Args:
            metrics: Local node metrics
            state: Distributed training state
            
        Returns:
            synchronized_metrics: Averaged metrics across all nodes
        """
        pass
    
    @abstractmethod
    def cleanup_distributed(self) -> None:
        """Clean up distributed training environment."""
        pass


class CheckpointManagerInterface(ABC):
    """Abstract interface for distributed checkpoint management."""
    
    @abstractmethod
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        state: DistributedState
    ) -> str:
        """
        Save distributed training checkpoint.
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
            state: Distributed training state
            
        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        pass
    
    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: DistributedState
    ) -> Tuple[int, int, Dict[str, float]]:
        """
        Load distributed training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            state: Distributed training state
            
        Returns:
            epoch, step, metrics: Restored training state
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self, experiment_id: str) -> List[str]:
        """
        List available checkpoints for experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            checkpoint_paths: List of available checkpoint paths
        """
        pass


class DistributedTrainer:
    """
    Main distributed training coordinator.
    
    This class orchestrates distributed training across multiple GPU nodes,
    handling model distribution, data sharding, gradient synchronization,
    and fault tolerance.
    """
    
    def __init__(
        self,
        trainer_impl: DistributedTrainerInterface,
        checkpoint_manager: CheckpointManagerInterface,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize distributed trainer.
        
        Args:
            trainer_impl: Distributed training implementation
            checkpoint_manager: Checkpoint management implementation
            logger: Optional logger instance
        """
        self.trainer_impl = trainer_impl
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger or logging.getLogger(__name__)
        self.state: Optional[DistributedState] = None
        self.training_metrics: List[TrainingMetrics] = []
    
    def setup(self, config: DistributedTrainingConfig) -> DistributedState:
        """
        Setup distributed training environment.
        
        Args:
            config: Distributed training configuration
            
        Returns:
            state: Distributed training state
        """
        self.logger.info("Setting up distributed training environment")
        self.state = self.trainer_impl.initialize_distributed(config)
        
        if self.state.is_master:
            self.logger.info(f"Initialized as master node (rank {self.state.rank})")
        else:
            self.logger.info(f"Initialized as worker node (rank {self.state.rank})")
        
        return self.state
    
    def prepare_model_and_data(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        config: DistributedTrainingConfig
    ) -> Tuple[torch.nn.Module, DataLoader]:
        """
        Prepare model and data loader for distributed training.
        
        Args:
            model: PyTorch model
            dataset: Training dataset
            config: Training configuration
            
        Returns:
            distributed_model, dataloader: Prepared model and data loader
        """
        if self.state is None:
            raise RuntimeError("Must call setup() before preparing model and data")
        
        # Setup distributed model
        distributed_model = self.trainer_impl.setup_model(model, self.state)
        
        # Setup distributed data loader
        dataloader = self.trainer_impl.setup_dataloader(dataset, config, self.state)
        
        self.logger.info(f"Prepared model and data loader for {self.state.world_size} nodes")
        return distributed_model, dataloader
    
    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epoch: int,
        config: DistributedTrainingConfig
    ) -> Dict[str, float]:
        """
        Train for one epoch with distributed coordination.
        
        Args:
            model: Distributed model
            dataloader: Distributed data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            config: Training configuration
            
        Returns:
            epoch_metrics: Aggregated metrics for the epoch
        """
        if self.state is None:
            raise RuntimeError("Must call setup() before training")
        
        model.train()
        epoch_metrics = {"loss": 0.0, "samples": 0}
        step_count = 0
        
        for step, batch in enumerate(dataloader):
            # Execute training step
            step_metrics = self.trainer_impl.train_step(
                model, batch, optimizer, criterion, self.state
            )
            
            # Accumulate metrics
            epoch_metrics["loss"] += step_metrics["loss"]
            epoch_metrics["samples"] += step_metrics.get("batch_size", config.batch_size_per_gpu)
            step_count += 1
            
            # Collect detailed metrics
            training_metric = TrainingMetrics(
                node_id=self.state.rank,
                epoch=epoch,
                step=step,
                loss=step_metrics["loss"],
                gpu_utilization=step_metrics.get("gpu_utilization", 0.0),
                gpu_memory_used=step_metrics.get("gpu_memory_used", 0.0),
                throughput_samples_per_sec=step_metrics.get("throughput", 0.0),
                communication_time_ms=step_metrics.get("communication_time", 0.0)
            )
            self.training_metrics.append(training_metric)
            
            # Save checkpoint periodically
            if step % config.checkpoint_frequency == 0 and self.state.is_master:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, step, step_metrics, self.state
                )
                self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Synchronize epoch metrics across all nodes
        if step_count > 0:
            epoch_metrics["loss"] /= step_count
        
        synchronized_metrics = self.trainer_impl.synchronize_metrics(epoch_metrics, self.state)
        
        if self.state.is_master:
            self.logger.info(f"Epoch {epoch} completed. Loss: {synchronized_metrics['loss']:.4f}")
        
        return synchronized_metrics
    
    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[int, int, Dict[str, float]]:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to restore
            optimizer: Optimizer to restore
            
        Returns:
            epoch, step, metrics: Restored training state
        """
        if self.state is None:
            raise RuntimeError("Must call setup() before resuming from checkpoint")
        
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        epoch, step, metrics = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, model, optimizer, self.state
        )
        
        self.logger.info(f"Resumed from epoch {epoch}, step {step}")
        return epoch, step, metrics
    
    def get_training_metrics(self) -> List[TrainingMetrics]:
        """Get collected training metrics."""
        return self.training_metrics.copy()
    
    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        self.logger.info("Cleaning up distributed training environment")
        self.trainer_impl.cleanup_distributed()
        self.state = None


class DistributedTrainingError(Exception):
    """Base exception for distributed training errors."""
    pass


class DistributedSetupError(DistributedTrainingError):
    """Exception raised during distributed training setup."""
    pass


class DistributedSynchronizationError(DistributedTrainingError):
    """Exception raised during gradient synchronization."""
    pass


class CheckpointError(DistributedTrainingError):
    """Exception raised during checkpoint operations."""
    pass