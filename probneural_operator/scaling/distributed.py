"""Concurrent processing and distributed computing utilities."""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import threading
import queue
import time
import os
import subprocess
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass
import logging
import socket
import psutil


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    timeout_minutes: int = 30


class MultiGPUTrainer:
    """Multi-GPU training with data parallelism and model parallelism support."""
    
    def __init__(self, 
                 model: nn.Module,
                 device_ids: Optional[List[int]] = None,
                 output_device: Optional[int] = None,
                 find_unused_parameters: bool = False):
        """Initialize multi-GPU trainer.
        
        Args:
            model: Model to train on multiple GPUs
            device_ids: List of GPU device IDs to use
            output_device: Primary GPU for outputs
            find_unused_parameters: Whether to find unused parameters in backward pass
        """
        self.model = model
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.output_device = output_device or self.device_ids[0]
        self.find_unused_parameters = find_unused_parameters
        
        if len(self.device_ids) == 0:
            raise ValueError("No GPU devices available")
        
        self.num_gpus = len(self.device_ids)
        self.is_distributed = False
        
        # Move model to primary device
        self.model = self.model.to(self.output_device)
        
        # Setup parallel training
        if self.num_gpus > 1:
            self._setup_data_parallel()
    
    def _setup_data_parallel(self):
        """Setup data parallel training."""
        if dist.is_available() and dist.is_initialized():
            # Use DistributedDataParallel
            self.model = DDP(
                self.model,
                device_ids=[self.output_device],
                output_device=self.output_device,
                find_unused_parameters=self.find_unused_parameters
            )
            self.is_distributed = True
        else:
            # Use DataParallel
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.device_ids,
                output_device=self.output_device
            )
    
    def train_step(self, 
                  dataloader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  criterion: nn.Module,
                  scaler: Optional[GradScaler] = None,
                  use_mixed_precision: bool = True) -> Dict[str, float]:
        """Perform training step across multiple GPUs.
        
        Args:
            dataloader: Training dataloader
            optimizer: Optimizer
            criterion: Loss function
            scaler: Gradient scaler for mixed precision
            use_mixed_precision: Whether to use automatic mixed precision
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        if scaler is None and use_mixed_precision:
            scaler = GradScaler()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to devices
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
                inputs = inputs.to(self.output_device, non_blocking=True)
                targets = targets.to(self.output_device, non_blocking=True)
            else:
                inputs = batch.to(self.output_device, non_blocking=True)
                targets = inputs  # Autoencoder case
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if use_mixed_precision and scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        end_time = time.time()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        training_time = end_time - start_time
        throughput = len(dataloader.dataset) / training_time if training_time > 0 else 0.0
        
        return {
            'avg_loss': avg_loss,
            'training_time': training_time,
            'throughput': throughput,
            'num_gpus_used': self.num_gpus,
            'batches_processed': num_batches
        }
    
    def get_memory_stats(self) -> Dict[int, Dict[str, float]]:
        """Get memory statistics for all GPUs."""
        stats = {}
        
        for device_id in self.device_ids:
            if torch.cuda.is_available():
                with torch.cuda.device(device_id):
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                    
                    stats[device_id] = {
                        'allocated_gb': allocated,
                        'reserved_gb': reserved,
                        'max_allocated_gb': max_allocated,
                        'free_gb': torch.cuda.get_device_properties(device_id).total_memory / 1024**3 - reserved
                    }
        
        return stats


class DistributedTrainer:
    """Distributed training across multiple nodes."""
    
    def __init__(self, config: DistributedConfig):
        """Initialize distributed trainer.
        
        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.is_initialized = False
        
    def setup(self):
        """Setup distributed training environment."""
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank,
            timeout=torch.distributed.distributed_c10d._timedelta_from_minutes(
                self.config.timeout_minutes
            )
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
        
        self.is_initialized = True
        
        # Log setup info
        logging.info(f"Distributed training setup complete. "
                    f"Rank: {self.config.rank}/{self.config.world_size}, "
                    f"Local rank: {self.config.local_rank}")
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for distributed training."""
        if not self.is_initialized:
            self.setup()
        
        device = torch.device(f"cuda:{self.config.local_rank}" 
                             if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Wrap in DistributedDataParallel
        model = DDP(model, device_ids=[self.config.local_rank])
        
        return model
    
    def prepare_dataloader(self, 
                          dataset,
                          batch_size: int,
                          **kwargs) -> DataLoader:
        """Prepare dataloader for distributed training."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=kwargs.get('shuffle', True)
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=kwargs.get('pin_memory', True),
            num_workers=kwargs.get('num_workers', 4),
            **{k: v for k, v in kwargs.items() 
               if k not in ['shuffle', 'sampler', 'pin_memory', 'num_workers']}
        )
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Average metrics across all processes."""
        if not self.is_initialized:
            return metrics
        
        averaged_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=f"cuda:{self.config.local_rank}"
                                 if torch.cuda.is_available() else "cpu")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            averaged_metrics[key] = tensor.item() / self.config.world_size
        
        return averaged_metrics


class ResourcePoolManager:
    """Manage pools of computational resources (GPUs, CPU threads, memory)."""
    
    def __init__(self):
        """Initialize resource pool manager."""
        self.gpu_pool = self._discover_gpus()
        self.cpu_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.memory_monitor = self._create_memory_monitor()
        
        self._resource_usage = {
            'gpu_utilization': {},
            'cpu_utilization': 0.0,
            'memory_usage_gb': 0.0
        }
        
        self._lock = threading.Lock()
    
    def _discover_gpus(self) -> List[Dict[str, Any]]:
        """Discover available GPUs."""
        gpus = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    'id': i,
                    'name': props.name,
                    'total_memory_gb': props.total_memory / 1024**3,
                    'multiprocessor_count': props.multi_processor_count,
                    'major': props.major,
                    'minor': props.minor,
                    'available': True
                })
        
        return gpus
    
    def _create_memory_monitor(self) -> threading.Thread:
        """Create background memory monitoring thread."""
        def monitor_memory():
            while True:
                try:
                    # CPU memory
                    memory_info = psutil.virtual_memory()
                    self._resource_usage['memory_usage_gb'] = memory_info.used / 1024**3
                    
                    # CPU utilization
                    self._resource_usage['cpu_utilization'] = psutil.cpu_percent(interval=1)
                    
                    # GPU utilization
                    for gpu in self.gpu_pool:
                        gpu_id = gpu['id']
                        if torch.cuda.is_available():
                            with torch.cuda.device(gpu_id):
                                allocated = torch.cuda.memory_allocated() / 1024**3
                                total = gpu['total_memory_gb']
                                utilization = allocated / total if total > 0 else 0.0
                                self._resource_usage['gpu_utilization'][gpu_id] = utilization
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    logging.warning(f"Memory monitoring error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=monitor_memory, daemon=True)
        thread.start()
        return thread
    
    def allocate_gpu(self, memory_gb: float = 1.0) -> Optional[int]:
        """Allocate GPU with specified memory requirement."""
        with self._lock:
            for gpu in self.gpu_pool:
                if gpu['available'] and gpu['total_memory_gb'] >= memory_gb:
                    current_usage = self._resource_usage['gpu_utilization'].get(gpu['id'], 0.0)
                    if current_usage < 0.8:  # Less than 80% utilized
                        gpu['available'] = False
                        return gpu['id']
        
        return None
    
    def release_gpu(self, gpu_id: int):
        """Release GPU back to pool."""
        with self._lock:
            for gpu in self.gpu_pool:
                if gpu['id'] == gpu_id:
                    gpu['available'] = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
    
    def get_optimal_device_placement(self, 
                                   models: List[nn.Module]) -> Dict[int, List[int]]:
        """Get optimal device placement for multiple models."""
        placement = {}
        
        # Estimate model memory requirements
        model_sizes = []
        for i, model in enumerate(models):
            size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
            model_sizes.append((i, size_gb))
        
        # Sort models by size (largest first)
        model_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Assign models to GPUs
        gpu_loads = {gpu['id']: 0.0 for gpu in self.gpu_pool}
        
        for model_idx, model_size in model_sizes:
            # Find GPU with least load
            best_gpu = min(gpu_loads.items(), key=lambda x: x[1])
            gpu_id, current_load = best_gpu
            
            # Check if GPU has enough memory
            gpu_capacity = next(gpu['total_memory_gb'] for gpu in self.gpu_pool 
                              if gpu['id'] == gpu_id)
            
            if current_load + model_size <= gpu_capacity * 0.9:  # Leave 10% buffer
                if gpu_id not in placement:
                    placement[gpu_id] = []
                placement[gpu_id].append(model_idx)
                gpu_loads[gpu_id] += model_size
        
        return placement
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource utilization statistics."""
        with self._lock:
            return {
                'available_gpus': sum(1 for gpu in self.gpu_pool if gpu['available']),
                'total_gpus': len(self.gpu_pool),
                'gpu_utilization': self._resource_usage['gpu_utilization'].copy(),
                'cpu_utilization_percent': self._resource_usage['cpu_utilization'],
                'memory_usage_gb': self._resource_usage['memory_usage_gb'],
                'total_memory_gb': psutil.virtual_memory().total / 1024**3
            }


class AsyncDataLoader:
    """Asynchronous data loader for improved I/O performance."""
    
    def __init__(self, 
                 dataset,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 buffer_size: int = 10,
                 prefetch_factor: int = 2):
        """Initialize async data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            num_workers: Number of worker processes
            buffer_size: Size of internal buffer
            prefetch_factor: Number of batches to prefetch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        
        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )
        
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._producer_thread = None
        self._stop_event = threading.Event()
    
    def _producer(self):
        """Background producer thread."""
        try:
            for batch in self._dataloader:
                if self._stop_event.is_set():
                    break
                
                self._buffer.put(batch, timeout=10)
                
        except Exception as e:
            logging.error(f"AsyncDataLoader producer error: {e}")
            self._buffer.put(e)  # Signal error to consumer
    
    def __iter__(self):
        """Start async iteration."""
        self._stop_event.clear()
        self._producer_thread = threading.Thread(target=self._producer, daemon=True)
        self._producer_thread.start()
        
        return self
    
    def __next__(self):
        """Get next batch asynchronously."""
        try:
            batch = self._buffer.get(timeout=30)
            
            if isinstance(batch, Exception):
                raise batch
            
            return batch
            
        except queue.Empty:
            # Check if producer is still running
            if self._producer_thread and self._producer_thread.is_alive():
                raise StopIteration("DataLoader timeout")
            else:
                raise StopIteration()
    
    def __len__(self):
        """Return number of batches."""
        return len(self._dataloader)
    
    def stop(self):
        """Stop async data loading."""
        self._stop_event.set()
        
        if self._producer_thread:
            self._producer_thread.join(timeout=5)


def launch_distributed_training(train_fn: Callable,
                               world_size: int,
                               master_addr: str = "localhost",
                               master_port: str = "12355",
                               backend: str = "nccl",
                               **kwargs):
    """Launch distributed training across multiple processes.
    
    Args:
        train_fn: Training function to execute
        world_size: Number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend
        **kwargs: Additional arguments for training function
    """
    def worker(rank: int):
        """Worker process for distributed training."""
        config = DistributedConfig(
            backend=backend,
            world_size=world_size,
            rank=rank,
            local_rank=rank % torch.cuda.device_count(),
            master_addr=master_addr,
            master_port=master_port
        )
        
        trainer = DistributedTrainer(config)
        
        try:
            trainer.setup()
            train_fn(trainer, **kwargs)
        finally:
            trainer.cleanup()
    
    # Launch worker processes
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank,))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()


async def async_posterior_sampling(model: nn.Module,
                                 input_data: torch.Tensor,
                                 num_samples: int = 100,
                                 batch_size: int = 10) -> torch.Tensor:
    """Asynchronous posterior sampling for uncertainty quantification.
    
    Args:
        model: Probabilistic neural operator model
        input_data: Input tensor
        num_samples: Number of posterior samples
        batch_size: Batch size for sampling
        
    Returns:
        Tensor of posterior samples
    """
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)
    
    def sample_batch(start_idx: int, end_idx: int) -> torch.Tensor:
        """Sample a batch of predictions."""
        model.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(start_idx, end_idx):
                if hasattr(model, 'sample'):
                    sample = model.sample(input_data)
                else:
                    # Fallback: add noise to prediction
                    pred = model(input_data)
                    noise_std = 0.1 * torch.std(pred)
                    sample = pred + torch.randn_like(pred) * noise_std
                
                samples.append(sample.unsqueeze(0))
        
        return torch.cat(samples, dim=0)
    
    # Create sampling tasks
    tasks = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        task = loop.run_in_executor(executor, sample_batch, i, end_idx)
        tasks.append(task)
    
    # Wait for all sampling tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Concatenate all samples
    all_samples = torch.cat(results, dim=0)
    
    return all_samples