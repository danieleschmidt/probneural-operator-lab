"""High-Performance Computing integration with SLURM and MPI support."""

import os
import subprocess
import time
import json
import logging
import threading
import pickle
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import socket
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import contextmanager
import yaml


@dataclass
class SLURMJobConfig:
    """Configuration for SLURM job submission."""
    job_name: str = "probneural_training"
    partition: str = "gpu"
    account: Optional[str] = None
    time_limit: str = "24:00:00"
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 8
    mem_per_cpu: str = "4G"
    gpus_per_node: int = 1
    gpu_type: Optional[str] = None
    
    # Advanced options
    exclusive: bool = False
    constraint: Optional[str] = None
    array_jobs: Optional[str] = None
    dependency: Optional[str] = None
    mail_type: Optional[str] = None
    mail_user: Optional[str] = None
    
    # Custom SLURM directives
    custom_directives: Dict[str, str] = field(default_factory=dict)
    
    # Environment setup
    modules_to_load: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    setup_commands: List[str] = field(default_factory=list)


@dataclass
class MPIConfig:
    """Configuration for MPI distributed training."""
    mpi_executable: str = "mpirun"
    num_processes: int = 1
    hosts: Optional[List[str]] = None
    hostfile: Optional[str] = None
    
    # MPI-specific options
    map_by: str = "slot"
    bind_to: str = "none"
    report_bindings: bool = False
    display_map: bool = False
    
    # OpenMPI specific
    mca_params: Dict[str, str] = field(default_factory=dict)
    
    # Custom MPI options
    custom_options: List[str] = field(default_factory=list)


class SLURMIntegration:
    """SLURM cluster integration for job submission and management."""
    
    def __init__(self, default_config: Optional[SLURMJobConfig] = None):
        """Initialize SLURM integration.
        
        Args:
            default_config: Default SLURM job configuration
        """
        self.default_config = default_config or SLURMJobConfig()
        self._job_history = []
        self._active_jobs = {}
        
        # Check SLURM availability
        self.slurm_available = self._check_slurm_availability()
        
        if not self.slurm_available:
            logging.warning("SLURM not available on this system")
    
    def _check_slurm_availability(self) -> bool:
        """Check if SLURM is available."""
        try:
            result = subprocess.run(['sinfo', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate_slurm_script(self, 
                            config: SLURMJobConfig,
                            python_script: str,
                            script_args: Optional[List[str]] = None) -> str:
        """Generate SLURM job script.
        
        Args:
            config: SLURM configuration
            python_script: Path to Python script to execute
            script_args: Arguments to pass to script
            
        Returns:
            SLURM script content
        """
        script_lines = ["#!/bin/bash"]
        
        # SLURM directives
        script_lines.extend([
            f"#SBATCH --job-name={config.job_name}",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --time={config.time_limit}",
            f"#SBATCH --nodes={config.nodes}",
            f"#SBATCH --ntasks-per-node={config.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem-per-cpu={config.mem_per_cpu}",
        ])
        
        if config.account:
            script_lines.append(f"#SBATCH --account={config.account}")
        
        if config.gpus_per_node > 0:
            if config.gpu_type:
                script_lines.append(f"#SBATCH --gpus-per-node={config.gpu_type}:{config.gpus_per_node}")
            else:
                script_lines.append(f"#SBATCH --gpus-per-node={config.gpus_per_node}")
        
        if config.exclusive:
            script_lines.append("#SBATCH --exclusive")
        
        if config.constraint:
            script_lines.append(f"#SBATCH --constraint={config.constraint}")
        
        if config.array_jobs:
            script_lines.append(f"#SBATCH --array={config.array_jobs}")
        
        if config.dependency:
            script_lines.append(f"#SBATCH --dependency={config.dependency}")
        
        if config.mail_type and config.mail_user:
            script_lines.extend([
                f"#SBATCH --mail-type={config.mail_type}",
                f"#SBATCH --mail-user={config.mail_user}"
            ])
        
        # Custom directives
        for key, value in config.custom_directives.items():
            script_lines.append(f"#SBATCH --{key}={value}")
        
        # Output/error files
        script_lines.extend([
            f"#SBATCH --output={config.job_name}_%j.out",
            f"#SBATCH --error={config.job_name}_%j.err"
        ])
        
        script_lines.append("")  # Empty line
        
        # Environment setup
        script_lines.extend([
            "# Environment setup",
            "set -e",  # Exit on error
            ""
        ])
        
        # Load modules
        if config.modules_to_load:
            for module in config.modules_to_load:
                script_lines.append(f"module load {module}")
            script_lines.append("")
        
        # Activate conda environment
        if config.conda_env:
            script_lines.extend([
                f"source activate {config.conda_env}",
                ""
            ])
        
        # Custom setup commands
        if config.setup_commands:
            script_lines.extend(config.setup_commands)
            script_lines.append("")
        
        # Set distributed training environment variables
        if config.nodes > 1 or config.ntasks_per_node > 1:
            script_lines.extend([
                "# Distributed training setup",
                "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)",
                "export MASTER_PORT=12355",
                "export WORLD_SIZE=$SLURM_NTASKS",
                "export RANK=$SLURM_PROCID",
                "export LOCAL_RANK=$SLURM_LOCALID",
                ""
            ])
        
        # Main execution
        script_lines.extend([
            "# Job execution",
            "echo \"Job started at $(date)\"",
            "echo \"Running on nodes: $SLURM_JOB_NODELIST\"",
            "echo \"Number of nodes: $SLURM_JOB_NUM_NODES\"",
            "echo \"Total tasks: $SLURM_NTASKS\"",
            ""
        ])
        
        # Python execution command
        python_cmd = f"python {python_script}"
        if script_args:
            python_cmd += " " + " ".join(script_args)
        
        if config.nodes > 1 or config.ntasks_per_node > 1:
            # Use srun for distributed execution
            script_lines.append(f"srun {python_cmd}")
        else:
            script_lines.append(python_cmd)
        
        script_lines.extend([
            "",
            "echo \"Job completed at $(date)\""
        ])
        
        return "\n".join(script_lines)
    
    def submit_job(self,
                  config: SLURMJobConfig,
                  python_script: str,
                  script_args: Optional[List[str]] = None,
                  script_output_dir: Optional[str] = None) -> Optional[str]:
        """Submit job to SLURM.
        
        Args:
            config: SLURM configuration
            python_script: Python script to execute
            script_args: Script arguments
            script_output_dir: Directory to save script and outputs
            
        Returns:
            Job ID if successful, None otherwise
        """
        if not self.slurm_available:
            logging.error("SLURM not available")
            return None
        
        # Create output directory
        if script_output_dir is None:
            script_output_dir = tempfile.mkdtemp(prefix="slurm_job_")
        else:
            os.makedirs(script_output_dir, exist_ok=True)
        
        # Generate script
        script_content = self.generate_slurm_script(config, python_script, script_args)
        script_path = os.path.join(script_output_dir, f"{config.job_name}.sbatch")
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logging.info(f"Generated SLURM script: {script_path}")
        
        # Submit job
        try:
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Extract job ID
                output_lines = result.stdout.strip().split('\n')
                job_id = None
                for line in output_lines:
                    if 'Submitted batch job' in line:
                        job_id = line.split()[-1]
                        break
                
                if job_id:
                    job_info = {
                        'job_id': job_id,
                        'config': config,
                        'script_path': script_path,
                        'python_script': python_script,
                        'script_args': script_args,
                        'submit_time': time.time(),
                        'output_dir': script_output_dir
                    }
                    
                    self._job_history.append(job_info)
                    self._active_jobs[job_id] = job_info
                    
                    logging.info(f"Submitted job {job_id}")
                    return job_id
                else:
                    logging.error(f"Could not extract job ID from: {result.stdout}")
            else:
                logging.error(f"Job submission failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            logging.error("Job submission timed out")
        except Exception as e:
            logging.error(f"Job submission error: {e}")
        
        return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status from SLURM.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status information
        """
        if not self.slurm_available:
            return None
        
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '--format=%i,%T,%r,%S,%E', '--noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse output
                fields = result.stdout.strip().split(',')
                if len(fields) >= 3:
                    return {
                        'job_id': fields[0],
                        'state': fields[1],
                        'reason': fields[2] if len(fields) > 2 else '',
                        'start_time': fields[3] if len(fields) > 3 else '',
                        'end_time': fields[4] if len(fields) > 4 else ''
                    }
            else:
                # Job might be completed, check sacct
                result = subprocess.run(
                    ['sacct', '-j', job_id, '--format=JobID,State,ExitCode,Start,End', '--noheader'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if not line.strip() or '.batch' in line or '.extern' in line:
                            continue
                        
                        fields = line.strip().split()
                        if len(fields) >= 3:
                            return {
                                'job_id': fields[0],
                                'state': fields[1],
                                'exit_code': fields[2] if len(fields) > 2 else '',
                                'start_time': fields[3] if len(fields) > 3 else '',
                                'end_time': fields[4] if len(fields) > 4 else ''
                            }
        
        except subprocess.TimeoutExpired:
            logging.warning(f"Timeout getting status for job {job_id}")
        except Exception as e:
            logging.error(f"Error getting job status: {e}")
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel SLURM job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if not self.slurm_available:
            return False
        
        try:
            result = subprocess.run(
                ['scancel', job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            success = result.returncode == 0
            if success:
                logging.info(f"Cancelled job {job_id}")
                if job_id in self._active_jobs:
                    del self._active_jobs[job_id]
            else:
                logging.error(f"Failed to cancel job {job_id}: {result.stderr}")
            
            return success
        
        except Exception as e:
            logging.error(f"Error cancelling job: {e}")
            return False
    
    def wait_for_job(self, 
                    job_id: str, 
                    timeout: Optional[float] = None,
                    check_interval: float = 30.0) -> Optional[Dict[str, Any]]:
        """Wait for job completion.
        
        Args:
            job_id: Job ID to wait for
            timeout: Timeout in seconds
            check_interval: Status check interval
            
        Returns:
            Final job status
        """
        start_time = time.time()
        
        while True:
            status = self.get_job_status(job_id)
            
            if status is None:
                logging.warning(f"Could not get status for job {job_id}")
                return None
            
            state = status.get('state', '').upper()
            
            # Check if job is finished
            if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']:
                if job_id in self._active_jobs:
                    del self._active_jobs[job_id]
                return status
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logging.warning(f"Timeout waiting for job {job_id}")
                return status
            
            # Wait before next check
            time.sleep(check_interval)
    
    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """Get cluster information.
        
        Returns:
            Cluster information
        """
        if not self.slurm_available:
            return None
        
        cluster_info = {}
        
        try:
            # Node information
            result = subprocess.run(
                ['sinfo', '--format=%N,%c,%m,%G,%T', '--noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                nodes = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        fields = line.split(',')
                        if len(fields) >= 5:
                            nodes.append({
                                'name': fields[0],
                                'cpus': fields[1],
                                'memory': fields[2],
                                'gpus': fields[3],
                                'state': fields[4]
                            })
                
                cluster_info['nodes'] = nodes
            
            # Queue information
            result = subprocess.run(
                ['squeue', '--format=%i,%u,%T,%r,%S', '--noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                jobs = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        fields = line.split(',')
                        if len(fields) >= 4:
                            jobs.append({
                                'job_id': fields[0],
                                'user': fields[1],
                                'state': fields[2],
                                'reason': fields[3],
                                'start_time': fields[4] if len(fields) > 4 else ''
                            })
                
                cluster_info['queue'] = jobs
        
        except Exception as e:
            logging.error(f"Error getting cluster info: {e}")
        
        return cluster_info
    
    def get_job_history(self) -> List[Dict[str, Any]]:
        """Get job submission history."""
        return self._job_history.copy()
    
    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get active jobs."""
        return self._active_jobs.copy()


class MPIDistributedTrainer:
    """MPI-based distributed training for neural operators."""
    
    def __init__(self, config: MPIConfig):
        """Initialize MPI distributed trainer.
        
        Args:
            config: MPI configuration
        """
        self.config = config
        self.is_initialized = False
        self.rank = 0
        self.world_size = 1
        
        # Check MPI availability
        self.mpi_available = self._check_mpi_availability()
        
        if not self.mpi_available:
            logging.warning("MPI not available on this system")
    
    def _check_mpi_availability(self) -> bool:
        """Check if MPI is available."""
        try:
            result = subprocess.run([self.config.mpi_executable, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def initialize_mpi(self):
        """Initialize MPI environment."""
        if not self.mpi_available:
            return False
        
        try:
            # Initialize PyTorch distributed with MPI
            if 'OMPI_COMM_WORLD_RANK' in os.environ:
                # Open MPI
                self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
                self.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
                local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
            elif 'PMI_RANK' in os.environ:
                # Intel MPI
                self.rank = int(os.environ['PMI_RANK'])
                self.world_size = int(os.environ['PMI_SIZE'])
                local_rank = int(os.environ.get('MPI_LOCALRANKID', '0'))
            else:
                # Fallback
                self.rank = 0
                self.world_size = 1
                local_rank = 0
            
            # Set up distributed training
            if self.world_size > 1:
                # Find master address
                if self.rank == 0:
                    master_addr = socket.gethostname()
                    master_port = "12355"
                else:
                    # Get from environment or default
                    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
                    master_port = os.environ.get('MASTER_PORT', '12355')
                
                os.environ['MASTER_ADDR'] = master_addr
                os.environ['MASTER_PORT'] = master_port
                
                # Initialize process group
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    rank=self.rank,
                    world_size=self.world_size
                )
                
                # Set CUDA device
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank % torch.cuda.device_count())
                
                self.is_initialized = True
                
                logging.info(f"MPI initialized: rank {self.rank}/{self.world_size}")
                return True
        
        except Exception as e:
            logging.error(f"MPI initialization failed: {e}")
            return False
        
        return False
    
    def cleanup_mpi(self):
        """Cleanup MPI environment."""
        if self.is_initialized:
            try:
                dist.destroy_process_group()
                self.is_initialized = False
                logging.info("MPI cleanup completed")
            except Exception as e:
                logging.error(f"MPI cleanup error: {e}")
    
    def generate_mpi_command(self, 
                           python_script: str,
                           script_args: Optional[List[str]] = None) -> List[str]:
        """Generate MPI execution command.
        
        Args:
            python_script: Python script to execute
            script_args: Script arguments
            
        Returns:
            MPI command as list of strings
        """
        cmd = [self.config.mpi_executable]
        
        # Number of processes
        cmd.extend(['-np', str(self.config.num_processes)])
        
        # Host specification
        if self.config.hosts:
            host_list = ','.join(self.config.hosts)
            cmd.extend(['-H', host_list])
        elif self.config.hostfile:
            cmd.extend(['-hostfile', self.config.hostfile])
        
        # Mapping and binding
        cmd.extend(['-map-by', self.config.map_by])
        cmd.extend(['-bind-to', self.config.bind_to])
        
        if self.config.report_bindings:
            cmd.append('-report-bindings')
        
        if self.config.display_map:
            cmd.append('-display-map')
        
        # MCA parameters (OpenMPI specific)
        for param, value in self.config.mca_params.items():
            cmd.extend(['-mca', param, value])
        
        # Custom options
        cmd.extend(self.config.custom_options)
        
        # Python command
        cmd.append('python')
        cmd.append(python_script)
        
        if script_args:
            cmd.extend(script_args)
        
        return cmd
    
    def run_distributed_training(self,
                                python_script: str,
                                script_args: Optional[List[str]] = None,
                                working_dir: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run distributed training with MPI.
        
        Args:
            python_script: Python script to execute
            script_args: Script arguments
            working_dir: Working directory
            
        Returns:
            Subprocess result
        """
        if not self.mpi_available:
            raise RuntimeError("MPI not available")
        
        cmd = self.generate_mpi_command(python_script, script_args)
        
        logging.info(f"Running MPI command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logging.info("MPI training completed successfully")
            else:
                logging.error(f"MPI training failed with return code {result.returncode}")
                logging.error(f"Error output: {result.stderr}")
            
            return result
        
        except Exception as e:
            logging.error(f"MPI execution error: {e}")
            raise


class CheckpointManager:
    """Advanced checkpoint management with fault tolerance."""
    
    def __init__(self,
                 checkpoint_dir: str,
                 max_checkpoints: int = 5,
                 checkpoint_frequency: int = 1,
                 async_save: bool = True):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_frequency: Frequency of checkpointing (epochs)
            async_save: Whether to save checkpoints asynchronously
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.checkpoint_frequency = checkpoint_frequency
        self.async_save = async_save
        
        self._checkpoint_history = []
        self._save_executor = ThreadPoolExecutor(max_workers=1) if async_save else None
        
    def save_checkpoint(self,
                       state: Dict[str, Any],
                       epoch: int,
                       is_best: bool = False,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint.
        
        Args:
            state: State dictionary to save
            epoch: Current epoch
            is_best: Whether this is the best checkpoint
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        if epoch % self.checkpoint_frequency != 0:
            return ""
        
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Add metadata
        full_state = {
            'epoch': epoch,
            'timestamp': time.time(),
            'is_best': is_best,
            'metadata': metadata or {},
            **state
        }
        
        if self.async_save:
            # Save asynchronously
            future = self._save_executor.submit(self._save_checkpoint_sync, full_state, checkpoint_path)
            
            # Don't wait, but log if there's an error
            def check_save(f):
                try:
                    f.result()
                    self._checkpoint_history.append({
                        'epoch': epoch,
                        'path': str(checkpoint_path),
                        'timestamp': time.time(),
                        'is_best': is_best
                    })
                    self._cleanup_old_checkpoints()
                    logging.info(f"Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    logging.error(f"Async checkpoint save failed: {e}")
            
            future.add_done_callback(check_save)
        else:
            # Save synchronously
            self._save_checkpoint_sync(full_state, checkpoint_path)
            self._checkpoint_history.append({
                'epoch': epoch,
                'path': str(checkpoint_path),
                'timestamp': time.time(),
                'is_best': is_best
            })
            self._cleanup_old_checkpoints()
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            if not self.async_save:
                shutil.copy2(checkpoint_path, best_path)
            else:
                self._save_executor.submit(shutil.copy2, checkpoint_path, best_path)
        
        return str(checkpoint_path)
    
    def _save_checkpoint_sync(self, state: Dict[str, Any], path: Path):
        """Save checkpoint synchronously."""
        # Save to temporary file first for atomic writes
        temp_path = path.with_suffix('.tmp')
        
        try:
            torch.save(state, temp_path)
            temp_path.rename(path)  # Atomic move
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False) -> Optional[Dict[str, Any]]:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path
            load_best: Whether to load best checkpoint
            
        Returns:
            Loaded state dictionary
        """
        if load_best:
            path = self.checkpoint_dir / "best_checkpoint.pth"
        elif checkpoint_path:
            path = Path(checkpoint_path)
        else:
            # Load latest checkpoint
            if not self._checkpoint_history:
                self._scan_checkpoint_directory()
            
            if not self._checkpoint_history:
                return None
            
            latest = max(self._checkpoint_history, key=lambda x: x['epoch'])
            path = Path(latest['path'])
        
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return None
        
        try:
            state = torch.load(path, map_location='cpu')
            logging.info(f"Loaded checkpoint: {path}")
            return state
        except Exception as e:
            logging.error(f"Failed to load checkpoint {path}: {e}")
            return None
    
    def _scan_checkpoint_directory(self):
        """Scan checkpoint directory to build history."""
        self._checkpoint_history = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_epoch_*.pth"):
            try:
                # Extract epoch from filename
                epoch_str = checkpoint_file.stem.split('_')[-1]
                epoch = int(epoch_str)
                
                self._checkpoint_history.append({
                    'epoch': epoch,
                    'path': str(checkpoint_file),
                    'timestamp': checkpoint_file.stat().st_mtime,
                    'is_best': False
                })
            except (ValueError, IndexError):
                continue
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self._checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by epoch and keep only the latest
        sorted_checkpoints = sorted(self._checkpoint_history, key=lambda x: x['epoch'])
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                self._checkpoint_history.remove(checkpoint)
            except Exception as e:
                logging.warning(f"Failed to remove old checkpoint {checkpoint['path']}: {e}")
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about available checkpoints."""
        if not self._checkpoint_history:
            self._scan_checkpoint_directory()
        
        return self._checkpoint_history.copy()
    
    def cleanup(self):
        """Cleanup resources."""
        if self._save_executor:
            self._save_executor.shutdown(wait=True)


class JobScheduler:
    """Job scheduling and resource management."""
    
    def __init__(self,
                 scheduler_type: str = "fifo",
                 max_concurrent_jobs: int = 4):
        """Initialize job scheduler.
        
        Args:
            scheduler_type: Scheduling algorithm ("fifo", "priority", "round_robin")
            max_concurrent_jobs: Maximum concurrent jobs
        """
        self.scheduler_type = scheduler_type
        self.max_concurrent_jobs = max_concurrent_jobs
        
        self._job_queue = []
        self._running_jobs = {}
        self._completed_jobs = []
        self._job_counter = 0
        
        self._scheduler_thread = None
        self._is_running = False
        self._lock = threading.Lock()
    
    def submit_job(self,
                  job_func: Callable,
                  job_args: Tuple = (),
                  job_kwargs: Dict[str, Any] = None,
                  priority: int = 0,
                  resources: Optional[Dict[str, Any]] = None) -> str:
        """Submit job to scheduler.
        
        Args:
            job_func: Function to execute
            job_args: Function arguments
            job_kwargs: Function keyword arguments
            priority: Job priority (higher = more priority)
            resources: Required resources
            
        Returns:
            Job ID
        """
        with self._lock:
            self._job_counter += 1
            job_id = f"job_{self._job_counter}"
            
            job_info = {
                'job_id': job_id,
                'job_func': job_func,
                'job_args': job_args,
                'job_kwargs': job_kwargs or {},
                'priority': priority,
                'resources': resources or {},
                'submit_time': time.time(),
                'status': 'queued'
            }
            
            self._job_queue.append(job_info)
            
            # Sort queue based on scheduler type
            if self.scheduler_type == "priority":
                self._job_queue.sort(key=lambda x: (-x['priority'], x['submit_time']))
            
            logging.info(f"Job {job_id} submitted to queue")
            return job_id
    
    def start_scheduler(self):
        """Start job scheduler."""
        if self._is_running:
            return
        
        self._is_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        logging.info(f"Job scheduler started ({self.scheduler_type})")
    
    def stop_scheduler(self):
        """Stop job scheduler."""
        self._is_running = False
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        
        # Wait for running jobs to complete
        executor = ThreadPoolExecutor(max_workers=1)
        
        def wait_for_jobs():
            while self._running_jobs:
                time.sleep(1)
        
        future = executor.submit(wait_for_jobs)
        try:
            future.result(timeout=30)
        except:
            logging.warning("Some jobs may still be running after scheduler stop")
        
        executor.shutdown()
        logging.info("Job scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._is_running:
            try:
                with self._lock:
                    # Check for completed jobs
                    completed_job_ids = []
                    for job_id, job_info in self._running_jobs.items():
                        future = job_info['future']
                        if future.done():
                            completed_job_ids.append(job_id)
                    
                    # Move completed jobs
                    for job_id in completed_job_ids:
                        job_info = self._running_jobs.pop(job_id)
                        future = job_info['future']
                        
                        try:
                            result = future.result()
                            job_info['status'] = 'completed'
                            job_info['result'] = result
                            job_info['end_time'] = time.time()
                        except Exception as e:
                            job_info['status'] = 'failed'
                            job_info['error'] = str(e)
                            job_info['end_time'] = time.time()
                        
                        self._completed_jobs.append(job_info)
                        logging.info(f"Job {job_id} {job_info['status']}")
                    
                    # Start new jobs if capacity available
                    while (len(self._running_jobs) < self.max_concurrent_jobs and 
                           self._job_queue):
                        
                        # Get next job based on scheduler type
                        if self.scheduler_type == "round_robin":
                            # Simple round-robin (actually FIFO in this implementation)
                            job_info = self._job_queue.pop(0)
                        else:
                            # FIFO or priority (already sorted)
                            job_info = self._job_queue.pop(0)
                        
                        # Check resource availability
                        if self._check_resource_availability(job_info['resources']):
                            self._start_job(job_info)
                        else:
                            # Put job back in queue
                            self._job_queue.insert(0, job_info)
                            break
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logging.error(f"Scheduler loop error: {e}")
                time.sleep(5)
    
    def _check_resource_availability(self, required_resources: Dict[str, Any]) -> bool:
        """Check if required resources are available."""
        # Simple resource check - can be extended
        if not required_resources:
            return True
        
        # Check GPU resources
        if 'gpu' in required_resources:
            required_gpus = required_resources['gpu']
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                used_gpus = sum(
                    job['resources'].get('gpu', 0) 
                    for job in self._running_jobs.values()
                )
                return (used_gpus + required_gpus) <= available_gpus
            else:
                return False
        
        return True
    
    def _start_job(self, job_info: Dict[str, Any]):
        """Start executing a job."""
        job_id = job_info['job_id']
        
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            job_info['job_func'],
            *job_info['job_args'],
            **job_info['job_kwargs']
        )
        
        job_info['future'] = future
        job_info['executor'] = executor
        job_info['status'] = 'running'
        job_info['start_time'] = time.time()
        
        self._running_jobs[job_id] = job_info
        logging.info(f"Started job {job_id}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        with self._lock:
            # Check running jobs
            if job_id in self._running_jobs:
                job_info = self._running_jobs[job_id].copy()
                job_info.pop('future', None)
                job_info.pop('executor', None)
                return job_info
            
            # Check completed jobs
            for job_info in self._completed_jobs:
                if job_info['job_id'] == job_id:
                    return job_info.copy()
            
            # Check queued jobs
            for job_info in self._job_queue:
                if job_info['job_id'] == job_id:
                    return job_info.copy()
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        with self._lock:
            return {
                'queued_jobs': len(self._job_queue),
                'running_jobs': len(self._running_jobs),
                'completed_jobs': len(self._completed_jobs),
                'scheduler_type': self.scheduler_type,
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'is_running': self._is_running
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            # Check if job is in queue
            for i, job_info in enumerate(self._job_queue):
                if job_info['job_id'] == job_id:
                    self._job_queue.pop(i)
                    job_info['status'] = 'cancelled'
                    job_info['end_time'] = time.time()
                    self._completed_jobs.append(job_info)
                    logging.info(f"Cancelled queued job {job_id}")
                    return True
            
            # Check if job is running
            if job_id in self._running_jobs:
                job_info = self._running_jobs.pop(job_id)
                future = job_info['future']
                executor = job_info['executor']
                
                # Try to cancel the future
                cancelled = future.cancel()
                
                if not cancelled:
                    # Force shutdown executor
                    executor.shutdown(wait=False)
                
                job_info['status'] = 'cancelled'
                job_info['end_time'] = time.time()
                self._completed_jobs.append(job_info)
                
                logging.info(f"Cancelled running job {job_id}")
                return True
        
        return False