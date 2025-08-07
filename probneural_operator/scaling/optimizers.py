"""Advanced optimization algorithms and hyperparameter optimization."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import warnings


@dataclass
class OptimizerConfig:
    """Configuration for advanced optimizers."""
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # AdamW specific
    amsgrad: bool = False
    
    # RMSprop specific
    alpha: float = 0.99
    momentum: float = 0.0
    centered: bool = False
    
    # LARS specific  
    trust_coefficient: float = 0.001
    eps: float = 1e-8
    
    # Lion specific
    beta: float = 0.95
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


class AdamWCustom(optim.Optimizer):
    """Enhanced AdamW with additional features."""
    
    def __init__(self, 
                 params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 amsgrad: bool = False,
                 maximize: bool = False,
                 rectify: bool = False,
                 lookahead: bool = False,
                 lookahead_k: int = 5,
                 lookahead_alpha: float = 0.5):
        """Enhanced AdamW optimizer.
        
        Args:
            rectify: Use rectified Adam (RAdam)
            lookahead: Use Lookahead optimization
            lookahead_k: Lookahead steps
            lookahead_alpha: Lookahead interpolation factor
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, maximize=maximize, rectify=rectify,
                       lookahead=lookahead, lookahead_k=lookahead_k, 
                       lookahead_alpha=lookahead_alpha)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                if p.grad.is_sparse:
                    raise RuntimeError('AdamWCustom does not support sparse gradients')
                
                params_with_grad.append(p)
                if p.grad.dtype in {torch.float16, torch.bfloat16}:
                    grads.append(p.grad.float())
                else:
                    grads.append(p.grad)
                
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['lookahead']:
                        state['slow_weights'] = p.data.clone()
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                else:
                    max_exp_avg_sqs.append(None)
                
                state['step'] += 1
                state_steps.append(state['step'])
            
            # Perform AdamW update
            self._adamw_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                rectify=group['rectify']
            )
            
            # Apply Lookahead if enabled
            if group['lookahead']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    if state['step'] % group['lookahead_k'] == 0:
                        # Lookahead update
                        slow_weights = state['slow_weights']
                        alpha = group['lookahead_alpha']
                        
                        slow_weights.data.add_(p.data - slow_weights.data, alpha=alpha)
                        p.data.copy_(slow_weights.data)
        
        return loss
    
    def _adamw_update(self, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                     state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps,
                     maximize, rectify):
        """Perform AdamW parameter update."""
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            if maximize:
                grad = -grad
            
            # Weight decay (L2 regularization)
            if weight_decay != 0:
                param.data.add_(param.data, alpha=-weight_decay * lr)
            
            # Exponential moving average of gradient values
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # Exponential moving average of squared gradient values
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            if rectify:
                # RAdam correction
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * beta2**step / bias_correction2
                
                if rho_t > 5:
                    # Variance rectification
                    variance_correction = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf / 
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    step_size = lr * variance_correction / bias_correction1
                else:
                    step_size = lr / bias_correction1
                    denom = torch.ones_like(denom)
            else:
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            
            param.data.addcdiv_(exp_avg, denom, value=-step_size)


class LARS(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer."""
    
    def __init__(self, 
                 params, 
                 lr: float = 1e-3, 
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 trust_coefficient: float = 0.001,
                 eps: float = 1e-8):
        """Initialize LARS optimizer."""
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform LARS optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coefficient = group['trust_coefficient']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_norm = p.data.norm()
                grad_norm = p.grad.data.norm()
                
                # Compute adaptive learning rate
                if param_norm > 0 and grad_norm > 0:
                    adaptive_lr = trust_coefficient * param_norm / (grad_norm + eps)
                    adaptive_lr = min(adaptive_lr, group['lr'])
                else:
                    adaptive_lr = group['lr']
                
                # Apply weight decay
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)
                
                # Apply momentum
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(p.grad.data)
                
                # Update parameters
                p.data.add_(buf, alpha=-adaptive_lr)
        
        return loss


class Lion(optim.Optimizer):
    """Lion optimizer (EvoLved Sign Momentum)."""
    
    def __init__(self, 
                 params, 
                 lr: float = 1e-4, 
                 betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        """Initialize Lion optimizer."""
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform Lion optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if group['weight_decay'] > 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update parameters
                update = exp_avg.mul(beta1).add_(grad, alpha=1-beta1).sign_()
                p.data.add_(update, alpha=-group['lr'])
                
                # Update exponential moving average
                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)
        
        return loss


class AdvancedOptimizerFactory:
    """Factory for creating advanced optimizers with configurations."""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: OptimizerConfig) -> optim.Optimizer:
        """Create optimizer from configuration.
        
        Args:
            model: PyTorch model
            config: Optimizer configuration
            
        Returns:
            Configured optimizer
        """
        params = model.parameters()
        
        if config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad
            )
        
        elif config.optimizer_type.lower() == "adamw_custom":
            return AdamWCustom(
                params,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
                **config.custom_params
            )
        
        elif config.optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                params,
                lr=config.learning_rate,
                alpha=config.alpha,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                centered=config.centered
            )
        
        elif config.optimizer_type.lower() == "lars":
            return LARS(
                params,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                trust_coefficient=config.trust_coefficient,
                eps=config.epsilon
            )
        
        elif config.optimizer_type.lower() == "lion":
            return Lion(
                params,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        
        elif config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
    
    @staticmethod
    def get_optimizer_recommendations(model_size: int, 
                                    dataset_size: int,
                                    task_type: str = "regression") -> List[OptimizerConfig]:
        """Get optimizer recommendations based on model and data characteristics.
        
        Args:
            model_size: Number of model parameters
            dataset_size: Size of training dataset
            task_type: Type of task (regression, classification, etc.)
            
        Returns:
            List of recommended optimizer configurations
        """
        recommendations = []
        
        if model_size < 1e6:  # Small models
            # AdamW with higher learning rate
            recommendations.append(OptimizerConfig(
                optimizer_type="adamw",
                learning_rate=3e-3,
                weight_decay=1e-4,
                beta1=0.9,
                beta2=0.999
            ))
            
            # SGD with momentum
            recommendations.append(OptimizerConfig(
                optimizer_type="sgd",
                learning_rate=1e-2,
                momentum=0.9,
                weight_decay=1e-4
            ))
        
        elif model_size < 1e8:  # Medium models
            # AdamW with moderate learning rate
            recommendations.append(OptimizerConfig(
                optimizer_type="adamw",
                learning_rate=1e-3,
                weight_decay=1e-4,
                beta1=0.9,
                beta2=0.999
            ))
            
            # Lion optimizer
            recommendations.append(OptimizerConfig(
                optimizer_type="lion",
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.99,
                weight_decay=1e-2
            ))
        
        else:  # Large models
            # AdamW with lower learning rate
            recommendations.append(OptimizerConfig(
                optimizer_type="adamw",
                learning_rate=3e-4,
                weight_decay=1e-2,
                beta1=0.9,
                beta2=0.999
            ))
            
            # LARS for large batch training
            recommendations.append(OptimizerConfig(
                optimizer_type="lars",
                learning_rate=1e-3,
                momentum=0.9,
                weight_decay=1e-4,
                trust_coefficient=0.001
            ))
        
        return recommendations


class WarmupCosineLRScheduler(_LRScheduler):
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, 
                 optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 eta_min: float = 0.0,
                 last_epoch: int = -1):
        """Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * progress)) / 2 
                   for base_lr in self.base_lrs]


class PolynomialLRScheduler(_LRScheduler):
    """Polynomial learning rate decay scheduler."""
    
    def __init__(self,
                 optimizer,
                 total_epochs: int,
                 power: float = 1.0,
                 eta_min: float = 0.0,
                 last_epoch: int = -1):
        """Initialize polynomial scheduler."""
        self.total_epochs = total_epochs
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate using polynomial decay."""
        if self.last_epoch >= self.total_epochs:
            return [self.eta_min for _ in self.base_lrs]
        
        decay_factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [self.eta_min + (base_lr - self.eta_min) * decay_factor 
               for base_lr in self.base_lrs]


class LearningRateScheduler:
    """Advanced learning rate scheduling strategies."""
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer,
                        scheduler_type: str,
                        **kwargs) -> _LRScheduler:
        """Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            scheduler_type: Type of scheduler
            **kwargs: Scheduler-specific parameters
            
        Returns:
            Learning rate scheduler
        """
        if scheduler_type == "warmup_cosine":
            return WarmupCosineLRScheduler(
                optimizer,
                warmup_epochs=kwargs.get('warmup_epochs', 10),
                total_epochs=kwargs.get('total_epochs', 100),
                eta_min=kwargs.get('eta_min', 0.0)
            )
        
        elif scheduler_type == "polynomial":
            return PolynomialLRScheduler(
                optimizer,
                total_epochs=kwargs.get('total_epochs', 100),
                power=kwargs.get('power', 1.0),
                eta_min=kwargs.get('eta_min', 0.0)
            )
        
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 0.0)
            )
        
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        
        elif scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                threshold=kwargs.get('threshold', 1e-4),
                min_lr=kwargs.get('min_lr', 0.0)
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class GradientManager:
    """Advanced gradient management including clipping and accumulation."""
    
    def __init__(self,
                 max_grad_norm: float = 1.0,
                 accumulation_steps: int = 1,
                 skip_threshold: Optional[float] = None):
        """Initialize gradient manager.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping
            accumulation_steps: Number of steps to accumulate gradients
            skip_threshold: Skip update if gradient norm exceeds this threshold
        """
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.skip_threshold = skip_threshold
        
        self._accumulation_count = 0
        self._gradient_norms = deque(maxlen=100)
    
    def accumulate_gradients(self, 
                           loss: torch.Tensor,
                           optimizer: optim.Optimizer,
                           model: nn.Module,
                           scaler: Optional[torch.cuda.amp.GradScaler] = None) -> bool:
        """Accumulate gradients and perform update when ready.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            model: Model
            scaler: Gradient scaler for mixed precision
            
        Returns:
            True if parameters were updated, False otherwise
        """
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self._accumulation_count += 1
        
        # Check if it's time to update
        if self._accumulation_count >= self.accumulation_steps:
            return self._perform_update(optimizer, model, scaler)
        
        return False
    
    def _perform_update(self,
                       optimizer: optim.Optimizer,
                       model: nn.Module,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None) -> bool:
        """Perform parameter update with gradient management."""
        # Calculate gradient norm before clipping
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        grad_norm = self._calculate_gradient_norm(model)
        self._gradient_norms.append(grad_norm)
        
        # Check if we should skip this update
        if self.skip_threshold is not None and grad_norm > self.skip_threshold:
            optimizer.zero_grad()
            self._accumulation_count = 0
            return False
        
        # Apply gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        # Perform optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        self._accumulation_count = 0
        
        return True
    
    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calculate total gradient norm."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return math.sqrt(total_norm) if param_count > 0 else 0.0
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self._gradient_norms:
            return {}
        
        norms = list(self._gradient_norms)
        
        return {
            'avg_gradient_norm': np.mean(norms),
            'std_gradient_norm': np.std(norms),
            'max_gradient_norm': np.max(norms),
            'min_gradient_norm': np.min(norms),
            'recent_gradient_norm': norms[-1]
        }


class HyperparameterOptimizer:
    """Hyperparameter optimization using various strategies."""
    
    def __init__(self, 
                 optimization_method: str = "optuna",
                 n_trials: int = 100,
                 timeout: Optional[int] = None):
        """Initialize hyperparameter optimizer.
        
        Args:
            optimization_method: Method for optimization ("optuna", "grid_search", "bayesian")
            n_trials: Number of trials
            timeout: Timeout in seconds
        """
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.timeout = timeout
        
        self._study = None
        self._trial_history = []
    
    def optimize(self,
                objective_fn: Callable,
                search_space: Dict[str, Any],
                direction: str = "minimize") -> Dict[str, Any]:
        """Optimize hyperparameters.
        
        Args:
            objective_fn: Function to optimize (returns metric to minimize/maximize)
            search_space: Search space definition
            direction: "minimize" or "maximize"
            
        Returns:
            Best hyperparameters found
        """
        if self.optimization_method == "optuna":
            return self._optimize_with_optuna(objective_fn, search_space, direction)
        elif self.optimization_method == "grid_search":
            return self._optimize_with_grid_search(objective_fn, search_space, direction)
        elif self.optimization_method == "bayesian":
            return self._optimize_with_bayesian(objective_fn, search_space, direction)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _optimize_with_optuna(self,
                             objective_fn: Callable,
                             search_space: Dict[str, Any],
                             direction: str) -> Dict[str, Any]:
        """Optimize using Optuna."""
        try:
            import optuna
            
            def objective(trial):
                # Sample parameters from search space
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Evaluate objective function
                try:
                    result = objective_fn(**params)
                    self._trial_history.append({
                        'params': params.copy(),
                        'value': result,
                        'timestamp': time.time()
                    })
                    return result
                except Exception as e:
                    logging.warning(f"Trial failed with error: {e}")
                    return float('inf') if direction == "minimize" else float('-inf')
            
            # Create study
            self._study = optuna.create_study(direction=direction)
            self._study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
            
            return self._study.best_params
            
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
    
    def _optimize_with_grid_search(self,
                                  objective_fn: Callable,
                                  search_space: Dict[str, Any],
                                  direction: str) -> Dict[str, Any]:
        """Optimize using grid search."""
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(search_space.keys())
        param_values = []
        
        for param_name in param_names:
            config = search_space[param_name]
            if config['type'] == 'categorical':
                param_values.append(config['choices'])
            elif config['type'] in ['float', 'int']:
                # Create grid of values
                low, high = config['low'], config['high']
                num_points = config.get('num_points', 10)
                
                if config['type'] == 'float':
                    if config.get('log', False):
                        values = np.logspace(np.log10(low), np.log10(high), num_points).tolist()
                    else:
                        values = np.linspace(low, high, num_points).tolist()
                else:  # int
                    values = np.linspace(low, high, num_points, dtype=int).tolist()
                
                param_values.append(values)
        
        # Evaluate all combinations
        best_params = None
        best_value = float('inf') if direction == "minimize" else float('-inf')
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            try:
                result = objective_fn(**params)
                self._trial_history.append({
                    'params': params.copy(),
                    'value': result,
                    'timestamp': time.time()
                })
                
                if ((direction == "minimize" and result < best_value) or
                    (direction == "maximize" and result > best_value)):
                    best_value = result
                    best_params = params.copy()
                
            except Exception as e:
                logging.warning(f"Trial failed with error: {e}")
        
        return best_params
    
    def _optimize_with_bayesian(self,
                               objective_fn: Callable,
                               search_space: Dict[str, Any],
                               direction: str) -> Dict[str, Any]:
        """Optimize using Bayesian optimization with Gaussian processes."""
        # Simplified Bayesian optimization
        # In production, consider using libraries like scikit-optimize
        
        param_names = list(search_space.keys())
        bounds = []
        
        # Extract bounds for continuous parameters
        for param_name in param_names:
            config = search_space[param_name]
            if config['type'] in ['float', 'int']:
                bounds.append((config['low'], config['high']))
            else:
                # For categorical, convert to integer indices
                bounds.append((0, len(config['choices']) - 1))
        
        # Initialize with random samples
        n_initial = min(10, self.n_trials // 4)
        X_samples = []
        y_samples = []
        
        for _ in range(n_initial):
            # Random sample
            sample = []
            params = {}
            
            for i, param_name in enumerate(param_names):
                config = search_space[param_name]
                low, high = bounds[i]
                
                if config['type'] == 'float':
                    if config.get('log', False):
                        value = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        value = np.random.uniform(low, high)
                elif config['type'] == 'int':
                    value = np.random.randint(low, high + 1)
                else:  # categorical
                    idx = np.random.randint(len(config['choices']))
                    value = config['choices'][idx]
                    sample.append(idx)  # Store index for GP
                    params[param_name] = value
                    continue
                
                sample.append(value)
                params[param_name] = value
            
            try:
                result = objective_fn(**params)
                X_samples.append(sample)
                y_samples.append(result)
                
                self._trial_history.append({
                    'params': params.copy(),
                    'value': result,
                    'timestamp': time.time()
                })
            except Exception as e:
                logging.warning(f"Initial sample failed: {e}")
        
        if not X_samples:
            return {}
        
        # Bayesian optimization loop
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        kernel = Matern(length_scale=1.0, nu=1.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        
        best_idx = np.argmin(y_samples) if direction == "minimize" else np.argmax(y_samples)
        best_params = None
        
        for trial in range(n_initial, self.n_trials):
            # Fit GP
            gp.fit(X_samples, y_samples)
            
            # Find next point using acquisition function (simplified)
            best_acquisition = float('-inf')
            next_sample = None
            
            # Random search for next point
            for _ in range(100):
                candidate = []
                candidate_params = {}
                
                for i, param_name in enumerate(param_names):
                    config = search_space[param_name]
                    low, high = bounds[i]
                    
                    if config['type'] == 'float':
                        if config.get('log', False):
                            value = np.exp(np.random.uniform(np.log(low), np.log(high)))
                        else:
                            value = np.random.uniform(low, high)
                    elif config['type'] == 'int':
                        value = np.random.randint(low, high + 1)
                    else:  # categorical
                        idx = np.random.randint(len(config['choices']))
                        value = config['choices'][idx]
                        candidate.append(idx)
                        candidate_params[param_name] = value
                        continue
                    
                    candidate.append(value)
                    candidate_params[param_name] = value
                
                # Acquisition function (Upper Confidence Bound)
                candidate = np.array(candidate).reshape(1, -1)
                mean, std = gp.predict(candidate, return_std=True)
                
                if direction == "minimize":
                    acquisition = -mean[0] + 2.0 * std[0]
                else:
                    acquisition = mean[0] + 2.0 * std[0]
                
                if acquisition > best_acquisition:
                    best_acquisition = acquisition
                    next_sample = candidate[0]
                    next_params = candidate_params.copy()
            
            if next_sample is not None:
                try:
                    result = objective_fn(**next_params)
                    X_samples = np.vstack([X_samples, next_sample])
                    y_samples = np.append(y_samples, result)
                    
                    self._trial_history.append({
                        'params': next_params.copy(),
                        'value': result,
                        'timestamp': time.time()
                    })
                    
                    # Update best
                    if ((direction == "minimize" and result < y_samples[best_idx]) or
                        (direction == "maximize" and result > y_samples[best_idx])):
                        best_idx = len(y_samples) - 1
                        best_params = next_params.copy()
                
                except Exception as e:
                    logging.warning(f"Bayesian trial failed: {e}")
        
        # Return best parameters found
        if best_params is None and self._trial_history:
            if direction == "minimize":
                best_trial = min(self._trial_history, key=lambda x: x['value'])
            else:
                best_trial = max(self._trial_history, key=lambda x: x['value'])
            best_params = best_trial['params']
        
        return best_params or {}
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._trial_history.copy()
    
    def get_best_trial(self, direction: str = "minimize") -> Optional[Dict[str, Any]]:
        """Get best trial from history."""
        if not self._trial_history:
            return None
        
        if direction == "minimize":
            return min(self._trial_history, key=lambda x: x['value'])
        else:
            return max(self._trial_history, key=lambda x: x['value'])


def create_optimizer_config_search_space() -> Dict[str, Any]:
    """Create search space for optimizer hyperparameters."""
    return {
        'optimizer_type': {
            'type': 'categorical',
            'choices': ['adamw', 'lion', 'lars', 'rmsprop']
        },
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-1,
            'log': True
        },
        'weight_decay': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-1,
            'log': True
        },
        'beta1': {
            'type': 'float',
            'low': 0.8,
            'high': 0.999
        },
        'beta2': {
            'type': 'float', 
            'low': 0.9,
            'high': 0.9999
        }
    }