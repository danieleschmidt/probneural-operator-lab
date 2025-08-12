"""Multi-fidelity Fourier Neural Operator implementation.

This module implements a novel multi-fidelity FNO architecture that can learn from
data at different fidelity levels and provide uncertainty estimates that account
for fidelity uncertainty.

Mathematical Foundation:
    Multi-fidelity modeling assumes a hierarchy of models with increasing fidelity:
    f_0 (low fidelity), f_1 (medium fidelity), ..., f_L (high fidelity)
    
    The multi-fidelity FNO learns the relationships:
    f_{l+1}(x) = f_l(x) + δ_l(x)
    
    Where δ_l represents the fidelity correction learned by a neural operator.

Key Innovations:
- Hierarchical uncertainty propagation across fidelity levels
- Transfer learning between fidelity levels
- Bayesian treatment of inter-fidelity correlations
- Optimal computational budget allocation

Research Applications:
- Computational fluid dynamics (coarse/fine mesh)
- Climate modeling (different resolution models)
- Materials science (DFT vs. empirical potentials)
"""

from typing import List, Dict, Tuple, Optional, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import ProbabilisticNeuralOperator
from ..base.layers import SpectralLayer, FeedForwardLayer
from ...utils.validation import validate_tensor_shape, validate_tensor_finite


class FidelityTransferLayer(nn.Module):
    """Transfer layer for learning fidelity corrections.
    
    This layer learns the correction δ_l(x) = f_{l+1}(x) - f_l(x)
    between adjacent fidelity levels.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 activation: str = "gelu"):
        """Initialize fidelity transfer layer.
        
        Args:
            input_dim: Input dimension (includes low-fidelity prediction)
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP for fidelity correction
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.correction_network = nn.Sequential(*layers)
        
        # Uncertainty estimation layer
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, 
                x: torch.Tensor,
                low_fidelity_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through fidelity transfer layer.
        
        Args:
            x: Input features
            low_fidelity_pred: Prediction from lower fidelity level
            
        Returns:
            Tuple of (correction, uncertainty)
        """
        # Concatenate input with low-fidelity prediction
        combined_input = torch.cat([x.flatten(start_dim=1), 
                                   low_fidelity_pred.flatten(start_dim=1)], dim=1)
        
        # Get features from hidden layers
        features = combined_input
        for layer in self.correction_network[:-1]:
            features = layer(features)
        
        # Compute correction
        correction = self.correction_network[-1](features)
        
        # Compute uncertainty
        uncertainty = self.uncertainty_head[:-1](features)
        uncertainty = self.uncertainty_head[-1](uncertainty)
        
        # Reshape to match low_fidelity_pred shape
        target_shape = low_fidelity_pred.shape
        if correction.shape != target_shape:
            correction = correction.view(target_shape)
            uncertainty = uncertainty.view(target_shape)
        
        return correction, uncertainty


class MultiFidelityFNO(ProbabilisticNeuralOperator):
    """Multi-fidelity Fourier Neural Operator with hierarchical uncertainty.
    
    This implements a novel architecture that learns from multiple fidelity levels
    and provides uncertainty estimates that account for both model uncertainty
    and fidelity uncertainty.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_fidelities: int = 3,
                 modes: List[int] = None,
                 widths: List[int] = None,
                 depths: List[int] = None,
                 activation: str = "gelu",
                 spatial_dim: int = 2,
                 posterior_type: str = "laplace",
                 prior_precision: float = 1.0,
                 fidelity_correlation: float = 0.8,
                 **kwargs):
        """Initialize Multi-fidelity FNO.
        
        Args:
            input_dim: Input function dimension
            output_dim: Output function dimension
            num_fidelities: Number of fidelity levels
            modes: List of Fourier modes for each fidelity level
            widths: List of hidden dimensions for each fidelity level
            depths: List of depths for each fidelity level
            activation: Activation function
            spatial_dim: Spatial dimension (1D, 2D, or 3D)
            posterior_type: Type of posterior approximation
            prior_precision: Prior precision for Bayesian inference
            fidelity_correlation: Correlation between fidelity levels (0-1)
        """
        super().__init__(
            input_dim, output_dim,
            posterior_type=posterior_type,
            prior_precision=prior_precision,
            **kwargs
        )
        
        self.num_fidelities = num_fidelities
        self.spatial_dim = spatial_dim
        self.fidelity_correlation = fidelity_correlation
        
        # Default parameters for each fidelity level
        if modes is None:
            modes = [8, 12, 16][:num_fidelities]  # Increasing complexity
        if widths is None:
            widths = [32, 64, 96][:num_fidelities]
        if depths is None:
            depths = [2, 3, 4][:num_fidelities]
        
        # Ensure lists match num_fidelities
        self.modes = (modes * num_fidelities)[:num_fidelities]
        self.widths = (widths * num_fidelities)[:num_fidelities]
        self.depths = (depths * num_fidelities)[:num_fidelities]
        
        # Build FNO architecture for each fidelity level
        self.fidelity_operators = nn.ModuleList()
        self.transfer_layers = nn.ModuleList()
        
        for i in range(num_fidelities):
            # Base FNO for this fidelity level
            fno_layers = self._build_fno_layers(
                input_dim, output_dim,
                modes=self.modes[i],
                width=self.widths[i],
                depth=self.depths[i],
                activation=activation
            )
            self.fidelity_operators.append(fno_layers)
            
            # Transfer layer (except for the lowest fidelity)
            if i > 0:
                transfer_layer = FidelityTransferLayer(
                    input_dim=input_dim + output_dim,  # Input + low-fidelity pred
                    output_dim=output_dim,
                    hidden_dim=self.widths[i],
                    num_layers=2,
                    activation=activation
                )
                self.transfer_layers.append(transfer_layer)
        
        # Cross-fidelity attention mechanism
        self.cross_fidelity_attention = nn.MultiheadAttention(
            embed_dim=max(self.widths),
            num_heads=4,
            batch_first=True
        )
        
        # Fidelity uncertainty quantification
        self.fidelity_uncertainty_head = nn.Sequential(
            nn.Linear(sum(self.widths), max(self.widths)),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(max(self.widths), output_dim),
            nn.Softplus()
        )
    
    def _build_fno_layers(self, input_dim: int, output_dim: int,
                          modes: int, width: int, depth: int,
                          activation: str) -> nn.Module:
        """Build FNO layers for a specific fidelity level."""
        layers = nn.ModuleDict()
        
        # Lifting layer
        layers['lift'] = nn.Linear(input_dim, width)
        
        # Spectral layers
        spectral_layers = nn.ModuleList([
            SpectralLayer(width, width, modes, self.spatial_dim)
            for _ in range(depth)
        ])
        layers['spectral'] = spectral_layers
        
        # Local layers
        local_layers = nn.ModuleList([
            FeedForwardLayer(width, width, activation)
            for _ in range(depth)
        ])
        layers['local'] = local_layers
        
        # Projection layer
        layers['project'] = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(width // 2, output_dim)
        )
        
        return layers
    
    def forward_fidelity(self, x: torch.Tensor, fidelity_level: int) -> torch.Tensor:
        """Forward pass for a specific fidelity level.
        
        Args:
            x: Input tensor
            fidelity_level: Fidelity level (0 = lowest)
            
        Returns:
            Output for the specified fidelity level
        """
        layers = self.fidelity_operators[fidelity_level]
        
        # Move channel dimension to last for linear layers
        x = x.permute(0, *range(2, x.ndim), 1)
        
        # Lifting
        x = layers['lift'](x)
        
        # Spectral + local layers
        for spectral, local in zip(layers['spectral'], layers['local']):
            # Spectral convolution
            x_spectral = x.permute(0, -1, *range(1, x.ndim-1))
            x_spectral = spectral(x_spectral)
            x_spectral = x_spectral.permute(0, *range(2, x_spectral.ndim), 1)
            
            # Local transformation
            x_local = local(x)
            
            # Residual connection
            x = x_spectral + x_local
        
        # Projection
        x = layers['project'](x)
        
        # Move channels back to second position
        return x.permute(0, -1, *range(1, x.ndim-1))
    
    def forward(self, x: torch.Tensor, target_fidelity: int = -1) -> torch.Tensor:
        """Forward pass through multi-fidelity architecture.
        
        Args:
            x: Input tensor
            target_fidelity: Target fidelity level (-1 for highest)
            
        Returns:
            Prediction at target fidelity level
        """
        if target_fidelity == -1:
            target_fidelity = self.num_fidelities - 1
        
        # Start with lowest fidelity
        prediction = self.forward_fidelity(x, 0)
        
        # Apply fidelity corrections up to target level
        for i in range(1, min(target_fidelity + 1, self.num_fidelities)):
            # Get correction from transfer layer
            correction, _ = self.transfer_layers[i-1](x, prediction)
            
            # Apply correction
            prediction = prediction + correction
        
        return prediction
    
    def predict_with_multifidelity_uncertainty(
            self, 
            x: torch.Tensor,
            return_all_fidelities: bool = False,
            num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Predict with multi-fidelity uncertainty quantification.
        
        Args:
            x: Input tensor
            return_all_fidelities: Whether to return predictions for all fidelities
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Dictionary containing predictions and uncertainties
        """
        self.eval()
        
        results = {
            'fidelity_predictions': [],
            'fidelity_uncertainties': [],
            'transfer_uncertainties': [],
            'total_uncertainty': None,
            'highest_fidelity_prediction': None
        }
        
        with torch.no_grad():
            # Get predictions for each fidelity level
            prediction = self.forward_fidelity(x, 0)
            results['fidelity_predictions'].append(prediction)
            
            # Estimate base uncertainty (simplified)
            base_uncertainty = torch.ones_like(prediction) * 0.1
            results['fidelity_uncertainties'].append(base_uncertainty)
            
            # Propagate through fidelity levels
            for i in range(1, self.num_fidelities):
                # Get correction and its uncertainty
                correction, transfer_uncertainty = self.transfer_layers[i-1](x, prediction)
                results['transfer_uncertainties'].append(transfer_uncertainty)
                
                # Update prediction
                prediction = prediction + correction
                results['fidelity_predictions'].append(prediction)
                
                # Propagate uncertainty
                # Total uncertainty = previous uncertainty + transfer uncertainty + correlation term
                prev_uncertainty = results['fidelity_uncertainties'][-1]
                correlation_term = self.fidelity_correlation * torch.sqrt(prev_uncertainty * transfer_uncertainty)
                
                total_fidelity_uncertainty = torch.sqrt(
                    prev_uncertainty.pow(2) + 
                    transfer_uncertainty.pow(2) + 
                    2 * correlation_term
                )
                results['fidelity_uncertainties'].append(total_fidelity_uncertainty)
            
            # Final results
            results['highest_fidelity_prediction'] = prediction
            results['total_uncertainty'] = results['fidelity_uncertainties'][-1]
            
            # Add epistemic uncertainty if posterior is fitted
            if self._is_fitted:
                _, epistemic_uncertainty = self.predict_with_uncertainty(
                    x, num_samples=num_samples
                )
                # Combine fidelity and epistemic uncertainties
                results['total_uncertainty'] = torch.sqrt(
                    results['total_uncertainty'].pow(2) + epistemic_uncertainty.pow(2)
                )
        
        # Keep only highest fidelity if not requested otherwise
        if not return_all_fidelities:
            results['fidelity_predictions'] = [results['highest_fidelity_prediction']]
            results['fidelity_uncertainties'] = [results['total_uncertainty']]
        
        return results
    
    def compute_fidelity_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute importance weights for each fidelity level.
        
        Args:
            x: Input tensor
            
        Returns:
            Importance weights for each fidelity level
        """
        # Get features from each fidelity level
        fidelity_features = []
        
        for i in range(self.num_fidelities):
            layers = self.fidelity_operators[i]
            
            # Get intermediate features
            x_temp = x.permute(0, *range(2, x.ndim), 1)
            features = layers['lift'](x_temp)
            
            # Pool features
            pooled_features = features.mean(dim=tuple(range(1, features.ndim-1)))
            fidelity_features.append(pooled_features)
        
        # Stack and apply attention
        features_stack = torch.stack(fidelity_features, dim=1)  # (batch, num_fidelities, features)
        
        # Self-attention to compute importance weights
        attended_features, attention_weights = self.cross_fidelity_attention(
            features_stack, features_stack, features_stack
        )
        
        # Return attention weights as importance scores
        return attention_weights.mean(dim=1)  # Average over heads
    
    def optimal_fidelity_selection(self, 
                                 x: torch.Tensor,
                                 computational_budget: float,
                                 fidelity_costs: List[float]) -> int:
        """Select optimal fidelity level given computational budget.
        
        Args:
            x: Input tensor
            computational_budget: Available computational budget
            fidelity_costs: Cost for each fidelity level
            
        Returns:
            Optimal fidelity level
        """
        # Get importance weights
        importance_weights = self.compute_fidelity_importance(x)
        
        # Compute cost-benefit ratio
        fidelity_costs = torch.tensor(fidelity_costs, device=x.device)
        cost_benefit = importance_weights.mean(dim=0) / fidelity_costs
        
        # Select highest fidelity within budget
        affordable_fidelities = torch.where(fidelity_costs <= computational_budget)[0]
        
        if len(affordable_fidelities) == 0:
            return 0  # Use lowest fidelity if budget is too low
        
        # Among affordable fidelities, select the one with best cost-benefit
        best_idx = torch.argmax(cost_benefit[affordable_fidelities])
        return affordable_fidelities[best_idx].item()
    
    def fit_multifidelity(self,
                         train_loaders: List[DataLoader],
                         val_loaders: Optional[List[DataLoader]] = None,
                         epochs: int = 100,
                         lr: float = 1e-3,
                         device: str = "auto") -> Dict[str, Any]:
        """Train multi-fidelity model with hierarchical learning.
        
        Args:
            train_loaders: List of training data loaders for each fidelity
            val_loaders: Optional list of validation data loaders
            epochs: Number of training epochs
            lr: Learning rate
            device: Training device
            
        Returns:
            Training history
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        
        # Separate optimizers for each fidelity level
        optimizers = []
        for i in range(self.num_fidelities):
            if i == 0:
                # Lowest fidelity: train base FNO
                optimizer = torch.optim.Adam(
                    self.fidelity_operators[i].parameters(), lr=lr
                )
            else:
                # Higher fidelities: train transfer layers
                params = list(self.transfer_layers[i-1].parameters())
                optimizer = torch.optim.Adam(params, lr=lr)
            optimizers.append(optimizer)
        
        criterion = nn.MSELoss()
        history = {'train_loss': [], 'val_loss': []}
        
        # Progressive training: train each fidelity level in sequence
        for fidelity_level in range(self.num_fidelities):
            print(f"Training fidelity level {fidelity_level}...")
            
            train_loader = train_loaders[fidelity_level]
            val_loader = val_loaders[fidelity_level] if val_loaders else None
            optimizer = optimizers[fidelity_level]
            
            fidelity_history = self._train_single_fidelity(
                fidelity_level, train_loader, val_loader,
                optimizer, criterion, epochs, device
            )
            
            # Accumulate history
            if fidelity_level == 0:
                history['train_loss'] = fidelity_history['train_loss']
                history['val_loss'] = fidelity_history['val_loss']
            else:
                # Combine losses from different fidelity levels
                for i, (train_loss, val_loss) in enumerate(
                    zip(fidelity_history['train_loss'], fidelity_history['val_loss'])
                ):
                    if i < len(history['train_loss']):
                        history['train_loss'][i] += train_loss
                        history['val_loss'][i] += val_loss
                    else:
                        history['train_loss'].append(train_loss)
                        history['val_loss'].append(val_loss)
        
        return history
    
    def _train_single_fidelity(self,
                             fidelity_level: int,
                             train_loader: DataLoader,
                             val_loader: Optional[DataLoader],
                             optimizer: torch.optim.Optimizer,
                             criterion: nn.Module,
                             epochs: int,
                             device: str) -> Dict[str, List[float]]:
        """Train a single fidelity level."""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                if fidelity_level == 0:
                    # Train base FNO
                    output = self.forward_fidelity(data, 0)
                else:
                    # Train transfer layer
                    with torch.no_grad():
                        low_fidelity_pred = self.forward(data, fidelity_level - 1)
                    
                    correction, _ = self.transfer_layers[fidelity_level - 1](data, low_fidelity_pred)
                    output = low_fidelity_pred + correction
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.eval()
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        
                        if fidelity_level == 0:
                            output = self.forward_fidelity(data, 0)
                        else:
                            low_fidelity_pred = self.forward(data, fidelity_level - 1)
                            correction, _ = self.transfer_layers[fidelity_level - 1](data, low_fidelity_pred)
                            output = low_fidelity_pred + correction
                        
                        val_loss += criterion(output, target).item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
            
            if epoch % 20 == 0:
                msg = f"Fidelity {fidelity_level}, Epoch {epoch}: Train Loss: {train_loss:.6f}"
                if val_loader:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
        
        return history