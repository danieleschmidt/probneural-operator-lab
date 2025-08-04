"""DeepONet implementation for learning operators between function spaces."""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from ..base import NeuralOperator, ProbabilisticNeuralOperator


class DeepONet(NeuralOperator):
    """Deep Operator Network (DeepONet) for learning operators.
    
    DeepONet learns mappings between function spaces by decomposing the
    learning into two neural networks:
    - Branch network: encodes the input function
    - Trunk network: encodes the evaluation locations
    
    The output is computed as the inner product of branch and trunk outputs.
    
    References:
        Lu et al. "Learning nonlinear operators via DeepONet based on the 
        universal approximation theorem of operators" Nature Machine Intelligence 2021.
    """
    
    def __init__(self,
                 branch_dim: int,
                 trunk_dim: int,
                 output_dim: int = 1,
                 branch_layers: List[int] = [128, 128, 128],
                 trunk_layers: List[int] = [128, 128, 128],
                 activation: str = "tanh",
                 use_bias: bool = True,
                 **kwargs):
        """Initialize DeepONet.
        
        Args:
            branch_dim: Input dimension for branch network (function values)
            trunk_dim: Input dimension for trunk network (coordinates)
            output_dim: Output dimension
            branch_layers: Hidden layer sizes for branch network
            trunk_layers: Hidden layer sizes for trunk network
            activation: Activation function
            use_bias: Whether to use bias in linear layers
        """
        super().__init__(branch_dim, output_dim, **kwargs)
        
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.activation = activation
        self.use_bias = use_bias
        
        # Determine activation function
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "swish":
            self.act = nn.SiLU() 
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build branch network (encodes input function)
        branch_arch = [branch_dim] + branch_layers + [branch_layers[-1]]
        self.branch_net = self._build_network(branch_arch, add_final_activation=True)
        
        # Build trunk network (encodes evaluation coordinates)
        trunk_arch = [trunk_dim] + trunk_layers + [trunk_layers[-1]]
        self.trunk_net = self._build_network(trunk_arch, add_final_activation=True)
        
        # Ensure branch and trunk output the same dimension
        assert branch_layers[-1] == trunk_layers[-1], \
            "Branch and trunk networks must have same output dimension"
        
        self.latent_dim = branch_layers[-1]
        
        # Final projection layer
        if output_dim > 1:
            self.output_projection = nn.Linear(self.latent_dim, output_dim, bias=use_bias)
        else:
            self.output_projection = None
    
    def _build_network(self, layer_sizes: List[int], add_final_activation: bool = False) -> nn.Module:
        """Build a feedforward network."""
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=self.use_bias))
            
            # Add activation except for the last layer (unless specified)
            if i < len(layer_sizes) - 2 or add_final_activation:
                layers.append(self.act)
        
        return nn.Sequential(*layers)
    
    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepONet.
        
        Args:
            branch_input: Input function values, shape (batch, branch_dim)
            trunk_input: Evaluation coordinates, shape (batch, n_points, trunk_dim)
            
        Returns:
            Output values at evaluation points, shape (batch, n_points, output_dim)
        """
        batch_size = branch_input.shape[0]
        n_points = trunk_input.shape[1]
        
        # Branch network processes input function
        branch_output = self.branch_net(branch_input)  # (batch, latent_dim)
        
        # Trunk network processes evaluation coordinates
        trunk_input_flat = trunk_input.reshape(-1, self.trunk_dim)  # (batch * n_points, trunk_dim)
        trunk_output = self.trunk_net(trunk_input_flat)  # (batch * n_points, latent_dim)
        trunk_output = trunk_output.reshape(batch_size, n_points, self.latent_dim)  # (batch, n_points, latent_dim)
        
        # Compute inner product between branch and trunk outputs
        # branch_output: (batch, latent_dim) -> (batch, 1, latent_dim)
        # trunk_output: (batch, n_points, latent_dim)
        branch_expanded = branch_output.unsqueeze(1)  # (batch, 1, latent_dim)
        
        # Element-wise multiplication and sum over latent dimension
        output = torch.sum(branch_expanded * trunk_output, dim=-1, keepdim=True)  # (batch, n_points, 1)
        
        # Apply output projection if multi-dimensional output
        if self.output_projection is not None:
            # Reshape for linear layer
            output_flat = output.reshape(-1, 1)  # (batch * n_points, 1)
            output_flat = self.output_projection(output_flat)  # (batch * n_points, output_dim)
            output = output_flat.reshape(batch_size, n_points, self.output_dim)
        
        return output
    
    def predict_on_grid(self, branch_input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Predict on a regular grid of evaluation points.
        
        Args:
            branch_input: Input function values
            grid: Grid coordinates, shape (..., trunk_dim)
            
        Returns:
            Predictions on grid
        """
        original_shape = grid.shape[:-1]
        grid_flat = grid.reshape(-1, self.trunk_dim)
        
        # Create batch dimension for grid if needed
        if grid_flat.ndim == 2 and branch_input.ndim == 2:
            batch_size = branch_input.shape[0]
            grid_batched = grid_flat.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            grid_batched = grid_flat.unsqueeze(0)
        
        with torch.no_grad():
            predictions = self(branch_input, grid_batched)
        
        # Reshape back to grid shape
        output_shape = (predictions.shape[0],) + original_shape + (self.output_dim,)
        return predictions.reshape(output_shape)


class ProbabilisticDeepONet(ProbabilisticNeuralOperator):
    """Probabilistic DeepONet with uncertainty quantification."""
    
    def __init__(self,
                 branch_dim: int,
                 trunk_dim: int,
                 output_dim: int = 1,
                 branch_layers: List[int] = [128, 128, 128],
                 trunk_layers: List[int] = [128, 128, 128],
                 activation: str = "tanh",
                 use_bias: bool = True,
                 posterior_type: str = "laplace",
                 prior_precision: float = 1.0,
                 **kwargs):
        """Initialize Probabilistic DeepONet.
        
        Args:
            branch_dim: Input dimension for branch network
            trunk_dim: Input dimension for trunk network  
            output_dim: Output dimension
            branch_layers: Hidden layer sizes for branch network
            trunk_layers: Hidden layer sizes for trunk network
            activation: Activation function
            use_bias: Whether to use bias
            posterior_type: Type of posterior approximation
            prior_precision: Prior precision for Bayesian inference
        """
        super().__init__(
            branch_dim, output_dim,
            posterior_type=posterior_type,
            prior_precision=prior_precision,
            **kwargs
        )
        
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.activation = activation
        self.use_bias = use_bias
        
        # Build the underlying DeepONet architecture
        self._build_architecture()
    
    def _build_architecture(self):
        """Build the DeepONet architecture."""
        # Determine activation function
        if self.activation == "tanh":
            self.act = nn.Tanh()
        elif self.activation == "relu":
            self.act = nn.ReLU()
        elif self.activation == "gelu":
            self.act = nn.GELU()
        elif self.activation == "swish":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        # Build branch network
        branch_arch = [self.branch_dim] + self.branch_layers + [self.branch_layers[-1]]
        self.branch_net = self._build_network(branch_arch, add_final_activation=True)
        
        # Build trunk network
        trunk_arch = [self.trunk_dim] + self.trunk_layers + [self.trunk_layers[-1]]
        self.trunk_net = self._build_network(trunk_arch, add_final_activation=True)
        
        # Ensure same output dimension
        assert self.branch_layers[-1] == self.trunk_layers[-1], \
            "Branch and trunk networks must have same output dimension"
        
        self.latent_dim = self.branch_layers[-1]
        
        # Final projection layer
        if self.output_dim > 1:
            self.output_projection = nn.Linear(self.latent_dim, self.output_dim, bias=self.use_bias)
        else:
            self.output_projection = None
    
    def _build_network(self, layer_sizes: List[int], add_final_activation: bool = False) -> nn.Module:
        """Build a feedforward network."""
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=self.use_bias))
            
            if i < len(layer_sizes) - 2 or add_final_activation:
                layers.append(self.act)
        
        return nn.Sequential(*layers)
    
    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through Probabilistic DeepONet."""
        batch_size = branch_input.shape[0]
        n_points = trunk_input.shape[1]
        
        # Branch network
        branch_output = self.branch_net(branch_input)
        
        # Trunk network
        trunk_input_flat = trunk_input.reshape(-1, self.trunk_dim)
        trunk_output = self.trunk_net(trunk_input_flat)
        trunk_output = trunk_output.reshape(batch_size, n_points, self.latent_dim)
        
        # Inner product
        branch_expanded = branch_output.unsqueeze(1)
        output = torch.sum(branch_expanded * trunk_output, dim=-1, keepdim=True)
        
        # Output projection
        if self.output_projection is not None:
            output_flat = output.reshape(-1, 1)
            output_flat = self.output_projection(output_flat)
            output = output_flat.reshape(batch_size, n_points, self.output_dim)
        
        return output
    
    def fit(self, 
            train_loader,
            val_loader=None,
            epochs: int = 100,
            lr: float = 1e-3,
            device: str = "auto"):
        """Custom fit method for DeepONet data format."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_idx, batch_data in enumerate(train_loader):
                if len(batch_data) == 3:
                    # DeepONet format: (branch_input, trunk_input, target)
                    branch_input, trunk_input, target = batch_data
                    branch_input = branch_input.to(device)
                    trunk_input = trunk_input.to(device)
                    target = target.to(device)
                else:
                    # Fallback: assume standard (input, target) format
                    # Try to split input into branch and trunk components
                    input_data, target = batch_data
                    input_data = input_data.to(device)
                    target = target.to(device)
                    
                    # Assume input is concatenated [branch_data, coordinates]
                    # This is a simplification - real usage should provide proper format
                    branch_input = input_data
                    
                    # Create dummy trunk coordinates (spatial grid)
                    batch_size = input_data.shape[0]
                    if target.ndim > 1:
                        # For time-series data like Burgers: target is (batch, time, spatial)
                        # Take final time step as target
                        if target.ndim == 3:  # (batch, time, spatial)
                            target = target[:, -1, :]  # Take final time step
                        
                        n_points = target.shape[-1]
                        trunk_input = torch.linspace(0, 1, n_points).unsqueeze(0).unsqueeze(-1)
                        trunk_input = trunk_input.expand(batch_size, -1, -1).to(device)
                        target = target.unsqueeze(-1)  # Add output dimension
                    else:
                        # Single point evaluation
                        trunk_input = torch.zeros(batch_size, 1, 1).to(device)
                        target = target.unsqueeze(-1).unsqueeze(-1)
                
                optimizer.zero_grad()
                output = self(branch_input, trunk_input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if len(train_loader) > 0:
                train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_data in val_loader:
                        if len(batch_data) == 3:
                            branch_input, trunk_input, target = batch_data
                            branch_input = branch_input.to(device)
                            trunk_input = trunk_input.to(device)
                            target = target.to(device)
                        else:
                            input_data, target = batch_data
                            input_data = input_data.to(device)
                            target = target.to(device)
                            
                            branch_input = input_data
                            batch_size = input_data.shape[0]
                            if target.ndim > 1:
                                # Handle time-series data
                                if target.ndim == 3:  # (batch, time, spatial)
                                    target = target[:, -1, :]  # Take final time step
                                
                                n_points = target.shape[-1]
                                trunk_input = torch.linspace(0, 1, n_points).unsqueeze(0).unsqueeze(-1)
                                trunk_input = trunk_input.expand(batch_size, -1, -1).to(device)
                                target = target.unsqueeze(-1)
                            else:
                                trunk_input = torch.zeros(batch_size, 1, 1).to(device)
                                target = target.unsqueeze(-1).unsqueeze(-1)
                        
                        output = self(branch_input, trunk_input)
                        val_loss += criterion(output, target).item()
                
                if len(val_loader) > 0:
                    val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
        
        return history
    
    @classmethod
    def from_config(cls, config: dict) -> "ProbabilisticDeepONet":
        """Create ProbabilisticDeepONet from configuration."""
        return cls(**config)