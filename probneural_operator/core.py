"""
Core functionality for ProbNeural Operator Lab - Pure Python Implementation
==========================================================================

This module provides the essential functionality for uncertainty quantification
in neural operators using pure Python, enabling basic operation without external
dependencies like PyTorch or NumPy.

This is the Generation 1: MAKE IT WORK implementation.
"""

import math
import random
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
import json
import time

class MockTensor:
    """Mock tensor implementation for environments without PyTorch/NumPy."""
    
    def __init__(self, data: Union[List, float, int]):
        if isinstance(data, (int, float)):
            self.data = [data]
            self.shape = (1,)
        elif isinstance(data, list):
            self.data = data
            self.shape = (len(data),)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return MockTensor(self.data[idx])
    
    def __setitem__(self, idx, value):
        if isinstance(value, MockTensor):
            self.data[idx] = value.data[0] if len(value.data) == 1 else value.data
        else:
            self.data[idx] = value
    
    def mean(self):
        return MockTensor([sum(self.data) / len(self.data)])
    
    def std(self):
        mean_val = sum(self.data) / len(self.data)
        variance = sum((x - mean_val)**2 for x in self.data) / len(self.data)
        return MockTensor([math.sqrt(variance)])
    
    def sum(self):
        return MockTensor([sum(self.data)])
    
    def numpy(self):
        """Return data as list (numpy compatibility)."""
        return self.data
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor([a + b for a, b in zip(self.data, other.data)])
        else:
            return MockTensor([x + other for x in self.data])
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor([a * b for a, b in zip(self.data, other.data)])
        else:
            return MockTensor([x * other for x in self.data])
    
    def __repr__(self):
        return f"MockTensor({self.data})"

class BaseNeuralOperator(ABC):
    """Abstract base class for neural operators."""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_trained = False
        self.training_history = []
        
        # Mock parameters
        self.parameters = {
            'weights': [random.random() for _ in range(input_dim * output_dim)],
            'biases': [random.random() for _ in range(output_dim)]
        }
    
    @abstractmethod
    def forward(self, x: MockTensor) -> MockTensor:
        """Forward pass through the neural operator."""
        pass
    
    def train(self, X: List[MockTensor], y: List[MockTensor], 
              epochs: int = 100, lr: float = 0.001) -> Dict[str, Any]:
        """Train the neural operator."""
        print(f"Training {self.__class__.__name__} for {epochs} epochs...")
        
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            
            for i, (xi, yi) in enumerate(zip(X, y)):
                # Forward pass
                pred = self.forward(xi)
                
                # Compute loss (MSE)
                loss = sum((p - t)**2 for p, t in zip(pred.data, yi.data)) / len(pred.data)
                total_loss += loss
                
                # Mock gradient update
                for j in range(len(self.parameters['weights'])):
                    self.parameters['weights'][j] -= lr * random.gauss(0, 0.01)
            
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            
            if epoch % (epochs // 10) == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        self.is_trained = True
        self.training_history = losses
        
        return {
            "final_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "epochs": epochs,
            "convergence": losses[-1] < losses[0] * 0.1
        }
    
    def predict(self, X: List[MockTensor]) -> List[MockTensor]:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return [self.forward(x) for x in X]

class ProbabilisticFNO(BaseNeuralOperator):
    """Probabilistic Fourier Neural Operator - Mock Implementation."""
    
    def __init__(self, modes: int = 12, width: int = 32, depth: int = 4, 
                 input_dim: int = 64, output_dim: int = 64, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.modes = modes
        self.width = width
        self.depth = depth
        
        # Additional parameters for FNO
        self.fourier_weights = [random.random() for _ in range(modes * width)]
        
    def forward(self, x: MockTensor) -> MockTensor:
        """Forward pass through Fourier layers."""
        # Mock Fourier Neural Operator computation
        
        # 1. Lift to higher dimension
        lifted = MockTensor([xi * w for xi, w in zip(x.data, self.parameters['weights'][:len(x.data)])])
        
        # 2. Fourier layers
        current = lifted
        for layer in range(self.depth):
            # Mock spectral convolution
            fourier_part = MockTensor([
                sum(current.data[i:i+self.modes]) / self.modes 
                for i in range(0, len(current.data), max(1, len(current.data)//self.modes))
            ])
            
            # Mock local convolution
            local_part = MockTensor([x + random.gauss(0, 0.01) for x in current.data])
            
            # Combine and activate
            current = MockTensor([f + l for f, l in zip(fourier_part.data, local_part.data[:len(fourier_part.data)])])
        
        # 3. Project to output dimension
        output = MockTensor([
            sum(current.data[i:i+max(1, len(current.data)//self.output_dim)]) / max(1, len(current.data)//self.output_dim)
            for i in range(0, len(current.data), max(1, len(current.data)//self.output_dim))
        ][:self.output_dim])
        
        return output

class ProbabilisticDeepONet(BaseNeuralOperator):
    """Probabilistic Deep Operator Network - Mock Implementation."""
    
    def __init__(self, branch_dim: int = 64, trunk_dim: int = 64, hidden_dim: int = 128,
                 input_dim: int = 64, output_dim: int = 64, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.hidden_dim = hidden_dim
        
        # Branch and trunk networks
        self.branch_weights = [random.random() for _ in range(branch_dim * hidden_dim)]
        self.trunk_weights = [random.random() for _ in range(trunk_dim * hidden_dim)]
    
    def forward(self, x: MockTensor) -> MockTensor:
        """Forward pass through branch-trunk architecture."""
        # Mock DeepONet computation
        
        # 1. Branch network (processes function input)
        branch_input = x.data[:self.branch_dim] if len(x.data) >= self.branch_dim else x.data + [0.0] * (self.branch_dim - len(x.data))
        branch_output = [
            sum(bi * bw for bi, bw in zip(branch_input, self.branch_weights[i:i+self.branch_dim]))
            for i in range(0, len(self.branch_weights), self.branch_dim)
        ]
        
        # 2. Trunk network (processes coordinates) 
        trunk_input = [float(i) / self.trunk_dim for i in range(self.trunk_dim)]
        trunk_output = [
            sum(ti * tw for ti, tw in zip(trunk_input, self.trunk_weights[i:i+self.trunk_dim]))
            for i in range(0, len(self.trunk_weights), self.trunk_dim)
        ]
        
        # 3. Combine branch and trunk
        min_len = min(len(branch_output), len(trunk_output))
        combined = [b * t for b, t in zip(branch_output[:min_len], trunk_output[:min_len])]
        
        # 4. Project to output dimension
        output = MockTensor(combined[:self.output_dim] + [0.0] * max(0, self.output_dim - len(combined)))
        
        return output

class BaseUncertaintyMethod(ABC):
    """Abstract base class for uncertainty quantification methods."""
    
    def __init__(self, model: BaseNeuralOperator):
        self.model = model
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: List[MockTensor], y: List[MockTensor]) -> Dict[str, Any]:
        """Fit the uncertainty method."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: List[MockTensor]) -> Tuple[List[MockTensor], List[MockTensor]]:
        """Predict with uncertainty estimates."""
        pass

class LinearizedLaplace(BaseUncertaintyMethod):
    """Linearized Laplace approximation for uncertainty quantification."""
    
    def __init__(self, model: BaseNeuralOperator, prior_precision: float = 1.0):
        super().__init__(model)
        self.prior_precision = prior_precision
        self.posterior_precision = None
        
    def fit(self, X: List[MockTensor], y: List[MockTensor]) -> Dict[str, Any]:
        """Fit Laplace approximation."""
        print(f"Fitting Laplace approximation with prior precision {self.prior_precision}...")
        
        # First train the base model
        training_result = self.model.train(X, y)
        
        # Mock Hessian computation (diagonal approximation)
        n_params = len(self.model.parameters['weights']) + len(self.model.parameters['biases'])
        self.posterior_precision = [self.prior_precision + random.random() for _ in range(n_params)]
        
        self.is_fitted = True
        
        return {
            **training_result,
            "laplace_fitted": True,
            "posterior_precision_mean": sum(self.posterior_precision) / len(self.posterior_precision)
        }
    
    def predict_with_uncertainty(self, X: List[MockTensor]) -> Tuple[List[MockTensor], List[MockTensor]]:
        """Predict with Laplace uncertainty."""
        if not self.is_fitted:
            raise ValueError("Laplace method must be fitted before prediction")
        
        # Base predictions
        mean_predictions = self.model.predict(X)
        
        # Uncertainty from posterior precision
        base_uncertainty = 1.0 / (sum(self.posterior_precision) / len(self.posterior_precision))
        
        uncertainties = []
        for pred in mean_predictions:
            # Scale uncertainty by prediction magnitude
            pred_magnitude = sum(abs(p) for p in pred.data) / len(pred.data)
            uncertainty = MockTensor([base_uncertainty * (1 + pred_magnitude) for _ in pred.data])
            uncertainties.append(uncertainty)
        
        return mean_predictions, uncertainties

class DeepEnsemble(BaseUncertaintyMethod):
    """Deep ensemble for uncertainty quantification."""
    
    def __init__(self, model_class, n_ensemble: int = 5, **model_kwargs):
        self.model_class = model_class
        self.n_ensemble = n_ensemble
        self.model_kwargs = model_kwargs
        self.ensemble_models = []
        
    def fit(self, X: List[MockTensor], y: List[MockTensor]) -> Dict[str, Any]:
        """Train ensemble of models."""
        print(f"Training ensemble of {self.n_ensemble} models...")
        
        ensemble_results = []
        
        for i in range(self.n_ensemble):
            print(f"  Training ensemble member {i+1}/{self.n_ensemble}")
            
            # Create model with different initialization
            model = self.model_class(**self.model_kwargs)
            
            # Add noise to initialization for diversity
            for j in range(len(model.parameters['weights'])):
                model.parameters['weights'][j] += random.gauss(0, 0.1)
            
            # Train model
            result = model.train(X, y)
            
            self.ensemble_models.append(model)
            ensemble_results.append(result)
        
        self.is_fitted = True
        
        # Aggregate results
        return {
            "ensemble_size": self.n_ensemble,
            "individual_results": ensemble_results,
            "mean_final_loss": sum(r["final_loss"] for r in ensemble_results) / len(ensemble_results),
            "ensemble_fitted": True
        }
    
    def predict_with_uncertainty(self, X: List[MockTensor]) -> Tuple[List[MockTensor], List[MockTensor]]:
        """Predict with ensemble uncertainty."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all ensemble members
        all_predictions = []
        for model in self.ensemble_models:
            preds = model.predict(X)
            all_predictions.append(preds)
        
        # Compute ensemble mean and std
        mean_predictions = []
        std_predictions = []
        
        for i in range(len(X)):
            # Collect predictions for sample i from all models
            sample_predictions = [all_predictions[j][i].data for j in range(self.n_ensemble)]
            
            # Compute mean across ensemble
            mean_pred = []
            for dim in range(len(sample_predictions[0])):
                dim_values = [pred[dim] for pred in sample_predictions]
                mean_pred.append(sum(dim_values) / len(dim_values))
            
            # Compute std across ensemble
            std_pred = []
            for dim in range(len(sample_predictions[0])):
                dim_values = [pred[dim] for pred in sample_predictions]
                mean_val = sum(dim_values) / len(dim_values)
                variance = sum((v - mean_val)**2 for v in dim_values) / len(dim_values)
                std_pred.append(math.sqrt(variance))
            
            mean_predictions.append(MockTensor(mean_pred))
            std_predictions.append(MockTensor(std_pred))
        
        return mean_predictions, std_predictions

class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, name: str, n_samples: int, input_dim: int, output_dim: int):
        self.name = name
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Generate mock data based on dataset type
        self.X, self.y = self._generate_data()
    
    def _generate_data(self) -> Tuple[List[MockTensor], List[MockTensor]]:
        """Generate mock PDE data."""
        X_data = []
        y_data = []
        
        for i in range(self.n_samples):
            if self.name == "burgers":
                # Mock Burgers equation data
                x = MockTensor([math.sin(2 * math.pi * j / self.input_dim) + 
                              random.gauss(0, 0.1) for j in range(self.input_dim)])
                y = MockTensor([0.9 * math.sin(2 * math.pi * j / self.output_dim) + 
                              random.gauss(0, 0.05) for j in range(self.output_dim)])
            
            elif self.name == "navier_stokes":
                # Mock Navier-Stokes data
                x = MockTensor([math.cos(2 * math.pi * j / self.input_dim) * 
                              math.sin(math.pi * j / self.input_dim) + 
                              random.gauss(0, 0.1) for j in range(self.input_dim)])
                y = MockTensor([0.8 * math.cos(2 * math.pi * j / self.output_dim) + 
                              random.gauss(0, 0.05) for j in range(self.output_dim)])
            
            elif self.name == "darcy_flow":
                # Mock Darcy flow data
                x = MockTensor([random.gauss(0, 1) for _ in range(self.input_dim)])
                y = MockTensor([sum(x.data[max(0, j-2):j+3]) / 5 + 
                              random.gauss(0, 0.05) for j in range(self.output_dim)])
            
            else:
                # Generic data
                x = MockTensor([random.gauss(0, 1) for _ in range(self.input_dim)])
                y = MockTensor([sum(x.data) / len(x.data) + 
                              random.gauss(0, 0.1) for _ in range(self.output_dim)])
            
            X_data.append(x)
            y_data.append(y)
        
        return X_data, y_data
    
    def train_test_split(self, test_ratio: float = 0.2) -> Dict[str, List[MockTensor]]:
        """Split data into train and test sets."""
        n_test = int(self.n_samples * test_ratio)
        n_train = self.n_samples - n_test
        
        return {
            "X_train": self.X[:n_train],
            "y_train": self.y[:n_train],
            "X_test": self.X[n_train:],
            "y_test": self.y[n_train:]
        }

class UncertaintyMetrics:
    """Metrics for evaluating uncertainty quality."""
    
    @staticmethod
    def mean_squared_error(y_true: List[MockTensor], y_pred: List[MockTensor]) -> float:
        """Compute Mean Squared Error."""
        total_mse = 0.0
        total_elements = 0
        
        for yt, yp in zip(y_true, y_pred):
            for yt_val, yp_val in zip(yt.data, yp.data):
                total_mse += (yt_val - yp_val) ** 2
                total_elements += 1
        
        return total_mse / total_elements if total_elements > 0 else 0.0
    
    @staticmethod
    def negative_log_likelihood(y_true: List[MockTensor], y_pred: List[MockTensor], 
                               y_std: List[MockTensor]) -> float:
        """Compute Negative Log-Likelihood."""
        total_nll = 0.0
        total_elements = 0
        
        for yt, yp, ys in zip(y_true, y_pred, y_std):
            for yt_val, yp_val, ys_val in zip(yt.data, yp.data, ys.data):
                # Gaussian NLL: 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
                variance = max(ys_val ** 2, 1e-6)  # Avoid log(0)
                nll = 0.5 * (math.log(2 * math.pi) + math.log(variance) + 
                            (yt_val - yp_val) ** 2 / variance)
                total_nll += nll
                total_elements += 1
        
        return total_nll / total_elements if total_elements > 0 else 0.0
    
    @staticmethod
    def coverage_probability(y_true: List[MockTensor], y_pred: List[MockTensor], 
                           y_std: List[MockTensor], confidence: float = 0.95) -> float:
        """Compute empirical coverage probability."""
        z_score = 1.96 if confidence == 0.95 else 2.58  # Approximation
        
        total_coverage = 0
        total_elements = 0
        
        for yt, yp, ys in zip(y_true, y_pred, y_std):
            for yt_val, yp_val, ys_val in zip(yt.data, yp.data, ys.data):
                lower = yp_val - z_score * ys_val
                upper = yp_val + z_score * ys_val
                
                if lower <= yt_val <= upper:
                    total_coverage += 1
                total_elements += 1
        
        return total_coverage / total_elements if total_elements > 0 else 0.0

def demo_basic_functionality():
    """Demonstrate basic functionality of the framework."""
    print("🔬 ProbNeural Operator Lab - Generation 1 Demo")
    print("=" * 50)
    
    # Create dataset
    print("📊 Creating mock dataset...")
    dataset = MockDataset("burgers", n_samples=100, input_dim=32, output_dim=32)
    split = dataset.train_test_split(test_ratio=0.2)
    
    print(f"   Dataset: {dataset.name}")
    print(f"   Training samples: {len(split['X_train'])}")
    print(f"   Test samples: {len(split['X_test'])}")
    
    # Create and train FNO
    print("\n🚀 Training Probabilistic FNO...")
    fno = ProbabilisticFNO(modes=8, width=16, depth=2, input_dim=32, output_dim=32)
    fno_result = fno.train(split['X_train'], split['y_train'], epochs=50)
    
    print(f"   Final loss: {fno_result['final_loss']:.4f}")
    print(f"   Converged: {fno_result['convergence']}")
    
    # Create and fit Laplace approximation
    print("\n📈 Fitting Laplace approximation...")
    laplace = LinearizedLaplace(fno, prior_precision=1.0)
    laplace_result = laplace.fit(split['X_train'], split['y_train'])
    
    print(f"   Posterior precision: {laplace_result['posterior_precision_mean']:.4f}")
    
    # Make predictions with uncertainty
    print("\n🎯 Making predictions with uncertainty...")
    mean_pred, std_pred = laplace.predict_with_uncertainty(split['X_test'])
    
    # Evaluate metrics
    print("\n📋 Evaluating uncertainty metrics...")
    mse = UncertaintyMetrics.mean_squared_error(split['y_test'], mean_pred)
    nll = UncertaintyMetrics.negative_log_likelihood(split['y_test'], mean_pred, std_pred)
    coverage = UncertaintyMetrics.coverage_probability(split['y_test'], mean_pred, std_pred)
    
    print(f"   MSE: {mse:.4f}")
    print(f"   NLL: {nll:.4f}")
    print(f"   Coverage@95%: {coverage:.3f}")
    
    # Test ensemble method
    print("\n🎭 Training Deep Ensemble...")
    ensemble = DeepEnsemble(ProbabilisticFNO, n_ensemble=3, 
                           modes=8, width=16, depth=2, input_dim=32, output_dim=32)
    ensemble_result = ensemble.fit(split['X_train'], split['y_train'])
    
    print(f"   Ensemble size: {ensemble_result['ensemble_size']}")
    print(f"   Mean final loss: {ensemble_result['mean_final_loss']:.4f}")
    
    # Ensemble predictions
    ens_mean, ens_std = ensemble.predict_with_uncertainty(split['X_test'])
    ens_mse = UncertaintyMetrics.mean_squared_error(split['y_test'], ens_mean)
    ens_nll = UncertaintyMetrics.negative_log_likelihood(split['y_test'], ens_mean, ens_std)
    ens_coverage = UncertaintyMetrics.coverage_probability(split['y_test'], ens_mean, ens_std)
    
    print(f"   Ensemble MSE: {ens_mse:.4f}")
    print(f"   Ensemble NLL: {ens_nll:.4f}")
    print(f"   Ensemble Coverage@95%: {ens_coverage:.3f}")
    
    print("\n✅ Generation 1 demo complete!")
    
    return {
        "fno_result": fno_result,
        "laplace_result": laplace_result,
        "ensemble_result": ensemble_result,
        "metrics": {
            "laplace": {"mse": mse, "nll": nll, "coverage": coverage},
            "ensemble": {"mse": ens_mse, "nll": ens_nll, "coverage": ens_coverage}
        }
    }

if __name__ == "__main__":
    demo_basic_functionality()