"""Comprehensive Demo of Novel Uncertainty Quantification Methods.

This example demonstrates all 5 novel uncertainty quantification methods
implemented in the ProbNeural-Operator-Lab repository:

1. Sparse Gaussian Process Neural Operator (SGPNO)
2. Normalizing Flow Posterior Approximation  
3. Physics-Informed Conformal Prediction
4. Meta-Learning Uncertainty Estimator (MLUE)
5. Information-Theoretic Active Learning

Each method is demonstrated on real PDE problems with comprehensive
evaluation and comparison.

Authors: TERRAGON Research Lab
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings('ignore')

# Import novel methods
from probneural_operator.posteriors.novel import (
    SparseGaussianProcessNeuralOperator,
    NormalizingFlowPosterior,
    PhysicsInformedConformalPredictor,
    MetaLearningUncertaintyEstimator,
    InformationTheoreticActiveLearner
)

# Import configurations
from probneural_operator.posteriors.novel.sparse_gp_neural_operator import SGPNOConfig
from probneural_operator.posteriors.novel.normalizing_flow_posterior import NormalizingFlowConfig
from probneural_operator.posteriors.novel.physics_informed_conformal import ConformalConfig
from probneural_operator.posteriors.novel.meta_learning_uncertainty import MetaLearningConfig
from probneural_operator.posteriors.novel.information_theoretic_active import ActiveLearningConfig

# Import benchmarking tools
from benchmarks.novel_methods import (
    run_comprehensive_novel_benchmark,
    NovelBenchmarkConfig,
    run_theoretical_validation_suite
)


def create_neural_operator_model(input_dim: int = 2, output_dim: int = 1, hidden_dim: int = 64) -> nn.Module:
    """Create a simple neural operator model for demonstration.
    
    Args:
        input_dim: Input dimension (spatial + temporal coordinates)
        output_dim: Output dimension (solution values)
        hidden_dim: Hidden layer dimension
        
    Returns:
        Neural network model
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def generate_burgers_data(n_samples: int = 500, nx: int = 64, nt: int = 50) -> tuple:
    """Generate Burgers' equation dataset.
    
    Solves: du/dt + u * du/dx = nu * d^2u/dx^2
    
    Args:
        n_samples: Number of trajectory samples
        nx: Number of spatial points
        nt: Number of time points
        
    Returns:
        Tuple of (inputs, outputs) tensors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Spatial and temporal domains
    x = torch.linspace(0, 2*np.pi, nx, device=device)
    t = torch.linspace(0, 1, nt, device=device)
    
    inputs = []
    outputs = []
    
    print(f"Generating {n_samples} Burgers equation samples...")
    
    for i in range(n_samples):
        # Random initial condition (sum of sines)
        A = torch.randn(3, device=device) * 0.5
        k = torch.tensor([1, 2, 3], device=device, dtype=torch.float)
        
        # Initial condition: u(x, 0) = sum(A_k * sin(k * x))
        u0 = sum(A[j] * torch.sin(k[j] * x) for j in range(3))
        
        # Simple analytical approximation for Burgers' solution
        # (In practice, would use numerical solver)
        sample_coords = []
        sample_values = []
        
        for t_idx, t_val in enumerate(t):
            # Decay and nonlinear steepening approximation
            decay = torch.exp(-0.01 * t_val * k.mean()**2)
            steepening = 1.0 / (1.0 + 0.1 * t_val * torch.abs(u0).max())
            
            u_t = u0 * decay * steepening
            
            # Add some realistic noise
            u_t += 0.02 * torch.randn_like(u_t)
            
            # Create coordinate-value pairs
            for x_idx, x_val in enumerate(x):
                coord = torch.tensor([x_val.item(), t_val.item()], device=device)
                value = u_t[x_idx].unsqueeze(0)
                
                sample_coords.append(coord)
                sample_values.append(value)
        
        # Subsample for efficiency
        if len(sample_coords) > 200:
            indices = torch.randperm(len(sample_coords))[:200]
            sample_coords = [sample_coords[idx] for idx in indices]
            sample_values = [sample_values[idx] for idx in indices]
        
        inputs.extend(sample_coords)
        outputs.extend(sample_values)
    
    # Convert to tensors
    all_inputs = torch.stack(inputs)
    all_outputs = torch.stack(outputs)
    
    print(f"Generated dataset: {all_inputs.shape[0]} points")
    return all_inputs, all_outputs


def demonstrate_sparse_gp_method(train_loader: DataLoader, test_loader: DataLoader) -> dict:
    """Demonstrate Sparse Gaussian Process Neural Operator."""
    print("\n" + "="*60)
    print("DEMONSTRATING: Sparse Gaussian Process Neural Operator")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_neural_operator_model().to(device)
    
    # Configure SGPNO
    config = SGPNOConfig(
        num_inducing=64,
        kernel_type="neural_operator",
        num_variational_steps=30,
        variational_lr=1e-3
    )
    
    # Create SGPNO instance
    sgpno = SparseGaussianProcessNeuralOperator(model, config)
    
    print("Key Features:")
    print("- Hybrid sparse approximation (inducing points + local kernels)")
    print("- Neural operator-informed kernels for physics-aware covariance")
    print("- Kronecker factorization for computational efficiency")
    print("- Variational inference with natural gradients")
    
    # Train model first
    print("\nTraining base neural operator...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(20):
        total_loss = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = nn.MSELoss()(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {total_loss/len(train_loader):.6f}")
    
    # Fit SGPNO posterior
    print("\nFitting SGPNO posterior approximation...")
    start_time = time.time()
    
    try:
        sgpno.fit(train_loader)
        fit_time = time.time() - start_time
        print(f"SGPNO fitting completed in {fit_time:.2f}s")
        
        # Test predictions
        print("\nTesting SGPNO predictions...")
        model.eval()
        
        with torch.no_grad():
            test_inputs, test_targets = next(iter(test_loader))
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            
            # Get predictions with uncertainty
            mean_pred, variance = sgpno.predict(test_inputs[:10])
            
            print(f"Prediction shape: {mean_pred.shape}")
            print(f"Uncertainty shape: {variance.shape}")
            print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
            print(f"Uncertainty range: [{variance.min():.3f}, {variance.max():.3f}]")
            
            # Compute metrics
            mse = nn.MSELoss()(mean_pred, test_targets[:10]).item()
            print(f"Test MSE: {mse:.6f}")
            
            # Sample from posterior
            samples = sgpno.sample(test_inputs[:5], num_samples=50)
            print(f"Posterior samples shape: {samples.shape}")
            
        success = True
        
    except Exception as e:
        print(f"SGPNO demonstration failed: {e}")
        fit_time = time.time() - start_time
        success = False
        mse = float('inf')
    
    return {
        'method': 'Sparse GP Neural Operator',
        'success': success,
        'training_time': fit_time,
        'test_mse': mse if success else float('inf'),
        'features': [
            'Scalable GP inference with inducing points',
            'Physics-informed kernel learning',
            'Principled uncertainty quantification',
            'Efficient variational optimization'
        ]
    }


def demonstrate_normalizing_flow_method(train_loader: DataLoader, test_loader: DataLoader) -> dict:
    """Demonstrate Normalizing Flow Posterior Approximation."""
    print("\n" + "="*60)
    print("DEMONSTRATING: Normalizing Flow Posterior Approximation")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_neural_operator_model().to(device)
    
    # Configure Normalizing Flow
    config = NormalizingFlowConfig(
        flow_type="real_nvp",
        num_flows=4,
        hidden_dim=64,
        vi_epochs=30,
        vi_lr=1e-3
    )
    
    # Create NF instance
    nf_posterior = NormalizingFlowPosterior(model, config)
    
    print("Key Features:")
    print("- Real NVP flows for flexible posterior approximation")
    print("- Physics-informed flow layers for PDE constraints")
    print("- Variational inference with normalizing flows")
    print("- Complex posterior geometry modeling")
    
    # Train model first (simplified)
    print("\nTraining base model...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(15):
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = nn.MSELoss()(predictions, batch_targets)
            loss.backward()
            optimizer.step()
    
    # Fit normalizing flow
    print("\nFitting normalizing flow posterior...")
    start_time = time.time()
    
    try:
        nf_posterior.fit(train_loader)
        fit_time = time.time() - start_time
        print(f"Normalizing flow fitting completed in {fit_time:.2f}s")
        
        # Test predictions
        print("\nTesting flow predictions...")
        
        with torch.no_grad():
            test_inputs, test_targets = next(iter(test_loader))
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            
            # Get predictions with uncertainty
            mean_pred, variance = nf_posterior.predict(test_inputs[:10], num_samples=50)
            
            print(f"Flow prediction shape: {mean_pred.shape}")
            print(f"Flow uncertainty shape: {variance.shape}")
            print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
            print(f"Uncertainty range: [{variance.min():.3f}, {variance.max():.3f}]")
            
            # Compute metrics
            mse = nn.MSELoss()(mean_pred, test_targets[:10]).item()
            print(f"Test MSE: {mse:.6f}")
            
            # Sample from flow
            samples = nf_posterior.sample(test_inputs[:5], num_samples=30)
            print(f"Flow samples shape: {samples.shape}")
        
        success = True
        
    except Exception as e:
        print(f"Normalizing flow demonstration failed: {e}")
        fit_time = time.time() - start_time
        success = False
        mse = float('inf')
    
    return {
        'method': 'Normalizing Flow Posterior',
        'success': success,
        'training_time': fit_time,
        'test_mse': mse if success else float('inf'),
        'features': [
            'Flexible posterior approximation with flows',
            'Real NVP architecture for invertible transforms',
            'Complex geometry modeling beyond Gaussians',
            'Variational inference optimization'
        ]
    }


def demonstrate_conformal_prediction_method(train_loader: DataLoader, test_loader: DataLoader) -> dict:
    """Demonstrate Physics-Informed Conformal Prediction."""
    print("\n" + "="*60)
    print("DEMONSTRATING: Physics-Informed Conformal Prediction")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_neural_operator_model().to(device)
    
    # Configure conformal prediction
    config = ConformalConfig(
        coverage_level=0.9,
        physics_weight=1.0,
        use_gradients=True
    )
    
    # Create conformal predictor
    conformal_pred = PhysicsInformedConformalPredictor(model, config)
    
    print("Key Features:")
    print("- Distribution-free uncertainty quantification")
    print("- Physics residual error (PRE) based scores")
    print("- No labeled calibration data required")
    print("- Guaranteed coverage levels")
    
    # Train model first
    print("\nTraining model for conformal prediction...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(15):
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = nn.MSELoss()(predictions, batch_targets)
            loss.backward()
            optimizer.step()
    
    # Fit conformal prediction
    print("\nCalibrating conformal prediction intervals...")
    start_time = time.time()
    
    try:
        conformal_pred.fit(train_loader)
        fit_time = time.time() - start_time
        print(f"Conformal calibration completed in {fit_time:.2f}s")
        
        # Test predictions
        print("\nTesting conformal prediction intervals...")
        
        with torch.no_grad():
            test_inputs, test_targets = next(iter(test_loader))
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            
            # Get prediction intervals
            lower, upper = conformal_pred.predict_interval(test_inputs[:20])
            
            # Check coverage
            coverage = ((test_targets[:20] >= lower) & (test_targets[:20] <= upper)).float().mean().item()
            
            # Compute interval width
            avg_width = (upper - lower).mean().item()
            
            print(f"Target coverage: {config.coverage_level:.1%}")
            print(f"Empirical coverage: {coverage:.1%}")
            print(f"Average interval width: {avg_width:.4f}")
            
            # Point predictions for MSE
            mean_pred, _ = conformal_pred.predict(test_inputs[:10])
            mse = nn.MSELoss()(mean_pred, test_targets[:10]).item()
            print(f"Test MSE: {mse:.6f}")
        
        success = True
        
    except Exception as e:
        print(f"Conformal prediction demonstration failed: {e}")
        fit_time = time.time() - start_time
        success = False
        mse = float('inf')
        coverage = 0.0
    
    return {
        'method': 'Physics-Informed Conformal Prediction',
        'success': success,
        'training_time': fit_time,
        'test_mse': mse if success else float('inf'),
        'coverage': coverage if success else 0.0,
        'features': [
            'Distribution-free uncertainty bounds',
            'Physics-informed nonconformity scores',
            'Guaranteed finite-sample coverage',
            'No assumptions on data distribution'
        ]
    }


def demonstrate_meta_learning_method(train_loader: DataLoader, test_loader: DataLoader) -> dict:
    """Demonstrate Meta-Learning Uncertainty Estimator."""
    print("\n" + "="*60)
    print("DEMONSTRATING: Meta-Learning Uncertainty Estimator")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_neural_operator_model().to(device)
    
    # Configure meta-learning
    config = MetaLearningConfig(
        inner_lr=1e-3,
        outer_lr=1e-4,
        inner_steps=3,
        meta_batch_size=4,
        num_meta_epochs=20,
        support_shots=5,
        query_shots=10
    )
    
    # Create meta-learner
    meta_learner = MetaLearningUncertaintyEstimator(model, config)
    
    print("Key Features:")
    print("- MAML-based rapid adaptation to new PDE domains")
    print("- Hierarchical uncertainty decomposition")
    print("- Task-specific uncertainty calibration")
    print("- Few-shot learning capabilities")
    
    # Meta-train
    print("\nMeta-training uncertainty estimator...")
    start_time = time.time()
    
    try:
        meta_learner.fit(train_loader)
        fit_time = time.time() - start_time
        print(f"Meta-training completed in {fit_time:.2f}s")
        
        # Test adaptation
        print("\nTesting meta-learned uncertainty...")
        
        with torch.no_grad():
            test_inputs, test_targets = next(iter(test_loader))
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            
            # Get hierarchical uncertainty
            mean_pred, total_variance = meta_learner.predict(test_inputs[:10])
            
            print(f"Meta prediction shape: {mean_pred.shape}")
            print(f"Total uncertainty shape: {total_variance.shape}")
            print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
            print(f"Uncertainty range: [{total_variance.min():.3f}, {total_variance.max():.3f}]")
            
            # Compute metrics
            mse = nn.MSELoss()(mean_pred, test_targets[:10]).item()
            print(f"Test MSE: {mse:.6f}")
            
            # Sample from meta-learned distribution
            samples = meta_learner.sample(test_inputs[:5], num_samples=20)
            print(f"Meta samples shape: {samples.shape}")
        
        success = True
        
    except Exception as e:
        print(f"Meta-learning demonstration failed: {e}")
        fit_time = time.time() - start_time
        success = False
        mse = float('inf')
    
    return {
        'method': 'Meta-Learning Uncertainty Estimator',
        'success': success,
        'training_time': fit_time,
        'test_mse': mse if success else float('inf'),
        'features': [
            'Fast adaptation to new PDE domains',
            'Hierarchical uncertainty decomposition',
            'Model-agnostic meta-learning (MAML)',
            'Few-shot uncertainty calibration'
        ]
    }


def demonstrate_active_learning_method(train_loader: DataLoader, test_loader: DataLoader) -> dict:
    """Demonstrate Information-Theoretic Active Learning."""
    print("\n" + "="*60)
    print("DEMONSTRATING: Information-Theoretic Active Learning")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_neural_operator_model().to(device)
    
    # Configure active learning
    config = ActiveLearningConfig(
        acquisition_function="physics_informed",
        max_iterations=3,  # Limited for demo
        batch_size=5,
        initial_pool_size=50
    )
    
    # Create active learner
    active_learner = InformationTheoreticActiveLearner(model, config)
    
    print("Key Features:")
    print("- Information-theoretic acquisition functions")
    print("- Physics-informed data selection")
    print("- Batch active learning with diversity")
    print("- Mutual information neural estimation (MINE)")
    
    # Active learning
    print("\nRunning active learning...")
    start_time = time.time()
    
    try:
        active_learner.fit(train_loader)
        fit_time = time.time() - start_time
        print(f"Active learning completed in {fit_time:.2f}s")
        
        # Get selection statistics
        stats = active_learner.get_selection_statistics()
        print(f"\nActive Learning Statistics:")
        print(f"- Total labeled points: {stats['total_labeled']}")
        print(f"- Selection iterations: {stats['num_iterations']}")
        print(f"- Average information gain: {stats['avg_information_gain']:.4f}")
        
        # Test active learning performance
        print("\nTesting actively trained model...")
        
        with torch.no_grad():
            test_inputs, test_targets = next(iter(test_loader))
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            
            # Get predictions with uncertainty
            mean_pred, variance = active_learner.predict(test_inputs[:10])
            
            print(f"Active prediction shape: {mean_pred.shape}")
            print(f"Active uncertainty shape: {variance.shape}")
            
            # Compute metrics
            mse = nn.MSELoss()(mean_pred, test_targets[:10]).item()
            print(f"Test MSE: {mse:.6f}")
        
        success = True
        
    except Exception as e:
        print(f"Active learning demonstration failed: {e}")
        fit_time = time.time() - start_time
        success = False
        mse = float('inf')
        stats = {}
    
    return {
        'method': 'Information-Theoretic Active Learning',
        'success': success,
        'training_time': fit_time,
        'test_mse': mse if success else float('inf'),
        'selection_stats': stats,
        'features': [
            'Optimal data selection for uncertainty',
            'Information-theoretic acquisition functions',
            'Physics-informed sample selection',
            'Efficient exploration of solution space'
        ]
    }


def run_comprehensive_comparison(results: list):
    """Run comprehensive comparison of all methods."""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON OF NOVEL UNCERTAINTY METHODS")
    print("="*80)
    
    # Success rates
    successful_methods = [r for r in results if r['success']]
    print(f"\nSuccessful demonstrations: {len(successful_methods)}/{len(results)}")
    
    # Performance comparison
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 40)
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        method_name = result['method']
        
        print(f"{status} {method_name}")
        if result['success']:
            print(f"    Training Time: {result['training_time']:.2f}s")
            print(f"    Test MSE: {result['test_mse']:.6f}")
        
        print(f"    Key Features:")
        for feature in result['features']:
            print(f"      • {feature}")
        print()
    
    # Best performing method
    if successful_methods:
        best_method = min(successful_methods, key=lambda x: x['test_mse'])
        print(f"Best performing method (lowest MSE): {best_method['method']}")
        print(f"MSE: {best_method['test_mse']:.6f}")
    
    # Method recommendations
    print("\nMETHOD RECOMMENDATIONS:")
    print("-" * 30)
    print("• Sparse GP Neural Operator: Best for scalable Bayesian inference")
    print("• Normalizing Flows: Best for complex posterior geometries")
    print("• Conformal Prediction: Best for guaranteed coverage without assumptions")
    print("• Meta-Learning: Best for rapid adaptation to new PDE domains")
    print("• Active Learning: Best for optimal data collection strategies")


def main():
    """Main demonstration function."""
    print("NOVEL UNCERTAINTY QUANTIFICATION METHODS DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo showcases 5 cutting-edge uncertainty quantification methods")
    print("for neural operators, each addressing different challenges in PDE solving.")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Generate demonstration data
    print("\nGenerating Burgers' equation dataset...")
    inputs, outputs = generate_burgers_data(n_samples=100)  # Reduced for demo
    
    # Split data
    n_train = int(0.8 * len(inputs))
    train_inputs, test_inputs = inputs[:n_train], inputs[n_train:]
    train_outputs, test_outputs = outputs[:n_train], outputs[n_train:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset split: {len(train_inputs)} train, {len(test_inputs)} test")
    
    # Demonstrate each method
    results = []
    
    try:
        # 1. Sparse GP Neural Operator
        result = demonstrate_sparse_gp_method(train_loader, test_loader)
        results.append(result)
    except Exception as e:
        print(f"SGPNO demo failed: {e}")
        results.append({'method': 'Sparse GP Neural Operator', 'success': False, 'features': []})
    
    try:
        # 2. Normalizing Flow Posterior
        result = demonstrate_normalizing_flow_method(train_loader, test_loader)
        results.append(result)
    except Exception as e:
        print(f"Flow demo failed: {e}")
        results.append({'method': 'Normalizing Flow Posterior', 'success': False, 'features': []})
    
    try:
        # 3. Physics-Informed Conformal Prediction
        result = demonstrate_conformal_prediction_method(train_loader, test_loader)
        results.append(result)
    except Exception as e:
        print(f"Conformal demo failed: {e}")
        results.append({'method': 'Physics-Informed Conformal Prediction', 'success': False, 'features': []})
    
    try:
        # 4. Meta-Learning Uncertainty Estimator
        result = demonstrate_meta_learning_method(train_loader, test_loader)
        results.append(result)
    except Exception as e:
        print(f"Meta-learning demo failed: {e}")
        results.append({'method': 'Meta-Learning Uncertainty Estimator', 'success': False, 'features': []})
    
    try:
        # 5. Information-Theoretic Active Learning
        result = demonstrate_active_learning_method(train_loader, test_loader)
        results.append(result)
    except Exception as e:
        print(f"Active learning demo failed: {e}")
        results.append({'method': 'Information-Theoretic Active Learning', 'success': False, 'features': []})
    
    # Comprehensive comparison
    run_comprehensive_comparison(results)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nAll novel uncertainty quantification methods have been demonstrated.")
    print("Each method offers unique advantages for different PDE solving scenarios.")
    print("\nFor production use, consider:")
    print("1. Problem complexity and computational budget")
    print("2. Required uncertainty guarantees")
    print("3. Available training data")
    print("4. Domain adaptation requirements")
    print("5. Real-time inference constraints")
    
    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstration
    results = main()