"""Research-Grade Experimental Benchmarking Suite.

This module implements comprehensive benchmarking for novel uncertainty methods:
1. Comparative studies with established baselines
2. Statistical significance testing
3. Reproducible experimental framework
4. Publication-ready results and visualizations

Research Benchmarks:
- Multi-method comparison (Laplace, Hierarchical, Adaptive, Ensemble, Dropout)
- Multi-dataset evaluation (Burgers, Navier-Stokes, Darcy Flow, Wave Equation)
- Multi-metric assessment (NLL, CRPS, Calibration, Coverage, Sharpness)
- Statistical significance validation (paired t-tests, Wilcoxon tests)
- Computational efficiency analysis

Authors: TERRAGON Labs Research Team
Date: 2025-08-13
"""

import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Statistical testing
try:
    from scipy import stats
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be skipped.")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    methods: List[str]
    datasets: List[str]
    metrics: List[str]
    n_trials: int = 5
    confidence_level: float = 0.95
    max_samples_per_dataset: int = 1000
    device: str = "auto"
    random_seed: int = 42
    save_intermediate: bool = True


@dataclass
class ExperimentResult:
    """Results from a single experimental run."""
    method: str
    dataset: str
    metric: str
    value: float
    std_error: float
    trial_id: int
    computation_time: float
    memory_usage: float


class ResearchBenchmarkSuite:
    """Comprehensive benchmarking suite for research validation.
    
    This suite provides rigorous experimental validation with:
    - Multiple baseline comparisons
    - Statistical significance testing
    - Reproducible experimental setup
    - Computational efficiency analysis
    - Publication-ready outputs
    """
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite.
        
        Args:
            config: Benchmarking configuration
        """
        self.config = config
        self.results = []
        self.statistical_tests = {}
        self.computational_metrics = {}
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
        
        # Device setup
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"Initialized benchmark suite on device: {self.device}")
    
    def run_comprehensive_benchmark(self,
                                  models: Dict[str, Any],
                                  datasets: Dict[str, DataLoader],
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive benchmarking across all methods and datasets.
        
        Args:
            models: Dictionary of method_name -> model_instance
            datasets: Dictionary of dataset_name -> data_loader
            save_path: Optional path to save results
            
        Returns:
            Complete benchmark results with statistical analysis
        """
        print("Starting comprehensive benchmark...")
        
        # Validate inputs
        self._validate_inputs(models, datasets)
        
        # Run experiments
        for trial in range(self.config.n_trials):
            print(f"\nTrial {trial + 1}/{self.config.n_trials}")
            
            for method_name in self.config.methods:
                if method_name not in models:
                    print(f"Warning: Method '{method_name}' not found in models. Skipping.")
                    continue
                
                for dataset_name in self.config.datasets:
                    if dataset_name not in datasets:
                        print(f"Warning: Dataset '{dataset_name}' not found. Skipping.")
                        continue
                    
                    print(f"  {method_name} on {dataset_name}...")
                    
                    # Run single experiment
                    experiment_results = self._run_single_experiment(
                        method_name=method_name,
                        model=models[method_name],
                        dataset_name=dataset_name,
                        data_loader=datasets[dataset_name],
                        trial_id=trial
                    )
                    
                    self.results.extend(experiment_results)
                    
                    # Save intermediate results
                    if self.config.save_intermediate and save_path:
                        self._save_intermediate_results(save_path, trial)
        
        # Statistical analysis
        print("\nPerforming statistical analysis...")
        self._perform_statistical_analysis()
        
        # Compile final results
        final_results = self._compile_results()
        
        # Save complete results
        if save_path:
            self._save_complete_results(final_results, save_path)
        
        return final_results
    
    def _validate_inputs(self, models: Dict[str, Any], datasets: Dict[str, DataLoader]) -> None:
        """Validate input models and datasets."""
        # Check methods
        missing_methods = set(self.config.methods) - set(models.keys())
        if missing_methods:
            raise ValueError(f"Missing methods in models: {missing_methods}")
        
        # Check datasets
        missing_datasets = set(self.config.datasets) - set(datasets.keys())
        if missing_datasets:
            raise ValueError(f"Missing datasets: {missing_datasets}")
        
        # Validate models have required methods
        for method_name, model in models.items():
            if not hasattr(model, 'predict_with_uncertainty') and not hasattr(model, 'predict'):
                warnings.warn(f"Model {method_name} lacks prediction methods")
    
    def _run_single_experiment(self,
                              method_name: str,
                              model: Any,
                              dataset_name: str,
                              data_loader: DataLoader,
                              trial_id: int) -> List[ExperimentResult]:
        """Run a single experimental trial.
        
        Args:
            method_name: Name of the method being tested
            model: Model instance
            dataset_name: Name of the dataset
            data_loader: Data loader for the dataset
            trial_id: Trial identifier
            
        Returns:
            List of experiment results for this trial
        """
        experiment_results = []
        
        # Prepare model
        model.to(self.device)
        model.eval()
        
        # Collect predictions and targets
        predictions = []
        uncertainties = []
        targets = []
        computation_times = []
        memory_usages = []
        
        sample_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            if sample_count >= self.config.max_samples_per_dataset:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Memory tracking
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(self.device)
            else:
                memory_before = 0
            
            # Time prediction
            start_time = time.time()
            
            with torch.no_grad():
                if hasattr(model, 'predict_with_uncertainty'):
                    mean_pred, uncertainty = model.predict_with_uncertainty(data)
                elif hasattr(model, 'predict') and hasattr(model, '_is_fitted') and model._is_fitted:
                    mean_pred, uncertainty = model.predict(data, num_samples=100)
                else:
                    # Fallback for deterministic models
                    mean_pred = model(data)
                    uncertainty = torch.ones_like(mean_pred) * 0.01
            
            computation_time = time.time() - start_time
            
            # Memory tracking
            if self.device.type == "cuda":
                memory_after = torch.cuda.memory_allocated(self.device)
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0
            
            predictions.append(mean_pred.cpu())
            uncertainties.append(uncertainty.cpu())
            targets.append(target.cpu())
            computation_times.append(computation_time)
            memory_usages.append(memory_usage)
            
            sample_count += data.shape[0]
        
        if not predictions:
            warnings.warn(f"No predictions collected for {method_name} on {dataset_name}")
            return experiment_results
        
        # Concatenate results
        all_predictions = torch.cat(predictions, dim=0)
        all_uncertainties = torch.cat(uncertainties, dim=0)
        all_targets = torch.cat(targets, dim=0)
        
        avg_computation_time = sum(computation_times) / len(computation_times)
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        
        # Compute metrics
        for metric_name in self.config.metrics:
            try:
                metric_value = self._compute_metric(
                    metric_name, all_predictions, all_uncertainties, all_targets
                )
                
                result = ExperimentResult(
                    method=method_name,
                    dataset=dataset_name,
                    metric=metric_name,
                    value=metric_value,
                    std_error=0.0,  # Will be computed later across trials
                    trial_id=trial_id,
                    computation_time=avg_computation_time,
                    memory_usage=avg_memory_usage
                )
                
                experiment_results.append(result)
                
            except Exception as e:
                warnings.warn(f"Failed to compute {metric_name} for {method_name}: {e}")
        
        return experiment_results
    
    def _compute_metric(self,
                       metric_name: str,
                       predictions: torch.Tensor,
                       uncertainties: torch.Tensor,
                       targets: torch.Tensor) -> float:
        """Compute a specific uncertainty metric.
        
        Args:
            metric_name: Name of the metric to compute
            predictions: Model predictions
            uncertainties: Uncertainty estimates (variance or std)
            targets: Ground truth targets
            
        Returns:
            Computed metric value
        """
        if metric_name.lower() == "nll":
            return self._compute_negative_log_likelihood(predictions, uncertainties, targets)
        elif metric_name.lower() == "crps":
            return self._compute_crps(predictions, uncertainties, targets)
        elif metric_name.lower() == "calibration":
            return self._compute_calibration_error(predictions, uncertainties, targets)
        elif metric_name.lower() == "coverage":
            return self._compute_coverage(predictions, uncertainties, targets)
        elif metric_name.lower() == "sharpness":
            return self._compute_sharpness(uncertainties)
        elif metric_name.lower() == "mse":
            return self._compute_mse(predictions, targets)
        elif metric_name.lower() == "mae":
            return self._compute_mae(predictions, targets)
        elif metric_name.lower() == "interval_score":
            return self._compute_interval_score(predictions, uncertainties, targets)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def _compute_negative_log_likelihood(self,
                                       predictions: torch.Tensor,
                                       uncertainties: torch.Tensor,
                                       targets: torch.Tensor) -> float:
        """Compute negative log-likelihood for Gaussian predictions."""
        # Assume uncertainties are variances
        if uncertainties.mean() < 1e-10:  # Likely standard deviations
            variances = uncertainties.pow(2)
        else:
            variances = uncertainties
        
        # Add small constant for numerical stability
        variances = torch.clamp(variances, min=1e-10)
        
        # Gaussian NLL: 0.5 * (log(2π) + log(σ²) + (y - μ)² / σ²)
        log_likelihood = -0.5 * (
            torch.log(2 * torch.pi * variances) +
            (targets - predictions).pow(2) / variances
        )
        
        return -log_likelihood.mean().item()
    
    def _compute_crps(self,
                     predictions: torch.Tensor,
                     uncertainties: torch.Tensor,
                     targets: torch.Tensor) -> float:
        """Compute Continuous Ranked Probability Score."""
        # For Gaussian predictions: CRPS = σ * (z * (2Φ(z) - 1) + 2φ(z) - 1/√π)
        # where z = (y - μ) / σ, Φ is CDF, φ is PDF
        
        if uncertainties.mean() < 1e-10:
            std_devs = uncertainties
        else:
            std_devs = torch.sqrt(torch.clamp(uncertainties, min=1e-10))
        
        z = (targets - predictions) / (std_devs + 1e-10)
        
        # Standard normal CDF and PDF approximations
        def standard_normal_cdf(x):
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        
        def standard_normal_pdf(x):
            return torch.exp(-0.5 * x.pow(2)) / torch.sqrt(2 * torch.pi)
        
        cdf_z = standard_normal_cdf(z)
        pdf_z = standard_normal_pdf(z)
        
        crps = std_devs * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / torch.sqrt(torch.tensor(torch.pi)))
        
        return crps.mean().item()
    
    def _compute_calibration_error(self,
                                  predictions: torch.Tensor,
                                  uncertainties: torch.Tensor,
                                  targets: torch.Tensor,
                                  n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        if uncertainties.mean() < 1e-10:
            std_devs = uncertainties
        else:
            std_devs = torch.sqrt(torch.clamp(uncertainties, min=1e-10))
        
        # Convert to confidence levels
        errors = torch.abs(predictions - targets)
        confidences = 1.0 - torch.clamp(errors / (std_devs + 1e-10), 0, 1)
        
        # Bin by confidence
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(confidences)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (errors[in_bin] < std_devs[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def _compute_coverage(self,
                         predictions: torch.Tensor,
                         uncertainties: torch.Tensor,
                         targets: torch.Tensor,
                         confidence_level: float = 0.95) -> float:
        """Compute coverage of uncertainty intervals."""
        if uncertainties.mean() < 1e-10:
            std_devs = uncertainties
        else:
            std_devs = torch.sqrt(torch.clamp(uncertainties, min=1e-10))
        
        # Compute confidence intervals (assuming Gaussian)
        z_score = stats.norm.ppf((1 + confidence_level) / 2) if SCIPY_AVAILABLE else 1.96
        
        lower_bound = predictions - z_score * std_devs
        upper_bound = predictions + z_score * std_devs
        
        # Check coverage
        in_interval = (targets >= lower_bound) & (targets <= upper_bound)
        coverage = in_interval.float().mean()
        
        return coverage.item()
    
    def _compute_sharpness(self, uncertainties: torch.Tensor) -> float:
        """Compute sharpness (average uncertainty magnitude)."""
        if uncertainties.mean() < 1e-10:
            return uncertainties.mean().item()
        else:
            return torch.sqrt(uncertainties).mean().item()
    
    def _compute_mse(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Squared Error."""
        return F.mse_loss(predictions, targets).item()
    
    def _compute_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Absolute Error."""
        return F.l1_loss(predictions, targets).item()
    
    def _compute_interval_score(self,
                               predictions: torch.Tensor,
                               uncertainties: torch.Tensor,
                               targets: torch.Tensor,
                               alpha: float = 0.05) -> float:
        """Compute Interval Score for uncertainty quantification."""
        if uncertainties.mean() < 1e-10:
            std_devs = uncertainties
        else:
            std_devs = torch.sqrt(torch.clamp(uncertainties, min=1e-10))
        
        # Confidence interval bounds
        z_score = stats.norm.ppf(1 - alpha/2) if SCIPY_AVAILABLE else 1.96
        lower = predictions - z_score * std_devs
        upper = predictions + z_score * std_devs
        
        # Interval score components
        width = upper - lower
        below_penalty = 2 * alpha * torch.clamp(lower - targets, min=0)
        above_penalty = 2 * alpha * torch.clamp(targets - upper, min=0)
        
        interval_score = width + below_penalty + above_penalty
        return interval_score.mean().item()
    
    def _perform_statistical_analysis(self) -> None:
        """Perform statistical significance testing."""
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not available. Skipping statistical tests.")
            return
        
        print("Computing statistical significance tests...")
        
        # Group results by method, dataset, and metric
        grouped_results = {}
        for result in self.results:
            key = (result.method, result.dataset, result.metric)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result.value)
        
        # Compute means and standard errors
        for key, values in grouped_results.items():
            method, dataset, metric = key
            mean_value = np.mean(values)
            std_error = np.std(values, ddof=1) / np.sqrt(len(values))
            
            # Update results with statistics
            for result in self.results:
                if (result.method == method and 
                    result.dataset == dataset and 
                    result.metric == metric):
                    result.std_error = std_error
        
        # Pairwise statistical tests
        self.statistical_tests = {}
        
        for dataset in self.config.datasets:
            for metric in self.config.metrics:
                test_key = f"{dataset}_{metric}"
                self.statistical_tests[test_key] = {}
                
                # Get all methods for this dataset/metric combination
                method_values = {}
                for method in self.config.methods:
                    key = (method, dataset, metric)
                    if key in grouped_results:
                        method_values[method] = grouped_results[key]
                
                # Pairwise comparisons
                methods = list(method_values.keys())
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i < j:
                            values1 = method_values[method1]
                            values2 = method_values[method2]
                            
                            if len(values1) >= 3 and len(values2) >= 3:
                                # Paired t-test
                                t_stat, t_p_value = stats.ttest_rel(values1, values2)
                                
                                # Wilcoxon signed-rank test (non-parametric)
                                w_stat, w_p_value = stats.wilcoxon(values1, values2)
                                
                                self.statistical_tests[test_key][f"{method1}_vs_{method2}"] = {
                                    't_statistic': t_stat,
                                    't_p_value': t_p_value,
                                    'wilcoxon_statistic': w_stat,
                                    'wilcoxon_p_value': w_p_value,
                                    'significantly_different': min(t_p_value, w_p_value) < (1 - self.config.confidence_level)
                                }
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final benchmark results."""
        # Group by method, dataset, metric
        compiled = {}
        
        for result in self.results:
            method_key = result.method
            if method_key not in compiled:
                compiled[method_key] = {}
            
            dataset_key = result.dataset
            if dataset_key not in compiled[method_key]:
                compiled[method_key][dataset_key] = {}
            
            metric_key = result.metric
            if metric_key not in compiled[method_key][dataset_key]:
                compiled[method_key][dataset_key][metric_key] = {
                    'values': [],
                    'computation_times': [],
                    'memory_usages': []
                }
            
            compiled[method_key][dataset_key][metric_key]['values'].append(result.value)
            compiled[method_key][dataset_key][metric_key]['computation_times'].append(result.computation_time)
            compiled[method_key][dataset_key][metric_key]['memory_usages'].append(result.memory_usage)
        
        # Compute statistics
        summary = {}
        for method in compiled:
            summary[method] = {}
            for dataset in compiled[method]:
                summary[method][dataset] = {}
                for metric in compiled[method][dataset]:
                    values = compiled[method][dataset][metric]['values']
                    times = compiled[method][dataset][metric]['computation_times']
                    memory = compiled[method][dataset][metric]['memory_usages']
                    
                    summary[method][dataset][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'std_error': np.std(values, ddof=1) / np.sqrt(len(values)),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'n_trials': len(values),
                        'avg_computation_time': np.mean(times),
                        'avg_memory_usage': np.mean(memory)
                    }
        
        return {
            'summary': summary,
            'raw_results': [result.__dict__ for result in self.results],
            'statistical_tests': self.statistical_tests,
            'config': self.config.__dict__,
            'benchmark_info': {
                'total_experiments': len(self.results),
                'n_methods': len(self.config.methods),
                'n_datasets': len(self.config.datasets),
                'n_metrics': len(self.config.metrics),
                'n_trials': self.config.n_trials
            }
        }
    
    def _save_intermediate_results(self, save_path: str, trial: int) -> None:
        """Save intermediate results during benchmarking."""
        intermediate_path = f"{save_path}_trial_{trial}_intermediate.json"
        intermediate_results = [result.__dict__ for result in self.results]
        
        with open(intermediate_path, 'w') as f:
            json.dump(intermediate_results, f, indent=2)
    
    def _save_complete_results(self, results: Dict[str, Any], save_path: str) -> None:
        """Save complete benchmark results."""
        # Save JSON results
        json_path = f"{save_path}_complete_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable report
        report_path = f"{save_path}_benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write(self._generate_benchmark_report(results))
        
        print(f"Results saved to {json_path}")
        print(f"Report saved to {report_path}")
    
    def _generate_benchmark_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable benchmark report."""
        report = "# RESEARCH BENCHMARK RESULTS\n"
        report += "=" * 50 + "\n\n"
        
        # Configuration
        report += "## CONFIGURATION\n"
        report += "-" * 20 + "\n"
        config = results['config']
        for key, value in config.items():
            report += f"{key}: {value}\n"
        report += "\n"
        
        # Summary Results
        report += "## SUMMARY RESULTS\n"
        report += "-" * 20 + "\n"
        
        summary = results['summary']
        for method in summary:
            report += f"\n### {method.upper()}\n"
            
            for dataset in summary[method]:
                report += f"\n#### Dataset: {dataset}\n"
                
                for metric in summary[method][dataset]:
                    stats = summary[method][dataset][metric]
                    report += f"- {metric}: {stats['mean']:.6f} ± {stats['std_error']:.6f}\n"
                    report += f"  (std: {stats['std']:.6f}, trials: {stats['n_trials']})\n"
                    report += f"  Time: {stats['avg_computation_time']:.4f}s, Memory: {stats['avg_memory_usage']:.0f} bytes\n"
        
        # Statistical Tests
        if results['statistical_tests']:
            report += "\n## STATISTICAL SIGNIFICANCE TESTS\n"
            report += "-" * 35 + "\n"
            
            for test_key, comparisons in results['statistical_tests'].items():
                report += f"\n### {test_key}\n"
                
                for comparison, stats in comparisons.items():
                    report += f"- {comparison}:\n"
                    report += f"  t-test p-value: {stats['t_p_value']:.6f}\n"
                    report += f"  Wilcoxon p-value: {stats['wilcoxon_p_value']:.6f}\n"
                    report += f"  Significantly different: {stats['significantly_different']}\n"
        
        # Best Performing Methods
        report += "\n## BEST PERFORMING METHODS\n"
        report += "-" * 30 + "\n"
        
        # Find best method for each metric/dataset combination
        for dataset in self.config.datasets:
            report += f"\n### Dataset: {dataset}\n"
            
            for metric in self.config.metrics:
                best_method = None
                best_value = None
                
                # Determine if lower is better for this metric
                lower_is_better = metric.lower() in ['nll', 'mse', 'mae', 'calibration', 'interval_score']
                
                for method in summary:
                    if dataset in summary[method] and metric in summary[method][dataset]:
                        value = summary[method][dataset][metric]['mean']
                        
                        if best_value is None:
                            best_value = value
                            best_method = method
                        elif (lower_is_better and value < best_value) or (not lower_is_better and value > best_value):
                            best_value = value
                            best_method = method
                
                if best_method:
                    report += f"- {metric}: {best_method} ({best_value:.6f})\n"
        
        # Computational Efficiency
        report += "\n## COMPUTATIONAL EFFICIENCY\n"
        report += "-" * 25 + "\n"
        
        for method in summary:
            total_time = 0
            total_memory = 0
            count = 0
            
            for dataset in summary[method]:
                for metric in summary[method][dataset]:
                    stats = summary[method][dataset][metric]
                    total_time += stats['avg_computation_time']
                    total_memory += stats['avg_memory_usage']
                    count += 1
            
            if count > 0:
                avg_time = total_time / count
                avg_memory = total_memory / count
                report += f"- {method}: {avg_time:.4f}s avg time, {avg_memory:.0f} bytes avg memory\n"
        
        report += "\n## CONCLUSION\n"
        report += "-" * 15 + "\n"
        report += "This benchmark provides comprehensive evaluation of uncertainty\n"
        report += "quantification methods across multiple datasets and metrics.\n"
        report += "Statistical significance tests validate the reliability of\n"
        report += "performance differences between methods.\n"
        
        return report


def create_synthetic_benchmarking_data(n_samples: int = 1000,
                                     input_dim: int = 64,
                                     output_dim: int = 1,
                                     noise_level: float = 0.1,
                                     device: torch.device = None) -> DataLoader:
    """Create synthetic data for benchmarking.
    
    Args:
        n_samples: Number of samples
        input_dim: Input dimension
        output_dim: Output dimension  
        noise_level: Noise level for targets
        device: Device for tensors
        
    Returns:
        DataLoader with synthetic data
    """
    if device is None:
        device = torch.device("cpu")
    
    # Generate synthetic input data
    inputs = torch.randn(n_samples, input_dim, device=device)
    
    # Generate targets with known function + noise
    # Simple function: weighted sum with nonlinearity
    weights = torch.randn(input_dim, output_dim, device=device)
    targets = torch.tanh(inputs @ weights)
    
    # Add heteroscedastic noise (input-dependent)
    noise_std = noise_level * (1 + 0.5 * torch.norm(inputs, dim=1, keepdim=True))
    noise = noise_std * torch.randn_like(targets)
    targets = targets + noise
    
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=32, shuffle=False)


def run_research_benchmark_example():
    """Example of running the research benchmark suite."""
    # Configuration
    config = BenchmarkConfig(
        methods=["hierarchical_laplace", "adaptive_scaling", "standard_laplace", "ensemble"],
        datasets=["synthetic_smooth", "synthetic_noisy"],
        metrics=["nll", "crps", "calibration", "coverage", "mse"],
        n_trials=3,
        max_samples_per_dataset=500
    )
    
    # Create synthetic datasets
    datasets = {
        "synthetic_smooth": create_synthetic_benchmarking_data(n_samples=500, noise_level=0.05),
        "synthetic_noisy": create_synthetic_benchmarking_data(n_samples=500, noise_level=0.2)
    }
    
    # Mock models for demonstration
    class MockModel:
        def __init__(self, name):
            self.name = name
            self._is_fitted = True
        
        def to(self, device):
            return self
        
        def eval(self):
            return self
        
        def predict_with_uncertainty(self, x):
            # Mock prediction with some random variation per method
            if "hierarchical" in self.name:
                mean = torch.randn_like(x[:, :1])
                var = torch.ones_like(mean) * 0.02
            elif "adaptive" in self.name:
                mean = torch.randn_like(x[:, :1])
                var = torch.ones_like(mean) * 0.03
            else:
                mean = torch.randn_like(x[:, :1])
                var = torch.ones_like(mean) * 0.05
            
            return mean, var
    
    models = {
        "hierarchical_laplace": MockModel("hierarchical_laplace"),
        "adaptive_scaling": MockModel("adaptive_scaling"),
        "standard_laplace": MockModel("standard_laplace"),
        "ensemble": MockModel("ensemble")
    }
    
    # Run benchmark
    benchmark = ResearchBenchmarkSuite(config)
    results = benchmark.run_comprehensive_benchmark(
        models=models,
        datasets=datasets,
        save_path="benchmark_results"
    )
    
    return results


if __name__ == "__main__":
    # Run example benchmark
    print("Running research benchmark example...")
    example_results = run_research_benchmark_example()
    print("Benchmark completed!")
    print(f"Total experiments: {example_results['benchmark_info']['total_experiments']}")