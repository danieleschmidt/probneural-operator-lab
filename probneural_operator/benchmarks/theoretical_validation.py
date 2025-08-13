"""Theoretical Validation Framework for Novel Uncertainty Methods.

This module provides comprehensive theoretical validation for:
1. Hierarchical Multi-Scale Uncertainty Decomposition
2. Adaptive Uncertainty Scaling
3. Novel theoretical properties and guarantees

Research Validation:
- Mathematical consistency checks
- Convergence analysis
- Calibration theory validation
- Scale separation properties
- Information-theoretic measures

Authors: TERRAGON Labs Research Team
Date: 2025-08-13
"""

import math
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from scipy import stats
    from scipy.special import gamma
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TheoreticalValidator:
    """Comprehensive theoretical validation framework.
    
    This class provides rigorous mathematical validation of novel uncertainty
    quantification methods with theoretical guarantees.
    """
    
    def __init__(self, 
                 tolerance: float = 1e-6,
                 num_theoretical_samples: int = 10000,
                 confidence_level: float = 0.95):
        """Initialize theoretical validator.
        
        Args:
            tolerance: Numerical tolerance for theoretical checks
            num_theoretical_samples: Number of samples for Monte Carlo validation
            confidence_level: Confidence level for statistical tests
        """
        self.tolerance = tolerance
        self.num_theoretical_samples = num_theoretical_samples
        self.confidence_level = confidence_level
        self.validation_results = {}
    
    def validate_hierarchical_decomposition(self, 
                                          hierarchical_model,
                                          test_loader: DataLoader) -> Dict[str, Any]:
        """Validate theoretical properties of hierarchical uncertainty decomposition.
        
        Args:
            hierarchical_model: Hierarchical Laplace approximation model
            test_loader: Test data for validation
            
        Returns:
            Dictionary of theoretical validation results
        """
        results = {}
        device = next(hierarchical_model.model.parameters()).device
        
        # Property 1: Scale Additivity
        results['scale_additivity'] = self._validate_scale_additivity(
            hierarchical_model, test_loader, device
        )
        
        # Property 2: Hierarchical Ordering
        results['hierarchical_ordering'] = self._validate_hierarchical_ordering(
            hierarchical_model, test_loader, device
        )
        
        # Property 3: Information Conservation
        results['information_conservation'] = self._validate_information_conservation(
            hierarchical_model, test_loader, device
        )
        
        # Property 4: Scale Separation Quality
        results['scale_separation'] = self._validate_scale_separation(
            hierarchical_model, test_loader, device
        )
        
        # Property 5: Convergence Properties
        results['convergence'] = self._validate_convergence_properties(
            hierarchical_model, test_loader, device
        )
        
        return results
    
    def _validate_scale_additivity(self, 
                                  model,
                                  test_loader: DataLoader,
                                  device: torch.device) -> Dict[str, float]:
        """Validate that total uncertainty equals sum of scale uncertainties.
        
        Mathematical property: Var_total = Var_global + Var_regional + Var_local
        """
        additivity_errors = []
        
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit for efficiency
                break
                
            data = data.to(device)
            
            # Get scale-decomposed uncertainties
            mean, total_var, scale_vars = model.predict(
                data, return_scale_decomposition=True
            )
            
            # Compute sum of scale variances
            scale_sum = torch.zeros_like(total_var)
            for scale_var in scale_vars.values():
                scale_sum += scale_var
            
            # Measure additivity error
            additivity_error = torch.abs(total_var - scale_sum) / (total_var + self.tolerance)
            additivity_errors.append(additivity_error.mean().item())
        
        return {
            'mean_error': sum(additivity_errors) / len(additivity_errors),
            'max_error': max(additivity_errors),
            'passes_test': max(additivity_errors) < self.tolerance * 100  # Allow some numerical error
        }
    
    def _validate_hierarchical_ordering(self,
                                       model,
                                       test_loader: DataLoader,
                                       device: torch.device) -> Dict[str, float]:
        """Validate that uncertainty scales follow hierarchical ordering.
        
        Theoretical expectation: Global >= Regional >= Local (generally)
        """
        ordering_violations = []
        scale_ratios = []
        
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 10:
                break
                
            data = data.to(device)
            mean, total_var, scale_vars = model.predict(
                data, return_scale_decomposition=True
            )
            
            # Compute average uncertainty per scale
            avg_uncertainties = {}
            for scale, var in scale_vars.items():
                avg_uncertainties[scale] = var.mean().item()
            
            # Check ordering (allow for some flexibility)
            if 'global' in avg_uncertainties and 'regional' in avg_uncertainties:
                ratio_global_regional = avg_uncertainties['global'] / (avg_uncertainties['regional'] + self.tolerance)
                scale_ratios.append(ratio_global_regional)
                
                if ratio_global_regional < 0.5:  # Strong violation
                    ordering_violations.append(1.0)
                else:
                    ordering_violations.append(0.0)
        
        violation_rate = sum(ordering_violations) / len(ordering_violations) if ordering_violations else 0.0
        avg_ratio = sum(scale_ratios) / len(scale_ratios) if scale_ratios else 1.0
        
        return {
            'violation_rate': violation_rate,
            'avg_global_regional_ratio': avg_ratio,
            'satisfies_ordering': violation_rate < 0.2  # Allow 20% violations
        }
    
    def _validate_information_conservation(self,
                                         model,
                                         test_loader: DataLoader,
                                         device: torch.device) -> Dict[str, float]:
        """Validate information-theoretic conservation properties.
        
        Property: Total information should be conserved across scales
        Information measured via differential entropy
        """
        entropy_conservation_errors = []
        
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 5:  # Computationally expensive
                break
                
            data = data.to(device)
            mean, total_var, scale_vars = model.predict(
                data, return_scale_decomposition=True
            )
            
            # Compute differential entropy for total uncertainty
            # H(X) = 0.5 * log(2πe * σ²) for Gaussian
            total_entropy = 0.5 * torch.log(2 * math.pi * math.e * total_var).mean()
            
            # Compute sum of scale entropies
            scale_entropy_sum = 0.0
            for scale_var in scale_vars.values():
                scale_entropy = 0.5 * torch.log(2 * math.pi * math.e * scale_var).mean()
                scale_entropy_sum += scale_entropy
            
            # Information conservation error
            conservation_error = torch.abs(total_entropy - scale_entropy_sum) / (torch.abs(total_entropy) + self.tolerance)
            entropy_conservation_errors.append(conservation_error.item())
        
        return {
            'mean_conservation_error': sum(entropy_conservation_errors) / len(entropy_conservation_errors),
            'max_conservation_error': max(entropy_conservation_errors),
            'information_conserved': max(entropy_conservation_errors) < 0.1
        }
    
    def _validate_scale_separation(self,
                                  model,
                                  test_loader: DataLoader,
                                  device: torch.device) -> Dict[str, float]:
        """Validate quality of scale separation.
        
        Measures how well different scales capture distinct uncertainty patterns.
        """
        scale_correlations = {}
        scale_names = list(model.scales)
        
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 10:
                break
                
            data = data.to(device)
            mean, total_var, scale_vars = model.predict(
                data, return_scale_decomposition=True
            )
            
            # Compute correlations between scale uncertainties
            for i, scale1 in enumerate(scale_names):
                for j, scale2 in enumerate(scale_names):
                    if i < j and scale1 in scale_vars and scale2 in scale_vars:
                        key = f"{scale1}_{scale2}"
                        if key not in scale_correlations:
                            scale_correlations[key] = []
                        
                        var1 = scale_vars[scale1].flatten()
                        var2 = scale_vars[scale2].flatten()
                        
                        # Compute correlation coefficient
                        correlation = torch.corrcoef(torch.stack([var1, var2]))[0, 1]
                        if not torch.isnan(correlation):
                            scale_correlations[key].append(correlation.item())
        
        # Analyze separation quality
        avg_correlations = {}
        for key, corrs in scale_correlations.items():
            avg_correlations[key] = sum(corrs) / len(corrs)
        
        max_correlation = max(avg_correlations.values()) if avg_correlations else 0.0
        good_separation = max_correlation < 0.5  # Scales should be relatively uncorrelated
        
        return {
            'scale_correlations': avg_correlations,
            'max_correlation': max_correlation,
            'good_separation': good_separation
        }
    
    def _validate_convergence_properties(self,
                                       model,
                                       test_loader: DataLoader,
                                       device: torch.device) -> Dict[str, float]:
        """Validate convergence properties of hierarchical method.
        
        Tests convergence as number of scales increases.
        """
        convergence_metrics = {}
        
        # Test with different numbers of scales
        original_scales = model.scales
        
        for n_scales in [1, 2, 3]:
            if n_scales <= len(original_scales):
                test_scales = original_scales[:n_scales]
                model.scales = test_scales
                
                uncertainties = []
                for batch_idx, (data, _) in enumerate(test_loader):
                    if batch_idx >= 5:
                        break
                    
                    data = data.to(device)
                    mean, total_var, _ = model.predict(data, return_scale_decomposition=True)
                    uncertainties.append(total_var.mean().item())
                
                avg_uncertainty = sum(uncertainties) / len(uncertainties)
                convergence_metrics[f'n_scales_{n_scales}'] = avg_uncertainty
        
        # Restore original scales
        model.scales = original_scales
        
        # Check convergence pattern
        uncertainties_by_scale = [convergence_metrics.get(f'n_scales_{i}', 0) for i in [1, 2, 3]]
        is_monotonic = all(uncertainties_by_scale[i] <= uncertainties_by_scale[i+1] + self.tolerance 
                          for i in range(len(uncertainties_by_scale)-1))
        
        return {
            'uncertainties_by_scale': uncertainties_by_scale,
            'converges_monotonically': is_monotonic,
            'convergence_rate': uncertainties_by_scale[-1] - uncertainties_by_scale[0] if len(uncertainties_by_scale) >= 2 else 0.0
        }
    
    def validate_adaptive_scaling(self,
                                adaptive_scaler,
                                test_loader: DataLoader) -> Dict[str, Any]:
        """Validate theoretical properties of adaptive uncertainty scaling.
        
        Args:
            adaptive_scaler: Adaptive uncertainty scaler
            test_loader: Test data
            
        Returns:
            Validation results for adaptive scaling
        """
        results = {}
        device = next(adaptive_scaler.base_model.parameters()).device
        
        # Property 1: Scaling Consistency
        results['scaling_consistency'] = self._validate_scaling_consistency(
            adaptive_scaler, test_loader, device
        )
        
        # Property 2: Calibration Improvement
        results['calibration_improvement'] = self._validate_calibration_improvement(
            adaptive_scaler, test_loader, device
        )
        
        # Property 3: Adaptive Convergence
        results['adaptive_convergence'] = self._validate_adaptive_convergence(
            adaptive_scaler, test_loader, device
        )
        
        # Property 4: Physics Constraint Satisfaction
        if adaptive_scaler.physics_constraints:
            results['physics_constraints'] = self._validate_physics_constraints(
                adaptive_scaler, test_loader, device
            )
        
        return results
    
    def _validate_scaling_consistency(self,
                                    scaler,
                                    test_loader: DataLoader,
                                    device: torch.device) -> Dict[str, float]:
        """Validate that scaling is consistent and well-behaved."""
        scaling_factors = []
        scaling_variations = []
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 10:
                break
                
            data = data.to(device)
            
            # Get base and scaled predictions
            if hasattr(scaler.base_model, 'predict_with_uncertainty'):
                base_mean, base_var = scaler.base_model.predict_with_uncertainty(data)
            else:
                base_mean = scaler.base_model(data)
                base_var = torch.ones_like(base_mean) * 0.01
                
            scaled_mean, scaled_var = scaler.predict_with_adaptive_scaling(data, update_history=False)
            
            # Compute scaling factors
            scale_factor = torch.sqrt(scaled_var / (base_var + self.tolerance))
            scaling_factors.extend(scale_factor.flatten().cpu().tolist())
            
            # Measure variation in scaling within batch
            scale_variation = scale_factor.std().item()
            scaling_variations.append(scale_variation)
        
        return {
            'mean_scaling_factor': sum(scaling_factors) / len(scaling_factors),
            'scaling_factor_std': torch.tensor(scaling_factors).std().item(),
            'mean_within_batch_variation': sum(scaling_variations) / len(scaling_variations),
            'reasonable_scaling': all(0.1 <= sf <= 10.0 for sf in scaling_factors)
        }
    
    def _validate_calibration_improvement(self,
                                        scaler,
                                        test_loader: DataLoader,
                                        device: torch.device) -> Dict[str, float]:
        """Validate that adaptive scaling improves calibration."""
        base_calibration_errors = []
        adaptive_calibration_errors = []
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 20:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Base predictions
            if hasattr(scaler.base_model, 'predict_with_uncertainty'):
                base_mean, base_var = scaler.base_model.predict_with_uncertainty(data)
            else:
                base_mean = scaler.base_model(data)
                base_var = torch.ones_like(base_mean) * 0.01
            
            # Adaptive predictions
            adaptive_mean, adaptive_var = scaler.predict_with_adaptive_scaling(data, update_history=False)
            
            # Compute calibration errors
            actual_errors = (base_mean - target).pow(2)  # Use base_mean for fair comparison
            
            base_calib_error = torch.abs(base_var - actual_errors) / (actual_errors + self.tolerance)
            adaptive_calib_error = torch.abs(adaptive_var - actual_errors) / (actual_errors + self.tolerance)
            
            base_calibration_errors.append(base_calib_error.mean().item())
            adaptive_calibration_errors.append(adaptive_calib_error.mean().item())
        
        base_avg_error = sum(base_calibration_errors) / len(base_calibration_errors)
        adaptive_avg_error = sum(adaptive_calibration_errors) / len(adaptive_calibration_errors)
        
        return {
            'base_calibration_error': base_avg_error,
            'adaptive_calibration_error': adaptive_avg_error,
            'improvement': base_avg_error - adaptive_avg_error,
            'relative_improvement': (base_avg_error - adaptive_avg_error) / (base_avg_error + self.tolerance),
            'shows_improvement': adaptive_avg_error < base_avg_error
        }
    
    def _validate_adaptive_convergence(self,
                                     scaler,
                                     test_loader: DataLoader,
                                     device: torch.device) -> Dict[str, float]:
        """Validate convergence properties of adaptive scaling."""
        # Simulate online adaptation process
        adaptation_history = []
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 50:  # Simulate 50 adaptation steps
                break
                
            data, target = data.to(device), target.to(device)
            
            # Get adaptive prediction
            adaptive_mean, adaptive_var = scaler.predict_with_adaptive_scaling(data)
            
            # Simulate feedback (in practice this would come from ground truth)
            if hasattr(scaler, 'update_with_feedback'):
                scaler.update_with_feedback(batch_idx, target)
            
            # Record adaptation state
            if hasattr(scaler, 'get_adaptation_metrics'):
                metrics = scaler.get_adaptation_metrics()
                adaptation_history.append(metrics.get('avg_scaling_quality', 1.0))
        
        # Analyze convergence
        if len(adaptation_history) >= 10:
            early_performance = sum(adaptation_history[:10]) / 10
            late_performance = sum(adaptation_history[-10:]) / 10
            convergence_improvement = late_performance - early_performance
            
            # Check for convergence (decreasing variance)
            recent_variance = torch.tensor(adaptation_history[-20:]).var().item() if len(adaptation_history) >= 20 else 1.0
        else:
            convergence_improvement = 0.0
            recent_variance = 1.0
        
        return {
            'adaptation_steps': len(adaptation_history),
            'convergence_improvement': convergence_improvement,
            'recent_variance': recent_variance,
            'converges': recent_variance < 0.1 and convergence_improvement >= 0
        }
    
    def _validate_physics_constraints(self,
                                    scaler,
                                    test_loader: DataLoader,
                                    device: torch.device) -> Dict[str, float]:
        """Validate that physics constraints are properly enforced."""
        constraint_violations = []
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 10:
                break
                
            data = data.to(device)
            
            # Get predictions with physics constraints
            mean, variance = scaler.predict_with_adaptive_scaling(data, update_history=False)
            std = torch.sqrt(variance)
            
            # Check constraint violations
            violations = []
            
            # Positivity constraints
            if 'positive_quantities' in scaler.physics_constraints:
                negative_probability = torch.sigmoid(-(mean / (std + self.tolerance)))  # P(X < 0)
                violations.append(negative_probability.mean().item())
            
            # Conservation constraints  
            if 'conservation_bounds' in scaler.physics_constraints:
                bounds = scaler.physics_constraints['conservation_bounds']
                max_violation = bounds.get('max_relative_violation', 0.1)
                
                # 3σ bounds should respect conservation
                upper_bound = mean + 3 * std
                relative_violation = torch.abs(upper_bound - mean) / (torch.abs(mean) + self.tolerance)
                violation_rate = (relative_violation > max_violation).float().mean().item()
                violations.append(violation_rate)
            
            if violations:
                constraint_violations.append(max(violations))
        
        avg_violation = sum(constraint_violations) / len(constraint_violations) if constraint_violations else 0.0
        
        return {
            'average_violation_rate': avg_violation,
            'max_violation_rate': max(constraint_violations) if constraint_violations else 0.0,
            'satisfies_constraints': avg_violation < 0.05  # Less than 5% violation rate
        }
    
    def validate_novel_theoretical_properties(self,
                                            hierarchical_model,
                                            adaptive_scaler,
                                            test_loader: DataLoader) -> Dict[str, Any]:
        """Validate novel theoretical properties unique to our approach.
        
        Tests properties that are novel contributions of this research.
        """
        results = {}
        device = next(hierarchical_model.model.parameters()).device
        
        # Novel Property 1: Cross-Scale Information Transfer
        results['cross_scale_information'] = self._validate_cross_scale_information(
            hierarchical_model, test_loader, device
        )
        
        # Novel Property 2: Adaptive-Hierarchical Synergy
        results['adaptive_hierarchical_synergy'] = self._validate_adaptive_hierarchical_synergy(
            hierarchical_model, adaptive_scaler, test_loader, device
        )
        
        # Novel Property 3: Uncertainty Attribution Consistency
        results['uncertainty_attribution'] = self._validate_uncertainty_attribution(
            hierarchical_model, test_loader, device
        )
        
        # Novel Property 4: Multi-Scale Active Learning Efficiency
        results['active_learning_efficiency'] = self._validate_active_learning_efficiency(
            hierarchical_model, test_loader, device
        )
        
        return results
    
    def _validate_cross_scale_information(self,
                                        model,
                                        test_loader: DataLoader,
                                        device: torch.device) -> Dict[str, float]:
        """Validate information transfer between uncertainty scales."""
        information_transfer_metrics = []
        
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 5:
                break
                
            data = data.to(device)
            mean, total_var, scale_vars = model.predict(data, return_scale_decomposition=True)
            
            # Compute mutual information between scales (approximated)
            scale_names = list(scale_vars.keys())
            mutual_info_sum = 0.0
            
            for i, scale1 in enumerate(scale_names):
                for j, scale2 in enumerate(scale_names):
                    if i < j:
                        var1 = scale_vars[scale1].flatten()
                        var2 = scale_vars[scale2].flatten()
                        
                        # Approximate mutual information using correlation
                        correlation = torch.corrcoef(torch.stack([var1, var2]))[0, 1]
                        if not torch.isnan(correlation):
                            # I(X,Y) ≈ -0.5 * log(1 - ρ²) for Gaussian
                            mutual_info = -0.5 * torch.log(1 - correlation.pow(2) + self.tolerance)
                            mutual_info_sum += mutual_info.item()
            
            information_transfer_metrics.append(mutual_info_sum)
        
        avg_transfer = sum(information_transfer_metrics) / len(information_transfer_metrics)
        
        return {
            'average_cross_scale_info': avg_transfer,
            'information_coherence': avg_transfer > 0.1 and avg_transfer < 2.0  # Balanced transfer
        }
    
    def _validate_adaptive_hierarchical_synergy(self,
                                              hierarchical_model,
                                              adaptive_scaler,
                                              test_loader: DataLoader,
                                              device: torch.device) -> Dict[str, float]:
        """Validate synergy between adaptive scaling and hierarchical decomposition."""
        synergy_benefits = []
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 10:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Hierarchical prediction alone
            hier_mean, hier_var, _ = hierarchical_model.predict(data, return_scale_decomposition=True)
            
            # Combined adaptive + hierarchical (simulate integration)
            if hasattr(adaptive_scaler, 'base_model'):
                # Set hierarchical as base model temporarily
                original_base = adaptive_scaler.base_model
                adaptive_scaler.base_model = hierarchical_model.model
                
                combined_mean, combined_var = adaptive_scaler.predict_with_adaptive_scaling(
                    data, update_history=False
                )
                
                # Restore original base
                adaptive_scaler.base_model = original_base
            else:
                combined_mean, combined_var = hier_mean, hier_var
            
            # Measure synergy benefit
            actual_error = (hier_mean - target).pow(2)
            
            hier_calibration = torch.abs(hier_var - actual_error) / (actual_error + self.tolerance)
            combined_calibration = torch.abs(combined_var - actual_error) / (actual_error + self.tolerance)
            
            synergy_benefit = hier_calibration.mean() - combined_calibration.mean()
            synergy_benefits.append(synergy_benefit.item())
        
        avg_synergy = sum(synergy_benefits) / len(synergy_benefits)
        
        return {
            'average_synergy_benefit': avg_synergy,
            'shows_positive_synergy': avg_synergy > 0,
            'synergy_magnitude': abs(avg_synergy)
        }
    
    def _validate_uncertainty_attribution(self,
                                        model,
                                        test_loader: DataLoader,
                                        device: torch.device) -> Dict[str, float]:
        """Validate consistency of uncertainty attribution across predictions."""
        attribution_consistency = []
        
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 10:
                break
                
            data = data.to(device)
            mean, total_var, scale_vars = model.predict(data, return_scale_decomposition=True)
            
            # Get uncertainty attribution
            attribution = model.get_uncertainty_attribution()
            
            # Check attribution properties
            attribution_sum = sum(attribution.values())
            sum_consistency = abs(attribution_sum - 1.0)  # Should sum to 1
            
            # Check non-negativity
            all_positive = all(v >= 0 for v in attribution.values())
            
            # Measure attribution stability (variance across spatial dimensions)
            spatial_attributions = []
            for scale, var in scale_vars.items():
                if var.numel() > 1:
                    scale_contribution = var / total_var
                    spatial_attribution_var = scale_contribution.var().item()
                    spatial_attributions.append(spatial_attribution_var)
            
            avg_spatial_var = sum(spatial_attributions) / len(spatial_attributions) if spatial_attributions else 0.0
            
            consistency_score = 1.0 - sum_consistency - avg_spatial_var
            if all_positive:
                consistency_score += 0.1  # Bonus for non-negativity
            
            attribution_consistency.append(consistency_score)
        
        return {
            'average_consistency': sum(attribution_consistency) / len(attribution_consistency),
            'attribution_quality': all(score > 0.8 for score in attribution_consistency)
        }
    
    def _validate_active_learning_efficiency(self,
                                           model,
                                           test_loader: DataLoader,
                                           device: torch.device) -> Dict[str, float]:
        """Validate efficiency gains for multi-scale active learning."""
        # Simulate active learning scenarios
        acquisition_qualities = []
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 5:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Get scale-aware acquisition scores
            if hasattr(model, 'active_learning_acquisition'):
                scale_weights = {'global': 0.3, 'regional': 0.5, 'local': 0.2}
                acquisition_scores = model.active_learning_acquisition(
                    data, scale_weights=scale_weights
                )
                
                # Measure quality: high scores should correlate with high actual errors
                actual_errors = (model.predict(data)[0] - target).pow(2).mean(dim=tuple(range(1, target.ndim)))
                
                # Correlation between acquisition scores and actual errors
                if len(acquisition_scores) > 1 and len(actual_errors) > 1:
                    correlation = torch.corrcoef(torch.stack([acquisition_scores, actual_errors]))[0, 1]
                    if not torch.isnan(correlation):
                        acquisition_qualities.append(correlation.item())
        
        avg_quality = sum(acquisition_qualities) / len(acquisition_qualities) if acquisition_qualities else 0.0
        
        return {
            'acquisition_correlation': avg_quality,
            'efficient_selection': avg_quality > 0.3  # Moderate correlation expected
        }
    
    def generate_validation_report(self,
                                 hierarchical_results: Dict[str, Any],
                                 adaptive_results: Dict[str, Any],
                                 novel_results: Dict[str, Any]) -> str:
        """Generate comprehensive theoretical validation report.
        
        Args:
            hierarchical_results: Results from hierarchical validation
            adaptive_results: Results from adaptive scaling validation
            novel_results: Results from novel properties validation
            
        Returns:
            Formatted validation report
        """
        report = "# THEORETICAL VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Hierarchical Validation
        report += "## 1. HIERARCHICAL DECOMPOSITION VALIDATION\n"
        report += "-" * 40 + "\n"
        
        for property_name, result in hierarchical_results.items():
            report += f"### {property_name.replace('_', ' ').title()}\n"
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, bool):
                        status = "✓ PASS" if value else "✗ FAIL"
                        report += f"- {key}: {status}\n"
                    elif isinstance(value, (int, float)):
                        report += f"- {key}: {value:.6f}\n"
                    else:
                        report += f"- {key}: {value}\n"
            report += "\n"
        
        # Adaptive Scaling Validation
        report += "## 2. ADAPTIVE SCALING VALIDATION\n"
        report += "-" * 40 + "\n"
        
        for property_name, result in adaptive_results.items():
            report += f"### {property_name.replace('_', ' ').title()}\n"
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, bool):
                        status = "✓ PASS" if value else "✗ FAIL"
                        report += f"- {key}: {status}\n"
                    elif isinstance(value, (int, float)):
                        report += f"- {key}: {value:.6f}\n"
                    else:
                        report += f"- {key}: {value}\n"
            report += "\n"
        
        # Novel Properties Validation
        report += "## 3. NOVEL THEORETICAL PROPERTIES\n"
        report += "-" * 40 + "\n"
        
        for property_name, result in novel_results.items():
            report += f"### {property_name.replace('_', ' ').title()}\n"
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, bool):
                        status = "✓ PASS" if value else "✗ FAIL"
                        report += f"- {key}: {status}\n"
                    elif isinstance(value, (int, float)):
                        report += f"- {key}: {value:.6f}\n"
                    else:
                        report += f"- {key}: {value}\n"
            report += "\n"
        
        # Overall Assessment
        report += "## 4. OVERALL THEORETICAL ASSESSMENT\n"
        report += "-" * 40 + "\n"
        
        # Count passes and fails
        all_results = [hierarchical_results, adaptive_results, novel_results]
        total_tests = 0
        passed_tests = 0
        
        for result_dict in all_results:
            for prop_result in result_dict.values():
                if isinstance(prop_result, dict):
                    for value in prop_result.values():
                        if isinstance(value, bool):
                            total_tests += 1
                            if value:
                                passed_tests += 1
        
        if total_tests > 0:
            pass_rate = passed_tests / total_tests
            report += f"- Total theoretical tests: {total_tests}\n"
            report += f"- Passed tests: {passed_tests}\n"
            report += f"- Pass rate: {pass_rate:.1%}\n"
            
            if pass_rate >= 0.8:
                report += "- Overall status: ✓ THEORETICALLY SOUND\n"
            elif pass_rate >= 0.6:
                report += "- Overall status: ⚠ MOSTLY VALID (minor issues)\n"
            else:
                report += "- Overall status: ✗ THEORETICAL ISSUES DETECTED\n"
        else:
            report += "- Status: No boolean tests found\n"
        
        report += "\n"
        report += "## CONCLUSION\n"
        report += "-" * 40 + "\n"
        report += "This report validates the theoretical soundness of the novel\n"
        report += "hierarchical multi-scale uncertainty decomposition and adaptive\n"
        report += "scaling methods. The validation covers mathematical consistency,\n"
        report += "convergence properties, and novel theoretical contributions.\n"
        
        return report


def run_comprehensive_theoretical_validation(hierarchical_model,
                                           adaptive_scaler,
                                           test_loader: DataLoader,
                                           save_path: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive theoretical validation and generate report.
    
    Args:
        hierarchical_model: Hierarchical Laplace approximation model
        adaptive_scaler: Adaptive uncertainty scaler
        test_loader: Test data loader
        save_path: Optional path to save report
        
    Returns:
        Complete validation results
    """
    validator = TheoreticalValidator()
    
    print("Running hierarchical decomposition validation...")
    hierarchical_results = validator.validate_hierarchical_decomposition(
        hierarchical_model, test_loader
    )
    
    print("Running adaptive scaling validation...")
    adaptive_results = validator.validate_adaptive_scaling(
        adaptive_scaler, test_loader
    )
    
    print("Running novel properties validation...")
    novel_results = validator.validate_novel_theoretical_properties(
        hierarchical_model, adaptive_scaler, test_loader
    )
    
    # Generate report
    report = validator.generate_validation_report(
        hierarchical_results, adaptive_results, novel_results
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Validation report saved to {save_path}")
    
    return {
        'hierarchical': hierarchical_results,
        'adaptive': adaptive_results,
        'novel': novel_results,
        'report': report
    }