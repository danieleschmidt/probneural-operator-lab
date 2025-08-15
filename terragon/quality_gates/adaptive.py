"""
Adaptive Quality Controller
===========================

Intelligent quality gate management with adaptive thresholds, 
self-learning capabilities, and context-aware optimization.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from .core import QualityGate, QualityGateResult, QualityGateStatus, GenerationType
from .monitoring import QualityMetrics


@dataclass
class AdaptiveThreshold:
    """Adaptive threshold configuration."""
    base_value: float
    current_value: float
    min_value: float
    max_value: float
    adaptation_rate: float = 0.1
    confidence: float = 0.5
    history: List[float] = field(default_factory=list)
    
    def update(self, success_rate: float, context: Dict[str, Any]) -> None:
        """Update threshold based on success rate and context."""
        # Simple adaptive logic - can be enhanced
        if success_rate > 0.9:  # Too easy
            self.current_value = min(self.max_value, 
                                   self.current_value + self.adaptation_rate * 10)
        elif success_rate < 0.7:  # Too hard
            self.current_value = max(self.min_value,
                                   self.current_value - self.adaptation_rate * 10)
        
        self.history.append(self.current_value)
        
        # Keep history manageable
        if len(self.history) > 50:
            self.history = self.history[-50:]
        
        # Update confidence based on stability
        if len(self.history) >= 5:
            recent_values = self.history[-5:]
            mean_val = sum(recent_values) / len(recent_values)
            variance = sum((x - mean_val) ** 2 for x in recent_values) / len(recent_values)
            recent_std = variance ** 0.5
            self.confidence = max(0.1, min(1.0, 1.0 - recent_std / 50.0))


@dataclass
class ContextualFactor:
    """Contextual factor affecting quality assessment."""
    name: str
    weight: float
    current_value: float
    impact_history: List[Tuple[float, float]] = field(default_factory=list)  # (value, impact)
    
    def learn_impact(self, value: float, quality_outcome: float) -> None:
        """Learn the impact of this factor on quality outcomes."""
        self.impact_history.append((value, quality_outcome))
        
        # Keep recent history
        if len(self.impact_history) > 100:
            self.impact_history = self.impact_history[-100:]
        
        # Update weight based on correlation
        if len(self.impact_history) >= 10:
            values = [h[0] for h in self.impact_history[-10:]]
            outcomes = [h[1] for h in self.impact_history[-10:]]
            
            # Simple correlation calculation
            if len(values) > 1:
                mean_values = sum(values) / len(values)
                mean_outcomes = sum(outcomes) / len(outcomes)
                
                numerator = sum((values[i] - mean_values) * (outcomes[i] - mean_outcomes) for i in range(len(values)))
                sum_sq_values = sum((values[i] - mean_values) ** 2 for i in range(len(values)))
                sum_sq_outcomes = sum((outcomes[i] - mean_outcomes) ** 2 for i in range(len(outcomes)))
                
                denominator = (sum_sq_values * sum_sq_outcomes) ** 0.5
                correlation = numerator / denominator if denominator > 0 else 0
            else:
                correlation = 0
            self.weight = abs(correlation) * 0.5  # Scale to reasonable range


class AdaptiveQualityController:
    """
    Adaptive quality controller with self-learning capabilities.
    
    Features:
    - Dynamic threshold adjustment based on performance
    - Context-aware quality assessment
    - Learning from historical patterns
    - Intelligent gate prioritization
    - Adaptive scheduling
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(".terragon/adaptive_quality.json")
        self.adaptive_thresholds: Dict[str, AdaptiveThreshold] = {}
        self.contextual_factors: Dict[str, ContextualFactor] = {}
        self.learning_history: List[Dict[str, Any]] = []
        self.performance_baseline: Dict[str, float] = {}
        
        self.config = {
            "learning_enabled": True,
            "adaptation_enabled": True,
            "context_weight": 0.3,
            "baseline_window": 20,
            "min_samples_for_adaptation": 10,
            "confidence_threshold": 0.6,
        }
        
        self._initialize_components()
        self._load_state()
        
        self.logger = logging.getLogger("AdaptiveQualityController")
    
    def _initialize_components(self) -> None:
        """Initialize adaptive components."""
        # Initialize default thresholds
        default_thresholds = {
            "test_coverage": AdaptiveThreshold(85.0, 85.0, 70.0, 95.0),
            "code_quality": AdaptiveThreshold(85.0, 85.0, 75.0, 95.0),
            "security_score": AdaptiveThreshold(90.0, 90.0, 80.0, 98.0),
            "performance_score": AdaptiveThreshold(80.0, 80.0, 70.0, 95.0),
        }
        
        self.adaptive_thresholds.update(default_thresholds)
        
        # Initialize contextual factors
        default_factors = {
            "time_of_day": ContextualFactor("time_of_day", 0.1, 0.0),
            "recent_changes": ContextualFactor("recent_changes", 0.2, 0.0),
            "developer_activity": ContextualFactor("developer_activity", 0.15, 0.0),
            "system_load": ContextualFactor("system_load", 0.1, 0.0),
            "project_phase": ContextualFactor("project_phase", 0.25, 0.0),
        }
        
        self.contextual_factors.update(default_factors)
    
    def _load_state(self) -> None:
        """Load adaptive state from disk."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    state = json.load(f)
                
                # Load thresholds
                if "thresholds" in state:
                    for name, threshold_data in state["thresholds"].items():
                        if name in self.adaptive_thresholds:
                            threshold = self.adaptive_thresholds[name]
                            threshold.current_value = threshold_data.get("current_value", threshold.current_value)
                            threshold.confidence = threshold_data.get("confidence", threshold.confidence)
                            threshold.history = threshold_data.get("history", [])
                
                # Load contextual factors
                if "factors" in state:
                    for name, factor_data in state["factors"].items():
                        if name in self.contextual_factors:
                            factor = self.contextual_factors[name]
                            factor.weight = factor_data.get("weight", factor.weight)
                            factor.impact_history = factor_data.get("impact_history", [])
                
                # Load learning history
                self.learning_history = state.get("learning_history", [])
                self.performance_baseline = state.get("performance_baseline", {})
                
                self.logger.info("Adaptive state loaded successfully")
                
            except Exception as e:
                self.logger.warning(f"Could not load adaptive state: {e}")
    
    def _save_state(self) -> None:
        """Save adaptive state to disk."""
        # Prepare state data
        state = {
            "last_updated": datetime.now().isoformat(),
            "thresholds": {},
            "factors": {},
            "learning_history": self.learning_history[-100:],  # Keep recent history
            "performance_baseline": self.performance_baseline,
            "config": self.config,
        }
        
        # Save thresholds
        for name, threshold in self.adaptive_thresholds.items():
            state["thresholds"][name] = {
                "base_value": threshold.base_value,
                "current_value": threshold.current_value,
                "min_value": threshold.min_value,
                "max_value": threshold.max_value,
                "confidence": threshold.confidence,
                "history": threshold.history[-20:],  # Keep recent history
            }
        
        # Save contextual factors
        for name, factor in self.contextual_factors.items():
            state["factors"][name] = {
                "weight": factor.weight,
                "current_value": factor.current_value,
                "impact_history": factor.impact_history[-50:],  # Keep recent history
            }
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save adaptive state: {e}")
    
    def get_adaptive_threshold(self, metric_name: str) -> float:
        """Get current adaptive threshold for a metric."""
        if metric_name in self.adaptive_thresholds:
            return self.adaptive_thresholds[metric_name].current_value
        
        # Return default if not found
        defaults = {
            "test_coverage": 85.0,
            "code_quality": 85.0,
            "security_score": 90.0,
            "performance_score": 80.0,
        }
        
        return defaults.get(metric_name, 75.0)
    
    def update_context(self, context: Dict[str, Any]) -> None:
        """Update contextual factors."""
        current_hour = datetime.now().hour
        self.contextual_factors["time_of_day"].current_value = current_hour / 24.0
        
        # Update other factors based on context
        if "recent_commits" in context:
            self.contextual_factors["recent_changes"].current_value = min(1.0, context["recent_commits"] / 10.0)
        
        if "system_metrics" in context:
            system_metrics = context["system_metrics"]
            if "cpu_usage" in system_metrics:
                self.contextual_factors["system_load"].current_value = system_metrics["cpu_usage"] / 100.0
        
        # Project phase (based on git history or explicit setting)
        project_phase_map = {"initial": 0.2, "development": 0.5, "testing": 0.7, "production": 0.9}
        project_phase = context.get("project_phase", "development")
        self.contextual_factors["project_phase"].current_value = project_phase_map.get(project_phase, 0.5)
    
    def calculate_contextual_adjustment(self) -> float:
        """Calculate contextual adjustment factor."""
        if not self.config["adaptation_enabled"]:
            return 1.0
        
        total_adjustment = 0.0
        total_weight = 0.0
        
        for factor in self.contextual_factors.values():
            # Calculate adjustment based on factor value and learned impact
            if len(factor.impact_history) >= 5:
                # Use learned relationship
                recent_impacts = [h[1] for h in factor.impact_history[-10:]]
                avg_impact = sum(recent_impacts) / len(recent_impacts)
                adjustment = (factor.current_value - 0.5) * avg_impact * factor.weight
            else:
                # Use default heuristic
                adjustment = (factor.current_value - 0.5) * factor.weight * 0.1
            
            total_adjustment += adjustment
            total_weight += factor.weight
        
        # Normalize and apply context weight
        if total_weight > 0:
            normalized_adjustment = total_adjustment / total_weight
            contextual_factor = 1.0 + (normalized_adjustment * self.config["context_weight"])
            return max(0.5, min(1.5, contextual_factor))  # Bound adjustment
        
        return 1.0
    
    async def adapt_gate_execution(self, gate: QualityGate, context: Dict[str, Any]) -> QualityGate:
        """Adapt gate execution based on learned patterns."""
        if not self.config["adaptation_enabled"]:
            return gate
        
        # Update context
        self.update_context(context)
        
        # Calculate contextual adjustments
        contextual_factor = self.calculate_contextual_adjustment()
        
        # Create adapted gate if needed
        if hasattr(gate, 'threshold') and gate.name in self.adaptive_thresholds:
            adapted_gate = gate.__class__()  # Create new instance
            adapted_gate.name = gate.name
            adapted_gate.critical = gate.critical
            adapted_gate.timeout = gate.timeout
            adapted_gate.dependencies = gate.dependencies.copy()
            
            # Adjust threshold
            base_threshold = self.adaptive_thresholds[gate.name].current_value
            adapted_threshold = base_threshold * contextual_factor
            adapted_gate.threshold = max(50.0, min(100.0, adapted_threshold))  # Bound threshold
            
            self.logger.info(f"Adapted gate '{gate.name}' threshold: {base_threshold:.1f} -> {adapted_threshold:.1f}")
            
            return adapted_gate
        
        return gate
    
    def learn_from_execution(self, gate_name: str, result: QualityGateResult, context: Dict[str, Any]) -> None:
        """Learn from gate execution results."""
        if not self.config["learning_enabled"]:
            return
        
        # Record execution data
        execution_data = {
            "timestamp": datetime.now().isoformat(),
            "gate_name": gate_name,
            "result": result.to_dict(),
            "context": context.copy(),
        }
        
        self.learning_history.append(execution_data)
        
        # Update adaptive thresholds
        if gate_name in self.adaptive_thresholds:
            threshold = self.adaptive_thresholds[gate_name]
            success_rate = self._calculate_recent_success_rate(gate_name)
            threshold.update(success_rate, context)
        
        # Update contextual factor learning
        for factor_name, factor in self.contextual_factors.items():
            factor.learn_impact(factor.current_value, result.percentage_score)
        
        # Update performance baseline
        self._update_performance_baseline(gate_name, result.percentage_score)
        
        # Save state periodically
        if len(self.learning_history) % 10 == 0:
            self._save_state()
        
        self.logger.debug(f"Learned from execution of gate '{gate_name}' with score {result.percentage_score:.1f}%")
    
    def _calculate_recent_success_rate(self, gate_name: str) -> float:
        """Calculate recent success rate for a gate."""
        recent_executions = [
            e for e in self.learning_history[-20:]  # Last 20 executions
            if e["gate_name"] == gate_name
        ]
        
        if not recent_executions:
            return 0.8  # Default success rate
        
        successes = sum(1 for e in recent_executions if e["result"]["status"] == "passed")
        return successes / len(recent_executions)
    
    def _update_performance_baseline(self, gate_name: str, score: float) -> None:
        """Update performance baseline for a gate."""
        if gate_name not in self.performance_baseline:
            self.performance_baseline[gate_name] = []
        
        self.performance_baseline[gate_name].append(score)
        
        # Keep recent baseline window
        window_size = self.config["baseline_window"]
        if len(self.performance_baseline[gate_name]) > window_size:
            self.performance_baseline[gate_name] = self.performance_baseline[gate_name][-window_size:]
    
    def get_gate_recommendations(self, generation: GenerationType) -> List[Dict[str, Any]]:
        """Get recommendations for gate execution order and prioritization."""
        recommendations = []
        
        if not self.learning_history:
            return [{"type": "info", "message": "Insufficient data for recommendations"}]
        
        # Analyze gate performance patterns
        gate_performance = {}
        for execution in self.learning_history[-50:]:  # Recent executions
            gate_name = execution["gate_name"]
            score = execution["result"]["percentage_score"]
            
            if gate_name not in gate_performance:
                gate_performance[gate_name] = []
            gate_performance[gate_name].append(score)
        
        # Generate recommendations based on patterns
        for gate_name, scores in gate_performance.items():
            if len(scores) >= 5:
                avg_score = sum(scores) / len(scores)
                score_variance = np.var(scores)
                
                if avg_score < 70:
                    recommendations.append({
                        "type": "priority",
                        "gate": gate_name,
                        "message": f"Gate '{gate_name}' consistently underperforming (avg: {avg_score:.1f}%)",
                        "suggested_action": "Review gate criteria or improve implementation",
                        "priority": "high"
                    })
                
                # Calculate variance manually
                mean_score = sum(scores) / len(scores)
                variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
                
                if variance > 400:  # High variance
                    recommendations.append({
                        "type": "stability",
                        "gate": gate_name,
                        "message": f"Gate '{gate_name}' showing high score variance",
                        "suggested_action": "Investigate inconsistent behavior",
                        "priority": "medium"
                    })
        
        # Contextual recommendations
        current_context = self.calculate_contextual_adjustment()
        if current_context < 0.8:
            recommendations.append({
                "type": "context",
                "message": "Current context suggests running lighter quality checks",
                "suggested_action": "Consider reducing gate timeout or threshold temporarily",
                "priority": "low"
            })
        elif current_context > 1.2:
            recommendations.append({
                "type": "context",
                "message": "Current context is favorable for comprehensive quality checks",
                "suggested_action": "Consider running additional optional gates",
                "priority": "low"
            })
        
        return recommendations
    
    def predict_gate_performance(self, gate_name: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict likely gate performance based on context and history."""
        if gate_name not in self.performance_baseline:
            return {"predicted_score": 75.0, "confidence": 0.1}
        
        baseline_scores = self.performance_baseline[gate_name]
        
        if len(baseline_scores) < self.config["min_samples_for_adaptation"]:
            return {
                "predicted_score": sum(baseline_scores) / len(baseline_scores),
                "confidence": 0.3
            }
        
        # Simple prediction based on recent performance and context
        recent_avg = sum(baseline_scores[-10:]) / len(baseline_scores[-10:])
        contextual_adjustment = self.calculate_contextual_adjustment()
        
        predicted_score = recent_avg * contextual_adjustment
        confidence = min(1.0, len(baseline_scores) / 20.0)  # More data = higher confidence
        
        return {
            "predicted_score": max(0.0, min(100.0, predicted_score)),
            "confidence": confidence
        }
    
    def generate_adaptation_report(self) -> Dict[str, Any]:
        """Generate comprehensive adaptation report."""
        report = {
            "summary": {
                "learning_enabled": self.config["learning_enabled"],
                "adaptation_enabled": self.config["adaptation_enabled"],
                "executions_learned": len(self.learning_history),
                "adaptive_thresholds": len(self.adaptive_thresholds),
                "contextual_factors": len(self.contextual_factors),
            },
            "threshold_status": {},
            "contextual_factors": {},
            "learning_insights": self._generate_learning_insights(),
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(),
        }
        
        # Threshold status
        for name, threshold in self.adaptive_thresholds.items():
            report["threshold_status"][name] = {
                "base_value": threshold.base_value,
                "current_value": threshold.current_value,
                "adaptation_amount": threshold.current_value - threshold.base_value,
                "confidence": threshold.confidence,
                "stability": 1.0 - (self._calculate_std(threshold.history[-10:]) / 10.0) if len(threshold.history) >= 10 else 0.5,
            }
        
        # Contextual factors
        for name, factor in self.contextual_factors.items():
            report["contextual_factors"][name] = {
                "current_value": factor.current_value,
                "weight": factor.weight,
                "learned_samples": len(factor.impact_history),
                "impact_correlation": self._calculate_factor_correlation(factor),
            }
        
        return report
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation manually."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _generate_learning_insights(self) -> List[str]:
        """Generate insights from learning history."""
        insights = []
        
        if len(self.learning_history) < 10:
            insights.append("Insufficient data for learning insights")
            return insights
        
        # Analyze trends
        recent_scores = [e["result"]["percentage_score"] for e in self.learning_history[-20:]]
        older_scores = [e["result"]["percentage_score"] for e in self.learning_history[-40:-20]] if len(self.learning_history) >= 40 else []
        
        if older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            
            if recent_avg > older_avg + 5:
                insights.append(f"Quality improving: {recent_avg:.1f}% vs {older_avg:.1f}% previously")
            elif recent_avg < older_avg - 5:
                insights.append(f"Quality declining: {recent_avg:.1f}% vs {older_avg:.1f}% previously")
            else:
                insights.append("Quality stable over recent period")
        
        # Analyze adaptation effectiveness
        adapted_gates = [e for e in self.learning_history[-30:] if "adapted" in e.get("context", {})]
        if adapted_gates:
            adapted_scores = [e["result"]["percentage_score"] for e in adapted_gates]
            non_adapted_scores = [e["result"]["percentage_score"] for e in self.learning_history[-30:] if e not in adapted_gates]
            
            if adapted_scores and non_adapted_scores:
                adapted_avg = sum(adapted_scores) / len(adapted_scores)
                non_adapted_avg = sum(non_adapted_scores) / len(non_adapted_scores)
                
                if adapted_avg > non_adapted_avg + 3:
                    insights.append(f"Adaptation improving performance: {adapted_avg:.1f}% vs {non_adapted_avg:.1f}%")
        
        return insights
    
    def _calculate_adaptation_effectiveness(self) -> float:
        """Calculate how effective the adaptation has been."""
        if len(self.learning_history) < 20:
            return 0.0
        
        # Compare recent performance with baseline
        recent_executions = self.learning_history[-20:]
        baseline_executions = self.learning_history[-40:-20] if len(self.learning_history) >= 40 else []
        
        if not baseline_executions:
            return 0.0
        
        recent_avg = sum(e["result"]["percentage_score"] for e in recent_executions) / len(recent_executions)
        baseline_avg = sum(e["result"]["percentage_score"] for e in baseline_executions) / len(baseline_executions)
        
        # Effectiveness as percentage improvement
        if baseline_avg > 0:
            effectiveness = ((recent_avg - baseline_avg) / baseline_avg) * 100
            return max(-50.0, min(50.0, effectiveness))  # Bound between -50% and +50%
        
        return 0.0
    
    def _calculate_factor_correlation(self, factor: ContextualFactor) -> float:
        """Calculate correlation between factor and quality outcomes."""
        if len(factor.impact_history) < 5:
            return 0.0
        
        values = [h[0] for h in factor.impact_history]
        outcomes = [h[1] for h in factor.impact_history]
        
        if len(set(values)) <= 1:  # No variance in values
            return 0.0
        
        if len(values) != len(outcomes) or len(values) < 2:
            return 0.0
        
        mean_values = sum(values) / len(values)
        mean_outcomes = sum(outcomes) / len(outcomes)
        
        numerator = sum((values[i] - mean_values) * (outcomes[i] - mean_outcomes) for i in range(len(values)))
        sum_sq_values = sum((values[i] - mean_values) ** 2 for i in range(len(values)))
        sum_sq_outcomes = sum((outcomes[i] - mean_outcomes) ** 2 for i in range(len(outcomes)))
        
        denominator = (sum_sq_values * sum_sq_outcomes) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation if abs(correlation) <= 1.0 else 0.0