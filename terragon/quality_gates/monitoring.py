"""
Continuous Quality Monitoring System
====================================

Real-time monitoring and adaptive quality control for progressive SDLC.
Provides continuous feedback, trend analysis, and proactive quality assurance.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import logging

from .core import QualityGateFramework, QualityGateResult, QualityGateStatus


@dataclass
class QualityMetrics:
    """Quality metrics tracking."""
    timestamp: datetime
    overall_score: float
    gate_scores: Dict[str, float] = field(default_factory=dict)
    trend_data: Dict[str, List[float]] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "gate_scores": self.gate_scores,
            "trend_data": self.trend_data,
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


class QualityTrendAnalyzer:
    """Analyzes quality trends and patterns."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: List[QualityMetrics] = []
    
    def add_metrics(self, metrics: QualityMetrics) -> None:
        """Add new metrics to history."""
        self.history.append(metrics)
        
        # Keep only recent history
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size:]
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends."""
        if len(self.history) < 3:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        recent_scores = [m.overall_score for m in self.history[-self.window_size:]]
        
        # Calculate trend direction
        if len(recent_scores) >= 3:
            first_third = sum(recent_scores[:len(recent_scores)//3]) / (len(recent_scores)//3)
            last_third = sum(recent_scores[-len(recent_scores)//3:]) / (len(recent_scores)//3)
            
            trend_direction = "improving" if last_third > first_third else "declining"
            trend_strength = abs(last_third - first_third) / 100.0
        else:
            trend_direction = "stable"
            trend_strength = 0.0
        
        # Volatility analysis
        if len(recent_scores) >= 2:
            volatility = sum(abs(recent_scores[i] - recent_scores[i-1]) 
                           for i in range(1, len(recent_scores))) / (len(recent_scores) - 1)
        else:
            volatility = 0.0
        
        return {
            "trend": trend_direction,
            "strength": trend_strength,
            "volatility": volatility,
            "confidence": min(1.0, len(recent_scores) / self.window_size),
            "recent_average": sum(recent_scores) / len(recent_scores),
            "recent_scores": recent_scores,
        }
    
    def detect_anomalies(self) -> List[str]:
        """Detect quality anomalies."""
        anomalies = []
        
        if len(self.history) < 5:
            return anomalies
        
        recent_metrics = self.history[-5:]
        
        # Check for sudden quality drops
        for i in range(1, len(recent_metrics)):
            score_drop = recent_metrics[i-1].overall_score - recent_metrics[i].overall_score
            if score_drop > 20:  # 20% drop
                anomalies.append(f"Sudden quality drop of {score_drop:.1f}% detected")
        
        # Check for consistently low scores
        recent_scores = [m.overall_score for m in recent_metrics]
        if all(score < 70 for score in recent_scores):
            anomalies.append("Consistently low quality scores detected")
        
        # Check for high volatility
        if len(recent_scores) >= 3:
            volatility = sum(abs(recent_scores[i] - recent_scores[i-1]) 
                           for i in range(1, len(recent_scores))) / (len(recent_scores) - 1)
            if volatility > 15:  # High volatility threshold
                anomalies.append(f"High quality volatility detected (±{volatility:.1f}%)")
        
        return anomalies
    
    def generate_recommendations(self) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if len(self.history) < 3:
            return ["Insufficient data for recommendations"]
        
        trends = self.analyze_trends()
        recent_metrics = self.history[-3:]
        
        # Analyze recent performance
        if trends["trend"] == "declining":
            recommendations.append("Quality trend is declining - review recent changes")
        
        if trends["volatility"] > 10:
            recommendations.append("Quality is unstable - consider improving test consistency")
        
        # Analyze specific gate performance
        gate_scores = {}
        for metrics in recent_metrics:
            for gate, score in metrics.gate_scores.items():
                if gate not in gate_scores:
                    gate_scores[gate] = []
                gate_scores[gate].append(score)
        
        # Find consistently poor performing gates
        for gate, scores in gate_scores.items():
            if len(scores) >= 2 and all(score < 70 for score in scores):
                recommendations.append(f"Gate '{gate}' consistently underperforming - needs attention")
        
        return recommendations


class ContinuousQualityMonitor:
    """
    Continuous quality monitoring system.
    
    Monitors quality gates execution, tracks trends, and provides
    real-time feedback and recommendations.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(".terragon/quality_monitor.json")
        self.framework = QualityGateFramework()
        self.trend_analyzer = QualityTrendAnalyzer()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[QualityMetrics] = []
        
        # Configuration
        self.config = {
            "monitoring_interval": 300,  # 5 minutes
            "alert_thresholds": {
                "quality_drop": 15.0,
                "volatility": 20.0,
                "low_score": 60.0,
            },
            "trend_window": 10,
            "max_history": 100,
            "auto_remediation": True,
        }
        
        self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> None:
        """Load monitoring configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
            except Exception as e:
                logging.warning(f"Could not load monitoring config: {e}")
    
    def _setup_logging(self) -> None:
        """Setup logging for monitoring."""
        log_dir = Path(".terragon/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "quality_monitor.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("QualityMonitor")
    
    def start_monitoring(self) -> None:
        """Start continuous quality monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Continuous quality monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous quality monitoring."""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("Continuous quality monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run quality assessment
                metrics = asyncio.run(self._assess_quality())
                
                # Add to trend analyzer
                self.trend_analyzer.add_metrics(metrics)
                self.metrics_history.append(metrics)
                
                # Keep history manageable
                if len(self.metrics_history) > self.config["max_history"]:
                    self.metrics_history = self.metrics_history[-self.config["max_history"]:]
                
                # Analyze trends and generate alerts
                self._process_metrics(metrics)
                
                # Save metrics
                self._save_metrics(metrics)
                
                # Sleep until next assessment
                time.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    async def _assess_quality(self) -> QualityMetrics:
        """Assess current quality state."""
        from .generations import Generation1Gates, Generation2Gates, Generation3Gates
        
        # Determine current generation and run appropriate gates
        current_generation = self.framework.current_generation
        
        if current_generation.value == "generation_1":
            gates = Generation1Gates().get_gates()
        elif current_generation.value == "generation_2":
            gates = Generation2Gates().get_gates()
        else:
            gates = Generation3Gates().get_gates()
        
        # Register and execute gates
        self.framework.register_gates(gates)
        results = await self.framework.execute_generation(current_generation)
        
        # Calculate metrics
        gate_scores = {name: result.percentage_score for name, result in results.items()}
        overall_score = sum(gate_scores.values()) / len(gate_scores) if gate_scores else 0
        
        # Generate alerts and recommendations
        alerts = self._generate_alerts(gate_scores, overall_score)
        recommendations = self._generate_recommendations(results)
        
        return QualityMetrics(
            timestamp=datetime.now(),
            overall_score=overall_score,
            gate_scores=gate_scores,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _generate_alerts(self, gate_scores: Dict[str, float], overall_score: float) -> List[str]:
        """Generate quality alerts."""
        alerts = []
        thresholds = self.config["alert_thresholds"]
        
        # Overall quality alerts
        if overall_score < thresholds["low_score"]:
            alerts.append(f"Overall quality score {overall_score:.1f}% below threshold")
        
        # Individual gate alerts
        for gate, score in gate_scores.items():
            if score < thresholds["low_score"]:
                alerts.append(f"Gate '{gate}' score {score:.1f}% below threshold")
        
        # Trend-based alerts
        if len(self.metrics_history) >= 2:
            prev_score = self.metrics_history[-1].overall_score
            score_change = overall_score - prev_score
            
            if score_change < -thresholds["quality_drop"]:
                alerts.append(f"Quality dropped by {abs(score_change):.1f}% since last assessment")
        
        # Anomaly detection
        anomalies = self.trend_analyzer.detect_anomalies()
        alerts.extend(anomalies)
        
        return alerts
    
    def _generate_recommendations(self, results: Dict[str, QualityGateResult]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Analyze failed gates
        failed_gates = [r for r in results.values() if not r.passed]
        
        if failed_gates:
            failure_types = {}
            for result in failed_gates:
                for error in result.errors:
                    failure_type = self._classify_failure_type(error)
                    failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
            
            # Recommend based on most common failures
            if failure_types:
                most_common = max(failure_types.items(), key=lambda x: x[1])
                recommendations.append(f"Focus on {most_common[0]} issues - {most_common[1]} failures detected")
        
        # Trend-based recommendations
        trend_recommendations = self.trend_analyzer.generate_recommendations()
        recommendations.extend(trend_recommendations)
        
        return recommendations
    
    def _classify_failure_type(self, error: str) -> str:
        """Classify failure type for recommendations."""
        error_lower = error.lower()
        
        if "test" in error_lower:
            return "testing"
        elif "security" in error_lower:
            return "security"
        elif "performance" in error_lower:
            return "performance"
        elif "style" in error_lower or "lint" in error_lower:
            return "code_quality"
        else:
            return "general"
    
    def _process_metrics(self, metrics: QualityMetrics) -> None:
        """Process metrics and trigger actions."""
        # Log alerts
        for alert in metrics.alerts:
            self.logger.warning(f"Quality Alert: {alert}")
        
        # Log recommendations
        for recommendation in metrics.recommendations:
            self.logger.info(f"Recommendation: {recommendation}")
        
        # Auto-remediation if enabled
        if self.config["auto_remediation"] and metrics.alerts:
            asyncio.run(self._attempt_auto_remediation(metrics))
    
    async def _attempt_auto_remediation(self, metrics: QualityMetrics) -> None:
        """Attempt automatic remediation of quality issues."""
        self.logger.info("Attempting auto-remediation of quality issues")
        
        # Implementation would include:
        # - Running automatic fixes for common issues
        # - Triggering re-execution of failed gates
        # - Adjusting thresholds if appropriate
        # - Scheduling follow-up assessments
        
        # For now, just log the attempt
        for alert in metrics.alerts:
            if "test" in alert.lower():
                self.logger.info("Would attempt to fix test issues")
            elif "security" in alert.lower():
                self.logger.info("Would attempt to fix security issues")
            elif "performance" in alert.lower():
                self.logger.info("Would attempt to optimize performance")
    
    def _save_metrics(self, metrics: QualityMetrics) -> None:
        """Save metrics to disk."""
        metrics_dir = Path(".terragon/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current metrics
        metrics_file = metrics_dir / f"quality_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save metrics: {e}")
        
        # Update summary file
        summary_file = metrics_dir / "quality_summary.json"
        summary = {
            "last_updated": metrics.timestamp.isoformat(),
            "overall_score": metrics.overall_score,
            "gate_scores": metrics.gate_scores,
            "alerts_count": len(metrics.alerts),
            "recommendations_count": len(metrics.recommendations),
            "trend_analysis": self.trend_analyzer.analyze_trends(),
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save summary: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if not self.metrics_history:
            return {"status": "no_data", "monitoring_active": self.monitoring_active}
        
        latest_metrics = self.metrics_history[-1]
        trends = self.trend_analyzer.analyze_trends()
        
        return {
            "status": "active" if self.monitoring_active else "stopped",
            "monitoring_active": self.monitoring_active,
            "latest_assessment": latest_metrics.timestamp.isoformat(),
            "overall_score": latest_metrics.overall_score,
            "alert_count": len(latest_metrics.alerts),
            "recommendation_count": len(latest_metrics.recommendations),
            "trend_analysis": trends,
            "assessment_count": len(self.metrics_history),
        }
    
    def get_historical_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical quality data."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics.to_dict() 
            for metrics in self.metrics_history 
            if metrics.timestamp >= cutoff_time
        ]
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.metrics_history:
            return {"error": "No quality data available"}
        
        latest = self.metrics_history[-1]
        trends = self.trend_analyzer.analyze_trends()
        
        # Calculate statistics
        recent_scores = [m.overall_score for m in self.metrics_history[-10:]]
        
        report = {
            "summary": {
                "report_generated": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active,
                "assessments_count": len(self.metrics_history),
                "current_score": latest.overall_score,
                "average_score": sum(recent_scores) / len(recent_scores),
                "best_score": max(recent_scores),
                "worst_score": min(recent_scores),
            },
            "current_status": {
                "overall_score": latest.overall_score,
                "gate_scores": latest.gate_scores,
                "active_alerts": latest.alerts,
                "recommendations": latest.recommendations,
            },
            "trend_analysis": trends,
            "historical_data": self.get_historical_data(24),
            "improvement_opportunities": self._identify_improvement_opportunities(),
        }
        
        return report
    
    def _identify_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify quality improvement opportunities."""
        if len(self.metrics_history) < 3:
            return []
        
        opportunities = []
        
        # Analyze gate performance over time
        gate_performance = {}
        for metrics in self.metrics_history[-5:]:  # Last 5 assessments
            for gate, score in metrics.gate_scores.items():
                if gate not in gate_performance:
                    gate_performance[gate] = []
                gate_performance[gate].append(score)
        
        # Find consistently underperforming gates
        for gate, scores in gate_performance.items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)
                if avg_score < 75:
                    opportunities.append({
                        "type": "gate_improvement",
                        "gate": gate,
                        "average_score": avg_score,
                        "priority": "high" if avg_score < 60 else "medium",
                        "description": f"Gate '{gate}' consistently scoring below expectations"
                    })
        
        # Find volatile areas
        for gate, scores in gate_performance.items():
            if len(scores) >= 3:
                volatility = sum(abs(scores[i] - scores[i-1]) for i in range(1, len(scores))) / (len(scores) - 1)
                if volatility > 15:
                    opportunities.append({
                        "type": "stability_improvement",
                        "gate": gate,
                        "volatility": volatility,
                        "priority": "medium",
                        "description": f"Gate '{gate}' showing high score volatility"
                    })
        
        return opportunities