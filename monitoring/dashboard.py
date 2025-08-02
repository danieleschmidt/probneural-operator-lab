"""Real-time monitoring dashboard for ProbNeural Operator Lab."""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import time

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from flask import Flask, render_template, jsonify, request
    import flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from probneural_operator.utils.metrics import MetricsCollector, get_global_metrics
from probneural_operator.utils.logging import get_logger
from monitoring.health_checks import HealthMonitor, get_health_monitor


class DashboardData:
    """Data provider for the monitoring dashboard."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 health_monitor: Optional[HealthMonitor] = None):
        self.metrics = metrics_collector or get_global_metrics()
        self.health = health_monitor or get_health_monitor()
        self.logger = get_logger("dashboard")
    
    def get_system_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get system metrics for the dashboard."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get all metrics since the specified time
        all_metrics = self.metrics.get_metrics(since=since)
        
        # Group metrics by type
        system_metrics = {}
        
        for metric in all_metrics:
            if metric.name.startswith("system."):
                metric_type = metric.name.replace("system.", "")
                
                if metric_type not in system_metrics:
                    system_metrics[metric_type] = {
                        "timestamps": [],
                        "values": [],
                        "unit": metric.unit
                    }
                
                system_metrics[metric_type]["timestamps"].append(metric.timestamp)
                system_metrics[metric_type]["values"].append(metric.value)
        
        return system_metrics
    
    def get_gpu_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get GPU metrics for the dashboard."""
        since = datetime.utcnow() - timedelta(hours=hours)
        all_metrics = self.metrics.get_metrics(since=since)
        
        gpu_metrics = {}
        
        for metric in all_metrics:
            if metric.name.startswith("gpu."):
                gpu_id = metric.tags.get("gpu", "0")
                metric_type = metric.name.replace("gpu.", "")
                
                key = f"gpu_{gpu_id}_{metric_type}"
                
                if key not in gpu_metrics:
                    gpu_metrics[key] = {
                        "timestamps": [],
                        "values": [],
                        "unit": metric.unit,
                        "gpu_id": gpu_id,
                        "metric_type": metric_type
                    }
                
                gpu_metrics[key]["timestamps"].append(metric.timestamp)
                gpu_metrics[key]["values"].append(metric.value)
        
        return gpu_metrics
    
    def get_experiment_metrics(self, experiment_name: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get experiment metrics for the dashboard."""
        since = datetime.utcnow() - timedelta(hours=hours)
        all_metrics = self.metrics.get_metrics(since=since)
        
        experiment_metrics = {}
        
        for metric in all_metrics:
            if metric.name.startswith("experiment."):
                exp_name = metric.tags.get("experiment")
                
                # Filter by experiment name if specified
                if experiment_name and exp_name != experiment_name:
                    continue
                
                metric_type = metric.name.replace("experiment.", "")
                
                if exp_name not in experiment_metrics:
                    experiment_metrics[exp_name] = {}
                
                if metric_type not in experiment_metrics[exp_name]:
                    experiment_metrics[exp_name][metric_type] = {
                        "timestamps": [],
                        "values": [],
                        "unit": metric.unit,
                        "tags": metric.tags
                    }
                
                experiment_metrics[exp_name][metric_type]["timestamps"].append(metric.timestamp)
                experiment_metrics[exp_name][metric_type]["values"].append(metric.value)
        
        return experiment_metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health.get_health_report()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dashboard."""
        # Recent metrics (last hour)
        recent_metrics = self.metrics.get_metrics(since=datetime.utcnow() - timedelta(hours=1))
        
        stats = {
            "total_metrics": len(recent_metrics),
            "unique_metric_names": len(set(m.name for m in recent_metrics)),
            "time_range": {
                "start": min(m.timestamp for m in recent_metrics).isoformat() if recent_metrics else None,
                "end": max(m.timestamp for m in recent_metrics).isoformat() if recent_metrics else None
            },
            "health_status": self.health.get_overall_status().value,
            "active_experiments": len(set(m.tags.get("experiment") for m in recent_metrics 
                                        if m.name.startswith("experiment.") and m.tags.get("experiment")))
        }
        
        return stats


class PlotlyDashboard:
    """Generate Plotly-based dashboard visualizations."""
    
    def __init__(self, data_provider: DashboardData):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for dashboard functionality")
        
        self.data = data_provider
        self.logger = get_logger("plotly_dashboard")
    
    def create_system_metrics_plot(self, hours: int = 1) -> str:
        """Create system metrics plot."""
        system_metrics = self.data.get_system_metrics(hours)
        
        if not system_metrics:
            return "<p>No system metrics available</p>"
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["CPU Usage", "Memory Usage", "Disk Usage", "Network I/O"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU Usage
        if "cpu.percent" in system_metrics:
            cpu_data = system_metrics["cpu.percent"]
            fig.add_trace(
                go.Scatter(x=cpu_data["timestamps"], y=cpu_data["values"],
                          name="CPU %", line=dict(color="red")),
                row=1, col=1
            )
        
        # Memory Usage
        if "memory.percent" in system_metrics:
            mem_data = system_metrics["memory.percent"]
            fig.add_trace(
                go.Scatter(x=mem_data["timestamps"], y=mem_data["values"],
                          name="Memory %", line=dict(color="blue")),
                row=1, col=2
            )
        
        # Disk Usage
        if "disk.percent" in system_metrics:
            disk_data = system_metrics["disk.percent"]
            fig.add_trace(
                go.Scatter(x=disk_data["timestamps"], y=disk_data["values"],
                          name="Disk %", line=dict(color="green")),
                row=2, col=1
            )
        
        fig.update_layout(
            title="System Metrics Dashboard",
            height=600,
            showlegend=False
        )
        
        return fig.to_html(include_plotlyjs=True, div_id="system-metrics-plot")
    
    def create_gpu_metrics_plot(self, hours: int = 1) -> str:
        """Create GPU metrics plot."""
        gpu_metrics = self.data.get_gpu_metrics(hours)
        
        if not gpu_metrics:
            return "<p>No GPU metrics available</p>"
        
        # Group by GPU
        gpus = set(data["gpu_id"] for data in gpu_metrics.values())
        
        if not gpus:
            return "<p>No GPU data found</p>"
        
        fig = make_subplots(
            rows=len(gpus), cols=2,
            subplot_titles=[f"GPU {gpu} Memory" for gpu in sorted(gpus)] + 
                          [f"GPU {gpu} Utilization" for gpu in sorted(gpus)],
            vertical_spacing=0.1
        )
        
        row = 1
        for gpu_id in sorted(gpus):
            # Memory usage
            memory_key = f"gpu_{gpu_id}_memory.allocated"
            if memory_key in gpu_metrics:
                data = gpu_metrics[memory_key]
                fig.add_trace(
                    go.Scatter(x=data["timestamps"], y=data["values"],
                              name=f"GPU {gpu_id} Memory", line=dict(color="purple")),
                    row=row, col=1
                )
            
            # Utilization
            util_key = f"gpu_{gpu_id}_utilization"
            if util_key in gpu_metrics:
                data = gpu_metrics[util_key]
                fig.add_trace(
                    go.Scatter(x=data["timestamps"], y=data["values"],
                              name=f"GPU {gpu_id} Util", line=dict(color="orange")),
                    row=row, col=2
                )
            
            row += 1
        
        fig.update_layout(
            title="GPU Metrics Dashboard",
            height=300 * len(gpus),
            showlegend=False
        )
        
        return fig.to_html(include_plotlyjs=True, div_id="gpu-metrics-plot")
    
    def create_experiment_metrics_plot(self, experiment_name: Optional[str] = None, hours: int = 24) -> str:
        """Create experiment metrics plot."""
        experiment_metrics = self.data.get_experiment_metrics(experiment_name, hours)
        
        if not experiment_metrics:
            return "<p>No experiment metrics available</p>"
        
        # Create subplots for each experiment
        experiments = list(experiment_metrics.keys())
        
        if not experiments:
            return "<p>No experiments found</p>"
        
        fig = make_subplots(
            rows=len(experiments), cols=1,
            subplot_titles=[f"Experiment: {exp}" for exp in experiments],
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        row = 1
        for exp_name in experiments:
            exp_data = experiment_metrics[exp_name]
            color_idx = 0
            
            for metric_name, metric_data in exp_data.items():
                if metric_name in ["loss", "accuracy"]:  # Focus on key metrics
                    fig.add_trace(
                        go.Scatter(
                            x=metric_data["timestamps"],
                            y=metric_data["values"],
                            name=f"{exp_name} - {metric_name}",
                            line=dict(color=colors[color_idx % len(colors)])
                        ),
                        row=row, col=1
                    )
                    color_idx += 1
            
            row += 1
        
        fig.update_layout(
            title="Experiment Metrics Dashboard",
            height=400 * len(experiments),
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=True, div_id="experiment-metrics-plot")


class FlaskDashboard:
    """Flask-based web dashboard."""
    
    def __init__(self, data_provider: DashboardData, host: str = "0.0.0.0", port: int = 8050):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web dashboard functionality")
        
        self.data = data_provider
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.logger = get_logger("flask_dashboard")
        
        if PLOTLY_AVAILABLE:
            self.plotly_dashboard = PlotlyDashboard(data_provider)
        else:
            self.plotly_dashboard = None
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route("/")
        def index():
            """Main dashboard page."""
            return self._render_dashboard()
        
        @self.app.route("/api/health")
        def health_api():
            """Health status API endpoint."""
            return jsonify(self.data.get_health_status())
        
        @self.app.route("/api/summary")
        def summary_api():
            """Summary statistics API endpoint."""
            return jsonify(self.data.get_summary_stats())
        
        @self.app.route("/api/system_metrics")
        def system_metrics_api():
            """System metrics API endpoint."""
            hours = int(request.args.get("hours", 1))
            return jsonify(self.data.get_system_metrics(hours))
        
        @self.app.route("/api/gpu_metrics")
        def gpu_metrics_api():
            """GPU metrics API endpoint."""
            hours = int(request.args.get("hours", 1))
            return jsonify(self.data.get_gpu_metrics(hours))
        
        @self.app.route("/api/experiment_metrics")
        def experiment_metrics_api():
            """Experiment metrics API endpoint."""
            hours = int(request.args.get("hours", 24))
            experiment = request.args.get("experiment")
            return jsonify(self.data.get_experiment_metrics(experiment, hours))
    
    def _render_dashboard(self) -> str:
        """Render the main dashboard HTML."""
        summary = self.data.get_summary_stats()
        health = self.data.get_health_status()
        
        # Generate plots if Plotly is available
        system_plot = ""
        gpu_plot = ""
        experiment_plot = ""
        
        if self.plotly_dashboard:
            try:
                system_plot = self.plotly_dashboard.create_system_metrics_plot()
                gpu_plot = self.plotly_dashboard.create_gpu_metrics_plot()
                experiment_plot = self.plotly_dashboard.create_experiment_metrics_plot()
            except Exception as e:
                self.logger.error(f"Error generating plots: {e}")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ProbNeural Operator Lab - Monitoring Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
                .summary-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .health-status {{ margin-bottom: 30px; }}
                .health-{health["overall_status"]} {{ 
                    color: {"green" if health["overall_status"] == "healthy" else "red" if health["overall_status"] == "critical" else "orange"}; 
                }}
                .plot-container {{ margin-bottom: 30px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .refresh-btn {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }}
                .refresh-btn:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 ProbNeural Operator Lab - Monitoring Dashboard</h1>
                <p>Real-time system and experiment monitoring</p>
                <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
            </div>
            
            <div class="summary">
                <div class="summary-card">
                    <h3>📊 Metrics</h3>
                    <p>Total: {summary.get("total_metrics", 0)}</p>
                    <p>Unique: {summary.get("unique_metric_names", 0)}</p>
                </div>
                <div class="summary-card">
                    <h3>🏥 Health Status</h3>
                    <p class="health-{health["overall_status"]}">
                        {health["overall_status"].upper()}
                    </p>
                </div>
                <div class="summary-card">
                    <h3>🧪 Experiments</h3>
                    <p>Active: {summary.get("active_experiments", 0)}</p>
                </div>
            </div>
            
            <div class="health-status">
                <h2>🏥 Health Checks</h2>
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    {self._render_health_table(health)}
                </div>
            </div>
            
            <div class="plot-container">
                <h2>🖥️ System Metrics</h2>
                {system_plot if system_plot else "<p>System metrics visualization not available (Plotly required)</p>"}
            </div>
            
            <div class="plot-container">
                <h2>🎮 GPU Metrics</h2>
                {gpu_plot if gpu_plot else "<p>GPU metrics visualization not available</p>"}
            </div>
            
            <div class="plot-container">
                <h2>🧪 Experiment Metrics</h2>
                {experiment_plot if experiment_plot else "<p>Experiment metrics visualization not available</p>"}
            </div>
            
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(function() {{
                    location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _render_health_table(self, health_data: Dict[str, Any]) -> str:
        """Render health check results as HTML table."""
        if not health_data.get("checks"):
            return "<p>No health check data available</p>"
        
        table_rows = []
        for check_name, check_data in health_data["checks"].items():
            status = check_data["status"]
            status_color = {
                "healthy": "green",
                "warning": "orange", 
                "critical": "red",
                "unknown": "gray"
            }.get(status, "gray")
            
            table_rows.append(f"""
                <tr>
                    <td>{check_name}</td>
                    <td style="color: {status_color}; font-weight: bold;">{status.upper()}</td>
                    <td>{check_data["message"]}</td>
                    <td>{check_data["duration"]:.3f}s</td>
                </tr>
            """)
        
        return f"""
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Check</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Status</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Message</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        """
    
    def run(self, debug: bool = False):
        """Run the Flask dashboard server."""
        self.logger.info(f"Starting dashboard server on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


def create_dashboard(metrics_collector: Optional[MetricsCollector] = None,
                    health_monitor: Optional[HealthMonitor] = None,
                    dashboard_type: str = "flask") -> FlaskDashboard:
    """Create and configure a monitoring dashboard.
    
    Args:
        metrics_collector: Metrics collector instance
        health_monitor: Health monitor instance
        dashboard_type: Type of dashboard ("flask")
    
    Returns:
        Dashboard instance
    """
    data_provider = DashboardData(metrics_collector, health_monitor)
    
    if dashboard_type == "flask":
        return FlaskDashboard(data_provider)
    else:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")


def start_dashboard_server(host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    """Start the monitoring dashboard server."""
    dashboard = create_dashboard()
    dashboard.run(debug=debug)