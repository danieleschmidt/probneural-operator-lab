"""Monitoring and observability package for ProbNeural Operator Lab."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from probneural_operator.utils.logging import setup_logging, get_logger
from probneural_operator.utils.metrics import get_global_metrics, start_system_monitoring
from monitoring.health_checks import get_health_monitor, add_default_health_checks


def load_monitoring_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load monitoring configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        # Return default configuration
        return {
            "logging": {"level": "INFO"},
            "metrics": {"enabled": True},
            "health_checks": {"enabled": True},
            "dashboard": {"enabled": True}
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_monitoring(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Setup comprehensive monitoring system based on configuration.
    
    Args:
        config_path: Path to monitoring configuration file
        
    Returns:
        Dictionary containing monitoring system components
    """
    # Load configuration
    config = load_monitoring_config(config_path)
    
    # Setup logging
    logging_config = config.get("logging", {})
    loggers = setup_logging(
        log_level=logging_config.get("level", "INFO"),
        log_dir=logging_config.get("log_dir"),
        console_output=logging_config.get("console_output", True),
        json_output=logging_config.get("json_output", False),
        file_output=logging_config.get("file_output", True),
        performance_logging=logging_config.get("performance_logging", True)
    )
    
    logger = get_logger("monitoring")
    logger.info("🔧 Setting up monitoring system")
    
    # Setup metrics collection
    metrics_config = config.get("metrics", {})
    metrics_collector = None
    
    if metrics_config.get("enabled", True):
        metrics_collector = get_global_metrics()
        
        # Start system monitoring if enabled
        system_monitoring = metrics_config.get("system_monitoring", {})
        if system_monitoring.get("enabled", True):
            interval = system_monitoring.get("interval", 5.0)
            start_system_monitoring(interval)
            logger.info(f"📊 Started system metrics collection (interval: {interval}s)")
    
    # Setup health checks
    health_config = config.get("health_checks", {})
    health_monitor = None
    
    if health_config.get("enabled", True):
        health_monitor = get_health_monitor()
        
        # Add default health checks
        add_default_health_checks()
        
        # Start health monitoring
        monitoring_interval = health_config.get("monitoring_interval", 60.0)
        health_monitor.start_monitoring(monitoring_interval)
        logger.info(f"🏥 Started health monitoring (interval: {monitoring_interval}s)")
    
    # Setup dashboard (optional, requires Flask)
    dashboard = None
    dashboard_config = config.get("dashboard", {})
    
    if dashboard_config.get("enabled", True):
        try:
            from monitoring.dashboard import create_dashboard
            dashboard = create_dashboard(metrics_collector, health_monitor)
            logger.info("📊 Dashboard created (call start_dashboard() to run server)")
        except ImportError as e:
            logger.warning(f"Dashboard not available: {e}")
    
    logger.info("✅ Monitoring system setup complete")
    
    return {
        "config": config,
        "loggers": loggers,
        "metrics_collector": metrics_collector,
        "health_monitor": health_monitor,
        "dashboard": dashboard
    }


def start_dashboard(monitoring_components: Optional[Dict[str, Any]] = None, 
                   host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    """Start the monitoring dashboard server.
    
    Args:
        monitoring_components: Components returned from setup_monitoring()
        host: Host to bind dashboard server
        port: Port to bind dashboard server
        debug: Enable debug mode
    """
    if monitoring_components is None:
        monitoring_components = setup_monitoring()
    
    dashboard = monitoring_components.get("dashboard")
    if dashboard is None:
        logger = get_logger("monitoring")
        logger.error("Dashboard not available. Ensure Flask is installed.")
        return
    
    # Override host/port from config if provided
    dashboard.host = host
    dashboard.port = port
    
    logger = get_logger("monitoring")
    logger.info(f"🚀 Starting dashboard server on http://{host}:{port}")
    
    dashboard.run(debug=debug)


def stop_monitoring(monitoring_components: Dict[str, Any]):
    """Stop all monitoring components.
    
    Args:
        monitoring_components: Components returned from setup_monitoring()
    """
    logger = get_logger("monitoring")
    logger.info("🛑 Stopping monitoring system")
    
    # Stop system monitoring
    from probneural_operator.utils.metrics import stop_system_monitoring
    stop_system_monitoring()
    
    # Stop health monitoring
    health_monitor = monitoring_components.get("health_monitor")
    if health_monitor:
        health_monitor.stop_monitoring()
    
    # Flush metrics
    metrics_collector = monitoring_components.get("metrics_collector")
    if metrics_collector:
        metrics_collector.flush()
    
    logger.info("✅ Monitoring system stopped")


# Convenience functions for quick setup
def quick_setup(log_level: str = "INFO", enable_dashboard: bool = True) -> Dict[str, Any]:
    """Quick monitoring setup with sensible defaults."""
    return setup_monitoring()


def start_monitoring_server(host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    """Start monitoring system and dashboard server in one call."""
    components = setup_monitoring()
    start_dashboard(components, host, port, debug)


# Auto-setup when imported (can be disabled by setting environment variable)
if not os.getenv("DISABLE_AUTO_MONITORING"):
    try:
        _monitoring_components = setup_monitoring()
    except Exception as e:
        logger = get_logger("monitoring")
        logger.warning(f"Auto-monitoring setup failed: {e}")