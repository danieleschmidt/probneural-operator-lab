# Monitoring & Observability Guide

This guide covers the comprehensive monitoring and observability system for the ProbNeural Operator Lab.

## Overview

The monitoring system provides:

- **Structured Logging**: JSON and colored console logging with performance tracking
- **Metrics Collection**: System, GPU, and experiment metrics with time-series storage
- **Health Checks**: Automated system health monitoring with alerts
- **Real-time Dashboard**: Web-based monitoring dashboard with visualizations
- **Experiment Tracking**: Comprehensive experiment lifecycle monitoring

## Quick Start

### 1. Basic Monitoring Setup

```python
from monitoring import setup_monitoring

# Setup with default configuration
components = setup_monitoring()

# Start dashboard server
from monitoring import start_dashboard
start_dashboard(components, host="0.0.0.0", port=8050)
```

### 2. Command Line Monitoring

```bash
# Start full monitoring system with dashboard
python scripts/start-monitoring.py

# Start on custom port
python scripts/start-monitoring.py --port 8080

# Metrics only (no dashboard)
python scripts/start-monitoring.py --metrics-only

# With custom configuration
python scripts/start-monitoring.py --config monitoring/custom-config.yaml
```

### 3. Docker Monitoring

```bash
# Start monitoring service
docker-compose --profile monitoring up

# Or with development profile
docker-compose --profile dev --profile monitoring up
```

## Configuration

### Configuration File

The monitoring system is configured via `monitoring/config.yaml`:

```yaml
# Logging Configuration
logging:
  level: INFO
  console_output: true
  file_output: true
  json_output: false
  log_dir: "logs"

# Metrics Collection
metrics:
  enabled: true
  buffer_size: 10000
  flush_interval: 60
  system_monitoring:
    enabled: true
    interval: 5.0

# Health Checks
health_checks:
  enabled: true
  monitoring_interval: 60.0
  checks:
    system_resources:
      enabled: true
      thresholds:
        cpu_threshold: 90.0
        memory_threshold: 90.0

# Dashboard
dashboard:
  enabled: true
  host: "0.0.0.0"
  port: 8050
```

### Environment Variables

```bash
# Logging level
export LOG_LEVEL=DEBUG

# Disable auto-monitoring setup
export DISABLE_AUTO_MONITORING=1

# Skip GPU tests/monitoring
export SKIP_GPU_TESTS=1
```

## Components

### 1. Structured Logging

#### Setup and Usage

```python
from probneural_operator.utils.logging import setup_logging, get_logger

# Setup logging
loggers = setup_logging(
    log_level="INFO",
    console_output=True,
    json_output=False,
    performance_logging=True
)

# Get logger
logger = get_logger("my_module")

# Basic logging
logger.info("Training started")
logger.error("Model failed to load")

# Performance logging with timer
with logger.perf.timer("model_inference"):
    output = model(input_data)

# Log model information
from probneural_operator.utils.logging import log_model_info
log_model_info(model, logger)
```

#### Log Files

- `logs/probneural_operator.log` - Main application log
- `logs/errors.log` - Error-only log
- `logs/performance.log` - Performance metrics log

### 2. Metrics Collection

#### Basic Usage

```python
from probneural_operator.utils.metrics import get_global_metrics, timer, record_metric

# Get metrics collector
metrics = get_global_metrics()

# Record metrics
record_metric("training.loss", 0.5, tags={"epoch": "1"})
record_metric("model.accuracy", 0.85, tags={"dataset": "validation"})

# Time operations
with timer("data_loading"):
    data = load_data()

# Counter metrics
metrics.increment_counter("api.requests", tags={"endpoint": "/predict"})
```

#### Experiment Tracking

```python
from probneural_operator.utils.metrics import ExperimentTracker

# Create experiment tracker
tracker = ExperimentTracker("uncertainty_experiment_v1")

# Set metadata
tracker.set_metadata(
    model_type="FNO",
    dataset="burgers_equation",
    config={"batch_size": 32}
)

# Record metrics during training
for epoch in range(epochs):
    # Training
    loss = train_epoch()
    tracker.record_loss(loss, step=epoch, phase="train")
    
    # Validation
    val_loss, val_acc = validate()
    tracker.record_loss(val_loss, step=epoch, phase="val")
    tracker.record_accuracy(val_acc, step=epoch, phase="val")

# Finish experiment
tracker.finish(status="completed")
tracker.save_summary()
```

#### System Metrics

Automatically collected metrics include:

- **System**: CPU %, Memory %, Disk %
- **GPU**: Memory usage, Utilization %
- **Process**: Memory usage, Thread count

### 3. Health Checks

#### Built-in Health Checks

```python
from monitoring.health_checks import get_health_monitor, SystemResourcesHealthCheck

# Get health monitor
monitor = get_health_monitor()

# Add custom health check
health_check = SystemResourcesHealthCheck(
    cpu_threshold=85.0,
    memory_threshold=80.0
)
monitor.add_check(health_check)

# Run checks manually
result = monitor.run_check("system_resources")
print(f"Status: {result.status}, Message: {result.message}")

# Get overall health
overall_status = monitor.get_overall_status()
```

#### Custom Health Checks

```python
from monitoring.health_checks import HealthCheck, HealthStatus

class CustomHealthCheck(HealthCheck):
    def __init__(self):
        super().__init__("custom_check", interval=30.0)
    
    def _check_impl(self):
        try:
            # Your health check logic
            if some_condition:
                return HealthStatus.HEALTHY, "All good", {}
            else:
                return HealthStatus.WARNING, "Minor issue", {"details": "..."}
        except Exception as e:
            return HealthStatus.CRITICAL, f"Failed: {e}", {}

# Add to monitor
monitor.add_check(CustomHealthCheck())
```

### 4. Dashboard

#### Access Dashboard

1. Start monitoring: `python scripts/start-monitoring.py`
2. Open browser: `http://localhost:8050`

#### Dashboard Features

- **Real-time Metrics**: Live system and GPU metrics
- **Health Status**: Current health check results
- **Experiment Tracking**: Training progress and results
- **System Overview**: Resource utilization summary

#### API Endpoints

```bash
# Health status
curl http://localhost:8050/api/health

# System metrics (last hour)
curl http://localhost:8050/api/system_metrics?hours=1

# Experiment metrics
curl http://localhost:8050/api/experiment_metrics?experiment=my_exp
```

## Integration Examples

### Training Script Integration

```python
import torch
from monitoring import setup_monitoring
from probneural_operator.utils.logging import get_logger
from probneural_operator.utils.metrics import ExperimentTracker

def main():
    # Setup monitoring
    setup_monitoring()
    logger = get_logger("training")
    
    # Start experiment tracking
    tracker = ExperimentTracker("fno_burgers_v1")
    tracker.set_metadata(
        model="FNO",
        dataset="burgers",
        optimizer="Adam",
        lr=1e-3
    )
    
    logger.info("🚀 Starting training")
    
    try:
        model = create_model()
        tracker.record_model_metrics(model)
        
        for epoch in range(num_epochs):
            # Training
            with logger.perf.timer(f"epoch_{epoch}_train"):
                train_loss = train_epoch(model, train_loader)
                tracker.record_loss(train_loss, step=epoch, phase="train")
            
            # Validation
            with logger.perf.timer(f"epoch_{epoch}_val"):
                val_loss, val_metrics = validate(model, val_loader)
                tracker.record_loss(val_loss, step=epoch, phase="val")
                tracker.record_uncertainty_metrics(val_metrics, step=epoch)
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        tracker.finish(status="completed")
        logger.info("✅ Training completed successfully")
        
    except Exception as e:
        tracker.finish(status="failed")
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### Model Server Integration

```python
from flask import Flask, request, jsonify
from monitoring import setup_monitoring
from probneural_operator.utils.logging import get_logger
from probneural_operator.utils.metrics import timer, record_metric

app = Flask(__name__)

# Setup monitoring
setup_monitoring()
logger = get_logger("model_server")

@app.route("/predict", methods=["POST"])
def predict():
    with timer("model_inference", tags={"endpoint": "/predict"}):
        try:
            data = request.json
            
            # Record request
            record_metric("api.requests", 1, tags={"endpoint": "/predict"})
            
            # Make prediction
            prediction = model.predict(data)
            
            # Record success
            record_metric("api.success", 1, tags={"endpoint": "/predict"})
            
            return jsonify({"prediction": prediction})
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            record_metric("api.errors", 1, tags={"endpoint": "/predict"})
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

## Best Practices

### 1. Logging

- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Include context in log messages
- Use structured logging for important events
- Don't log sensitive information

### 2. Metrics

- Use consistent naming conventions
- Add relevant tags for filtering
- Don't create too many unique metric names
- Aggregate high-frequency metrics

### 3. Health Checks

- Keep checks lightweight and fast
- Set appropriate thresholds
- Test health checks regularly
- Document check failure scenarios

### 4. Performance

- Use timers for critical operations
- Monitor resource usage
- Set up alerts for anomalies
- Regular cleanup of old data

## Troubleshooting

### Common Issues

1. **Dashboard not starting**
   ```bash
   # Install Flask dependencies
   pip install flask plotly

   # Check port availability
   netstat -tulpn | grep 8050
   ```

2. **Metrics not appearing**
   ```python
   # Check if system monitoring is enabled
   from probneural_operator.utils.metrics import get_global_metrics
   metrics = get_global_metrics()
   print(len(metrics.get_metrics()))
   ```

3. **Health checks failing**
   ```python
   # Run health checks manually
   from monitoring.health_checks import get_health_monitor
   monitor = get_health_monitor()
   results = monitor.run_all_checks()
   for name, result in results.items():
       print(f"{name}: {result.status} - {result.message}")
   ```

### Log Analysis

```bash
# View recent errors
tail -f logs/errors.log

# Search for specific patterns
grep "CRITICAL" logs/probneural_operator.log

# Analyze performance logs
grep "timer" logs/performance.log | tail -20
```

### Metrics Analysis

```bash
# View recent metrics
ls -la metrics/

# Analyze metrics with jq
cat metrics/metrics_*.json | jq '.[] | select(.name == "system.cpu.percent")'
```

## Extensions

### Custom Dashboards

Create custom dashboard components by extending the `PlotlyDashboard` class:

```python
from monitoring.dashboard import PlotlyDashboard

class CustomDashboard(PlotlyDashboard):
    def create_custom_plot(self):
        # Your custom visualization logic
        pass
```

### External Integrations

The monitoring system can be extended to integrate with:

- **Prometheus** for metrics collection
- **Grafana** for advanced dashboards
- **ELK Stack** for log analysis
- **MLflow** for experiment tracking
- **Weights & Biases** for ML experiment management

## Security Considerations

- Dashboard runs on all interfaces by default (0.0.0.0)
- No authentication enabled by default
- Consider using reverse proxy with authentication for production
- Logs may contain sensitive information - configure retention appropriately