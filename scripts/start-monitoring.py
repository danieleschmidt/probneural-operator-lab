#!/usr/bin/env python3
"""Start the ProbNeural Operator Lab monitoring system."""

import argparse
import signal
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring import setup_monitoring, start_dashboard, stop_monitoring
from probneural_operator.utils.logging import get_logger


def signal_handler(sig, frame, monitoring_components):
    """Handle shutdown signals gracefully."""
    logger = get_logger("monitoring")
    logger.info("Received shutdown signal, stopping monitoring...")
    stop_monitoring(monitoring_components)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Start ProbNeural Operator Lab monitoring system")
    
    # Configuration options
    parser.add_argument("--config", "-c", type=str, help="Path to monitoring configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    # Dashboard options
    parser.add_argument("--dashboard", action="store_true", default=True, 
                       help="Enable dashboard (default: True)")
    parser.add_argument("--no-dashboard", action="store_true", 
                       help="Disable dashboard")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Dashboard host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8050, 
                       help="Dashboard port (default: 8050)")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    
    # Monitoring options
    parser.add_argument("--metrics-only", action="store_true", 
                       help="Only start metrics collection (no dashboard)")
    parser.add_argument("--health-only", action="store_true", 
                       help="Only start health monitoring (no dashboard)")
    parser.add_argument("--system-monitoring", action="store_true", default=True,
                       help="Enable system metrics monitoring (default: True)")
    parser.add_argument("--no-system-monitoring", action="store_true",
                       help="Disable system metrics monitoring")
    
    # Output options
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Suppress non-error output")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Resolve dashboard setting
    enable_dashboard = args.dashboard and not args.no_dashboard and not args.metrics_only and not args.health_only
    
    # Setup monitoring
    config_path = Path(args.config) if args.config else None
    
    try:
        print("🚀 Starting ProbNeural Operator Lab monitoring system...")
        
        # Setup monitoring components
        monitoring_components = setup_monitoring(config_path)
        
        logger = get_logger("monitoring")
        
        if args.verbose:
            logger.info("Monitoring components initialized:")
            for component, obj in monitoring_components.items():
                if obj is not None:
                    logger.info(f"  ✅ {component}")
                else:
                    logger.info(f"  ❌ {component} (disabled or unavailable)")
        
        # Setup signal handlers for graceful shutdown
        def signal_wrapper(sig, frame):
            signal_handler(sig, frame, monitoring_components)
        
        signal.signal(signal.SIGINT, signal_wrapper)
        signal.signal(signal.SIGTERM, signal_wrapper)
        
        if enable_dashboard:
            # Start dashboard server (blocking)
            print(f"🌐 Starting dashboard server on http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop...")
            
            try:
                start_dashboard(monitoring_components, args.host, args.port, args.debug)
            except KeyboardInterrupt:
                pass
        else:
            # Just run monitoring in background
            print("📊 Monitoring system started (no dashboard)")
            print("Metrics and health checks are running in the background")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    except Exception as e:
        print(f"❌ Failed to start monitoring: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        print("🛑 Stopping monitoring system...")
        if 'monitoring_components' in locals():
            stop_monitoring(monitoring_components)
        print("✅ Monitoring system stopped")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())