"""
Command Line Interface for Terragon Quality Gates
=================================================

CLI tool for executing and managing progressive quality gates.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from .core import QualityGateFramework, GenerationType, run_quality_gates
from .monitoring import ContinuousQualityMonitor
from .adaptive import AdaptiveQualityController


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Terragon Progressive Quality Gates System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  terragon-quality run gen1              # Run Generation 1 gates
  terragon-quality run gen2              # Run Generation 2 gates  
  terragon-quality run gen3              # Run Generation 3 gates
  terragon-quality run all               # Run all generations sequentially
  terragon-quality monitor start         # Start continuous monitoring
  terragon-quality monitor status        # Check monitoring status
  terragon-quality report                # Generate quality report
  terragon-quality adapt --threshold     # Show adaptive thresholds
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Execute quality gates')
    run_parser.add_argument(
        'generation',
        choices=['gen1', 'gen2', 'gen3', 'all', 'research'],
        help='Generation to run'
    )
    run_parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path'
    )
    run_parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Overall timeout in seconds (default: 1800)'
    )
    run_parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run gates in parallel where possible'
    )
    run_parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first critical failure'
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Quality monitoring')
    monitor_subparsers = monitor_parser.add_subparsers(dest='monitor_action')
    
    monitor_subparsers.add_parser('start', help='Start continuous monitoring')
    monitor_subparsers.add_parser('stop', help='Stop continuous monitoring')
    monitor_subparsers.add_parser('status', help='Show monitoring status')
    monitor_subparsers.add_parser('report', help='Generate monitoring report')
    
    # Adapt command
    adapt_parser = subparsers.add_parser('adapt', help='Adaptive quality control')
    adapt_parser.add_argument(
        '--show-thresholds',
        action='store_true',
        help='Show current adaptive thresholds'
    )
    adapt_parser.add_argument(
        '--show-factors',
        action='store_true',
        help='Show contextual factors'
    )
    adapt_parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset adaptive learning'
    )
    adapt_parser.add_argument(
        '--report',
        action='store_true',
        help='Generate adaptation report'
    )
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate quality reports')
    report_parser.add_argument(
        '--format',
        choices=['json', 'markdown', 'html'],
        default='json',
        help='Report format'
    )
    report_parser.add_argument(
        '--output',
        type=Path,
        help='Output file path'
    )
    report_parser.add_argument(
        '--include-history',
        action='store_true',
        help='Include historical data'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file to validate'
    )
    
    return parser


async def run_generation_command(args) -> int:
    """Execute run generation command."""
    print("🚀 Terragon Progressive Quality Gates")
    print("=" * 50)
    
    if args.generation == 'all':
        # Run all generations sequentially
        generations = [GenerationType.GENERATION_1, GenerationType.GENERATION_2, GenerationType.GENERATION_3]
        
        overall_success = True
        for generation in generations:
            print(f"\n🎯 Starting {generation.value.replace('_', ' ').title()}")
            success = await run_quality_gates(generation)
            
            if not success:
                overall_success = False
                if args.fail_fast:
                    print(f"❌ {generation.value} failed - stopping execution (fail-fast enabled)")
                    return 1
                else:
                    print(f"❌ {generation.value} failed - continuing to next generation")
        
        return 0 if overall_success else 1
    
    elif args.generation == 'research':
        # Run research-specific gates
        from .research import ResearchQualityGates
        
        framework = QualityGateFramework(args.config)
        research_gates = ResearchQualityGates().get_gates()
        framework.register_gates(research_gates)
        
        print("\n🔬 Running Research Quality Gates")
        
        # Execute research gates
        results = {}
        for gate in research_gates:
            try:
                result = await gate.execute(framework.context)
                results[gate.name] = result
                
                status_emoji = "✅" if result.passed else "❌"
                print(f"{status_emoji} {gate.name}: {result.percentage_score:.1f}%")
                
            except Exception as e:
                print(f"💥 {gate.name}: ERROR - {str(e)}")
                return 1
        
        # Evaluate overall success
        passed_gates = sum(1 for r in results.values() if r.passed)
        total_gates = len(results)
        success_rate = (passed_gates / total_gates) * 100 if total_gates > 0 else 0
        
        print(f"\n📊 Research Quality Summary: {passed_gates}/{total_gates} gates passed ({success_rate:.1f}%)")
        
        return 0 if success_rate >= 80 else 1
    
    else:
        # Run specific generation
        generation_map = {
            'gen1': GenerationType.GENERATION_1,
            'gen2': GenerationType.GENERATION_2,
            'gen3': GenerationType.GENERATION_3,
        }
        
        generation = generation_map[args.generation]
        success = await run_quality_gates(generation)
        
        return 0 if success else 1


def run_monitor_command(args) -> int:
    """Execute monitor command."""
    monitor = ContinuousQualityMonitor()
    
    if args.monitor_action == 'start':
        print("🔍 Starting continuous quality monitoring...")
        monitor.start_monitoring()
        print("✅ Monitoring started. Use 'terragon-quality monitor status' to check progress.")
        return 0
    
    elif args.monitor_action == 'stop':
        print("🛑 Stopping continuous quality monitoring...")
        monitor.stop_monitoring()
        print("✅ Monitoring stopped.")
        return 0
    
    elif args.monitor_action == 'status':
        status = monitor.get_current_status()
        
        print("📊 Quality Monitoring Status")
        print("=" * 30)
        print(f"Active: {'Yes' if status['monitoring_active'] else 'No'}")
        
        if status['status'] != 'no_data':
            print(f"Latest Assessment: {status.get('latest_assessment', 'N/A')}")
            print(f"Overall Score: {status.get('overall_score', 0):.1f}%")
            print(f"Active Alerts: {status.get('alert_count', 0)}")
            print(f"Recommendations: {status.get('recommendation_count', 0)}")
            print(f"Total Assessments: {status.get('assessment_count', 0)}")
            
            if 'trend_analysis' in status:
                trends = status['trend_analysis']
                print(f"Quality Trend: {trends.get('trend', 'unknown').title()}")
                print(f"Trend Confidence: {trends.get('confidence', 0):.1%}")
        
        return 0
    
    elif args.monitor_action == 'report':
        report = monitor.generate_quality_report()
        
        print("📄 Quality Monitoring Report")
        print("=" * 35)
        print(json.dumps(report, indent=2))
        
        return 0
    
    return 1


def run_adapt_command(args) -> int:
    """Execute adapt command."""
    controller = AdaptiveQualityController()
    
    if args.show_thresholds:
        print("🎯 Adaptive Thresholds")
        print("=" * 25)
        
        for name, threshold in controller.adaptive_thresholds.items():
            change = threshold.current_value - threshold.base_value
            change_str = f"({change:+.1f})" if abs(change) > 0.1 else ""
            
            print(f"{name}: {threshold.current_value:.1f}% {change_str}")
            print(f"  Base: {threshold.base_value:.1f}%")
            print(f"  Range: [{threshold.min_value:.1f}%, {threshold.max_value:.1f}%]")
            print(f"  Confidence: {threshold.confidence:.1%}")
            print()
        
        return 0
    
    elif args.show_factors:
        print("🌍 Contextual Factors")
        print("=" * 22)
        
        for name, factor in controller.contextual_factors.items():
            correlation = controller._calculate_factor_correlation(factor)
            
            print(f"{name}: {factor.current_value:.3f}")
            print(f"  Weight: {factor.weight:.3f}")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Samples: {len(factor.impact_history)}")
            print()
        
        return 0
    
    elif args.reset:
        print("🔄 Resetting adaptive learning...")
        
        # Reset to base values
        for threshold in controller.adaptive_thresholds.values():
            threshold.current_value = threshold.base_value
            threshold.confidence = 0.5
            threshold.history = []
        
        for factor in controller.contextual_factors.values():
            factor.weight = 0.1
            factor.impact_history = []
        
        controller.learning_history = []
        controller.performance_baseline = {}
        controller._save_state()
        
        print("✅ Adaptive learning reset successfully.")
        return 0
    
    elif args.report:
        report = controller.generate_adaptation_report()
        
        print("📊 Adaptation Report")
        print("=" * 20)
        print(json.dumps(report, indent=2))
        
        return 0
    
    return 1


def run_report_command(args) -> int:
    """Execute report command."""
    framework = QualityGateFramework()
    report = framework.get_execution_report()
    
    if args.include_history:
        monitor = ContinuousQualityMonitor()
        monitor_report = monitor.generate_quality_report()
        report["monitoring_data"] = monitor_report
    
    if args.format == 'json':
        output = json.dumps(report, indent=2)
    elif args.format == 'markdown':
        output = generate_markdown_report(report)
    elif args.format == 'html':
        output = generate_html_report(report)
    else:
        output = json.dumps(report, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"📄 Report saved to {args.output}")
    else:
        print(output)
    
    return 0


def generate_markdown_report(report: dict) -> str:
    """Generate markdown format report."""
    md = ["# Quality Gates Report", ""]
    
    summary = report.get("summary", {})
    md.extend([
        "## Summary",
        f"- **Total Executions:** {summary.get('total_executions', 0)}",
        f"- **Pass Rate:** {summary.get('pass_rate', 0):.1f}%",
        f"- **Current Generation:** {summary.get('current_generation', 'Unknown')}",
        ""
    ])
    
    if "performance_metrics" in report:
        metrics = report["performance_metrics"]
        md.extend([
            "## Performance Metrics",
            f"- **Average Execution Time:** {metrics.get('avg_execution_time', 0):.2f}s",
            f"- **Average Score:** {metrics.get('avg_score', 0):.1f}%",
            ""
        ])
    
    if "recommendations" in report:
        md.extend(["## Recommendations"])
        for rec in report["recommendations"]:
            md.append(f"- {rec}")
        md.append("")
    
    return "\n".join(md)


def generate_html_report(report: dict) -> str:
    """Generate HTML format report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Gates Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ margin: 10px 0; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
    </style>
</head>
<body>
    <h1>Quality Gates Report</h1>
    
    <h2>Summary</h2>
    <div class="metric">Total Executions: {report.get('summary', {}).get('total_executions', 0)}</div>
    <div class="metric">Pass Rate: {report.get('summary', {}).get('pass_rate', 0):.1f}%</div>
    
    <h2>Recent Executions</h2>
    <ul>
"""
    
    for execution in report.get("recent_executions", []):
        status_class = "passed" if execution.get("status") == "passed" else "failed"
        html += f'<li class="{status_class}">{execution.get("gate_name", "Unknown")}: {execution.get("percentage_score", 0):.1f}%</li>'
    
    html += """
    </ul>
</body>
</html>
"""
    
    return html


def run_validate_command(args) -> int:
    """Execute validate command."""
    config_path = args.config or Path(".terragon/quality_gates.json")
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"✅ Configuration file is valid: {config_path}")
        
        # Basic validation
        required_sections = ["context", "thresholds"]
        for section in required_sections:
            if section in config:
                print(f"  ✓ {section} section found")
            else:
                print(f"  ⚠️  {section} section missing")
        
        return 0
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return 1


async def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'run':
            return await run_generation_command(args)
        elif args.command == 'monitor':
            return run_monitor_command(args)
        elif args.command == 'adapt':
            return run_adapt_command(args)
        elif args.command == 'report':
            return run_report_command(args)
        elif args.command == 'validate':
            return run_validate_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


def cli_entry_point():
    """Entry point for CLI installation."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_point()