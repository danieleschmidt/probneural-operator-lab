#!/usr/bin/env python3
"""
Terragon Quality Gates Execution Script
=======================================

Main execution script for running progressive quality gates.
Can be run directly or imported as a module.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

from terragon.quality_gates.core import QualityGateFramework, GenerationType, run_quality_gates
from terragon.quality_gates.generations import Generation1Gates, Generation2Gates, Generation3Gates
from terragon.quality_gates.research import ResearchQualityGates
from terragon.quality_gates.monitoring import ContinuousQualityMonitor
from terragon.quality_gates.adaptive import AdaptiveQualityController


async def run_all_generations():
    """Run all generations of quality gates sequentially."""
    print("🚀 Terragon Progressive Quality Gates - Full Execution")
    print("=" * 60)
    
    generations = [
        (GenerationType.GENERATION_1, "MAKE IT WORK (Simple)"),
        (GenerationType.GENERATION_2, "MAKE IT ROBUST (Reliable)"),
        (GenerationType.GENERATION_3, "MAKE IT SCALE (Optimized)"),
    ]
    
    overall_success = True
    generation_results = {}
    
    for generation_type, description in generations:
        print(f"\n🎯 {description}")
        print("-" * 50)
        
        success = await run_quality_gates(generation_type)
        generation_results[generation_type.value] = success
        
        if success:
            print(f"✅ {description} - COMPLETED SUCCESSFULLY")
        else:
            print(f"❌ {description} - FAILED")
            overall_success = False
    
    # Run research gates if all generations passed
    if overall_success:
        print(f"\n🔬 RESEARCH VALIDATION")
        print("-" * 50)
        
        framework = QualityGateFramework()
        research_gates = ResearchQualityGates().get_gates()
        framework.register_gates(research_gates)
        
        research_results = {}
        research_success = True
        
        for gate in research_gates:
            try:
                result = await gate.execute(framework.context)
                research_results[gate.name] = result.passed
                
                status_emoji = "✅" if result.passed else "❌"
                print(f"{status_emoji} {gate.name}: {result.percentage_score:.1f}%")
                
                if not result.passed:
                    research_success = False
                    
            except Exception as e:
                print(f"💥 {gate.name}: CRITICAL ERROR - {str(e)}")
                research_success = False
        
        generation_results["research"] = research_success
        if not research_success:
            overall_success = False
    
    # Final summary
    print(f"\n📊 TERRAGON SDLC EXECUTION SUMMARY")
    print("=" * 60)
    
    for gen_name, success in generation_results.items():
        status = "PASSED" if success else "FAILED"
        emoji = "✅" if success else "❌"
        print(f"{emoji} {gen_name.replace('_', ' ').title()}: {status}")
    
    if overall_success:
        print(f"\n🎉 ALL QUALITY GATES PASSED - AUTONOMOUS SDLC COMPLETE!")
        print("🚀 Project is ready for production deployment")
    else:
        print(f"\n⚠️  QUALITY GATES FAILED - REMEDIATION REQUIRED")
        print("🔧 Review failed gates and implement fixes")
    
    return overall_success


async def run_with_monitoring():
    """Run quality gates with continuous monitoring."""
    print("🔍 Starting Quality Gates with Monitoring")
    print("=" * 45)
    
    # Initialize monitoring
    monitor = ContinuousQualityMonitor()
    controller = AdaptiveQualityController()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        print("✅ Continuous monitoring started")
        
        # Run quality gates
        success = await run_all_generations()
        
        # Generate reports
        print(f"\n📄 Generating Reports...")
        
        monitor_report = monitor.generate_quality_report()
        adapt_report = controller.generate_adaptation_report()
        
        # Save reports
        reports_dir = Path(".terragon/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(reports_dir / f"monitoring_report_{timestamp}.json", 'w') as f:
            json.dump(monitor_report, f, indent=2)
        
        with open(reports_dir / f"adaptation_report_{timestamp}.json", 'w') as f:
            json.dump(adapt_report, f, indent=2)
        
        print(f"📄 Reports saved to {reports_dir}")
        
        return success
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        print("🛑 Monitoring stopped")


async def quick_validation():
    """Run quick validation of basic functionality."""
    print("⚡ Quick Quality Validation")
    print("=" * 30)
    
    # Run only Generation 1 gates
    success = await run_quality_gates(GenerationType.GENERATION_1)
    
    if success:
        print("✅ Quick validation PASSED")
        print("🎯 Ready for full quality gate execution")
    else:
        print("❌ Quick validation FAILED")
        print("🔧 Fix basic issues before proceeding")
    
    return success


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            success = asyncio.run(quick_validation())
        elif command == "monitor":
            success = asyncio.run(run_with_monitoring())
        elif command == "gen1":
            success = asyncio.run(run_quality_gates(GenerationType.GENERATION_1))
        elif command == "gen2":
            success = asyncio.run(run_quality_gates(GenerationType.GENERATION_2))
        elif command == "gen3":
            success = asyncio.run(run_quality_gates(GenerationType.GENERATION_3))
        elif command == "research":
            # Run research gates only
            framework = QualityGateFramework()
            research_gates = ResearchQualityGates().get_gates()
            framework.register_gates(research_gates)
            
            async def run_research():
                results = {}
                for gate in research_gates:
                    result = await gate.execute(framework.context)
                    results[gate.name] = result
                    print(f"{'✅' if result.passed else '❌'} {gate.name}: {result.percentage_score:.1f}%")
                
                success_rate = sum(1 for r in results.values() if r.passed) / len(results)
                return success_rate >= 0.8
            
            success = asyncio.run(run_research())
        elif command == "help":
            print_help()
            return
        else:
            print(f"❌ Unknown command: {command}")
            print_help()
            sys.exit(1)
    else:
        # Default: run all generations
        success = asyncio.run(run_all_generations())
    
    sys.exit(0 if success else 1)


def print_help():
    """Print help information."""
    print("""
🚀 Terragon Quality Gates - Usage

Commands:
  python quality_gates.py           # Run all generations (default)
  python quality_gates.py quick     # Quick validation (Gen 1 only)
  python quality_gates.py monitor   # Run with continuous monitoring
  python quality_gates.py gen1      # Run Generation 1 gates only
  python quality_gates.py gen2      # Run Generation 2 gates only  
  python quality_gates.py gen3      # Run Generation 3 gates only
  python quality_gates.py research  # Run research gates only
  python quality_gates.py help      # Show this help

Generations:
  Generation 1: MAKE IT WORK (Simple)
    - Basic syntax and import validation
    - Core functionality checks
    
  Generation 2: MAKE IT ROBUST (Reliable)
    - Comprehensive testing with coverage
    - Security vulnerability scanning
    - Code quality and style validation
    
  Generation 3: MAKE IT SCALE (Optimized)
    - Performance benchmarking
    - Scalability testing
    - Production readiness assessment
    
  Research Gates:
    - Reproducibility validation
    - Statistical significance testing
    - Baseline comparison verification
    - Publication readiness assessment
    - Novelty validation

For more detailed control, use the CLI:
  pip install -e .
  terragon-quality --help
""")


if __name__ == "__main__":
    main()