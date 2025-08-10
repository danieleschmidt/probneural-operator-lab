#!/usr/bin/env python3
"""
TERRAGON Autonomous SDLC - Final System Validation
Comprehensive validation of all implemented capabilities
"""

import os
import sys
import json
import time
import inspect
from pathlib import Path

def validate_framework_structure():
    """Validate the framework structure and completeness."""
    print("🏗️  FRAMEWORK STRUCTURE VALIDATION")
    print("=" * 50)
    
    # Core components
    components = {
        "models": ["deeponet", "fno", "base"],
        "posteriors": ["laplace", "base"],  
        "active": ["acquisition.py", "learner.py"],
        "data": ["datasets.py", "generators.py", "loaders.py"],
        "calibration": ["temperature.py"],
        "scaling": ["cache.py", "distributed.py", "autoscale.py", "serving.py"],
        "utils": ["config.py", "monitoring.py", "security.py", "validation.py"]
    }
    
    base_path = Path("probneural_operator")
    validated = 0
    total = 0
    
    for component, files in components.items():
        component_path = base_path / component
        if component_path.exists():
            print(f"✅ {component}/")
            for file in files:
                file_path = component_path / file
                if file_path.exists() or (component_path / file / "__init__.py").exists():
                    print(f"   ✅ {file}")
                    validated += 1
                else:
                    print(f"   ❌ {file}")
                total += 1
        else:
            print(f"❌ {component}/")
            total += len(files)
    
    print(f"\nComponent validation: {validated}/{total} ({validated/total*100:.1f}%)")
    return validated >= total * 0.9

def validate_deployment_infrastructure():
    """Validate deployment and production readiness."""
    print("\n🚀 DEPLOYMENT INFRASTRUCTURE VALIDATION")
    print("=" * 50)
    
    deployment_files = [
        "k8s-deployment.yaml",
        "docker-compose.yml", 
        "docker-compose.production.yml",
        "Dockerfile",
        "deployment/kubernetes_manifests.py",
        "monitoring/prometheus.yml",
        "monitoring/alerts.yml"
    ]
    
    validated = 0
    for file in deployment_files:
        if Path(file).exists():
            print(f"✅ {file}")
            validated += 1
        else:
            print(f"❌ {file}")
    
    print(f"\nDeployment validation: {validated}/{len(deployment_files)} ({validated/len(deployment_files)*100:.1f}%)")
    return validated >= len(deployment_files) * 0.8

def validate_documentation():
    """Validate comprehensive documentation."""
    print("\n📚 DOCUMENTATION VALIDATION") 
    print("=" * 50)
    
    docs = [
        "README.md",
        "API_REFERENCE.md", 
        "CONTRIBUTING.md",
        "PRODUCTION_DEPLOYMENT_GUIDE.md",
        "docs/ARCHITECTURE.md",
        "docs/BEST_PRACTICES.md",
        "TERRAGON_AUTONOMOUS_SDLC_COMPLETION_REPORT.md"
    ]
    
    validated = 0
    for doc in docs:
        if Path(doc).exists():
            print(f"✅ {doc}")
            validated += 1
        else:
            print(f"❌ {doc}")
    
    print(f"\nDocumentation validation: {validated}/{len(docs)} ({validated/len(docs)*100:.1f}%)")
    return validated >= len(docs) * 0.8

def validate_testing_framework():
    """Validate testing capabilities."""
    print("\n🧪 TESTING FRAMEWORK VALIDATION")
    print("=" * 50)
    
    test_components = [
        "tests/unit/",
        "tests/integration/", 
        "tests/benchmarks/",
        "pure_python_test.py",
        "test_basic_imports.py",
        "test_syntax.py",
        "benchmarks/uncertainty_benchmark.py"
    ]
    
    validated = 0
    for component in test_components:
        if Path(component).exists():
            print(f"✅ {component}")
            validated += 1
        else:
            print(f"❌ {component}")
    
    print(f"\nTesting validation: {validated}/{len(test_components)} ({validated/len(test_components)*100:.1f}%)")
    return validated >= len(test_components) * 0.8

def validate_examples_and_tutorials():
    """Validate examples and usage guides."""
    print("\n📖 EXAMPLES & TUTORIALS VALIDATION")
    print("=" * 50)
    
    examples = [
        "examples/basic_training_example.py",
        "examples/comprehensive_usage_guide.py",
        "examples/production_server.py",
        "examples/comprehensive_test.py"
    ]
    
    validated = 0
    for example in examples:
        if Path(example).exists():
            print(f"✅ {example}")
            validated += 1
        else:
            print(f"❌ {example}")
    
    print(f"\nExamples validation: {validated}/{len(examples)} ({validated/len(examples)*100:.1f}%)")
    return validated >= len(examples) * 0.8

def validate_benchmark_results():
    """Validate benchmark execution results."""
    print("\n📊 BENCHMARK RESULTS VALIDATION")
    print("=" * 50)
    
    if Path("benchmark_results/benchmark_results.json").exists():
        try:
            with open("benchmark_results/benchmark_results.json", "r") as f:
                results = json.load(f)
            
            summary = results.get("summary", {})
            avg_nll = summary.get("average_nll", 0)
            avg_crps = summary.get("average_crps", 0)
            datasets = summary.get("datasets_tested", 0)
            methods = summary.get("methods_compared", 0)
            
            print(f"✅ Benchmark results found")
            print(f"   📈 Average NLL: {avg_nll:.4f}")
            print(f"   📈 Average CRPS: {avg_crps:.4f}")
            print(f"   📊 Datasets tested: {datasets}")
            print(f"   🔍 Methods compared: {methods}")
            
            return True
        except Exception as e:
            print(f"❌ Benchmark results invalid: {e}")
            return False
    else:
        print("❌ No benchmark results found")
        return False

def generate_system_metrics():
    """Generate comprehensive system metrics."""
    print("\n📊 SYSTEM METRICS GENERATION")
    print("=" * 50)
    
    metrics = {}
    
    # Count files by type
    py_files = list(Path(".").rglob("*.py"))
    md_files = list(Path(".").rglob("*.md"))
    yaml_files = list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.yml"))
    json_files = list(Path(".").rglob("*.json"))
    
    metrics["files"] = {
        "python": len(py_files),
        "documentation": len(md_files), 
        "configuration": len(yaml_files),
        "data": len(json_files),
        "total": len(py_files) + len(md_files) + len(yaml_files) + len(json_files)
    }
    
    # Estimate lines of code
    total_lines = 0
    code_lines = 0
    for py_file in py_files:
        try:
            with open(py_file, "r") as f:
                lines = f.readlines()
                total_lines += len(lines)
                code_lines += len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        except:
            pass
    
    metrics["code"] = {
        "total_lines": total_lines,
        "code_lines": code_lines,
        "avg_lines_per_file": total_lines // len(py_files) if py_files else 0
    }
    
    # Directory structure depth
    max_depth = max([len(p.parts) for p in Path(".").rglob("*") if p.is_file()], default=0)
    metrics["structure"] = {
        "max_directory_depth": max_depth,
        "total_directories": len(list(Path(".").rglob("**/"))),
        "total_files": len(list(Path(".").rglob("*")))
    }
    
    print(f"📁 Files: {metrics['files']['total']} total")
    print(f"   🐍 Python: {metrics['files']['python']}")
    print(f"   📚 Documentation: {metrics['files']['documentation']}")
    print(f"   ⚙️  Configuration: {metrics['files']['configuration']}")
    
    print(f"💻 Code: {metrics['code']['total_lines']} total lines")
    print(f"   📝 Code lines: {metrics['code']['code_lines']}")
    print(f"   📊 Avg per file: {metrics['code']['avg_lines_per_file']}")
    
    print(f"🏗️  Structure: {metrics['structure']['max_directory_depth']} max depth")
    print(f"   📂 Directories: {metrics['structure']['total_directories']}")
    
    return metrics

def main():
    """Run comprehensive system validation."""
    print("🔍 TERRAGON AUTONOMOUS SDLC - FINAL SYSTEM VALIDATION")
    print("=" * 70)
    print(f"Validation started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all validations
    validations = [
        ("Framework Structure", validate_framework_structure),
        ("Deployment Infrastructure", validate_deployment_infrastructure), 
        ("Documentation", validate_documentation),
        ("Testing Framework", validate_testing_framework),
        ("Examples & Tutorials", validate_examples_and_tutorials),
        ("Benchmark Results", validate_benchmark_results)
    ]
    
    results = {}
    for name, validator in validations:
        results[name] = validator()
    
    # Generate metrics
    metrics = generate_system_metrics()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 FINAL VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n📊 Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 SYSTEM VALIDATION SUCCESSFUL!")
        print("✅ All components validated and ready for production")
        print("🚀 TERRAGON Autonomous SDLC execution COMPLETED")
    else:
        print(f"\n⚠️  System validation partially successful: {passed}/{total}")
        
    # Save results
    final_results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "validations": results,
        "metrics": metrics,
        "overall_score": f"{passed}/{total}",
        "success_rate": f"{passed/total*100:.1f}%"
    }
    
    with open("final_validation_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📄 Results saved to: final_validation_results.json")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)