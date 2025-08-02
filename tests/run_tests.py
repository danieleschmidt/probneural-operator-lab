#!/usr/bin/env python3
"""Test runner script with various testing options."""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} passed")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"❌ {description} failed")
        print(f"   Error: {result.stderr.strip()}")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests with various options")
    
    # Test selection
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests only")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    
    # Coverage options
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--coverage-html", action="store_true", help="Generate HTML coverage report")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    parser.add_argument("--junit", action="store_true", help="Generate JUnit XML report")
    
    # Performance options
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel processes")
    parser.add_argument("--fast", action="store_true", help="Fast mode (skip slow tests)")
    
    # Debugging options
    parser.add_argument("--pdb", action="store_true", help="Drop into debugger on failures")
    parser.add_argument("--lf", "--last-failed", action="store_true", help="Run only last failed tests")
    parser.add_argument("--ff", "--failed-first", action="store_true", help="Run failed tests first")
    
    # Specific test selection
    parser.add_argument("pattern", nargs="?", help="Test pattern to match")
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.benchmark:
        cmd.extend(["-m", "benchmark"])
    
    # Marker handling
    markers = []
    if not args.slow and not args.benchmark:
        markers.append("not slow")
    if not args.gpu:
        markers.append("not gpu")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Coverage
    if args.coverage or args.coverage_html:
        cmd.extend(["--cov=probneural_operator"])
        cmd.extend(["--cov-report=term-missing"])
        
        if args.coverage_html:
            cmd.extend(["--cov-report=html:htmlcov"])
    
    # Output options
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    if args.junit:
        cmd.extend(["--junitxml=test-results.xml"])
    
    # Performance
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    elif args.fast:
        cmd.extend(["-n", "auto"])
    
    # Debugging
    if args.pdb:
        cmd.append("--pdb")
    if args.lf:
        cmd.append("--lf")
    if args.ff:
        cmd.append("--ff")
    
    # Test pattern
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    # Default test directory
    if not any(arg.startswith("tests/") for arg in cmd):
        cmd.append("tests/")
    
    # Run tests
    print("🧪 Running tests...")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    
    # Summary
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {result.returncode}")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())