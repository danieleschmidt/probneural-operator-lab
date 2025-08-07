#!/usr/bin/env python3
"""
Basic import test to verify framework structure.
This test checks imports without running heavy computations.
"""

import sys
import traceback

def test_import(module_name, description=""):
    """Test importing a module."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except Exception as e:
        print(f"✗ {module_name} {description}: {e}")
        return False

def main():
    """Run basic import tests."""
    print("Testing basic imports for ProbNeural-Operator-Lab\n")
    
    success_count = 0
    total_count = 0
    
    # Test basic Python modules first
    tests = [
        ("os", "basic system module"),
        ("sys", "basic system module"), 
        ("abc", "abstract base classes"),
        ("typing", "type hints"),
        ("pathlib", "path utilities"),
    ]
    
    for module, desc in tests:
        total_count += 1
        if test_import(module, desc):
            success_count += 1
    
    print("\nTesting framework structure (without heavy dependencies):")
    
    # Test framework imports that don't require torch/numpy immediately
    framework_tests = [
        ("probneural_operator.models.base", "base model classes"),
        ("probneural_operator.posteriors.base", "base posterior classes"),
        ("probneural_operator.data", "data module structure"),
        ("probneural_operator.active", "active learning module"),
        ("probneural_operator.calibration", "calibration module"),
    ]
    
    for module, desc in framework_tests:
        total_count += 1
        if test_import(module, desc):
            success_count += 1
    
    print(f"\nResults: {success_count}/{total_count} imports successful")
    
    if success_count == total_count:
        print("✓ All basic imports working!")
        return 0
    else:
        print("✗ Some imports failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())