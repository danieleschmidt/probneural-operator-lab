#!/usr/bin/env python3
"""
Check the framework file structure.
"""

import os
from pathlib import Path

def check_file_exists(path, description=""):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {path} {description}")
        return True
    else:
        print(f"✗ {path} {description} - NOT FOUND")
        return False

def main():
    """Check the framework structure."""
    print("Checking ProbNeural-Operator-Lab file structure...\n")
    
    success_count = 0
    total_count = 0
    
    # Core structure files
    files_to_check = [
        ("probneural_operator/__init__.py", "main package init"),
        
        # Models
        ("probneural_operator/models/__init__.py", "models package"),
        ("probneural_operator/models/base/__init__.py", "base models"),
        ("probneural_operator/models/base/neural_operator.py", "base neural operator"),
        ("probneural_operator/models/base/layers.py", "neural operator layers"),
        ("probneural_operator/models/fno/__init__.py", "FNO package"),
        ("probneural_operator/models/fno/fno.py", "FNO implementation"),
        ("probneural_operator/models/deeponet/__init__.py", "DeepONet package"),
        ("probneural_operator/models/deeponet/deeponet.py", "DeepONet implementation"),
        
        # Posteriors
        ("probneural_operator/posteriors/__init__.py", "posteriors package"),
        ("probneural_operator/posteriors/base/__init__.py", "base posteriors"),
        ("probneural_operator/posteriors/base/posterior.py", "posterior base class"),
        ("probneural_operator/posteriors/base/factory.py", "posterior factory"),
        ("probneural_operator/posteriors/laplace/__init__.py", "Laplace package"),
        ("probneural_operator/posteriors/laplace/laplace.py", "Laplace implementation"),
        
        # Data
        ("probneural_operator/data/__init__.py", "data package"),
        ("probneural_operator/data/datasets.py", "dataset classes"),
        ("probneural_operator/data/loaders.py", "data loaders"),
        ("probneural_operator/data/transforms.py", "data transforms"),
        ("probneural_operator/data/generators.py", "data generators"),
        
        # Active Learning
        ("probneural_operator/active/__init__.py", "active learning package"),
        ("probneural_operator/active/learner.py", "active learner"),
        ("probneural_operator/active/acquisition.py", "acquisition functions"),
        
        # Calibration
        ("probneural_operator/calibration/__init__.py", "calibration package"),
        ("probneural_operator/calibration/temperature.py", "temperature scaling"),
        
        # Utilities
        ("probneural_operator/utils/__init__.py", "utilities package"),
    ]
    
    for file_path, description in files_to_check:
        total_count += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    print(f"\nResults: {success_count}/{total_count} files found")
    
    if success_count == total_count:
        print("✓ All framework files present!")
        
        # Check for basic Python syntax
        print("\nChecking for basic Python syntax issues...")
        syntax_issues = 0
        
        for file_path, _ in files_to_check:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        compile(f.read(), file_path, 'exec')
                except SyntaxError as e:
                    print(f"✗ Syntax error in {file_path}: {e}")
                    syntax_issues += 1
        
        if syntax_issues == 0:
            print("✓ No syntax errors found!")
        else:
            print(f"✗ {syntax_issues} syntax errors found")
        
        return 0
    else:
        print("✗ Some framework files missing")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())