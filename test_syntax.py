#!/usr/bin/env python3
"""Syntax validation script for the ProbNeural-Operator-Lab framework."""

import sys
import os
import importlib.util

# Mock torch and related modules
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()
    
    def __iter__(self):
        return iter([])

# Create comprehensive mock for torch
mock_torch = MockModule()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch
sys.modules['torch.nn.functional'] = mock_torch
sys.modules['torch.optim'] = mock_torch
sys.modules['torch.autograd'] = mock_torch
sys.modules['torch.distributions'] = mock_torch
sys.modules['torch.fft'] = mock_torch
sys.modules['numpy'] = MockModule()
sys.modules['scipy'] = MockModule()
sys.modules['scipy.sparse'] = MockModule()
sys.modules['scipy.linalg'] = MockModule()
sys.modules['sklearn'] = MockModule()
sys.modules['sklearn.model_selection'] = MockModule()
sys.modules['matplotlib'] = MockModule()
sys.modules['matplotlib.pyplot'] = MockModule()
sys.modules['tqdm'] = MockModule()

def test_syntax_validation():
    """Test syntax validation for all Python files."""
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk('/root/repo/probneural_operator'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to validate")
    
    failed_files = []
    
    for file_path in python_files:
        try:
            # Load and compile the module
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ {file_path}")
        except Exception as e:
            print(f"❌ {file_path}: {e}")
            failed_files.append((file_path, str(e)))
    
    print(f"\n=== SUMMARY ===")
    print(f"Total files: {len(python_files)}")
    print(f"Successful: {len(python_files) - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\n=== FAILED FILES ===")
        for file_path, error in failed_files:
            print(f"{file_path}: {error}")
        return False
    else:
        print("\n✅ All files passed syntax validation!")
        return True

if __name__ == "__main__":
    success = test_syntax_validation()
    sys.exit(0 if success else 1)