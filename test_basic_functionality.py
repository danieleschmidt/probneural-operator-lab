#!/usr/bin/env python3
"""Basic functionality test for the probabilistic neural operator framework."""

import sys
import os
sys.path.insert(0, '/root/repo')

# Test basic imports that should work
def test_basic_imports():
    """Test that basic modules can be imported."""
    print("Testing basic imports...")
    
    try:
        # Test utility imports (should work without external deps)
        from probneural_operator.utils.exceptions import ModelInitializationError
        from probneural_operator.utils.validation import validate_tensor_shape
        from probneural_operator.utils.internationalization import MultilingualSupport
        print("✅ Utils imports successful")
        
        # Test scaling imports (should work without torch)
        from probneural_operator.scaling.cache import AdvancedCache
        from probneural_operator.scaling.production_server import ProductionServer
        print("✅ Scaling imports successful")
        
        # Test security imports
        from probneural_operator.utils.security_framework import SecurityFramework
        from probneural_operator.utils.compliance import ComplianceFramework
        print("✅ Security imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test exception system
        from probneural_operator.utils.exceptions import ModelInitializationError
        try:
            raise ModelInitializationError("Test error")
        except ModelInitializationError as e:
            print(f"✅ Exception system works: {e}")
        
        # Test internationalization
        from probneural_operator.utils.internationalization import MultilingualSupport
        ml = MultilingualSupport()
        text = ml.get_text("model_training", "en")
        print(f"✅ Internationalization works: {text}")
        
        # Test cache system
        from probneural_operator.scaling.cache import AdvancedCache
        cache = AdvancedCache(max_size_gb=1.0)
        cache.put("test_key", {"data": "test_value"})
        cached_value = cache.get("test_key")
        print(f"✅ Cache system works: {cached_value}")
        
        # Test security framework
        from probneural_operator.utils.security_framework import SecurityFramework
        security = SecurityFramework()
        print(f"✅ Security framework initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧠 PROBNEURAL OPERATOR - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test imports
    import_success = test_basic_imports()
    
    # Test functionality  
    if import_success:
        func_success = test_basic_functionality()
    else:
        func_success = False
    
    print("\n" + "=" * 60)
    if import_success and func_success:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("Framework core functionality is working.")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Check dependency installation and import paths.")
        return 1

if __name__ == "__main__":
    sys.exit(main())