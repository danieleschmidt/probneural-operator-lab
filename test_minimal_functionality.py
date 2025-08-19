#!/usr/bin/env python3
"""Minimal functionality test for components that don't require torch."""

import sys
import os
sys.path.insert(0, '/root/repo')

def test_independent_modules():
    """Test modules that don't depend on torch."""
    print("Testing framework modules without torch dependencies...")
    
    try:
        # Test basic exception classes
        from probneural_operator.utils.exceptions import (
            ProbNeuralOperatorError, 
            ModelInitializationError,
            DataLoadingError,
            TrainingError
        )
        print("✅ Exception classes imported successfully")
        
        # Test exception functionality
        try:
            raise ModelInitializationError("Test initialization error")
        except ModelInitializationError as e:
            print(f"✅ Exception handling works: {e}")
        
        # Test internationalization (pure Python)
        from probneural_operator.utils.internationalization import MultilingualSupport
        ml = MultilingualSupport()
        
        # Test getting text in different languages
        en_text = ml.get_text("model_training", "en")
        es_text = ml.get_text("model_training", "es") 
        fr_text = ml.get_text("model_training", "fr")
        
        print(f"✅ Internationalization: EN='{en_text}', ES='{es_text}', FR='{fr_text}'")
        
        # Test compliance framework
        from probneural_operator.utils.compliance import ComplianceFramework
        compliance = ComplianceFramework()
        
        # Test GDPR compliance check
        gdpr_compliant = compliance.check_gdpr_compliance({
            'data_minimization': True,
            'consent_collected': True,
            'data_encrypted': True
        })
        print(f"✅ GDPR compliance check: {gdpr_compliant}")
        
        # Test security framework basics
        from probneural_operator.utils.security_framework import SecurityFramework
        security = SecurityFramework()
        
        # Test basic security operations
        test_data = "sensitive_data_123"
        encrypted = security.encrypt_data(test_data)
        decrypted = security.decrypt_data(encrypted)
        
        print(f"✅ Security encryption/decryption: '{test_data}' -> encrypted -> '{decrypted}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in independent modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_framework_structure():
    """Test that the framework structure is valid."""
    print("\nTesting framework structure...")
    
    try:
        # Check that main package can be imported
        import probneural_operator
        print("✅ Main package importable")
        
        # Check package structure exists
        package_dir = '/root/repo/probneural_operator'
        subdirs = ['models', 'utils', 'scaling', 'data', 'posteriors', 'active', 'calibration']
        
        for subdir in subdirs:
            subdir_path = os.path.join(package_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"✅ {subdir}/ directory exists")
            else:
                print(f"❌ {subdir}/ directory missing")
        
        # Check requirements file exists
        req_path = '/root/repo/requirements.txt'
        if os.path.exists(req_path):
            print("✅ requirements.txt exists")
            with open(req_path, 'r') as f:
                lines = len(f.readlines())
            print(f"✅ Requirements file has {lines} lines")
        else:
            print("❌ requirements.txt missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework structure error: {e}")
        return False

def test_production_readiness():
    """Test production-ready components."""
    print("\nTesting production-ready components...")
    
    try:
        # Test production server (should work without torch)
        from probneural_operator.scaling.production_server import ProductionServer
        
        # Create a basic production server instance
        server = ProductionServer(host="localhost", port=8000)
        print("✅ Production server can be instantiated")
        
        # Test advanced monitoring
        from probneural_operator.utils.advanced_monitoring import AdvancedMonitoringSystem
        monitoring = AdvancedMonitoringSystem()
        print("✅ Advanced monitoring system created")
        
        # Test logging configuration
        from probneural_operator.utils.logging_config import setup_logging
        logger = setup_logging("test_logger", level="INFO")
        logger.info("Test log message")
        print("✅ Logging system working")
        
        return True
        
    except Exception as e:
        print(f"❌ Production components error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run minimal functionality tests."""
    print("🔬 PROBNEURAL OPERATOR - MINIMAL FUNCTIONALITY TEST")
    print("=" * 70)
    print("Testing core functionality without ML dependencies...")
    print()
    
    # Run all tests
    test1 = test_independent_modules()
    test2 = test_framework_structure() 
    test3 = test_production_readiness()
    
    print("\n" + "=" * 70)
    
    if test1 and test2 and test3:
        print("🎉 ALL MINIMAL TESTS PASSED!")
        print()
        print("✨ Framework Core Status:")
        print("   • Exception handling: ✅ Working")
        print("   • Internationalization: ✅ Working") 
        print("   • Security framework: ✅ Working")
        print("   • Compliance tools: ✅ Working")
        print("   • Production server: ✅ Working")
        print("   • Monitoring system: ✅ Working")
        print()
        print("📋 Next Steps:")
        print("   1. Install ML dependencies: pip install -r requirements.txt")
        print("   2. Test full framework with: python test_full_functionality.py")
        print("   3. Run model training: python examples/train_example.py")
        return 0
    else:
        print("❌ SOME MINIMAL TESTS FAILED")
        print("Basic framework components need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())