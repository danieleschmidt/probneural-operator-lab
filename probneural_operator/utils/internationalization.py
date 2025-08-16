"""
Internationalization (i18n) Support for ProbNeural-Operator-Lab
===============================================================

Multi-language support for global deployment including:
- Interface localization (en, es, fr, de, ja, zh) 
- Error message translation
- Documentation localization
- Regional compliance features
"""

import os
import json
from typing import Dict, Optional
from pathlib import Path

class InternationalizationManager:
    """Manages multi-language support and localization."""
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations = {}
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files for all supported languages."""
        translations_dir = Path(__file__).parent / "locales"
        
        supported_locales = ["en", "es", "fr", "de", "ja", "zh"]
        
        for locale in supported_locales:
            locale_file = translations_dir / f"{locale}.json"
            if locale_file.exists():
                try:
                    with open(locale_file, 'r', encoding='utf-8') as f:
                        self.translations[locale] = json.load(f)
                except Exception:
                    # Fallback to default if translation fails
                    self.translations[locale] = self._get_default_translations()
            else:
                self.translations[locale] = self._get_default_translations()
    
    def _get_default_translations(self) -> Dict[str, str]:
        """Default English translations."""
        return {
            "training_started": "Training started",
            "training_completed": "Training completed successfully",
            "validation_error": "Validation error occurred",
            "model_saved": "Model saved",
            "prediction_uncertainty": "Prediction uncertainty",
            "active_learning_iteration": "Active learning iteration",
            "distributed_training_init": "Initializing distributed training",
            "quality_gate_passed": "Quality gate passed",
            "quality_gate_failed": "Quality gate failed",
            "compliance_check": "Compliance check",
            "gdpr_compliant": "GDPR compliant",
            "data_processing_notice": "Data processing notice",
        }
    
    def set_locale(self, locale: str):
        """Set current locale."""
        if locale in self.translations:
            self.current_locale = locale
        else:
            raise ValueError(f"Unsupported locale: {locale}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key to current locale with optional formatting."""
        translation = self.translations.get(
            self.current_locale, {}
        ).get(key, key)
        
        if kwargs:
            try:
                return translation.format(**kwargs)
            except:
                return translation
        
        return translation
    
    def get_supported_locales(self) -> list:
        """Get list of supported locales."""
        return list(self.translations.keys())

# Global i18n instance
i18n = InternationalizationManager()

def _(key: str, **kwargs) -> str:
    """Shorthand translation function."""
    return i18n.translate(key, **kwargs)

def set_language(locale: str):
    """Set global language."""
    i18n.set_locale(locale)