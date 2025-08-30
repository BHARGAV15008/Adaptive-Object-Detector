"""
Configuration Management Module
Handles loading and managing configuration from YAML files with environment variable support
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from functools import lru_cache
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class AppConfig:
    """Application configuration data class"""
    name: str
    version: str
    description: str
    environment: Environment
    debug: bool
    host: str
    port: int
    workers: int


class ConfigManager:
    """
    Advanced configuration manager with environment variable interpolation,
    validation, and hot reloading capabilities
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        self._config: Optional[Dict[str, Any]] = None
        self._watchers = []
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file with environment variable interpolation"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_content = file.read()
                
            # Interpolate environment variables
            config_content = self._interpolate_env_vars(config_content)
            
            # Parse YAML
            self._config = yaml.safe_load(config_content)
            
            # Validate configuration
            self._validate_config()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _interpolate_env_vars(self, content: str) -> str:
        """Replace ${VAR_NAME} patterns with environment variables"""
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, f"${{{var_name}}}")  # Keep original if not found
        
        return re.sub(pattern, replacer, content)
    
    def _validate_config(self) -> None:
        """Validate configuration structure and required fields"""
        if not self._config:
            raise ValueError("Configuration is empty")
        
        required_sections = ['app', 'models', 'api', 'storage']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Required configuration section missing: {section}")
    
    @lru_cache(maxsize=128)
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation (e.g., 'app.name')
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self._config:
            return default
        
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except Exception:
            return default
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration as data class"""
        app_config = self.get('app', {})
        return AppConfig(
            name=app_config.get('name', 'Object Detection System'),
            version=app_config.get('version', '1.0.0'),
            description=app_config.get('description', ''),
            environment=Environment(app_config.get('environment', 'development')),
            debug=app_config.get('debug', True),
            host=app_config.get('host', '0.0.0.0'),
            port=app_config.get('port', 8000),
            workers=app_config.get('workers', 4)
        )
    
    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model configuration for specified model or default"""
        if not model_name:
            model_name = self.get('models.default_model', 'yolov8n')
        
        available_models = self.get('models.available_models', {})
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        return available_models[model_name]
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get('database', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get('api', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get('monitoring', {})
    
    def get_cv_features_config(self) -> Dict[str, Any]:
        """Get computer vision features configuration"""
        return self.get('cv_features', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get('security', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.get('hardware', {})
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        if not self._config:
            self._config = {}
        
        keys = key_path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        
        # Clear cache
        self.get.cache_clear()
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.get(f'features.{feature_name}', False)
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.get('app.environment') == Environment.DEVELOPMENT.value
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.get('app.environment') == Environment.PRODUCTION.value
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self.get.cache_clear()
        self.load_config()
        logger.info("Configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config.copy() if self._config else {}
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file"""
        if not path:
            path = self.config_path
        
        with open(path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {path}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    return config_manager


def get(key_path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return config_manager.get(key_path, default)


def set_config(key_path: str, value: Any) -> None:
    """Convenience function to set configuration value"""
    config_manager.set(key_path, value)


def is_feature_enabled(feature_name: str) -> bool:
    """Convenience function to check if feature is enabled"""
    return config_manager.is_feature_enabled(feature_name)
