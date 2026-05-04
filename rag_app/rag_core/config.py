"""
Config Module - Load and manage application configuration from YAML file
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """
    Configuration loader for the RAG application.
    
    Loads settings from config.yaml file and provides attribute-based access.
    """
    
    def __init__(self) -> None:
        """
        Load configuration from YAML file.
        
        The config.yaml file should be located in the parent directory
        (same level as manage.py in Django project).
        """
        config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        
        with open(config_path, "r", encoding="utf-8") as file:
            self._data: Dict[str, Any] = yaml.safe_load(file)
    
    def __getattr__(self, name: str) -> Any:
        """
        Access configuration values as attributes. """
        return self._data.get(name, {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default. """
        return self._data.get(key, default)


# Singleton instance for module-level import
config = Config()