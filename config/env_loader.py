"""
Environment loader for managing configuration from .env files.
Loads environment variables from .env.dev or .env.example files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class EnvLoader:
    """Load and manage environment variables from .env files."""
    
    _loaded = False
    
    @classmethod
    def load(cls, env_file=None):
        """
        Load environment variables from .env file.
        
        Args:
            env_file (str): Path to .env file. If None, tries .env.dev, then .env.example
        """
        if cls._loaded:
            return
        
        if env_file is None:
            # Try loading from .env.dev first, then .env.example
            workspace_root = Path(__file__).parent.parent
            env_files = [
                workspace_root / ".env.dev",
                workspace_root / ".env.example",
            ]
            
            for env_file_path in env_files:
                if env_file_path.exists():
                    env_file = str(env_file_path)
                    logger.info(f"Loading environment from: {env_file}")
                    break
        
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            cls._loaded = True
            logger.info(f"Successfully loaded environment variables from {env_file}")
        else:
            logger.warning(f"No .env file found at {env_file}. Using system environment variables.")
    
    @staticmethod
    def get(key, default=None):
        """
        Get environment variable with optional default value.
        
        Args:
            key (str): Environment variable name
            default: Default value if variable not found
        
        Returns:
            Environment variable value or default
        """
        EnvLoader.load()
        return os.getenv(key, default)
    
    @staticmethod
    def get_int(key, default=None):
        """Get environment variable as integer."""
        value = EnvLoader.get(key, default)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {key}={value} to int, using default {default}")
            return default
    
    @staticmethod
    def get_float(key, default=None):
        """Get environment variable as float."""
        value = EnvLoader.get(key, default)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {key}={value} to float, using default {default}")
            return default
    
    @staticmethod
    def get_bool(key, default=False):
        """Get environment variable as boolean."""
        value = EnvLoader.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_list(key, delimiter=',', default=None):
        """Get environment variable as list."""
        value = EnvLoader.get(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(delimiter) if item.strip()]
