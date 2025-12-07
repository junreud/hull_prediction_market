"""
Utility functions for reproducibility, logging, and timing.
"""

import os
import random
import time
import logging
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Any

import numpy as np
import yaml


# Global shared logger instance
_SHARED_LOGGER = None
_SHARED_LOG_FILE = None


def get_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_str: Optional[str] = None,
    name: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a shared logger instance.
    
    All modules use the same logger to consolidate logs into one file.
    The first call sets the log file, and subsequent calls reuse it.
    
    Args:
        log_file: Path to log file (optional, uses first caller's value)
        level: Logging level
        format_str: Custom format string (optional)
        name: Logger name (ignored, uses shared logger)
        
    Returns:
        Shared logger instance
    """
    global _SHARED_LOGGER, _SHARED_LOG_FILE
    
    # If logger doesn't exist yet, create it
    if _SHARED_LOGGER is None:
        _SHARED_LOG_FILE = log_file or 'logs/app.log'
        _SHARED_LOGGER = setup_logging(_SHARED_LOG_FILE, level, format_str, logger_name='shared_logger')
    
    return _SHARED_LOGGER


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deep learning frameworks (if used)
    # Note: Disabled PyTorch/TensorFlow to avoid potential internet access issues in Kaggle
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # except ImportError:
    #     pass
    
def load_config(config_path: str = "conf/params.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_str: Optional[str] = None,
    logger_name: str = 'prediction_market'
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_str: Custom format string (optional)
        logger_name: Name for the logger
        
    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_str)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


class Timer:
    """
    Context manager for timing code execution.
    
    Usage:
        with Timer("Processing data"):
            # your code here
            pass
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger('prediction_market')
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.name}")
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger.info(f"Completed: {self.name} - Time: {elapsed:.2f}s")
        
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


def timeit(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Usage:
        @timeit
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger('prediction_market')
        start_time = time.time()
        logger.info(f"Starting function: {func.__name__}")
        
        result = func(*args, **kwargs)
        
        elapsed = time.time() - start_time
        logger.info(f"Completed function: {func.__name__} - Time: {elapsed:.2f}s")
        
        return result
    
    return wrapper


def create_directories(config: dict) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    dirs_to_create = [
        config.get('paths', {}).get('artifacts', 'artifacts/'),
        config.get('paths', {}).get('models', 'models/'),
        config.get('paths', {}).get('submissions', 'submissions/'),
        config.get('paths', {}).get('logs', 'logs/'),
        'data/raw/',
        'data/processed/',
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    print(f"Random number test: {np.random.rand()}")
    
    # Test config loading
    try:
        config = load_config("conf/params.yaml")
        print(f"Config loaded successfully. Seed: {config.get('seed')}")
    except FileNotFoundError:
        print("Config file not found (expected in test)")
    
    # Test logging with global logger
    logger = get_logger(log_file="logs/test_utils.log", level="INFO")
    logger.info("Test log message")
    
    # Test timer
    with Timer("Test operation", logger):
        time.sleep(0.1)
    
    # Test that getting logger again doesn't create duplicates
    logger2 = get_logger()
    logger2.info("Second logger call (should use same instance)")
    
    print("All tests completed!")
