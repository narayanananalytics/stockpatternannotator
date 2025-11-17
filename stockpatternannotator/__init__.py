"""
Stock Pattern Annotator
A flexible tool to annotate OHLC price data with candlestick patterns and support/resistance levels.
"""

from .annotator import PatternAnnotator
from .patterns import PatternConfig, get_default_patterns
from .pivots import PivotDetector
from .polygon_client import PolygonClient, create_polygon_client
from .database import DatabaseManager, create_database_manager
from .pipeline import DataPipeline
from .config import Config, load_config, setup_environment
from .validation import PatternValidator

# RL components (optional dependencies)
try:
    from .rl_environment import PatternTradingEnv
    from .rl_features import FeatureEngineer
    from .rl_agent import RLTradingAgent, create_training_agent
    from .rl_pipeline import RLPipeline
    from .rl_gpu_utils import (
        check_gpu_availability,
        print_gpu_info,
        get_gpu_optimized_config,
        enable_gpu_optimizations
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

__version__ = "0.4.0"
__all__ = [
    "PatternAnnotator",
    "PatternConfig",
    "get_default_patterns",
    "PivotDetector",
    "PolygonClient",
    "create_polygon_client",
    "DatabaseManager",
    "create_database_manager",
    "DataPipeline",
    "Config",
    "load_config",
    "setup_environment",
    "PatternValidator"
]

# Add RL components to __all__ if available
if RL_AVAILABLE:
    __all__.extend([
        "PatternTradingEnv",
        "FeatureEngineer",
        "RLTradingAgent",
        "create_training_agent",
        "RLPipeline",
        "check_gpu_availability",
        "print_gpu_info",
        "get_gpu_optimized_config",
        "enable_gpu_optimizations"
    ])
