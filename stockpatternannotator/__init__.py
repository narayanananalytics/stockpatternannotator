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

__version__ = "0.3.0"
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
