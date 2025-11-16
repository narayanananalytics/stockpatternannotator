"""
Stock Pattern Annotator
A flexible tool to annotate OHLC price data with candlestick patterns and support/resistance levels.
"""

from .annotator import PatternAnnotator
from .patterns import PatternConfig, get_default_patterns
from .pivots import PivotDetector

__version__ = "0.1.0"
__all__ = ["PatternAnnotator", "PatternConfig", "get_default_patterns", "PivotDetector"]
