"""
Pattern definitions and configuration for candlestick patterns.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class PatternConfig:
    """Configuration for pattern detection."""

    # Single-candle patterns
    single_candle_patterns: List[str] = field(default_factory=lambda: [
        "DOJI",
        "HAMMER",
        "INVERTED_HAMMER",
        "SHOOTING_STAR",
        "HANGING_MAN",
        "SPINNING_TOP",
        "MARUBOZU",
        "DRAGONFLY_DOJI",
        "GRAVESTONE_DOJI"
    ])

    # Multi-candle patterns
    multi_candle_patterns: List[str] = field(default_factory=lambda: [
        "ENGULFING",
        "BULLISH_ENGULFING",
        "BEARISH_ENGULFING",
        "HARAMI",
        "BULLISH_HARAMI",
        "BEARISH_HARAMI",
        "PIERCING_LINE",
        "DARK_CLOUD_COVER",
        "MORNING_STAR",
        "EVENING_STAR",
        "THREE_WHITE_SOLDIERS",
        "THREE_BLACK_CROWS",
        "RISING_THREE_METHODS",
        "FALLING_THREE_METHODS"
    ])

    # Detection parameters
    min_similarity: float = 0.85
    window_size: int = 1  # Window size for pattern search
    max_window_size: int = 5  # Maximum window for multi-candle patterns

    # Parallel processing settings
    use_parallel: bool = True
    n_jobs: Optional[int] = None  # None means use all available cores
    show_progress: bool = False

    def get_all_patterns(self) -> List[str]:
        """Get all enabled patterns."""
        return self.single_candle_patterns + self.multi_candle_patterns

    def get_pattern_window_size(self, pattern_name: str) -> int:
        """Get the appropriate window size for a pattern."""
        # Single candle patterns use window size of 1
        if pattern_name in self.single_candle_patterns:
            return 1

        # Multi-candle patterns - determine window size based on pattern name
        pattern_windows = {
            "ENGULFING": 2,
            "BULLISH_ENGULFING": 2,
            "BEARISH_ENGULFING": 2,
            "HARAMI": 2,
            "BULLISH_HARAMI": 2,
            "BEARISH_HARAMI": 2,
            "PIERCING_LINE": 2,
            "DARK_CLOUD_COVER": 2,
            "MORNING_STAR": 3,
            "EVENING_STAR": 3,
            "THREE_WHITE_SOLDIERS": 3,
            "THREE_BLACK_CROWS": 3,
            "RISING_THREE_METHODS": 5,
            "FALLING_THREE_METHODS": 5
        }

        return pattern_windows.get(pattern_name, self.window_size)


def get_default_patterns() -> PatternConfig:
    """Get default pattern configuration."""
    return PatternConfig()


def create_custom_pattern_config(
    single_patterns: Optional[List[str]] = None,
    multi_patterns: Optional[List[str]] = None,
    min_similarity: float = 0.85,
    use_parallel: bool = True
) -> PatternConfig:
    """
    Create a custom pattern configuration.

    Args:
        single_patterns: List of single-candle pattern names to detect
        multi_patterns: List of multi-candle pattern names to detect
        min_similarity: Minimum similarity threshold for pattern matching
        use_parallel: Whether to use parallel processing

    Returns:
        PatternConfig instance
    """
    config = PatternConfig()

    if single_patterns is not None:
        config.single_candle_patterns = single_patterns
    if multi_patterns is not None:
        config.multi_candle_patterns = multi_patterns

    config.min_similarity = min_similarity
    config.use_parallel = use_parallel

    return config


# Pattern similarity templates (for reference and future use)
# These would be used with vectorbtpro's pattern search functionality
PATTERN_TEMPLATES = {
    "DOJI": {
        "description": "Opening and closing prices nearly equal, small body",
        "criteria": "abs(open - close) / (high - low) < 0.1"
    },
    "HAMMER": {
        "description": "Small body at upper end, long lower shadow",
        "criteria": "Long lower shadow (2x body), small/no upper shadow"
    },
    "SHOOTING_STAR": {
        "description": "Small body at lower end, long upper shadow",
        "criteria": "Long upper shadow (2x body), small/no lower shadow"
    },
    "ENGULFING": {
        "description": "Second candle completely engulfs first candle",
        "criteria": "Second body completely contains first body"
    },
    "MORNING_STAR": {
        "description": "Three-candle bullish reversal pattern",
        "criteria": "Long bearish, small body, long bullish"
    },
    "EVENING_STAR": {
        "description": "Three-candle bearish reversal pattern",
        "criteria": "Long bullish, small body, long bearish"
    }
}
