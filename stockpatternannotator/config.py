"""
Configuration management for API keys and database connections.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the stock pattern annotator."""

    # Polygon.io API configuration
    polygon_api_key: Optional[str] = None

    # Database configuration
    database_url: str = "sqlite:///stockpatterns.db"

    # Rate limiting
    polygon_rate_limit_delay: float = 0.1  # seconds between requests

    # Default timespan for data fetching
    default_timespan: str = "day"
    default_multiplier: int = 1

    # Pattern detection defaults
    default_min_similarity: float = 0.85
    default_use_parallel: bool = True

    # Pivot detection defaults
    default_pivot_left_bars: int = 5
    default_pivot_right_bars: int = 5
    default_pivot_min_strength: float = 0.3

    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration from environment variables.

        Environment variables:
            POLYGON_API_KEY: Polygon.io API key
            DATABASE_URL: Database connection string
            POLYGON_RATE_LIMIT: Rate limit delay in seconds
        """
        return cls(
            polygon_api_key=os.getenv('POLYGON_API_KEY'),
            database_url=os.getenv('DATABASE_URL', 'sqlite:///stockpatterns.db'),
            polygon_rate_limit_delay=float(os.getenv('POLYGON_RATE_LIMIT', '0.1'))
        )

    def validate(self) -> tuple[bool, str]:
        """
        Validate configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check polygon API key if needed
        if not self.polygon_api_key and os.getenv('POLYGON_API_KEY') is None:
            return False, "Polygon API key not set. Set POLYGON_API_KEY environment variable or pass api_key parameter."

        # Validate database URL format
        if not self.database_url:
            return False, "Database URL is required"

        valid_prefixes = ['sqlite://', 'postgresql://', 'mysql://', 'mysql+pymysql://']
        if not any(self.database_url.startswith(prefix) for prefix in valid_prefixes):
            return False, f"Invalid database URL. Must start with one of: {valid_prefixes}"

        return True, ""


def load_config(
    polygon_api_key: Optional[str] = None,
    database_url: Optional[str] = None,
    use_env: bool = True
) -> Config:
    """
    Load configuration with priority: parameters > environment > defaults.

    Args:
        polygon_api_key: Polygon.io API key
        database_url: Database connection string
        use_env: Whether to load from environment variables

    Returns:
        Config instance
    """
    if use_env:
        config = Config.from_env()
    else:
        config = Config()

    # Override with explicit parameters
    if polygon_api_key is not None:
        config.polygon_api_key = polygon_api_key

    if database_url is not None:
        config.database_url = database_url

    return config


def setup_environment(
    polygon_api_key: Optional[str] = None,
    database_url: Optional[str] = None
):
    """
    Setup environment variables for configuration.

    Args:
        polygon_api_key: Polygon.io API key
        database_url: Database connection string
    """
    if polygon_api_key:
        os.environ['POLYGON_API_KEY'] = polygon_api_key

    if database_url:
        os.environ['DATABASE_URL'] = database_url


# Example .env file content
ENV_TEMPLATE = """
# Polygon.io Configuration
POLYGON_API_KEY=your_api_key_here

# Database Configuration
# SQLite (default)
DATABASE_URL=sqlite:///stockpatterns.db

# PostgreSQL
# DATABASE_URL=postgresql://user:password@localhost:5432/stockpatterns

# MySQL
# DATABASE_URL=mysql+pymysql://user:password@localhost:3306/stockpatterns

# Rate Limiting
POLYGON_RATE_LIMIT=0.1
"""


def create_env_template(filepath: str = '.env.template'):
    """
    Create a template .env file.

    Args:
        filepath: Path to create the template file
    """
    with open(filepath, 'w') as f:
        f.write(ENV_TEMPLATE.strip())

    print(f"Created environment template at: {filepath}")
    print("Copy this to .env and fill in your values")
