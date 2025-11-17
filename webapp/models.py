"""Data models for the web application."""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComputeType(str, Enum):
    """Compute type enum."""
    CPU = "cpu"
    GPU = "gpu"


class TrainingJob(BaseModel):
    """Training job model."""
    job_id: str
    user_email: str

    # Input parameters
    polygon_api_key: str
    symbol: str
    timeframe: str  # 1M, 3M, 5M, 1H, 1D
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD

    # Training parameters
    total_timesteps: int = Field(default=100000, ge=1000, le=5000000)
    compute_type: ComputeType = ComputeType.CPU
    forecast_horizon: int = Field(default=5, ge=1, le=50)

    # Environment parameters
    initial_balance: float = Field(default=10000.0, gt=0)
    transaction_cost: float = Field(default=0.001, ge=0, le=0.1)
    allow_short: bool = False

    # Job metadata
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0-100
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    model_path: Optional[str] = None  # GCS path
    results_path: Optional[str] = None  # GCS path
    metrics: Optional[Dict[str, Any]] = None

    # Cloud Run Job details
    cloud_run_job_name: Optional[str] = None
    cloud_run_execution_name: Optional[str] = None

    class Config:
        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore."""
        data = self.dict()
        # Convert datetime to timestamp
        for key in ['created_at', 'updated_at', 'started_at', 'completed_at']:
            if data.get(key):
                data[key] = data[key].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingJob':
        """Create from Firestore dictionary."""
        # Convert ISO strings back to datetime
        for key in ['created_at', 'updated_at', 'started_at', 'completed_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class JobResults(BaseModel):
    """Job results model."""
    job_id: str

    # Training metrics
    mean_return: float
    std_return: float
    mean_win_rate: float
    mean_trades: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Backtest summary
    total_trades: int
    winning_trades: int
    losing_trades: int
    final_balance: float
    total_return_pct: float

    # Timestamps
    training_duration_seconds: float
    data_candles: int

    # Equity curve (for plotting)
    equity_curve: Optional[list] = None  # List of {timestamp, balance}
    trades: Optional[list] = None  # List of trade dicts

    class Config:
        arbitrary_types_allowed = True
