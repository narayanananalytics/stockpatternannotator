"""
Database module for storing OHLC data and annotations.
"""

from typing import Optional, List, Union
import pandas as pd
from datetime import datetime
import warnings

try:
    import sqlalchemy
    from sqlalchemy import create_engine, Table, Column, Integer, String, Float, DateTime, MetaData, Index
    from sqlalchemy.pool import NullPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    warnings.warn("sqlalchemy not available. Install it to use database functionality.")


class DatabaseManager:
    """Manager for database operations."""

    def __init__(self, database_url: str = "sqlite:///stockpatterns.db"):
        """
        Initialize database manager.

        Args:
            database_url: Database connection string (SQLAlchemy format)
                         Examples:
                         - SQLite: "sqlite:///stockpatterns.db"
                         - PostgreSQL: "postgresql://user:password@localhost:5432/dbname"
                         - MySQL: "mysql+pymysql://user:password@localhost:3306/dbname"
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy is required for database functionality. Install with: pip install sqlalchemy")

        self.database_url = database_url
        self.engine = create_engine(database_url, poolclass=NullPool)
        self.metadata = MetaData()

        # Define tables
        self._define_tables()

        # Create tables if they don't exist
        self.metadata.create_all(self.engine)

    def _define_tables(self):
        """Define database tables."""
        # OHLC data table
        self.ohlc_table = Table(
            'ohlc_data',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('symbol', String(20), nullable=False, index=True),
            Column('timeframe', String(10), nullable=False, index=True),
            Column('timestamp', DateTime, nullable=False, index=True),
            Column('open', Float, nullable=False),
            Column('high', Float, nullable=False),
            Column('low', Float, nullable=False),
            Column('close', Float, nullable=False),
            Column('volume', Float),
            Column('transactions', Integer),
            Column('created_at', DateTime, default=datetime.utcnow),
            Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp', unique=True)
        )

        # Annotations table
        self.annotations_table = Table(
            'annotations',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('symbol', String(20), nullable=False, index=True),
            Column('timeframe', String(10), nullable=False, index=True),
            Column('pattern_name', String(50), nullable=False, index=True),
            Column('start_timestamp', DateTime, nullable=False, index=True),
            Column('end_timestamp', DateTime, nullable=False),
            Column('confidence', Float),
            Column('created_at', DateTime, default=datetime.utcnow),
            Index('idx_ann_symbol_timeframe', 'symbol', 'timeframe'),
            Index('idx_ann_pattern', 'pattern_name')
        )

    def save_ohlc_data(
        self,
        data: pd.DataFrame,
        if_exists: str = 'append'
    ) -> int:
        """
        Save OHLC data to database.

        Args:
            data: DataFrame with OHLC data (must have columns: symbol, timeframe, timestamp, open, high, low, close)
            if_exists: How to behave if table exists ('append', 'replace', 'fail')

        Returns:
            Number of rows saved
        """
        # Validate required columns
        required_cols = ['symbol', 'timeframe', 'open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing}")

        # Prepare data
        df = data.copy()

        # Ensure timestamp is in the data
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                raise ValueError("Data must have timestamp column or DatetimeIndex")

        # Add created_at if not present
        if 'created_at' not in df.columns:
            df['created_at'] = datetime.utcnow()

        # Select only columns that exist in the table
        table_columns = [c.name for c in self.ohlc_table.columns if c.name != 'id']
        df_columns = [col for col in table_columns if col in df.columns]
        df = df[df_columns]

        # Handle duplicates based on if_exists strategy
        if if_exists == 'append':
            # Remove duplicates that already exist in database
            df = self._remove_existing_ohlc_duplicates(df)

        # Save to database
        rows_saved = df.to_sql(
            'ohlc_data',
            self.engine,
            if_exists=if_exists,
            index=False,
            method='multi',
            chunksize=1000
        )

        return len(df) if rows_saved is None else rows_saved

    def save_annotations(
        self,
        annotations: pd.DataFrame,
        if_exists: str = 'append'
    ) -> int:
        """
        Save annotations to database.

        Args:
            annotations: DataFrame with annotations
            if_exists: How to behave if table exists ('append', 'replace', 'fail')

        Returns:
            Number of rows saved
        """
        # Validate required columns
        required_cols = ['symbol', 'timeframe', 'pattern_name', 'start_timestamp', 'end_timestamp']
        if not all(col in annotations.columns for col in required_cols):
            missing = [col for col in required_cols if col not in annotations.columns]
            raise ValueError(f"Missing required columns: {missing}")

        # Prepare data
        df = annotations.copy()

        # Add created_at if not present
        if 'created_at' not in df.columns:
            df['created_at'] = datetime.utcnow()

        # Select only columns that exist in the table
        table_columns = [c.name for c in self.annotations_table.columns if c.name != 'id']
        df_columns = [col for col in table_columns if col in df.columns]
        df = df[df_columns]

        # Save to database
        rows_saved = df.to_sql(
            'annotations',
            self.engine,
            if_exists=if_exists,
            index=False,
            method='multi',
            chunksize=1000
        )

        return len(df) if rows_saved is None else rows_saved

    def load_ohlc_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load OHLC data from database.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of rows to return

        Returns:
            DataFrame with OHLC data
        """
        # Build query
        query = "SELECT symbol, timeframe, timestamp, open, high, low, close, volume, transactions FROM ohlc_data WHERE 1=1"
        params = {}

        if symbol:
            query += " AND symbol = :symbol"
            params['symbol'] = symbol

        if timeframe:
            query += " AND timeframe = :timeframe"
            params['timeframe'] = timeframe

        if start_date:
            query += " AND timestamp >= :start_date"
            params['start_date'] = start_date

        if end_date:
            query += " AND timestamp <= :end_date"
            params['end_date'] = end_date

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        df = pd.read_sql(query, self.engine, params=params)

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        return df

    def load_annotations(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load annotations from database.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            pattern_name: Filter by pattern name
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of rows to return

        Returns:
            DataFrame with annotations
        """
        # Build query
        query = "SELECT * FROM annotations WHERE 1=1"
        params = {}

        if symbol:
            query += " AND symbol = :symbol"
            params['symbol'] = symbol

        if timeframe:
            query += " AND timeframe = :timeframe"
            params['timeframe'] = timeframe

        if pattern_name:
            query += " AND pattern_name = :pattern_name"
            params['pattern_name'] = pattern_name

        if start_date:
            query += " AND start_timestamp >= :start_date"
            params['start_date'] = start_date

        if end_date:
            query += " AND start_timestamp <= :end_date"
            params['end_date'] = end_date

        query += " ORDER BY start_timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        df = pd.read_sql(query, self.engine, params=params)

        if not df.empty:
            df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
            df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])

        return df

    def _remove_existing_ohlc_duplicates(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows from new_data that already exist in the database.

        Args:
            new_data: DataFrame with new OHLC data

        Returns:
            DataFrame with duplicates removed
        """
        if new_data.empty:
            return new_data

        # Get unique symbol-timeframe combinations
        combinations = new_data[['symbol', 'timeframe']].drop_duplicates()

        existing_timestamps = []

        for _, row in combinations.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']

            # Get min and max timestamps for this combination in new data
            subset = new_data[(new_data['symbol'] == symbol) & (new_data['timeframe'] == timeframe)]
            min_ts = subset['timestamp'].min()
            max_ts = subset['timestamp'].max()

            # Query existing timestamps
            query = """
                SELECT timestamp FROM ohlc_data
                WHERE symbol = :symbol AND timeframe = :timeframe
                AND timestamp >= :min_ts AND timestamp <= :max_ts
            """
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'min_ts': min_ts,
                'max_ts': max_ts
            }

            existing = pd.read_sql(query, self.engine, params=params)
            if not existing.empty:
                existing['timestamp'] = pd.to_datetime(existing['timestamp'])
                existing_timestamps.extend(existing['timestamp'].tolist())

        # Remove duplicates
        if existing_timestamps:
            new_data = new_data[~new_data['timestamp'].isin(existing_timestamps)]

        return new_data

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in the database.

        Returns:
            List of symbol strings
        """
        query = "SELECT DISTINCT symbol FROM ohlc_data ORDER BY symbol"
        df = pd.read_sql(query, self.engine)
        return df['symbol'].tolist()

    def get_available_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """
        Get list of available timeframes.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of timeframe strings
        """
        query = "SELECT DISTINCT timeframe FROM ohlc_data"
        params = {}

        if symbol:
            query += " WHERE symbol = :symbol"
            params['symbol'] = symbol

        query += " ORDER BY timeframe"

        df = pd.read_sql(query, self.engine, params=params)
        return df['timeframe'].tolist()

    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary of available data.

        Returns:
            DataFrame with data summary (symbol, timeframe, count, min_date, max_date)
        """
        query = """
            SELECT
                symbol,
                timeframe,
                COUNT(*) as record_count,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM ohlc_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """

        df = pd.read_sql(query, self.engine)
        if not df.empty:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])

        return df

    def get_annotation_summary(self) -> pd.DataFrame:
        """
        Get summary of annotations.

        Returns:
            DataFrame with annotation summary
        """
        query = """
            SELECT
                symbol,
                timeframe,
                pattern_name,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM annotations
            GROUP BY symbol, timeframe, pattern_name
            ORDER BY symbol, timeframe, pattern_name
        """

        return pd.read_sql(query, self.engine)

    def delete_ohlc_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Delete OHLC data.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Number of rows deleted
        """
        from sqlalchemy import delete

        stmt = delete(self.ohlc_table)

        if symbol:
            stmt = stmt.where(self.ohlc_table.c.symbol == symbol)
        if timeframe:
            stmt = stmt.where(self.ohlc_table.c.timeframe == timeframe)
        if start_date:
            stmt = stmt.where(self.ohlc_table.c.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(self.ohlc_table.c.timestamp <= end_date)

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            conn.commit()
            return result.rowcount

    def delete_annotations(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern_name: Optional[str] = None
    ) -> int:
        """
        Delete annotations.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            pattern_name: Filter by pattern name

        Returns:
            Number of rows deleted
        """
        from sqlalchemy import delete

        stmt = delete(self.annotations_table)

        if symbol:
            stmt = stmt.where(self.annotations_table.c.symbol == symbol)
        if timeframe:
            stmt = stmt.where(self.annotations_table.c.timeframe == timeframe)
        if pattern_name:
            stmt = stmt.where(self.annotations_table.c.pattern_name == pattern_name)

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            conn.commit()
            return result.rowcount

    def close(self):
        """Close database connection."""
        self.engine.dispose()


def create_database_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """
    Create a database manager with URL from environment if not provided.

    Args:
        database_url: Database connection string (if None, will use default SQLite)

    Returns:
        DatabaseManager instance
    """
    if database_url is None:
        import os
        database_url = os.getenv('DATABASE_URL', 'sqlite:///stockpatterns.db')

    return DatabaseManager(database_url)
