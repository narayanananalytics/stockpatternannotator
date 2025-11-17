"""
Complete RL Training Pipeline.

Integrates data fetching, pattern validation, feature engineering,
training, and evaluation.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from .database import DatabaseManager
from .validation import PatternValidator
from .rl_features import FeatureEngineer
from .rl_environment import PatternTradingEnv
from .rl_agent import RLTradingAgent, create_training_agent


class RLPipeline:
    """
    Complete pipeline for RL-based pattern trading.

    Workflow:
        1. Load OHLC data and annotations from database
        2. Validate patterns and calculate probabilities
        3. Calculate technical indicators
        4. Prepare features for RL environment
        5. Split train/test data
        6. Train RL agent
        7. Evaluate on test set
        8. Backtest results
    """

    def __init__(
        self,
        database_url: str = 'sqlite:///stockpatterns.db',
        forecast_horizon: int = 5,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize RL pipeline.

        Args:
            database_url: Database connection string
            forecast_horizon: Forecast horizon for pattern probabilities
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.db = DatabaseManager(database_url)
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.random_state = random_state

        self.feature_engineer = FeatureEngineer()
        self.pattern_validator = None
        self.agent = None

        # Data storage
        self.ohlc_data = None
        self.annotations = None
        self.validation_results = None
        self.probabilities = None
        self.pattern_probs_df = None
        self.tech_indicators = None

        # Train/test split
        self.train_data = None
        self.test_data = None

    def load_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> 'RLPipeline':
        """
        Load data from database.

        Args:
            symbol: Symbol to load (None for all)
            timeframe: Timeframe to load (None for all)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Self for chaining
        """
        print("Loading data from database...")

        # Load OHLC data
        self.ohlc_data = self.db.load_ohlc_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if self.ohlc_data.empty:
            raise ValueError("No OHLC data found in database")

        print(f"Loaded {len(self.ohlc_data)} OHLC records")

        # Load annotations
        self.annotations = self.db.load_annotations(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if self.annotations.empty:
            warnings.warn("No annotations found in database. Pattern probabilities will be limited.")
        else:
            print(f"Loaded {len(self.annotations)} annotations")

        return self

    def validate_patterns(self) -> 'RLPipeline':
        """
        Validate patterns and calculate probabilities.

        Returns:
            Self for chaining
        """
        if self.annotations.empty:
            warnings.warn("No annotations to validate")
            return self

        print(f"\nValidating patterns with horizon={self.forecast_horizon}...")

        # Create validator
        self.pattern_validator = PatternValidator(
            forecast_horizons=[self.forecast_horizon],
            price_change_threshold=0.0,
            require_minimum_samples=3
        )

        # Validate patterns
        self.validation_results = self.pattern_validator.validate_patterns(
            ohlc_data=self.ohlc_data,
            annotations=self.annotations,
            price_column='close'
        )

        print(f"Validated {len(self.validation_results)} pattern instances")

        # Calculate probabilities
        if not self.validation_results.empty:
            self.probabilities = self.pattern_validator.calculate_probabilities()
            print(f"Calculated probabilities for {len(self.probabilities)} pattern types")

            # Prepare pattern probabilities DataFrame
            self.pattern_probs_df = self.feature_engineer.prepare_pattern_probabilities(
                self.validation_results,
                self.probabilities,
                self.forecast_horizon
            )
        else:
            warnings.warn("No validation results")

        return self

    def calculate_features(self) -> 'RLPipeline':
        """
        Calculate technical indicators.

        Returns:
            Self for chaining
        """
        print("\nCalculating technical indicators...")

        self.tech_indicators = self.feature_engineer.calculate_all_features(
            self.ohlc_data
        )

        print(f"Calculated {len(self.tech_indicators.columns)} technical indicators")

        return self

    def prepare_environments(
        self,
        env_config: Optional[Dict] = None
    ) -> Tuple[PatternTradingEnv, PatternTradingEnv]:
        """
        Prepare train and test environments.

        Args:
            env_config: Environment configuration

        Returns:
            train_env, test_env
        """
        print("\nPreparing training and test environments...")

        # Split data
        split_idx = int(len(self.ohlc_data) * (1 - self.test_size))

        train_ohlc = self.ohlc_data.iloc[:split_idx]
        test_ohlc = self.ohlc_data.iloc[split_idx:]

        train_indicators = self.tech_indicators.iloc[:split_idx]
        test_indicators = self.tech_indicators.iloc[split_idx:]

        # Pattern probabilities
        if self.pattern_probs_df is not None and not self.pattern_probs_df.empty:
            train_pattern_probs = self.pattern_probs_df[
                self.pattern_probs_df.index.isin(train_ohlc.index)
            ]
            test_pattern_probs = self.pattern_probs_df[
                self.pattern_probs_df.index.isin(test_ohlc.index)
            ]
        else:
            # Create empty DataFrames with required structure
            train_pattern_probs = pd.DataFrame(index=train_ohlc.index)
            train_pattern_probs['bullish_prob'] = 50.0
            train_pattern_probs['bearish_prob'] = 50.0
            train_pattern_probs['confidence'] = 0.5

            test_pattern_probs = pd.DataFrame(index=test_ohlc.index)
            test_pattern_probs['bullish_prob'] = 50.0
            test_pattern_probs['bearish_prob'] = 50.0
            test_pattern_probs['confidence'] = 0.5

        # Default environment config
        default_config = {
            'initial_balance': 10000,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'max_drawdown': 0.2,
            'holding_penalty_weight': 0.001,
            'pattern_bonus_weight': 0.1,
            'pattern_prob_threshold': 0.55,
            'max_holding_period': 20,
            'allow_short': True
        }

        if env_config:
            default_config.update(env_config)

        # Create environments
        train_env = PatternTradingEnv(
            df=train_ohlc,
            pattern_probabilities=train_pattern_probs,
            technical_indicators=train_indicators,
            **default_config
        )

        test_env = PatternTradingEnv(
            df=test_ohlc,
            pattern_probabilities=test_pattern_probs,
            technical_indicators=test_indicators,
            **default_config
        )

        self.train_env = train_env
        self.test_env = test_env

        print(f"Train environment: {len(train_ohlc)} steps")
        print(f"Test environment: {len(test_ohlc)} steps")

        return train_env, test_env

    def train_agent(
        self,
        total_timesteps: int = 100000,
        hyperparameters: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> RLTradingAgent:
        """
        Train RL agent.

        Args:
            total_timesteps: Total training timesteps
            hyperparameters: Custom hyperparameters
            save_path: Path to save trained model

        Returns:
            Trained agent
        """
        if not hasattr(self, 'train_env'):
            raise ValueError("Must call prepare_environments() first")

        print(f"\nTraining RL agent for {total_timesteps} timesteps...")

        # Create agent
        self.agent = create_training_agent(self.train_env, hyperparameters)

        # Train
        self.agent.train(
            total_timesteps=total_timesteps,
            eval_env=self.test_env,
            eval_freq=max(10000, total_timesteps // 10),
            n_eval_episodes=5
        )

        # Save model
        if save_path:
            self.agent.save(save_path)
            print(f"Model saved to {save_path}")

        return self.agent

    def evaluate_agent(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict:
        """
        Evaluate trained agent on test set.

        Args:
            n_episodes: Number of test episodes
            render: Whether to render

        Returns:
            Evaluation metrics
        """
        if self.agent is None:
            raise ValueError("Must train agent first")

        if not hasattr(self, 'test_env'):
            raise ValueError("Must prepare environments first")

        print(f"\nEvaluating agent on test set ({n_episodes} episodes)...")

        results = self.agent.evaluate(
            self.test_env,
            n_episodes=n_episodes,
            render=render
        )

        print("\nTest Results:")
        print(f"Mean Return: {results['mean_return']:.4f} ± {results['std_return']:.4f}")
        print(f"Mean Trades: {results['mean_trades']:.1f}")
        print(f"Mean Win Rate: {results['mean_win_rate']:.2%}")
        print(f"Min Return: {results['min_return']:.4f}")
        print(f"Max Return: {results['max_return']:.4f}")

        return results

    def backtest(
        self,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run detailed backtest on test set.

        Args:
            save_path: Path to save backtest results

        Returns:
            DataFrame with backtest results
        """
        if self.agent is None:
            raise ValueError("Must train agent first")

        print("\nRunning detailed backtest...")

        backtest_results = self.agent.backtest(self.test_env)

        if save_path:
            backtest_results.to_csv(save_path, index=False)
            print(f"Backtest results saved to {save_path}")

        # Calculate metrics
        final_value = backtest_results['total_value'].iloc[-1]
        initial_value = self.test_env.initial_balance
        total_return = (final_value - initial_value) / initial_value

        total_trades = backtest_results['position'].diff().abs().sum() / 2

        print(f"\nBacktest Summary:")
        print(f"Initial Balance: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Total Trades: {int(total_trades)}")

        return backtest_results

    def run_full_pipeline(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        total_timesteps: int = 100000,
        env_config: Optional[Dict] = None,
        hyperparameters: Optional[Dict] = None,
        model_save_path: str = './rl_models/trained_agent',
        backtest_save_path: str = './rl_results/backtest.csv'
    ) -> Dict:
        """
        Run complete pipeline from data loading to backtesting.

        Args:
            symbol: Symbol to trade
            timeframe: Timeframe
            total_timesteps: Training timesteps
            env_config: Environment configuration
            hyperparameters: Agent hyperparameters
            model_save_path: Where to save trained model
            backtest_save_path: Where to save backtest results

        Returns:
            Dictionary with all results
        """
        # Create directories
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(backtest_save_path).parent.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        self.load_data(symbol=symbol, timeframe=timeframe)
        self.validate_patterns()
        self.calculate_features()
        train_env, test_env = self.prepare_environments(env_config)

        # Train
        agent = self.train_agent(
            total_timesteps=total_timesteps,
            hyperparameters=hyperparameters,
            save_path=model_save_path
        )

        # Evaluate
        eval_results = self.evaluate_agent(n_episodes=10)

        # Backtest
        backtest_results = self.backtest(save_path=backtest_save_path)

        # Compile results
        results = {
            'evaluation': eval_results,
            'backtest': backtest_results,
            'probabilities': self.probabilities,
            'agent': agent,
            'train_env': train_env,
            'test_env': test_env
        }

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        return results

    def close(self):
        """Close database connection."""
        self.db.close()
