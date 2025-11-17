"""
Reinforcement Learning Environment for Pattern-Based Trading.

This module implements a custom gym environment that integrates pattern probabilities
with technical indicators for RL-based trading.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PatternTradingEnv(gym.Env):
    """
    Custom trading environment that uses pattern probabilities as signals.

    State Space:
        - Pattern probabilities (bullish/bearish for each detected pattern)
        - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - Current P&L (unrealized)
        - Max P&L achieved in current position
        - Pattern confidence scores
        - Current position (long/short/neutral)

    Action Space:
        - 0: Hold
        - 1: Buy (go long)
        - 2: Sell (go short or close long)

    Reward:
        Custom reward combining:
        - Profit/loss from trades
        - Penalty for holding too long
        - Penalty for excessive trading (transaction costs)
        - Bonus for following high-probability patterns
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        pattern_probabilities: pd.DataFrame,
        technical_indicators: pd.DataFrame,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,  # 0.1%
        max_position_size: float = 1.0,   # 100% of balance
        max_drawdown: float = 0.2,        # 20%
        holding_penalty_weight: float = 0.001,
        pattern_bonus_weight: float = 0.1,
        pattern_prob_threshold: float = 0.5,
        max_holding_period: int = 20,
        signal_weights: Optional[Dict[str, float]] = None,
        allow_short: bool = True
    ):
        """
        Initialize the trading environment.

        Args:
            df: OHLC data with datetime index
            pattern_probabilities: Pattern probability data
            technical_indicators: Technical indicator data
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            max_position_size: Maximum position size as fraction of balance
            max_drawdown: Maximum drawdown before stopping (0.2 = 20%)
            holding_penalty_weight: Weight for holding penalty in reward
            pattern_bonus_weight: Weight for pattern bonus in reward
            pattern_prob_threshold: Minimum probability to get bonus
            max_holding_period: Maximum candles to hold before penalty increases
            signal_weights: Weights for different signals (hyperparameter)
            allow_short: Whether to allow short positions
        """
        super(PatternTradingEnv, self).__init__()

        # Data
        self.df = df.copy()
        self.pattern_probs = pattern_probabilities.copy()
        self.tech_indicators = technical_indicators.copy()

        # Align all data to same index
        self._align_data()

        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.allow_short = allow_short

        # Reward parameters
        self.holding_penalty_weight = holding_penalty_weight
        self.pattern_bonus_weight = pattern_bonus_weight
        self.pattern_prob_threshold = pattern_prob_threshold
        self.max_holding_period = max_holding_period

        # Signal weights (hyperparameters)
        if signal_weights is None:
            self.signal_weights = {
                'pattern_prob': 0.4,
                'technical': 0.3,
                'momentum': 0.3
            }
        else:
            self.signal_weights = signal_weights

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.position_size = 0
        self.entry_price = 0
        self.max_balance = initial_balance
        self.current_pnl = 0
        self.max_pnl_in_position = 0
        self.holding_period = 0
        self.entry_pattern_prob = 0
        self.total_trades = 0
        self.winning_trades = 0

        # Action space: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)

        # Observation space
        self._setup_observation_space()

    def _align_data(self):
        """Align all dataframes to same index."""
        # Find common index
        common_index = self.df.index.intersection(
            self.pattern_probs.index
        ).intersection(
            self.tech_indicators.index
        )

        self.df = self.df.loc[common_index]
        self.pattern_probs = self.pattern_probs.loc[common_index]
        self.tech_indicators = self.tech_indicators.loc[common_index]

        self.max_steps = len(self.df) - 1

    def _setup_observation_space(self):
        """Setup the observation space dimensions."""
        # Count features
        n_pattern_features = len([col for col in self.pattern_probs.columns
                                  if 'prob' in col.lower()])
        n_tech_features = len(self.tech_indicators.columns)

        # State features:
        # - Pattern probabilities
        # - Technical indicators
        # - Current position (1)
        # - Current P&L (1)
        # - Max P&L in position (1)
        # - Holding period (1)
        # - Pattern confidence (1)
        # - Normalized price (1)

        n_features = (
            n_pattern_features +
            n_tech_features +
            6  # position, pnl, max_pnl, holding, confidence, price
        )

        # Observation space is continuous
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.max_balance = self.initial_balance
        self.current_pnl = 0
        self.max_pnl_in_position = 0
        self.holding_period = 0
        self.entry_pattern_prob = 0
        self.total_trades = 0
        self.winning_trades = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)."""
        obs = []

        # Pattern probabilities
        pattern_row = self.pattern_probs.iloc[self.current_step]
        for col in self.pattern_probs.columns:
            if 'prob' in col.lower():
                obs.append(pattern_row[col] / 100.0)  # Normalize to 0-1

        # Technical indicators (assume already normalized)
        tech_row = self.tech_indicators.iloc[self.current_step]
        obs.extend(tech_row.values)

        # Trading state
        obs.append(float(self.position))  # -1, 0, or 1
        obs.append(self.current_pnl / self.initial_balance)  # Normalized P&L
        obs.append(self.max_pnl_in_position / self.initial_balance)
        obs.append(self.holding_period / self.max_holding_period)  # Normalized

        # Pattern confidence (if available)
        confidence = pattern_row.get('confidence', 0.5)
        obs.append(confidence)

        # Normalized price
        current_price = self.df.iloc[self.current_step]['close']
        if self.current_step > 0:
            prev_price = self.df.iloc[self.current_step - 1]['close']
            price_change = (current_price - prev_price) / prev_price
        else:
            price_change = 0
        obs.append(price_change)

        return np.array(obs, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.df.iloc[self.current_step]['close']

        # Execute action
        reward = 0
        trade_executed = False

        if action == 1:  # Buy
            reward = self._execute_buy(current_price)
            trade_executed = True

        elif action == 2:  # Sell
            reward = self._execute_sell(current_price)
            trade_executed = True

        else:  # Hold
            reward = self._execute_hold(current_price)

        # Update holding period
        if self.position != 0:
            self.holding_period += 1

        # Update current P&L
        if self.position != 0:
            self.current_pnl = (current_price - self.entry_price) * self.position * self.position_size
            self.max_pnl_in_position = max(self.max_pnl_in_position, self.current_pnl)
        else:
            self.current_pnl = 0
            self.max_pnl_in_position = 0

        # Update max balance
        total_value = self.balance + self.current_pnl
        self.max_balance = max(self.max_balance, total_value)

        # Move to next step
        self.current_step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        # Check max drawdown
        drawdown = (self.max_balance - total_value) / self.max_balance
        if drawdown > self.max_drawdown:
            terminated = True
            reward -= 10  # Large penalty for exceeding drawdown

        # Get next observation
        obs = self._get_observation() if not (terminated or truncated) else self._get_observation()

        # Info dict
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_pnl': self.current_pnl,
            'total_value': total_value,
            'holding_period': self.holding_period,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'trade_executed': trade_executed
        }

        return obs, reward, terminated, truncated, info

    def _execute_buy(self, current_price: float) -> float:
        """Execute buy action."""
        reward = 0

        # Close short position if exists
        if self.position == -1:
            pnl = (self.entry_price - current_price) * self.position_size
            self.balance += pnl
            reward += pnl / self.initial_balance  # Normalized reward

            # Transaction cost
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            reward -= cost / self.initial_balance

            if pnl > 0:
                self.winning_trades += 1
            self.total_trades += 1

            self.position = 0
            self.position_size = 0

        # Open long position or do nothing if already long
        if self.position == 0:
            # Calculate position size
            self.position_size = (self.balance * self.max_position_size) / current_price
            self.position = 1
            self.entry_price = current_price
            self.holding_period = 0
            self.max_pnl_in_position = 0

            # Get pattern probability for bonus
            pattern_row = self.pattern_probs.iloc[self.current_step]
            bullish_cols = [col for col in pattern_row.index if 'bullish' in col.lower() and 'prob' in col.lower()]
            if bullish_cols:
                self.entry_pattern_prob = pattern_row[bullish_cols].mean() / 100.0
            else:
                self.entry_pattern_prob = 0.5

            # Transaction cost
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            reward -= cost / self.initial_balance

            # Bonus for high probability patterns
            if self.entry_pattern_prob > self.pattern_prob_threshold:
                bonus = (self.entry_pattern_prob - self.pattern_prob_threshold) * self.pattern_bonus_weight
                reward += bonus

        return reward

    def _execute_sell(self, current_price: float) -> float:
        """Execute sell action."""
        reward = 0

        # Close long position if exists
        if self.position == 1:
            pnl = (current_price - self.entry_price) * self.position_size
            self.balance += pnl
            reward += pnl / self.initial_balance

            # Transaction cost
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            reward -= cost / self.initial_balance

            if pnl > 0:
                self.winning_trades += 1
            self.total_trades += 1

            self.position = 0
            self.position_size = 0

        # Open short position if allowed
        if self.position == 0 and self.allow_short:
            self.position_size = (self.balance * self.max_position_size) / current_price
            self.position = -1
            self.entry_price = current_price
            self.holding_period = 0
            self.max_pnl_in_position = 0

            # Get pattern probability
            pattern_row = self.pattern_probs.iloc[self.current_step]
            bearish_cols = [col for col in pattern_row.index if 'bearish' in col.lower() and 'prob' in col.lower()]
            if bearish_cols:
                self.entry_pattern_prob = pattern_row[bearish_cols].mean() / 100.0
            else:
                self.entry_pattern_prob = 0.5

            # Transaction cost
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            reward -= cost / self.initial_balance

            # Bonus for high probability patterns
            if self.entry_pattern_prob > self.pattern_prob_threshold:
                bonus = (self.entry_pattern_prob - self.pattern_prob_threshold) * self.pattern_bonus_weight
                reward += bonus

        return reward

    def _execute_hold(self, current_price: float) -> float:
        """Execute hold action."""
        reward = 0

        # Penalty for holding too long
        if self.position != 0 and self.holding_period > self.max_holding_period:
            penalty = (self.holding_period - self.max_holding_period) * self.holding_penalty_weight
            reward -= penalty

        # Check for automatic exit based on max P&L drop
        if self.position != 0 and self.max_pnl_in_position > 0:
            pnl_drop = self.max_pnl_in_position - self.current_pnl
            # If we've given back too much profit, penalize
            if pnl_drop > self.max_pnl_in_position * 0.5:  # Given back >50% of max gain
                reward -= 0.1

        return reward

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            current_price = self.df.iloc[self.current_step]['close']
            total_value = self.balance + self.current_pnl

            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Position: {self.position} ({'Long' if self.position == 1 else 'Short' if self.position == -1 else 'Neutral'})")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Current P&L: ${self.current_pnl:.2f}")
            print(f"Total Value: ${total_value:.2f}")
            print(f"Return: {((total_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
            print(f"Total Trades: {self.total_trades}")
            if self.total_trades > 0:
                print(f"Win Rate: {(self.winning_trades / self.total_trades * 100):.2f}%")
            print("-" * 50)

    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.balance + self.current_pnl

    def get_metrics(self) -> Dict:
        """Get trading metrics."""
        total_value = self.get_portfolio_value()
        total_return = (total_value - self.initial_balance) / self.initial_balance

        return {
            'total_return': total_return,
            'total_value': total_value,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'max_balance': self.max_balance,
            'max_drawdown': (self.max_balance - total_value) / self.max_balance if self.max_balance > 0 else 0
        }
