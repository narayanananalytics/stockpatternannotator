"""
Feature engineering for RL trading environment.

Calculates technical indicators and prepares pattern probability signals.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Calculate technical indicators for RL trading."""

    def __init__(self):
        """Initialize feature engineer."""
        pass

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: OHLC DataFrame
            indicators: List of indicators to calculate (None = all)

        Returns:
            DataFrame with technical indicators
        """
        if indicators is None:
            indicators = [
                'rsi', 'macd', 'bollinger', 'sma', 'ema',
                'atr', 'obv', 'momentum', 'roc', 'stochastic'
            ]

        features = pd.DataFrame(index=df.index)

        if 'rsi' in indicators:
            features = features.join(self.calculate_rsi(df))

        if 'macd' in indicators:
            features = features.join(self.calculate_macd(df))

        if 'bollinger' in indicators:
            features = features.join(self.calculate_bollinger_bands(df))

        if 'sma' in indicators:
            features = features.join(self.calculate_sma(df))

        if 'ema' in indicators:
            features = features.join(self.calculate_ema(df))

        if 'atr' in indicators:
            features = features.join(self.calculate_atr(df))

        if 'obv' in indicators:
            features = features.join(self.calculate_obv(df))

        if 'momentum' in indicators:
            features = features.join(self.calculate_momentum(df))

        if 'roc' in indicators:
            features = features.join(self.calculate_roc(df))

        if 'stochastic' in indicators:
            features = features.join(self.calculate_stochastic(df))

        # Normalize features
        features = self.normalize_features(features)

        return features

    def calculate_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """Calculate RSI indicator."""
        close = df['close']
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return pd.DataFrame({'rsi': rsi}, index=df.index)

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        close = df['close']

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist
        }, index=df.index)

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        close = df['close']

        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        # Calculate position relative to bands
        bb_position = (close - lower_band) / (upper_band - lower_band)

        return pd.DataFrame({
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_position': bb_position
        }, index=df.index)

    def calculate_sma(
        self,
        df: pd.DataFrame,
        periods: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        close = df['close']
        result = pd.DataFrame(index=df.index)

        for period in periods:
            sma = close.rolling(window=period).mean()
            # Calculate position relative to SMA
            result[f'sma_{period}_ratio'] = close / sma - 1

        return result

    def calculate_ema(
        self,
        df: pd.DataFrame,
        periods: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        close = df['close']
        result = pd.DataFrame(index=df.index)

        for period in periods:
            ema = close.ewm(span=period, adjust=False).mean()
            result[f'ema_{period}_ratio'] = close / ema - 1

        return result

    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Normalize by price
        atr_pct = atr / close

        return pd.DataFrame({'atr': atr_pct}, index=df.index)

    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

        # Normalize
        obv_normalized = (obv - obv.rolling(window=50).mean()) / obv.rolling(window=50).std()

        return pd.DataFrame({'obv': obv_normalized}, index=df.index)

    def calculate_momentum(
        self,
        df: pd.DataFrame,
        period: int = 10
    ) -> pd.DataFrame:
        """Calculate momentum."""
        close = df['close']
        momentum = close.pct_change(period)

        return pd.DataFrame({'momentum': momentum}, index=df.index)

    def calculate_roc(
        self,
        df: pd.DataFrame,
        period: int = 12
    ) -> pd.DataFrame:
        """Calculate Rate of Change."""
        close = df['close']
        roc = ((close - close.shift(period)) / close.shift(period)) * 100

        return pd.DataFrame({'roc': roc}, index=df.index)

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        high = df['high']
        low = df['low']
        close = df['close']

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return pd.DataFrame({
            'stoch_k': k,
            'stoch_d': d
        }, index=df.index)

    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [-1, 1] range."""
        normalized = features.copy()

        for col in features.columns:
            # Skip if already in 0-100 range (like RSI)
            if col in ['rsi', 'stoch_k', 'stoch_d']:
                normalized[col] = (features[col] - 50) / 50  # Center around 0
            # Skip if already normalized
            elif 'ratio' in col or 'position' in col or 'obv' in col:
                normalized[col] = features[col].clip(-3, 3) / 3  # Clip outliers
            else:
                # Standard normalization
                mean = features[col].rolling(window=50).mean()
                std = features[col].rolling(window=50).std()
                normalized[col] = ((features[col] - mean) / std).clip(-3, 3) / 3

        # Fill NaN with 0
        normalized = normalized.fillna(0)

        return normalized

    def prepare_pattern_probabilities(
        self,
        validation_results: pd.DataFrame,
        probabilities: pd.DataFrame,
        forecast_horizon: int = 5
    ) -> pd.DataFrame:
        """
        Prepare pattern probabilities for the environment.

        Args:
            validation_results: Validation results from PatternValidator
            probabilities: Pattern probabilities
            forecast_horizon: Which horizon to use

        Returns:
            DataFrame with pattern probabilities indexed by timestamp
        """
        if validation_results.empty:
            return pd.DataFrame()

        # Get relevant columns for the forecast horizon
        prob_data = []

        for timestamp in validation_results['pattern_timestamp'].unique():
            patterns_at_time = validation_results[
                validation_results['pattern_timestamp'] == timestamp
            ]

            row = {'timestamp': timestamp}

            # Get probabilities for each pattern at this timestamp
            for _, pattern_row in patterns_at_time.iterrows():
                pattern_name = pattern_row['pattern_name']

                # Find probability for this pattern
                pattern_prob = probabilities[
                    probabilities['pattern_name'] == pattern_name
                ]

                if not pattern_prob.empty:
                    bullish_col = f'h{forecast_horizon}_bullish_prob'
                    bearish_col = f'h{forecast_horizon}_bearish_prob'
                    confidence_col = 'confidence' if 'confidence' in pattern_row else None

                    if bullish_col in pattern_prob.columns:
                        row[f'{pattern_name}_bullish_prob'] = pattern_prob[bullish_col].iloc[0]
                    if bearish_col in pattern_prob.columns:
                        row[f'{pattern_name}_bearish_prob'] = pattern_prob[bearish_col].iloc[0]
                    if confidence_col:
                        row['confidence'] = pattern_row[confidence_col]

            prob_data.append(row)

        prob_df = pd.DataFrame(prob_data)

        if not prob_df.empty:
            prob_df = prob_df.set_index('timestamp')

        return prob_df

    def calculate_weighted_signal(
        self,
        pattern_probs: pd.DataFrame,
        tech_indicators: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate weighted signal from multiple sources.

        Args:
            pattern_probs: Pattern probabilities
            tech_indicators: Technical indicators
            weights: Weights for each signal type

        Returns:
            DataFrame with weighted signals
        """
        signals = pd.DataFrame(index=pattern_probs.index)

        # Pattern signal (bullish - bearish)
        bullish_cols = [col for col in pattern_probs.columns if 'bullish' in col.lower()]
        bearish_cols = [col for col in pattern_probs.columns if 'bearish' in col.lower()]

        if bullish_cols and bearish_cols:
            pattern_signal = (
                pattern_probs[bullish_cols].mean(axis=1) -
                pattern_probs[bearish_cols].mean(axis=1)
            ) / 100.0  # Normalize to -1 to 1
        else:
            pattern_signal = 0

        # Technical signal (from RSI, MACD, etc.)
        tech_signal = 0
        if 'rsi' in tech_indicators.columns:
            # RSI signal: >0.5 = overbought (bearish), <-0.5 = oversold (bullish)
            tech_signal += -tech_indicators['rsi'] * 0.3

        if 'macd_hist' in tech_indicators.columns:
            tech_signal += tech_indicators['macd_hist'] * 0.3

        if 'bb_position' in tech_indicators.columns:
            # BB position: >1 = above upper band (bearish), <0 = below lower (bullish)
            tech_signal += (0.5 - tech_indicators['bb_position']) * 0.4

        # Momentum signal
        momentum_signal = 0
        if 'momentum' in tech_indicators.columns:
            momentum_signal += tech_indicators['momentum'] * 0.5

        if 'roc' in tech_indicators.columns:
            momentum_signal += tech_indicators['roc'] / 100.0 * 0.5

        # Weighted combination
        signals['weighted_signal'] = (
            pattern_signal * weights.get('pattern_prob', 0.4) +
            tech_signal * weights.get('technical', 0.3) +
            momentum_signal * weights.get('momentum', 0.3)
        )

        signals['pattern_signal'] = pattern_signal
        signals['technical_signal'] = tech_signal
        signals['momentum_signal'] = momentum_signal

        return signals
