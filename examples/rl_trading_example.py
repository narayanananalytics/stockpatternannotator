"""
Reinforcement Learning Trading Example.

This script demonstrates the complete RL trading workflow:
1. Load data from database
2. Validate patterns and calculate probabilities
3. Calculate technical indicators
4. Train RL agent using pattern probabilities as signals
5. Evaluate and backtest

GPU Acceleration:
- Automatically detects and optimizes for available GPU
- For 16GB GPU: Uses batch_size=512, n_steps=8192, mixed precision
- For 8-16GB GPU: Uses batch_size=256, n_steps=4096
- Falls back to CPU if no GPU detected

Requirements:
- Run polygon_pipeline_example.py first to populate database
- Install RL dependencies: pip install gymnasium stable-baselines3 torch
- For GPU: Install CUDA-enabled PyTorch (see pytorch.org)
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from stockpatternannotator import RLPipeline
    from stockpatternannotator import RL_AVAILABLE

    if not RL_AVAILABLE:
        print("ERROR: RL dependencies not available.")
        print("Install with: pip install gymnasium stable-baselines3 torch")
        sys.exit(1)

except ImportError as e:
    print(f"ERROR: {e}")
    print("Install required packages: pip install gymnasium stable-baselines3 torch")
    sys.exit(1)


def main():
    print("=" * 70)
    print("Stock Pattern Annotator - RL Trading Example")
    print("=" * 70)
    print()

    # Check for database
    if not os.path.exists('stockpatterns_example.db'):
        print("ERROR: Database not found!")
        print("Please run polygon_pipeline_example.py and pattern_validation_example.py first.")
        print("This will create the database with OHLC data and pattern annotations.")
        return

    print("NOTE: GPU optimization is enabled by default.")
    print("The pipeline will automatically detect your GPU and optimize hyperparameters.")
    print("For your 16GB GPU, expect: batch_size=512, n_steps=8192, mixed precision")
    print()

    # Configuration
    symbol = None  # None = use all symbols in database
    timeframe = '1D'  # Daily data
    total_timesteps = 50000  # Training timesteps (increase for better results)

    # Environment configuration
    env_config = {
        'initial_balance': 10000,
        'transaction_cost': 0.001,  # 0.1% per trade
        'max_position_size': 1.0,    # 100% of balance
        'max_drawdown': 0.2,         # 20% max drawdown
        'holding_penalty_weight': 0.001,
        'pattern_bonus_weight': 0.1,
        'pattern_prob_threshold': 0.55,  # Bonus above 55% probability
        'max_holding_period': 20,
        'allow_short': True,
        'signal_weights': {
            'pattern_prob': 0.4,  # 40% weight to pattern probabilities
            'technical': 0.3,     # 30% weight to technical indicators
            'momentum': 0.3       # 30% weight to momentum
        }
    }

    # Agent hyperparameters (optional - GPU optimization happens automatically)
    # Note: n_steps and batch_size will be auto-optimized for your GPU
    # You can override them here if needed
    hyperparameters = {
        'learning_rate': 3e-4,
        # 'n_steps': 2048,       # Auto-optimized based on GPU memory
        # 'batch_size': 64,      # Auto-optimized based on GPU memory
        'n_epochs': 10,
        'gamma': 0.99,           # Discount factor
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,        # Entropy coefficient (exploration)
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    }

    print("Configuration:")
    print(f"  Symbol: {symbol if symbol else 'All'}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Training timesteps: {total_timesteps:,}")
    print(f"  Initial balance: ${env_config['initial_balance']:,}")
    print(f"  Transaction cost: {env_config['transaction_cost']:.1%}")
    print(f"  Max drawdown: {env_config['max_drawdown']:.0%}")
    print()

    # Create pipeline
    print("=" * 70)
    print("INITIALIZING RL PIPELINE")
    print("=" * 70)
    print()

    # Pipeline will automatically detect GPU and show info
    pipeline = RLPipeline(
        database_url='sqlite:///stockpatterns_example.db',
        forecast_horizon=5,  # Use 5-candle forecast probabilities
        test_size=0.2,       # 20% for testing
        random_state=42,
        show_gpu_info=True   # Shows GPU detection and optimization info
    )

    try:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            symbol=symbol,
            timeframe=timeframe,
            total_timesteps=total_timesteps,
            env_config=env_config,
            hyperparameters=hyperparameters,
            model_save_path='./rl_models/pattern_trading_agent',
            backtest_save_path='./rl_results/backtest.csv'
        )

        # Display results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print()

        eval_results = results['evaluation']
        print("Test Set Evaluation (10 episodes):")
        print(f"  Mean Return: {eval_results['mean_return']:.2%} ± {eval_results['std_return']:.2%}")
        print(f"  Mean Trades: {eval_results['mean_trades']:.1f}")
        print(f"  Mean Win Rate: {eval_results['mean_win_rate']:.2%}")
        print(f"  Best Return: {eval_results['max_return']:.2%}")
        print(f"  Worst Return: {eval_results['min_return']:.2%}")
        print()

        # Pattern probabilities used
        if results['probabilities'] is not None and not results['probabilities'].empty:
            print("Pattern Probabilities Used in Training:")
            print("-" * 70)
            prob_df = results['probabilities'][['pattern_name', 'total_samples', 'h5_bullish_prob', 'h5_bearish_prob', 'h5_win_rate']]
            prob_df = prob_df.dropna(subset=['h5_win_rate'])
            if not prob_df.empty:
                print(prob_df.to_string(index=False))
            print()

        # Backtest summary
        backtest = results['backtest']
        if not backtest.empty:
            final_value = backtest['total_value'].iloc[-1]
            initial_value = env_config['initial_balance']
            total_return = (final_value - initial_value) / initial_value
            total_trades = (backtest['position'].diff().abs() > 0).sum()

            print("Detailed Backtest:")
            print(f"  Initial Balance: ${initial_value:,.2f}")
            print(f"  Final Value: ${final_value:,.2f}")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Total Trades: {total_trades}")
            print()

            print("Backtest results saved to: ./rl_results/backtest.csv")
            print("Model saved to: ./rl_models/pattern_trading_agent.zip")
            print()

        # Actionable insights
        print("=" * 70)
        print("ACTIONABLE INSIGHTS")
        print("=" * 70)
        print()

        if eval_results['mean_return'] > 0:
            print("✓ Agent achieved positive returns on test set")
            if eval_results['mean_win_rate'] > 0.5:
                print("✓ Win rate above 50% - agent learned profitable patterns")
        else:
            print("⚠ Agent did not achieve positive returns")
            print("  Consider:")
            print("  - Training for more timesteps")
            print("  - Adjusting hyperparameters")
            print("  - Using more historical data")
            print("  - Tuning reward function weights")

        print()
        print("Next steps:")
        print("  1. Analyze backtest.csv for detailed trade history")
        print("  2. Visualize results with tensorboard: tensorboard --logdir=./rl_tensorboard")
        print("  3. Fine-tune hyperparameters for better performance")
        print("  4. Test on different time periods or symbols")
        print("  5. Implement paper trading to validate in real-time")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        pipeline.close()

    print()
    print("=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
