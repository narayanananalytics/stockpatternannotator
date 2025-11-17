"""
Reinforcement Learning Agent for Pattern-Based Trading.

Uses PPO (Proximal Policy Optimization) algorithm from stable-baselines3.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    import warnings
    warnings.warn("stable-baselines3 not available. Install it to use RL features.")

from .rl_environment import PatternTradingEnv


class TradingCallback(BaseCallback):
    """Custom callback for tracking trading performance during training."""

    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super(TradingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_trades = []
        self.episode_win_rates = []

    def _on_step(self) -> bool:
        # Get info from last step
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]

            # Check if episode ended
            if self.locals.get('dones', [False])[0]:
                if hasattr(self.training_env.envs[0], 'get_metrics'):
                    metrics = self.training_env.envs[0].get_metrics()

                    self.episode_returns.append(metrics['total_return'])
                    self.episode_trades.append(metrics['total_trades'])
                    self.episode_win_rates.append(metrics['win_rate'])

                    if self.num_timesteps % self.log_freq == 0:
                        if self.episode_returns:
                            avg_return = np.mean(self.episode_returns[-10:])
                            avg_trades = np.mean(self.episode_trades[-10:])
                            avg_win_rate = np.mean(self.episode_win_rates[-10:])

                            print(f"\nTimestep: {self.num_timesteps}")
                            print(f"Avg Return (last 10): {avg_return:.4f}")
                            print(f"Avg Trades (last 10): {avg_trades:.1f}")
                            print(f"Avg Win Rate (last 10): {avg_win_rate:.2%}")

        return True


class RLTradingAgent:
    """RL Trading Agent using PPO algorithm."""

    def __init__(
        self,
        env: PatternTradingEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        device: str = 'auto',
        verbose: int = 1
    ):
        """
        Initialize RL Trading Agent.

        Args:
            env: Trading environment
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epoch when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for GAE
            clip_range: Clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            use_sde: Whether to use State Dependent Exploration
            device: Device (cpu, cuda, auto)
            verbose: Verbosity level
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

        self.env = env
        self.verbose = verbose

        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            device=device,
            verbose=verbose,
            tensorboard_log="./rl_tensorboard/"
        )

    def train(
        self,
        total_timesteps: int = 100000,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10,
        eval_env: Optional[PatternTradingEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5
    ) -> 'RLTradingAgent':
        """
        Train the agent.

        Args:
            total_timesteps: Total number of timesteps to train
            callback: Custom callback
            log_interval: Log interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of episodes for evaluation

        Returns:
            Self
        """
        callbacks = []

        # Add trading callback
        trading_callback = TradingCallback(verbose=self.verbose)
        callbacks.append(trading_callback)

        # Add evaluation callback if eval_env provided
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./rl_models/best_model',
                log_path='./rl_logs/',
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        # Add custom callback if provided
        if callback is not None:
            callbacks.append(callback)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval
        )

        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Any]:
        """
        Predict action for given observation.

        Args:
            observation: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            action, state
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def evaluate(
        self,
        env: PatternTradingEnv,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the agent.

        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes
            deterministic: Whether to use deterministic policy
            render: Whether to render

        Returns:
            Dictionary with evaluation metrics
        """
        episode_returns = []
        episode_trades = []
        episode_win_rates = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_length = 0

            while not (done or truncated):
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_length += 1

                if render:
                    env.render()

            # Get episode metrics
            metrics = env.get_metrics()
            episode_returns.append(metrics['total_return'])
            episode_trades.append(metrics['total_trades'])
            episode_win_rates.append(metrics['win_rate'])
            episode_lengths.append(episode_length)

        return {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_trades': np.mean(episode_trades),
            'mean_win_rate': np.mean(episode_win_rates),
            'mean_length': np.mean(episode_lengths),
            'min_return': np.min(episode_returns),
            'max_return': np.max(episode_returns)
        }

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
        if self.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, env: PatternTradingEnv) -> 'RLTradingAgent':
        """Load a trained model."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required")

        agent = cls(env, verbose=0)
        agent.model = PPO.load(path, env=env)
        return agent

    def backtest(
        self,
        env: PatternTradingEnv,
        deterministic: bool = True
    ) -> pd.DataFrame:
        """
        Run backtest and return detailed results.

        Args:
            env: Environment to backtest on
            deterministic: Whether to use deterministic policy

        Returns:
            DataFrame with backtest results
        """
        results = []
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = self.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)

            # Record state
            results.append({
                'step': env.current_step,
                'timestamp': env.df.index[env.current_step] if env.current_step < len(env.df) else None,
                'price': env.df.iloc[env.current_step]['close'] if env.current_step < len(env.df) else None,
                'action': action,
                'position': info['position'],
                'balance': info['balance'],
                'current_pnl': info['current_pnl'],
                'total_value': info['total_value'],
                'reward': reward,
                'holding_period': info['holding_period']
            })

        return pd.DataFrame(results)

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities for given observation."""
        # Get action distribution
        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()
        return probs[0]


def create_training_agent(
    env: PatternTradingEnv,
    hyperparameters: Optional[Dict] = None
) -> RLTradingAgent:
    """
    Create a trading agent with default or custom hyperparameters.

    Args:
        env: Trading environment
        hyperparameters: Optional hyperparameters to override defaults

    Returns:
        RLTradingAgent instance
    """
    default_params = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'device': 'auto'
    }

    if hyperparameters:
        default_params.update(hyperparameters)

    return RLTradingAgent(env, **default_params)


def hyperparameter_search(
    env_train: PatternTradingEnv,
    env_val: PatternTradingEnv,
    param_grid: Dict[str, list],
    n_trials: int = 10,
    timesteps_per_trial: int = 50000
) -> Dict:
    """
    Simple hyperparameter search.

    Args:
        env_train: Training environment
        env_val: Validation environment
        param_grid: Grid of parameters to search
        n_trials: Number of random trials
        timesteps_per_trial: Timesteps to train each trial

    Returns:
        Best parameters and results
    """
    import random

    best_score = -np.inf
    best_params = None
    results = []

    for trial in range(n_trials):
        # Sample random parameters
        params = {}
        for key, values in param_grid.items():
            params[key] = random.choice(values)

        print(f"\nTrial {trial + 1}/{n_trials}")
        print(f"Parameters: {params}")

        # Create and train agent
        agent = create_training_agent(env_train, params)
        agent.train(total_timesteps=timesteps_per_trial, log_interval=10)

        # Evaluate
        eval_results = agent.evaluate(env_val, n_episodes=5)
        score = eval_results['mean_return']

        results.append({
            'params': params.copy(),
            'score': score,
            'results': eval_results
        })

        print(f"Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params.copy()
            print(f"New best score: {best_score:.4f}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }
