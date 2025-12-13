"""
Optuna Hyperparameter Tuning for DQN

This script uses Optuna to systematically search for optimal hyperparameters
for DQN on the RSA environment.

TUNING STRATEGY:
---------------
- Bayesian optimization with TPE (Tree-structured Parzen Estimator)
- Minimize blocking rate objective
- Parallel trials support for faster optimization
- Pruning of unpromising trials to save computation

HYPERPARAMETERS TUNED:
---------------------
- Learning rate (log scale)
- Network architecture (width and depth)
- Buffer size
- Learning starts
- Batch size
- Target update interval
- Exploration schedule (initial/final epsilon, fraction)
- Tau (soft update coefficient)
- Gamma (discount factor)

OUTPUTS:
-------
- Best hyperparameters: JSON file with optimal configuration
- Study database: SQLite database for resuming/analyzing
- Visualization: Plots showing optimization history
"""

import os
import glob
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from rsaenv import RSAEnv
import json
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')


class TrialEvalCallback(BaseCallback):
    """
    Callback for Optuna trial evaluation.
    
    Reports intermediate values to Optuna for pruning and tracks
    final performance metrics.
    """
    def __init__(self, trial, eval_freq=5, n_eval_episodes=5):
        super().__init__()
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_idx = 0
        self.is_pruned = False
        
        # Track metrics
        self.episode_rewards = []
        self.episode_blocking_rates = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Track episode metrics
        self.current_episode_reward += self.locals['rewards'][0]
        
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            blocking_rate = info.get('blocking_rate', 0.0)
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_blocking_rates.append(blocking_rate)
            self.current_episode_reward = 0
            
            # Report intermediate value every eval_freq episodes
            if len(self.episode_blocking_rates) % self.n_eval_episodes == 0:
                mean_blocking = np.mean(self.episode_blocking_rates[-self.n_eval_episodes:])
                self.trial.report(mean_blocking, self.eval_idx)
                self.eval_idx += 1
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
        
        return True
    
    def get_final_metrics(self):
        """Return final performance metrics"""
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'mean_blocking_rate': np.mean(self.episode_blocking_rates),
            'std_blocking_rate': np.std(self.episode_blocking_rates)
        }


def load_training_requests(data_dir='data/train', max_files=None):
    """Load training request files"""
    files = sorted(glob.glob(os.path.join(data_dir, 'requests-*.csv')))
    if max_files:
        files = files[:max_files]
    return files


def create_env(capacity, request_files):
    """Create wrapped environment"""
    class MultiFileEnv(RSAEnv):
        def __init__(self, capacity, files):
            self.files = files
            self.file_idx = 0
            super().__init__(capacity=capacity, request_file=self.files[self.file_idx])
        
        def reset(self, seed=None, options=None):
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.requests = self._load_requests(self.files[self.file_idx])
            return super().reset(seed=seed, options=options)
    
    env = MultiFileEnv(capacity, request_files)
    env = Monitor(env)
    return env


def objective(trial, capacity, request_files, n_training_episodes=200):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        capacity: Link capacity
        request_files: Training data files
        n_training_episodes: Number of episodes for quick evaluation
        
    Returns:
        Mean blocking rate (lower is better)
    """
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical('buffer_size', [50000, 100000, 200000, 500000])
    learning_starts = trial.suggest_int('learning_starts', 500, 5000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    tau = trial.suggest_float('tau', 0.001, 0.02)
    gamma = trial.suggest_float('gamma', 0.95, 0.9999)
    train_freq = trial.suggest_categorical('train_freq', [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 2, 4])
    target_update_interval = trial.suggest_int('target_update_interval', 500, 5000)
    
    # Exploration schedule
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    exploration_initial_eps = trial.suggest_float('exploration_initial_eps', 0.9, 1.0)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
    
    # Network architecture
    net_arch_size = trial.suggest_categorical('net_arch_size', [64, 128, 256, 512])
    net_arch_depth = trial.suggest_int('net_arch_depth', 2, 4)
    net_arch = [net_arch_size] * net_arch_depth
    
    # Activation function
    activation_fn_name = trial.suggest_categorical('activation_fn', ['tanh', 'relu'])
    activation_fn = nn.Tanh if activation_fn_name == 'tanh' else nn.ReLU
    
    # Create environment
    env = create_env(capacity, request_files[:n_training_episodes])
    
    # Create callback
    eval_callback = TrialEvalCallback(trial, eval_freq=5, n_eval_episodes=5)
    
    try:
        # Create model with suggested hyperparameters
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=activation_fn
            ),
            verbose=0
        )
        
        # Train
        total_timesteps = n_training_episodes * 100
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False
        )
        
        # Check if pruned
        if eval_callback.is_pruned:
            raise optuna.TrialPruned()
        
        # Get final metrics
        metrics = eval_callback.get_final_metrics()
        mean_blocking = float(metrics['mean_blocking_rate'])
        
        # Store additional metrics as user attributes (convert to native Python types)
        trial.set_user_attr('mean_reward', float(metrics['mean_reward']))
        trial.set_user_attr('std_blocking_rate', float(metrics['std_blocking_rate']))
        
        return mean_blocking
        
    except optuna.TrialPruned:
        # Trial was pruned by the pruner - this is normal, just re-raise
        raise
    except Exception as e:
        # Actual error occurred - log it and mark as failed
        print(f"Trial failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.TrialPruned()
    finally:
        env.close()


def optimize_hyperparameters(capacity, n_trials=50, n_training_episodes=200):
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        capacity: Link capacity
        n_trials: Number of trials to run
        n_training_episodes: Episodes per trial (keep small for speed)
        
    Returns:
        Study object with results
    """
    print(f"\n{'='*70}")
    print(f"Optuna Hyperparameter Optimization - DQN Capacity {capacity}")
    print(f"{'='*70}\n")
    
    # Load training data
    request_files = load_training_requests()
    print(f"Loaded {len(request_files)} training files")
    print(f"Using {n_training_episodes} episodes per trial for quick evaluation\n")
    
    # Create study
    study_name = f"dqn_capacity_{capacity}"
    storage_name = f"sqlite:///optuna_studies/dqn_study_capacity_{capacity}.db"
    
    os.makedirs('optuna_studies', exist_ok=True)
    
    # Sampler and pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=5)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',  # Minimize blocking rate
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Optimize
    print(f"Starting optimization with {n_trials} trials...")
    print("(Pruning enabled: unpromising trials will be stopped early)\n")
    
    study.optimize(
        lambda trial: objective(trial, capacity, request_files, n_training_episodes),
        n_trials=n_trials,
        n_jobs=-1,  # Set to -1 for parallel trials (requires more memory)
        show_progress_bar=True
    )
    
    return study


def analyze_and_visualize_study(study, capacity):
    """
    Analyze optimization results and create visualizations.
    
    Args:
        study: Completed Optuna study
        capacity: Link capacity
    """
    print(f"\n{'='*70}")
    print(f"Optimization Results - DQN Capacity {capacity}")
    print(f"{'='*70}\n")
    
    # Best trial
    best_trial = study.best_trial
    print(f"Best Trial: #{best_trial.number}")
    print(f"  Blocking Rate: {best_trial.value:.4f}")
    print(f"  Mean Reward: {best_trial.user_attrs.get('mean_reward', 'N/A')}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters
    os.makedirs('results', exist_ok=True)
    best_params_file = f'results/best_dqn_hyperparameters_capacity_{capacity}.json'
    
    # Convert all values to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_trial_number': int(best_trial.number),
            'best_blocking_rate': float(best_trial.value),
            'best_params': convert_to_native(best_trial.params),
            'user_attrs': convert_to_native(best_trial.user_attrs)
        }, f, indent=2)
    print(f"\nBest hyperparameters saved to {best_params_file}")
    
    # Visualizations
    os.makedirs('plots', exist_ok=True)
    
    # 1. Optimization history
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(f'plots/dqn_optuna_history_capacity_{capacity}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nOptimization visualizations saved to 'plots/' directory")
    
    # Statistics
    print(f"\nStudy Statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")


def train_final_model_with_best_params(capacity, best_params, num_episodes=1000):
    """
    Train final model with best hyperparameters from Optuna.
    
    Args:
        capacity: Link capacity
        best_params: Best hyperparameters from Optuna study
        num_episodes: Number of episodes for full training
    """
    print(f"\n{'='*70}")
    print(f"Training Final DQN Model with Best Hyperparameters - Capacity {capacity}")
    print(f"{'='*70}\n")
    
    # Load training data
    request_files = load_training_requests(max_files=num_episodes)
    print(f"Loaded {len(request_files)} training files")
    
    # Create environment
    env = create_env(capacity, request_files)
    
    # Extract network architecture
    net_arch_size = best_params['net_arch_size']
    net_arch_depth = best_params['net_arch_depth']
    net_arch = [net_arch_size] * net_arch_depth
    
    activation_fn_name = best_params['activation_fn']
    activation_fn = nn.Tanh if activation_fn_name == 'tanh' else nn.ReLU
    
    # Create model with best hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=best_params['learning_rate'],
        buffer_size=best_params['buffer_size'],
        learning_starts=best_params['learning_starts'],
        batch_size=best_params['batch_size'],
        tau=best_params['tau'],
        gamma=best_params['gamma'],
        train_freq=best_params['train_freq'],
        gradient_steps=best_params['gradient_steps'],
        target_update_interval=best_params['target_update_interval'],
        exploration_fraction=best_params['exploration_fraction'],
        exploration_initial_eps=best_params['exploration_initial_eps'],
        exploration_final_eps=best_params['exploration_final_eps'],
        policy_kwargs=dict(
            net_arch=net_arch,
            activation_fn=activation_fn
        ),
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/dqn_optimized_capacity_{capacity}/"
    )
    
    # Train
    total_timesteps = num_episodes * 100
    print(f"\nTraining for {total_timesteps} timesteps (~{num_episodes} episodes)...")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # Save model
    save_name = f'models/dqn_optimized_capacity_{capacity}'
    model.save(save_name)
    print(f"\nOptimized model saved to {save_name}.zip")
    
    return model


def main():
    """Main optimization workflow"""
    
    # Configuration
    capacities = [20, 10]
    n_trials = 50  # Number of hyperparameter configurations to try
    n_training_episodes_per_trial = 200  # Quick evaluation per trial
    n_final_training_episodes = 1000  # Full training with best params
    
    print("\n" + "="*70)
    print("DQN Hyperparameter Optimization with Optuna")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Trials per capacity: {n_trials}")
    print(f"  Episodes per trial: {n_training_episodes_per_trial} (quick evaluation)")
    print(f"  Final training episodes: {n_final_training_episodes} (full training)")
    print(f"\nOptimization will take approximately:")
    print(f"  {n_trials * n_training_episodes_per_trial * len(capacities) / 60:.0f} minutes for all trials")
    print("="*70)
    
    for capacity in capacities:
        # Run optimization
        study = optimize_hyperparameters(
            capacity=capacity,
            n_trials=n_trials,
            n_training_episodes=n_training_episodes_per_trial
        )
        
        # Analyze results
        analyze_and_visualize_study(study, capacity)
        
        # Train final model with best hyperparameters
        train_final = input(f"\nTrain final model with best hyperparameters for capacity {capacity}? (y/n): ")
        if train_final.lower() == 'y':
            train_final_model_with_best_params(
                capacity=capacity,
                best_params=study.best_params,
                num_episodes=n_final_training_episodes
            )
    
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    print("\nResults saved in:")
    print("  - results/best_dqn_hyperparameters_capacity_*.json")
    print("  - plots/dqn_optuna_*.png")
    print("  - optuna_studies/dqn_study_*.db")
    print("\nOptimized models (if trained):")
    print("  - models/dqn_optimized_capacity_*.zip")


if __name__ == '__main__':
    main()
