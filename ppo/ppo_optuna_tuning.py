"""
Optuna Hyperparameter Tuning for PPO
Uses Optuna & pruning to minimize blocking rate on RSA env.

Tuning Strategy:
- Bayesian optimization with TPE (Tree-structured Parzen Estimator)
- Minimize blocking rate objective
- Parallel trials support for faster optimization
- Pruning of unpromising trials to save computation

Hyperparams:
- Learning rate (log scale)
- Network architecture (width and depth)
- n_steps (rollout length)
- Batch size (minibatch size)
- Gamma (discount factor)
- GAE lambda
- Clip range
- Entropy coefficient
- Value function coefficient
- Number of PPO epochs
- Max gradient norm

Output:
- Best hyperparameters: json file
- Study database: sqlite
- Visualization: plot imgs
"""

import os
import glob
import json
import warnings
import numpy as np
import optuna
import matplotlib.pyplot as plt
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import argparse
import sys
import shutil

# Resolve project root and ensure it is on sys.path so we can import rsaenv
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
from rsaenv import RSAEnv
warnings.filterwarnings("ignore")

class TrialEvalCallback(BaseCallback):
    # Callback for Optuna trial evaluation
    # Reports intermediate vals to Optuna for pruning
    # Tracks final performance metrics

    def __init__(self, trial, eval_freq=5, n_eval_episodes=5):
        super().__init__()
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        # Internal counters
        self.eval_idx = 0
        self.is_pruned = False
        # Metrics
        self.episode_rewards = []
        self.episode_blocking_rates = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Called at every environment step
        # Track episode metrics
        # Accumulate reward for current episode
        self.current_episode_reward += self.locals["rewards"][0]
        
        if self.locals["dones"][0]: # If episode finished
            info = self.locals["infos"][0]
            blocking_rate = info.get("blocking_rate", 0.0)

            # Store metrics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_blocking_rates.append(blocking_rate)
            self.current_episode_reward = 0.0

            # Every eval_freq episodes, report to Optuna
            if len(self.episode_blocking_rates) % self.eval_freq == 0:
                mean_blocking = np.mean(self.episode_blocking_rates[-self.n_eval_episodes :])

                # Report intermediate objective value
                self.trial.report(mean_blocking, self.eval_idx)
                self.eval_idx += 1

                # Prune if Optuna decides this trial is bad
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
        return True

    def get_final_metrics(self): # Return final aggregated metrics
        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "mean_blocking_rate": float(np.mean(self.episode_blocking_rates)),
            "std_blocking_rate": float(np.std(self.episode_blocking_rates)),
        }

def load_training_requests(data_dir=None, max_files=None):
    """Load training request files.

    The default `data_dir` is '<project_root>/data/train'. If a relative `data_dir`
    is passed, it will be resolved relative to the project root. This function
    raises FileNotFoundError with a clear error message when no files are found.
    """
    # default path under the project root
    if data_dir is None: data_dir = os.path.join(project_root, 'data', 'train')
    elif not os.path.isabs(data_dir):
        candidate = os.path.join(project_root, data_dir)
        if os.path.exists(candidate): data_dir = candidate

    files = sorted(glob.glob(os.path.join(data_dir, "requests-*.csv")))
    if not files: raise FileNotFoundError(f"No training files found in '{data_dir}'. Check that 'data/train' exists and is accessible from the project root.")
    if max_files: files = files[:max_files]
    return files


def create_env(capacity, request_files):
    # Create wrapped environment
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
    """Optuna objective function for hyperparameter optimization

    Args:
        trial: Optuna trial object
        capacity: Link capacity
        request_files: Training data files
        n_training_episodes: Number of episodes for quick evaluation
        
    Returns: Mean blocking rate (lower is better)
    """

    # PPO hyperparams
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)

    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.02, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)

    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    # Ensure PPO constraint
    if batch_size > n_steps: raise optuna.TrialPruned()

    # Network Architecture
    net_arch_size = trial.suggest_categorical("net_arch_size", [64, 128, 256, 512])
    net_arch_depth = trial.suggest_int("net_arch_depth", 2, 4)
    net_arch = [net_arch_size] * net_arch_depth

    # Activation function
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = nn.Tanh if activation_fn_name == "tanh" else nn.ReLU

    # Ensure we have at least one training file
    if not request_files: raise ValueError('No training files available for Optuna trials - check data/train')

    # Create callback
    env = create_env(capacity, request_files[:n_training_episodes])
    callback = TrialEvalCallback(trial)

    try:
        # Create model with suggested hyperparameters
        # PPO Model
        # Create per-trial tensorboard path for traceability and debugging
        tb_log_dir = os.path.join(script_dir, f"tensorboard_logs/ppo_optuna_capacity_{capacity}/trial_{trial.number}")
        os.makedirs(tb_log_dir, exist_ok=True)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            max_grad_norm=max_grad_norm,
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
            ),
            verbose=0,
            tensorboard_log=tb_log_dir,
        )

        # Train PPO
        model.learn(
            total_timesteps=n_training_episodes * 100,
            callback=callback,
            progress_bar=False,
        )

        # Check if pruned
        if callback.is_pruned: raise optuna.TrialPruned()
        metrics = callback.get_final_metrics()

        # Store additional metrics as user attributes (convert to native Python types)
        trial.set_user_attr("mean_reward", metrics["mean_reward"])
        trial.set_user_attr("std_blocking_rate", metrics["std_blocking_rate"])

        return metrics["mean_blocking_rate"]

    except optuna.TrialPruned: raise # Trial was pruned by the pruner - this is normal, just re-raise
    except Exception as e: # Actual error occurred - log it and mark as failed
        print(f"Trial failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.TrialPruned()
    finally: env.close()


def optimize_hyperparameters(capacity, n_trials=50, n_training_episodes=200, n_jobs=-1, data_dir=None):
    """Run Optuna hyperparameter optimization.
    
    Args:
        capacity: Link capacity
        n_trials: Number of trials to run
        n_training_episodes: Episodes per trial (keep small for speed)
        
    Returns: Study object with results
    """
    print("\n" + "=" * 70)
    print(f"Optuna Hyperparameter Optimization - PPO Capacity {capacity}")
    print("=" * 70)

    # Load training data
    request_files = load_training_requests(data_dir=data_dir)
    print(f"Loaded {len(request_files)} training files")
    if n_training_episodes > len(request_files):
        print(f"Warning: Requested {n_training_episodes} episodes per trial but only {len(request_files)} files available. Capping to {len(request_files)}")
        n_training_episodes = len(request_files)
    print(f"Using {n_training_episodes} episodes per trial for quick evaluation\n")
    
    # Create study
    study_name = f"ppo_capacity_{capacity}"
    optuna_dir = os.path.join(script_dir, 'optuna_studies')
    os.makedirs(optuna_dir, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(optuna_dir, f'ppo_study_capacity_{capacity}.db')}"
    
    # Sampler and pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=5)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize', # Minimize blocking rate
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    # Optimize
    print(f"Starting optimization with {n_trials} trials...")
    print("(Pruning enabled: unpromising trials will be stopped early)\n")
    study.optimize(
        lambda t: objective(t, capacity, request_files, n_training_episodes),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    return study



def analyze_and_visualize_study(study, capacity):
    """Analyze optimization results and create visualizations.
    
    Args:
        study: Completed Optuna study
        capacity: Link capacity
    """
    print(f"\n{'='*70}")
    print(f"Optimization Results - PPO Capacity {capacity}")
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
    os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)
    best_params_file = os.path.join(script_dir, f'results/best_ppo_hyperparameters_capacity_{capacity}.json')
    
    # Convert all values to native Python types for JSON serialization
    def convert_to_native(obj): # Convert numpy types to native Python types
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list): return [convert_to_native(item) for item in obj]
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
    os.makedirs(os.path.join(script_dir, 'plots'), exist_ok=True)
    
    # Optimization history
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'plots/ppo_optuna_history_capacity_{capacity}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nOptimization visualizations saved to '{os.path.join(script_dir, 'plots')}' directory")
    
    # Statistics
    print(f"\nStudy Statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")


def train_final_model_with_best_params(capacity, best_params, num_episodes=1000, save_copy_as_standard=True, data_dir=None):
    """Train final PPO model with best hyperparameters from Optuna

    Args:
        capacity: Link capacity
        best_params: Best hyperparameters from Optuna study
        num_episodes: Number of episodes for full training
    """
    print(f"\n{'='*70}")
    print(f"Training Final PPO Model with Best Hyperparameters - Capacity {capacity}")
    print(f"{'='*70}\n")

    # Load training request files (One request file = one episode)
    request_files = load_training_requests(data_dir=data_dir, max_files=num_episodes)
    if not request_files: raise ValueError('No training files found for final training - check data_dir or project root data/train')
    print(f"Loaded {len(request_files)} training files")

    # Create training environment
    env = create_env(capacity, request_files)

    # Extract network architecture from Optuna parameters
    # (PPO uses a shared policy/value MLP)
    net_arch_size = best_params["net_arch_size"]
    net_arch_depth = best_params["net_arch_depth"]
    # Example: net_arch_size=256, depth=3 => [256, 256, 256]
    net_arch = [net_arch_size] * net_arch_depth

    # Activation function
    activation_fn_name = best_params["activation_fn"]
    activation_fn = nn.Tanh if activation_fn_name == "tanh" else nn.ReLU

    # Create PPO model with best hyperparameters
    tb_log_dir = os.path.join(script_dir, f"tensorboard_logs/ppo_optimized_capacity_{capacity}")
    os.makedirs(tb_log_dir, exist_ok=True)
    model = PPO(
        "MlpPolicy",
        env,
        # Core PPO hyperparameters
        learning_rate=best_params["learning_rate"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        # PPO-specific optimization parameters
        clip_range=best_params["clip_range"],
        ent_coef=best_params["ent_coef"],
        vf_coef=best_params["vf_coef"],
        # Rollout and update parameters
        n_steps=best_params["n_steps"], # Steps per rollout
        batch_size=best_params["batch_size"], # Minibatch size
        n_epochs=best_params["n_epochs"], # Gradient epochs per update
        max_grad_norm=best_params["max_grad_norm"],
        # Policy network configuration
        policy_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn),
        # Logging
        verbose=1,
        tensorboard_log=tb_log_dir,
    )

    # Train PPO model
    total_timesteps = num_episodes * 100
    print(f"\nTraining for {total_timesteps} timesteps (~{num_episodes} episodes)...")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save trained PPO model
    save_name = os.path.join(script_dir, f"models/ppo_optimized_capacity_{capacity}")
    os.makedirs(os.path.join(script_dir, 'models'), exist_ok=True)
    model.save(save_name)
    print(f"\nOptimized PPO model saved to {save_name}.zip")
    if save_copy_as_standard:
        # Also save a copy matching the standard model naming convention for the evaluation script
        standard_name = os.path.join(script_dir, f'models/ppo_capacity_{capacity}.zip')
        optimized_name = f"{save_name}.zip"
        try:
            shutil.copyfile(optimized_name, standard_name)
            print(f"Also saved a copy to {standard_name}")
        except Exception:
            print('Warning: Could not copy optimized model to standard model name')
    env.close() # Clean up environment
    return model


def main(): # Main optimization workflow
    parser = argparse.ArgumentParser(description='PPO Optuna hyperparameter tuning')
    parser.add_argument('--capacities', type=str, default='20,10', help='Comma-separated capacities to optimize (default: 20,10)')
    parser.add_argument('--data-dir', type=str, default=None, help='Optional data directory (relative to project root or absolute). Defaults to project_root/data/train')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials per capacity')
    parser.add_argument('--n-training-episodes-per-trial', type=int, default=200, help='Episodes per trial for quick evaluation')
    parser.add_argument('--n-final-training-episodes', type=int, default=1000, help='Episodes for final training with best params')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs for Optuna (n_jobs)')
    parser.add_argument('--train-final', action='store_true', help='Automatically train final models with the best hyperparameters')
    parser.add_argument('--no-prompt', action='store_true', help='Suppress interactive prompts; implies --train-final if prompt would be used')
    args = parser.parse_args()

    capacities = [int(c) for c in args.capacities.split(',')]
    n_trials = args.n_trials
    n_training_episodes_per_trial = args.n_training_episodes_per_trial
    n_final_training_episodes = args.n_final_training_episodes
    print("\n" + "="*70)
    print("PPO Hyperparameter Optimization with Optuna")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Trials per capacity: {n_trials}")
    print(f"  Episodes per trial: {n_training_episodes_per_trial} (quick evaluation)")
    print(f"  Final training episodes: {n_final_training_episodes} (full training)")
    print(f"  n_jobs (parallel): {args.n_jobs}")
    print(f"\nOptimization will take approximately:")
    print(f"  {n_trials * n_training_episodes_per_trial * len(capacities) / 60:.0f} minutes for all trials")
    print("="*70)

    for capacity in capacities:
        # Run optimization
        study = optimize_hyperparameters(
            capacity=capacity,
            n_trials=n_trials,
            n_training_episodes=n_training_episodes_per_trial,
            n_jobs=args.n_jobs,
            data_dir=args.data_dir
        )
        
        # Analyze results
        analyze_and_visualize_study(study, capacity)
        
        # Train final model with best hyperparameters
        if args.train_final or args.no_prompt:
            train_final_model_with_best_params(
                capacity=capacity,
                best_params=study.best_params,
                num_episodes=n_final_training_episodes,
                data_dir=args.data_dir
            )
        else:
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
    print("  - results/best_ppo_hyperparameters_capacity_*.json")
    print("  - plots/ppo_optuna_*.png")
    print("  - optuna_studies/ppo_study_*.db")
    print("\nOptimized models (if trained):")
    print("  - models/ppo_optimized_capacity_*.zip")


if __name__ == '__main__': main()