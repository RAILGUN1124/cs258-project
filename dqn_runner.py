"""
DQN Agent Training Script for RSA Environment

This script trains Deep Q-Network agents using stable-baselines3 for
the Routing and Spectrum Allocation problem with different link capacities.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from rsaenv import RSAEnv
from nwutil import generate_sample_graph
import json


class TrainingMetricsCallback(BaseCallback):
    """
    Callback for collecting training metrics during DQN training.
    Tracks episode rewards and blocking rates.
    """
    
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_blocking_rates = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Accumulate reward
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            # Get blocking rate from info
            info = self.locals['infos'][0]
            blocking_rate = info.get('blocking_rate', 0.0)
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_blocking_rates.append(blocking_rate)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_blocking = np.mean(self.episode_blocking_rates[-10:])
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Avg Reward (last 10) = {avg_reward:.2f}, "
                      f"Avg Blocking Rate (last 10) = {avg_blocking:.3f}")
        
        return True
    
    def get_metrics(self):
        """Return collected metrics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_blocking_rates': self.episode_blocking_rates,
            'episode_lengths': self.episode_lengths
        }


def load_training_requests(data_dir='data/train', max_files=None):
    """Load all training request files"""
    files = sorted(glob.glob(os.path.join(data_dir, 'requests-*.csv')))
    if max_files:
        files = files[:max_files]
    return files


def create_env(capacity, request_files):
    """Create a wrapped RSA environment with multiple episodes"""
    class MultiFileEnv(RSAEnv):
        def __init__(self, capacity, files):
            self.files = files
            self.file_idx = 0
            super().__init__(capacity=capacity, request_file=self.files[self.file_idx])
        
        def reset(self, seed=None, options=None):
            # Load next file
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.requests = self._load_requests(self.files[self.file_idx])
            return super().reset(seed=seed, options=options)
    
    env = MultiFileEnv(capacity, request_files)
    env = Monitor(env)
    return env


def train_dqn(capacity, num_episodes=1000, save_name='dqn_model', verbose=1):
    """
    Train a DQN agent for RSA with specified link capacity.
    
    Args:
        capacity: Link capacity (number of wavelengths)
        num_episodes: Number of training episodes
        save_name: Name for saving the trained model
        verbose: Verbosity level
    
    Returns:
        Trained model and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training DQN with capacity={capacity}")
    print(f"{'='*60}\n")
    
    # Load training data
    request_files = load_training_requests(max_files=num_episodes)
    print(f"Loaded {len(request_files)} training files")
    
    # Create environment
    env = create_env(capacity, request_files)
    
    # Create DQN agent with tuned hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200000,  # Increased from 100k to 200k
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=verbose,
        tensorboard_log=f"./tensorboard_logs/dqn_capacity_{capacity}/"
    )
    
    # Set up callback for metrics
    callback = TrainingMetricsCallback(verbose=verbose)
    
    # Train the agent
    total_timesteps = num_episodes * 100  # Assuming ~100 requests per episode
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True
    )
    
    # Save the model
    model.save(save_name)
    print(f"\nModel saved to {save_name}.zip")
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Save metrics
    metrics_file = f"{save_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'episode_rewards': [float(r) for r in metrics['episode_rewards']],
            'episode_blocking_rates': [float(b) for b in metrics['episode_blocking_rates']],
            'episode_lengths': [int(l) for l in metrics['episode_lengths']]
        }, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    return model, metrics


def plot_training_results(metrics, capacity, save_path=None):
    """
    Plot training learning curves.
    
    Args:
        metrics: Dictionary with training metrics
        capacity: Link capacity used
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    episodes = range(1, len(metrics['episode_rewards']) + 1)
    
    # Plot 1: Episode Rewards (with moving average)
    ax1.plot(episodes, metrics['episode_rewards'], alpha=0.3, label='Episode Reward')
    
    # Compute moving average
    window = 10
    if len(metrics['episode_rewards']) >= window:
        moving_avg = np.convolve(metrics['episode_rewards'], 
                                 np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(episodes) + 1), moving_avg, 
                label=f'Moving Avg (window={window})', linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title(f'Learning Curve - Capacity {capacity}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Blocking Rate (with moving average)
    ax2.plot(episodes, metrics['episode_blocking_rates'], alpha=0.3, label='Blocking Rate')
    
    if len(metrics['episode_blocking_rates']) >= window:
        moving_avg_blocking = np.convolve(metrics['episode_blocking_rates'],
                                         np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(episodes) + 1), moving_avg_blocking,
                label=f'Moving Avg (window={window})', linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Blocking Rate (B)')
    ax2.set_title(f'Blocking Rate Over Training - Capacity {capacity}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {save_path}")
    
    plt.show()


def main():
    """Main training function"""
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Training parameters
    num_train_episodes = 1000  # Adjust based on available training data
    
    # Part 1: Train with capacity=20
    print("\n" + "="*70)
    print("PART 1: Training with Link Capacity = 20")
    print("="*70)
    
    model_20, metrics_20 = train_dqn(
        capacity=20,
        num_episodes=num_train_episodes,
        save_name='models/dqn_capacity_20',
        verbose=1
    )
    
    plot_training_results(
        metrics_20,
        capacity=20,
        save_path='plots/training_capacity_20.png'
    )
    
    # Part 2: Train with capacity=10
    print("\n" + "="*70)
    print("PART 2: Training with Link Capacity = 10")
    print("="*70)
    
    model_10, metrics_10 = train_dqn(
        capacity=10,
        num_episodes=num_train_episodes,
        save_name='models/dqn_capacity_10',
        verbose=1
    )
    
    plot_training_results(
        metrics_10,
        capacity=10,
        save_path='plots/training_capacity_10.png'
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nTrained models saved in 'models/' directory")
    print(f"Training plots saved in 'plots/' directory")
    print(f"Metrics saved in 'models/' directory")
    print(f"\nRun the evaluation script to test on the eval dataset.")


if __name__ == '__main__':
    main()
