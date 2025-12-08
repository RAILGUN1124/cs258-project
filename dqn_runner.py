"""
DQN Agent Training Script for RSA Environment

This script trains Deep Q-Network agents using stable-baselines3 for
the Routing and Spectrum Allocation problem with different link capacities.

TRAINING PIPELINE:
-----------------
1. Load training request files (sequences of demands)
2. Create multi-file environment wrapper (cycles through files)
3. Initialize DQN with tuned hyperparameters
4. Train for specified timesteps with metric tracking
5. Save trained model and generate learning curves

KEY DESIGN CHOICES:
------------------
- DQN algorithm: Effective for discrete action spaces (8 paths)
- MLP policy: Fully connected network [input → 64 → 64 → 8 Q-values]
- Experience replay: Breaks correlation, improves sample efficiency
- Target network: Stabilizes learning (updated every 1000 steps)
- Epsilon-greedy: Balances exploration vs. exploitation

OUTPUTS:
-------
- Trained models: models/dqn_capacity_{20,10}.zip
- Training metrics: JSON files with episode rewards and blocking rates
- Learning curves: PNG plots showing training progress
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
    Custom callback for collecting detailed training metrics during DQN training.
    
    stable-baselines3 provides basic logging, but we need domain-specific metrics:
    - Episode rewards: Cumulative reward per episode (negative blocking count)
    - Blocking rates: Percentage of requests blocked (key RSA metric)
    - Episode lengths: Number of requests processed per episode
    
    WHY THIS IS IMPORTANT:
    - Standard RL metrics (episode reward) don't directly show RSA performance
    - Blocking rate is the primary metric for optical network quality
    - Tracking both helps diagnose learning: rewards ↑ ⇔ blocking ↓
    - Enables detailed analysis and visualization of learning curves
    
    This callback is invoked after every environment step during training,
    accumulating statistics and logging progress.
    """
    
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        # Lists to store metrics across all episodes
        self.episode_rewards = []  # Cumulative reward per episode
        self.episode_blocking_rates = []  # Blocking rate per episode
        self.episode_lengths = []  # Number of requests per episode
        
        # Accumulators for current episode (reset after each episode)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """
        Called by stable-baselines3 after every environment step.
        
        This method has access to self.locals dict containing:
        - 'rewards': Reward received this step
        - 'dones': Whether episode terminated
        - 'infos': Additional info from environment (including blocking_rate)
        
        We accumulate rewards across the episode, then log when episode ends.
        
        Returns:
            bool: True to continue training, False to stop early
        """
        # Accumulate reward for current episode
        # rewards[0] because vectorized env returns array (we use single env)
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended (all requests processed)
        if self.locals['dones'][0]:
            # Extract final blocking rate from environment info dict
            info = self.locals['infos'][0]
            blocking_rate = info.get('blocking_rate', 0.0)
            
            # Store episode-level metrics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_blocking_rates.append(blocking_rate)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset accumulators for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Print progress every 10 episodes (if verbose enabled)
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_blocking = np.mean(self.episode_blocking_rates[-10:])
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Avg Reward (last 10) = {avg_reward:.2f}, "
                      f"Avg Blocking Rate (last 10) = {avg_blocking:.3f}")
        
        return True  # Continue training
    
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
    """
    Create a wrapped RSA environment that cycles through multiple request files.
    
    WHY THIS IS NEEDED:
    - Each request file contains ~100 requests (one episode)
    - Training requires 1000+ episodes for convergence
    - We need to automatically load new files for each episode
    
    The MultiFileEnv wrapper:
    - Inherits from RSAEnv (same interface)
    - Overrides reset() to load next file in sequence
    - Cycles through files (wraps around with modulo)
    - Provides diverse training scenarios (different request patterns)
    
    This ensures the agent sees varied traffic patterns and learns
    robust policies rather than overfitting to a single sequence.
    
    Args:
        capacity: Number of wavelengths per link
        request_files: List of paths to CSV request files
    
    Returns:
        Monitor-wrapped MultiFileEnv for training
    """
    class MultiFileEnv(RSAEnv):
        """Environment wrapper that automatically cycles through request files."""
        def __init__(self, capacity, files):
            self.files = files
            self.file_idx = 0  # Start with first file
            super().__init__(capacity=capacity, request_file=self.files[self.file_idx])
        
        def reset(self, seed=None, options=None):
            """Load next request file and reset environment."""
            # Advance to next file (circular)
            self.file_idx = (self.file_idx + 1) % len(self.files)
            # Load requests from new file
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
    
    # Create DQN agent with carefully tuned hyperparameters
    # Each hyperparameter choice affects learning speed, stability, and final performance
    model = DQN(
        "MlpPolicy",  # Multi-layer perceptron: [input→64→64→8 Q-values]
        env,
        
        # LEARNING RATE (1e-4 = 0.0001)
        # - Controls how quickly weights are updated
        # - Too high: unstable, diverges
        # - Too low: slow convergence
        # - 1e-4 is standard for DQN
        learning_rate=1e-4,
        
        # REPLAY BUFFER SIZE (200,000 transitions)
        # - Stores (state, action, reward, next_state) tuples
        # - Larger buffer = more diverse training data, better sample efficiency
        # - Increased from 100k to 200k for improved performance
        # - 200k can store ~2000 episodes worth of experience
        buffer_size=200000,
        
        # LEARNING STARTS (wait 1000 steps before training)
        # - Fill buffer with random exploration first
        # - Ensures diverse initial experiences
        # - Prevents early overfitting to limited data
        learning_starts=1000,
        
        # BATCH SIZE (64 samples per gradient update)
        # - Number of transitions sampled from replay buffer for each update
        # - Larger = more stable gradients but slower
        # - 64 is a good balance for this problem size
        batch_size=64,
        
        # TAU (0.005 = soft update coefficient)
        # - Controls target network update: θ_target ← τ*θ + (1-τ)*θ_target
        # - Small tau = slow target updates = more stable learning
        # - Prevents target network from changing too quickly
        tau=0.005,
        
        # GAMMA (0.99 = discount factor)
        # - Importance of future rewards: Q(s,a) = r + γ*max Q(s',a')
        # - 0.99 means agent values long-term consequences
        # - Critical for RSA: current decisions affect future capacity
        gamma=0.99,
        
        # TRAIN FREQ (update every 4 steps)
        # - How often to perform gradient descent
        # - More frequent = faster learning but more computation
        train_freq=4,
        
        # GRADIENT STEPS (1 update per training trigger)
        # - How many gradient updates to perform when training
        # - More steps = learn more from buffer but risk overfitting
        gradient_steps=1,
        
        # TARGET UPDATE INTERVAL (every 1000 steps)
        # - How often to update target network (used for computing target Q-values)
        # - Larger = more stable but slower to track policy improvements
        target_update_interval=1000,
        
        # EXPLORATION SCHEDULE (epsilon-greedy)
        # - exploration_fraction=0.3: Anneal epsilon over first 30% of training
        # - Start with random exploration (eps=1.0)
        # - End with mostly exploitation (eps=0.05, 5% random)
        # - Allows agent to explore initially, then refine learned policy
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        
        verbose=verbose,
        tensorboard_log=f"./tensorboard_logs/dqn_capacity_{capacity}/"
    )
    
    # Set up callback for metrics collection
    # This callback will be invoked after every environment step during training
    callback = TrainingMetricsCallback(verbose=verbose)
    
    # Train the agent
    # total_timesteps = number of environment steps (requests processed)
    # Each episode has ~100 requests, so 1000 episodes ≈ 100,000 timesteps
    total_timesteps = num_episodes * 100  # Assuming ~100 requests per episode
    
    # Main training loop (handled by stable-baselines3)
    # This will:
    # 1. Interact with environment (select actions using epsilon-greedy)
    # 2. Store experiences in replay buffer
    # 3. Periodically sample batch and perform gradient descent
    # 4. Update target network every target_update_interval steps
    # 5. Call our callback to collect metrics
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,  # Print logging info every 10 updates
        progress_bar=True  # Show progress bar during training
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
    Generate comprehensive visualizations of training progress.
    
    Creates two subplots:
    1. Episode Rewards: Shows learning progress (rewards should increase)
    2. Blocking Rate: Shows RSA performance (blocking should decrease)
    
    Both plots include:
    - Raw episode values (noisy, shows variance)
    - Moving average (smoothed, shows trends)
    - Mean lines (overall performance)
    
    WHY BOTH METRICS:
    - Episode reward is the RL objective (what agent optimizes)
    - Blocking rate is the domain metric (what we actually care about)
    - They should be inversely correlated: reward ↑ ⇔ blocking ↓
    
    Args:
        metrics: Dictionary with keys 'episode_rewards', 'episode_blocking_rates'
        capacity: Link capacity used (for plot title)
        save_path: Path to save the PNG plot
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
