"""
PPO Training Script for RSA Environment

Follows DQN training setup but uses PPO
Separate agents are trained for capacity=20 and capacity=10
Models, metrics, and plots saved for eval run
"""
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ppo_env import create_env


class TrainingMetricsCallback(BaseCallback):
    # Collect metrics for each episode during PPO training
    # Track cumulative rewards, blocking rates, and episode lengths

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Lists to hold metrics for all episodes
        self.episode_rewards = [] # Total reward per episode
        self.episode_blocking_rates = [] # Blocking rate (blocked/total)
        self.episode_lengths = [] # Number of steps per episode

        # Running totals for the current episode
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Called after each env step during training
        # Accumulates rewards and tracks episode completions

        # Add reward from this step to the running total
        self.current_episode_reward += float(self.locals['rewards'][0])
        self.current_episode_length += 1 # Count this step

        # Check if the episode ended on this step
        if self.locals['dones'][0]:
            # Extract extra info from the environment (blocking_rate)
            info = self.locals['infos'][0]
            blocking_rate = info.get('blocking_rate', 0.0)

            # Store metrics for the finished episode
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_blocking_rates.append(blocking_rate)
            self.episode_lengths.append(self.current_episode_length)

            # Reset counters for the next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # Occasionally print a short summary for every 10 episodes
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                avg_r = np.mean(self.episode_rewards[-10:])
                avg_b = np.mean(self.episode_blocking_rates[-10:])
                print(f"Episode {len(self.episode_rewards)}: Avg Reward (last 10)={avg_r:.2f}, Avg Blocking={avg_b:.3f}")

        return True  # Continue training

    def get_metrics(self):
        # Return all collected metrics as dictionary
        return {
            'episode_rewards': self.episode_rewards,
            'episode_blocking_rates': self.episode_blocking_rates,
            'episode_lengths': self.episode_lengths
        }


def load_training_requests(data_dir=None, max_files=None):
    # Load CSV files with training requests
    # Resolve relative paths relative to project root
    # Supports limiting  # of files for quicker experiments

    # Determine project root relative to this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if data_dir is None: data_dir = os.path.join(project_root, 'data', 'train')

    # If data_dir is relative, resolve to absolute path
    if not os.path.isabs(data_dir):
        candidate = os.path.join(project_root, data_dir)
        if os.path.exists(candidate): data_dir = candidate

    # Find all CSV files matching the naming pattern
    files = sorted(glob.glob(os.path.join(data_dir, 'requests-*.csv')))
    if not files: raise FileNotFoundError(f"No training files found in '{data_dir}'. Check that data/train/ exists.")

    # Optionally limit number of files
    if max_files: files = files[:max_files]
    return files


def train_ppo(capacity, num_episodes=1000, save_name='models/ppo_model', verbose=1):
    """Train PPO agent for a given capacity

    Args:
        capacity: Number of wavelengths per link (e.g., 10 or 20)
        num_episodes: How many episodes to train (roughly one file per episode)
        save_name: Where to save the trained model
        verbose: Verbosity level for logging

    Returns:
        model: Trained PPO agent
        metrics: Dictionary of episode metrics
    """
    print(f"\n=== Training PPO: Capacity={capacity} ===\n")

    # Load CSV files with network requests
    request_files = load_training_requests(max_files=num_episodes)
    print(f"Loaded {len(request_files)} request files")

    # Create environment that cycles through request files
    env = create_env(capacity, request_files)

    # Set up PPO agent with reasonable hyperparameters
    model = PPO(
        "MlpPolicy",  # Fully connected network
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=verbose,
        tensorboard_log=f"tensorboard_logs/ppo_capacity_{capacity}/"
    )

    # Callback to collect episode metrics
    callback = TrainingMetricsCallback(verbose=verbose)

    # Estimate total timesteps (roughly 100 per episode)
    total_timesteps = num_episodes * 100

    # Start training
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    model.save(save_name)
    print(f"Model saved to {save_name}.zip")

    # Save training metrics
    metrics = callback.get_metrics()
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
    # Plot cumulative rewards and blocking rate over episodes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    episodes = range(1, len(metrics['episode_rewards']) + 1)

    # Plot cumulative reward per episode
    ax1.plot(episodes, metrics['episode_rewards'], alpha=0.3, label='Episode Reward')
    window = 10  # Moving average window
    if len(metrics['episode_rewards']) >= window:
        ma = np.convolve(metrics['episode_rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(episodes)+1), ma, label=f'MA (w={window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title(f'PPO Learning - Capacity {capacity}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot blocking rate
    ax2.plot(episodes, metrics['episode_blocking_rates'], alpha=0.3, label='Blocking Rate')
    if len(metrics['episode_blocking_rates']) >= window:
        ma_b = np.convolve(metrics['episode_blocking_rates'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(episodes)+1), ma_b, label=f'MA (w={window})', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Blocking Rate')
    ax2.set_title(f'Blocking over Training - Capacity {capacity}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {save_path}")
    plt.show()


def main():
    # Ensure all dirs exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('tensorboard_logs', exist_ok=True)

    num_train_episodes = 1000  # Default number of episodes

    # Train PPO with capacity 20
    model_20, metrics_20 = train_ppo(
        capacity=20,
        num_episodes=num_train_episodes,
        save_name='models/ppo_capacity_20',
        verbose=1
    )
    plot_training_results(metrics_20, capacity=20, save_path='plots/ppo_training_capacity_20.png')

    # Train PPO with capacity 10
    model_10, metrics_10 = train_ppo(
        capacity=10,
        num_episodes=num_train_episodes,
        save_name='models/ppo_capacity_10',
        verbose=1
    )
    plot_training_results(metrics_10, capacity=10, save_path='plots/ppo_training_capacity_10.png')
    print("PPO training complete. Models and plots saved.")


if __name__ == '__main__':  main()