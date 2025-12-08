"""
PPO Training Script for RSA Environment

This mirrors the DQN training pipeline but uses Stable-Baselines3 PPO.
It trains separate agents for capacity=20 and capacity=10 and saves
models, metrics, and training plots.
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
    """Callback to collect episode-level statistics during PPO training.

    This callback mirrors the one used for DQN training. It accumulates the
    per-step rewards and captures episode-level blocking rate information
    from the environment "info" dict when an episode ends.
    """

    def __init__(self, verbose=0):
        # Initialize parent callback with verbosity setting
        super().__init__(verbose)
        # Store lists of episode-level metrics
        self.episode_rewards = [] # cumulative reward per episode
        self.episode_blocking_rates = [] # blocking rate (blocked/total) per episode
        self.episode_lengths = [] # length (#requests) per episode

        # Running accumulators for the current episode
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called by SB3 after every environment step.

        The self.locals dict contains training internals such as rewards,
        dones, and infos. For vectorized envs these are arrays. Here we
        use index 0 because we operate with a single Monitor-wrapped env.
        """
        # Accumulate reward for the current episode
        self.current_episode_reward += float(self.locals['rewards'][0])
        # Increment step count for current episode
        self.current_episode_length += 1

        # Check if the episode finished on this step
        if self.locals['dones'][0]:
            # Extract environment-provided info (contains blocking_rate)
            info = self.locals['infos'][0]
            blocking_rate = info.get('blocking_rate', 0.0)

            # Append the episode summary metrics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_blocking_rates.append(blocking_rate)
            self.episode_lengths.append(self.current_episode_length)

            # Reset running accumulators for the next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # Occasionally print a progress summary when verbose
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                avg_r = np.mean(self.episode_rewards[-10:])
                avg_b = np.mean(self.episode_blocking_rates[-10:])
                print(f"Episode {len(self.episode_rewards)}: Avg Reward (last 10)={avg_r:.2f}, Avg Blocking={avg_b:.3f}")

        # Returning True tells SB3 to continue training
        return True

    def get_metrics(self):
        """Return the collected training metrics as a dictionary."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_blocking_rates': self.episode_blocking_rates,
            'episode_lengths': self.episode_lengths
        }


def load_training_requests(data_dir=None, max_files=None):
    """Load training request files.

    If called from inside ppo/, the default training directory is assumed
    to be the project-root data/train/ directory. This helper resolves the
    correct path so the script works both when run from the project root or
    when run from inside ppo/.
    """
    # Resolve project root (parent of this file's directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data', 'train')

    # If provided data_dir is relative, resolve it against project root
    if not os.path.isabs(data_dir):
        candidate = os.path.join(project_root, data_dir)
        if os.path.exists(candidate):
            data_dir = candidate

    files = sorted(glob.glob(os.path.join(data_dir, 'requests-*.csv')))
    if not files:
        raise FileNotFoundError(f"No training files found in '{data_dir}'. Ensure dataset exists under data/train/")

    if max_files:
        files = files[:max_files]
    return files


def train_ppo(capacity, num_episodes=1000, save_name='models/ppo_model', verbose=1):
    """Train a PPO agent on the RSA environment.

    Args:
        capacity: Number of wavelengths per link (10 or 20).
        num_episodes: Number of episodes to train (each episode = one CSV file).
        save_name: Path (under ppo/ when running from ppo/) to save the model.
        verbose: Verbosity level forwarded to Stable-Baselines3.

    Returns:
        Tuple (trained_model, metrics_dict)
    """

    # Print header for clarity
    print(f"\nTraining PPO with capacity={capacity}\n")

    # Load list of training files (resolves project root when needed)
    request_files = load_training_requests(max_files=num_episodes)
    print(f"Loaded {len(request_files)} training files")

    # Create the environment wrapper that cycles through CSV files
    env = create_env(capacity, request_files)

    # Configure PPO hyperparameters (these are reasonable defaults)
    model = PPO(
        "MlpPolicy",  # Fully-connected policy network
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=verbose,
        tensorboard_log=f"tensorboard_logs/ppo_capacity_{capacity}/"
    )

    # Callback collects episode rewards and blocking rates
    callback = TrainingMetricsCallback(verbose=verbose)

    # Estimate total timesteps: ~100 timesteps per episode (approx)
    total_timesteps = num_episodes * 100

    # Start learning (this will take time for large num_episodes)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Ensure save directory exists and persist the model
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    model.save(save_name)
    print(f"Saved model to {save_name}.zip")

    # Retrieve and persist collected metrics
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    episodes = range(1, len(metrics['episode_rewards']) + 1)

    ax1.plot(episodes, metrics['episode_rewards'], alpha=0.3, label='Episode Reward')
    window = 10
    if len(metrics['episode_rewards']) >= window:
        mv = np.convolve(metrics['episode_rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(episodes)+1), mv, label=f'MA (w={window})', linewidth=2)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Cumulative Reward'); ax1.set_title(f'PPO Learning - Capacity {capacity}')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, metrics['episode_blocking_rates'], alpha=0.3, label='Blocking Rate')
    if len(metrics['episode_blocking_rates']) >= window:
        mvb = np.convolve(metrics['episode_blocking_rates'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(episodes)+1), mvb, label=f'MA (w={window})', linewidth=2)
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Blocking Rate'); ax2.set_title(f'Blocking over Training - Capacity {capacity}')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {save_path}")
    plt.show()


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('tensorboard_logs', exist_ok=True)

    num_train_episodes = 1000

    # Capacity 20
    model_20, metrics_20 = train_ppo(capacity=20, num_episodes=num_train_episodes, save_name='models/ppo_capacity_20', verbose=1)
    plot_training_results(metrics_20, capacity=20, save_path='plots/ppo_training_capacity_20.png')

    # Capacity 10
    model_10, metrics_10 = train_ppo(capacity=10, num_episodes=num_train_episodes, save_name='models/ppo_capacity_10', verbose=1)
    plot_training_results(metrics_10, capacity=10, save_path='plots/ppo_training_capacity_10.png')

    print("PPO training complete. Models and plots saved.")


if __name__ == '__main__':
    main()
