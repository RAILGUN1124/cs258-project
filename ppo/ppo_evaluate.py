"""
Evaluation utilities for PPO-trained agents on the RSA environment.

This script is responsible for:
- Loading evaluation request files (CSV) from data/eval/ (resolved
    relative to the project root so the script can be run from ppo/).
- Loading a saved PPO model from ppo/models/.
- Running the model deterministically on each evaluation episode and
    recording the episode blocking rate and cumulative reward.
- Saving results to ppo/results/ and generating plots in ppo/plots/.
"""

import os
import glob
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add project root to sys.path so we can import rsaenv when running
# from the ppo/ subdirectory (this matches how run_all.sh invokes scripts).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rsaenv import RSAEnv


def load_eval_requests(data_dir='data/eval'):
    """Return a sorted list of evaluation CSV files.

    Behavior:
    - If data_dir is None, default to <project_root>/data/eval.
    - If a relative path is given, prefer the location under project root.
    - Raises a clear FileNotFoundError if no matching files are found.

    Args:
        data_dir: Optional path to evaluation data (defaults to 'data/eval').

    Returns:
        List[str]: Sorted list of file paths matching requests-*.csv.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Default to project_root/data/eval when running from inside ppo/
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data', 'eval')
    # If provided a relative path, prefer the folder under project_root
    elif not os.path.isabs(data_dir):
        candidate = os.path.join(project_root, data_dir)
        if os.path.exists(candidate):
            data_dir = candidate

    # Find files matching the naming convention
    files = sorted(glob.glob(os.path.join(data_dir, 'requests-*.csv')))
    if not files:
        # Fail fast with a descriptive error to help debugging
        raise FileNotFoundError(f"No evaluation files found in '{data_dir}'. Ensure dataset exists under data/eval/")
    return files


def evaluate_model(model, capacity, eval_files, deterministic=True):
    """Run the model on each evaluation file and collect metrics.

    For each CSV file (one episode) we:
    1. Construct a fresh RSAEnv with the file's requests.
    2. Reset the env to get the initial observation.
    3. Step through the episode until termination, using the model's
       deterministic policy (if requested).
    4. Record the final blocking_rate and cumulative reward.

    Args:
        model: Loaded Stable-Baselines3 PPO model.
        capacity: Link capacity (10 or 20) to construct the env.
        eval_files: Iterable of CSV file paths to evaluate.
        deterministic: Whether to use deterministic policy (recommended).

    Returns:
        dict: Contains lists of per-episode blocking rates and rewards and
              summary statistics (mean/std blocking rate).
    """

    # Storage for per-episode results
    episode_blocking_rates = []
    episode_rewards = []

    print(f"\nEvaluating on {len(eval_files)} episodes...")

    # Iterate over each evaluation episode file
    for i, req_file in enumerate(eval_files):
        # Create a fresh environment for this episode so we start from
        # all links free and the exact request sequence in the CSV file.
        env = RSAEnv(capacity=capacity, request_file=req_file)

        # Reset returns (obs, info) per Gymnasium API used in this project
        obs, info = env.reset()

        done = False
        episode_reward = 0

        # Run the episode until termination (or truncation)
        while not done:
            # Query the model for an action. deterministic=True uses the
            # greedy policy (no exploration), which is desired for eval.
            action, _ = model.predict(obs, deterministic=deterministic)

            # Apply the action in the environment and collect result
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # At episode end the env provides a 'blocking_rate' in info
        episode_blocking_rates.append(info['blocking_rate'])
        episode_rewards.append(episode_reward)

        # Provide progress feedback for large eval sets
        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(eval_files)} episodes")

    # Compute summary statistics and return detailed lists
    return {
        'episode_blocking_rates': episode_blocking_rates,
        'episode_rewards': episode_rewards,
        'mean_blocking_rate': float(np.mean(episode_blocking_rates)),
        'std_blocking_rate': float(np.std(episode_blocking_rates))
    }


def plot_evaluation_results(eval_metrics, capacity, save_path=None):
    episodes = range(1, len(eval_metrics['episode_blocking_rates']) + 1)
    blocking_rates = eval_metrics['episode_blocking_rates']

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, blocking_rates, alpha=0.5, label='Episode Blocking Rate')
    window = 10
    if len(blocking_rates) >= window:
        mv = np.convolve(blocking_rates, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(episodes) + 1), mv, label=f'Moving Avg (w={window})', color='red')

    mean_blocking = eval_metrics['mean_blocking_rate']
    plt.axhline(y=mean_blocking, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean_blocking:.4f}')
    plt.xlabel('Episode'); plt.ylabel('Blocking Rate (B)')
    plt.title(f'PPO Evaluation - Capacity {capacity} (Mean={mean_blocking:.4f})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    # Ensure local output directories exist under ppo/
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load evaluation request files (from project_root/data/eval)
    eval_files = load_eval_requests()
    print(f"Loaded {len(eval_files)} evaluation files")

    # Evaluate both capacity configurations (20 and 10)
    for capacity in [20, 10]:
        # Model filenames are expected to be in ppo/models/ when running
        # from ppo/ working directory.
        model_path = f'models/ppo_capacity_{capacity}.zip'
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skip")
            continue

        # Load the trained PPO model from disk
        print(f"Loading model {model_path}")
        model = PPO.load(model_path)

        # Run evaluation: deterministic policy, no learning
        eval_metrics = evaluate_model(model, capacity, eval_files, deterministic=True)

        # Save detailed results to ppo/results/
        results_file = f'results/ppo_eval_capacity_{capacity}.json'
        with open(results_file, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"Saved results to {results_file}")

        # Generate and save plot to ppo/plots/
        plot_evaluation_results(eval_metrics, capacity, save_path=f'plots/ppo_evaluation_capacity_{capacity}.png')

    print('PPO evaluation complete.')


if __name__ == '__main__':
    main()
