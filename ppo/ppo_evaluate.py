"""
Evaluate PPO agents on RSA environment.
Load CSV eval files from data/eval/
Loads saved PPO models from models/
Runs models deterministically & collects blocking rates & rewards
Save results and plots under results/ & plots/
"""

import os, glob, sys, json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# make sure project root is on path so rsaenv can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rsaenv import RSAEnv  # actual environment class

def load_eval_requests(data_dir='data/eval'):
    """Return sorted list of CSV eval files.
    Default: project_root/data/eval (if given path is relative)
    Raises FileNotFoundError if none found.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # default folder if None
    if data_dir is None: data_dir = os.path.join(project_root, 'data', 'eval')
    # relative paths: try folder under project root first
    elif not os.path.isabs(data_dir):
        candidate = os.path.join(project_root, data_dir)
        if os.path.exists(candidate): data_dir = candidate

    # glob all CSV files that start with 'requests-'
    files = sorted(glob.glob(os.path.join(data_dir, 'requests-*.csv')))
    if not files: raise FileNotFoundError(f"No evaluation files found in '{data_dir}'")
    return files


def evaluate_model(model, capacity, eval_files, deterministic=True):
    """Run PPO model on each CSV episode and collect blocking rate & reward.
    Arguments:
        model: PPO agent
        capacity: link capacity to use in the env
        eval_files: list of CSVs, each file = one episode
        deterministic: use greedy policy if True

    Returns: dict with episode metrics and mean/std blocking rate
    """
    # store per-episode data
    episode_blocking_rates = []
    episode_rewards = []

    print(f"\nEvaluating on {len(eval_files)} episodes...")
    for i, req_file in enumerate(eval_files):
        # create fresh env for each episode to reset link states
        env = RSAEnv(capacity=capacity, request_file=req_file)
        obs, info = env.reset() # Gym API returns (obs, info)
        done = False
        ep_reward = 0 # track cumulative reward

        # run one episode
        while not done:
            # get action from model, deterministic avoids exploration
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        # at episode end, env provides 'blocking_rate' in info dict
        episode_blocking_rates.append(info['blocking_rate'])
        episode_rewards.append(ep_reward)

        # simple progress feedback every 50 episodes
        if (i + 1) % 50 == 0: print(f"  Evaluated {i+1}/{len(eval_files)} episodes")

    # compute summary stats and return
    return {
        'episode_blocking_rates': episode_blocking_rates,
        'episode_rewards': episode_rewards,
        'mean_blocking_rate': float(np.mean(episode_blocking_rates)),
        'std_blocking_rate': float(np.std(episode_blocking_rates))
    }


def plot_evaluation_results(eval_metrics, capacity, save_path=None):
    """Plot episode blocking rates with moving average and mean line."""
    episodes = range(1, len(eval_metrics['episode_blocking_rates']) + 1)
    blocking_rates = eval_metrics['episode_blocking_rates']

    plt.figure(figsize=(10, 6))
    # raw episode blocking rates
    plt.plot(episodes, blocking_rates, alpha=0.5, label='Episode Blocking Rate')

    # moving average to smooth out short-term fluctuations
    window = 10
    if len(blocking_rates) >= window:
        mv = np.convolve(blocking_rates, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(episodes) + 1), mv, label=f'MA (w={window})', color='red')

    # mean blocking rate line
    mean_blocking = eval_metrics['mean_blocking_rate']
    plt.axhline(mean_blocking, color='green', linestyle='--', linewidth=2, label=f'Mean={mean_blocking:.4f}')
    plt.xlabel('Episode')
    plt.ylabel('Blocking Rate')
    plt.title(f'PPO Evaluation - Capacity {capacity}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    # ensure output folders exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # load eval files
    eval_files = load_eval_requests()
    print(f"Loaded {len(eval_files)} evaluation files")

    # evaluate models for both capacity configurations
    for capacity in [20, 10]:
        # expect model to be saved in models/ folder
        model_path = f'models/ppo_capacity_{capacity}.zip'
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping")
            continue
        print(f"Loading model {model_path}")
        model = PPO.load(model_path)

        # run evaluation
        metrics = evaluate_model(model, capacity, eval_files, deterministic=True)

        # save metrics as JSON for record keeping
        results_file = f'results/ppo_eval_capacity_{capacity}.json'
        with open(results_file, 'w') as f: json.dump(metrics, f, indent=2)
        print(f"Saved results to {results_file}")

        # also save plot
        plot_evaluation_results(metrics, capacity, save_path=f'plots/ppo_evaluation_capacity_{capacity}.png')
    print("PPO evaluation done.")

if __name__ == '__main__': main()