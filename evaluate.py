"""
Evaluation Script for Trained DQN Models

This script evaluates trained DQN agents on held-out test data to measure
real-world performance and generalization.

EVALUATION vs. TRAINING:
-----------------------
Key differences from training:
1. DETERMINISTIC POLICY: No exploration (epsilon=0), agent always chooses best action
2. NO LEARNING: Weights frozen, no gradient updates
3. UNSEEN DATA: Evaluation on different request files (tests generalization)
4. MULTIPLE RUNS: 1000 episodes for statistical confidence

WHY THIS MATTERS:
----------------
- Training performance can be misleading (agent may overfit)
- Evaluation shows true capability on novel scenarios
- Deterministic policy reveals what agent actually learned
- Large sample size (1000 episodes) gives robust statistics

OUTPUTS:
-------
- Evaluation metrics: JSON files with per-episode blocking rates
- Visualization: Plots showing blocking rate distribution
- Statistics: Mean, std, min, max blocking rates
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from rsaenv import RSAEnv
import json


def load_eval_requests(data_dir='data/eval'):
    """Load all evaluation request files"""
    files = sorted(glob.glob(os.path.join(data_dir, 'requests-*.csv')))
    return files


def evaluate_model(model, capacity, eval_files, deterministic=True):
    """
    Evaluate a trained model on held-out evaluation dataset.
    
    This function runs the trained DQN agent on unseen request sequences to
    measure its performance on novel scenarios. Unlike training:
    - No exploration: Agent uses greedy policy (argmax Q-values)
    - No learning: Model weights are frozen
    - Fresh environment: Each episode uses a different request file
    
    EVALUATION PROCESS:
    1. For each evaluation file:
       a. Create new environment with that file's requests
       b. Reset to initial state
       c. Run episode: agent selects actions, environment responds
       d. Record final blocking rate
    2. Aggregate statistics across all episodes
    
    WHY DETERMINISTIC=TRUE:
    - During training, agent explores (sometimes chooses suboptimal actions)
    - During evaluation, we want to see the learned policy's true performance
    - deterministic=True means: always choose action with highest Q-value
    - This is the policy the agent will use in deployment
    
    Args:
        model: Trained DQN model (loaded from .zip file)
        capacity: Link capacity (10 or 20 wavelengths)
        eval_files: List of paths to evaluation request files
        deterministic: If True, use greedy policy (recommended for evaluation)
    
    Returns:
        dict: Contains:
            - episode_blocking_rates: List of blocking rate per episode
            - episode_rewards: List of cumulative reward per episode
            - mean_blocking_rate: Average across all episodes
            - std_blocking_rate: Standard deviation (shows consistency)
    """
    # Storage for results across all episodes
    episode_blocking_rates = []  # Per-episode blocking percentage
    episode_rewards = []  # Per-episode cumulative reward
    
    print(f"\nEvaluating on {len(eval_files)} episodes...")
    
    # Iterate through all evaluation files (each file = one episode)
    for i, req_file in enumerate(eval_files):
        # Create fresh environment for this episode
        # Each file has different request sequence → tests generalization
        env = RSAEnv(capacity=capacity, request_file=req_file)
        
        # Reset environment to initial state (all wavelengths free)
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        # Run episode: process all requests in this file
        while not done:
            # Get action from trained model
            # deterministic=True → chooses action with max Q(s,a)
            # _states is for recurrent policies (not used in MLP)
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward  # Accumulate rewards
            done = terminated or truncated
        
        # Episode finished - record final statistics
        # info['blocking_rate'] is computed by environment: blocked / total requests
        episode_blocking_rates.append(info['blocking_rate'])
        episode_rewards.append(episode_reward)
        
        # Print progress every 50 episodes
        if (i + 1) % 50 == 0:
            avg_blocking = np.mean(episode_blocking_rates[-50:])
            print(f"  Evaluated {i+1}/{len(eval_files)} episodes, "
                  f"Avg Blocking Rate (last 50) = {avg_blocking:.4f}")
    
    return {
        'episode_blocking_rates': episode_blocking_rates,
        'episode_rewards': episode_rewards,
        'mean_blocking_rate': np.mean(episode_blocking_rates),
        'std_blocking_rate': np.std(episode_blocking_rates)
    }


def plot_evaluation_results(eval_metrics, capacity, save_path=None):
    """
    Visualize evaluation results to show performance distribution.
    
    Creates a plot showing:
    - Raw blocking rates per episode (scatter, shows variability)
    - Moving average (smoothed trend line)
    - Mean blocking rate (horizontal line, overall performance)
    
    WHY THIS VISUALIZATION:
    - Raw values show episode-to-episode variability (some request
      sequences are harder than others)
    - Moving average reveals if there are trends or patterns
    - Mean line shows overall expected performance
    - Spread of points indicates robustness/consistency
    
    INTERPRETING THE PLOT:
    - Flat trend = consistent performance across scenarios
    - Low variance = robust agent (handles diverse traffic well)
    - Points near zero = excellent performance
    - High points = challenging scenarios (agent struggles)
    
    Args:
        eval_metrics: Dictionary with 'episode_blocking_rates', 'mean_blocking_rate',
                     'std_blocking_rate'
        capacity: Link capacity (10 or 20, used in title)
        save_path: If provided, save plot to this file path
    """
    episodes = range(1, len(eval_metrics['episode_blocking_rates']) + 1)
    blocking_rates = eval_metrics['episode_blocking_rates']
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw blocking rates with transparency to show density
    # alpha=0.5 makes overlapping points visible
    plt.plot(episodes, blocking_rates, alpha=0.5, label='Episode Blocking Rate')
    
    # Plot moving average to show smoothed trend
    # Helps identify if performance improves/degrades over eval set
    # (Though order shouldn't matter since eval files are independent)
    window = 10
    if len(blocking_rates) >= window:
        # Convolve = sliding window average
        # mode='valid' excludes edges where window doesn't fully fit
        moving_avg = np.convolve(blocking_rates, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(episodes) + 1), moving_avg,
                label=f'Moving Avg (window={window})', linewidth=2, color='red')
    
    # Add horizontal line at mean blocking rate
    # This is the single-number summary of performance
    mean_blocking = eval_metrics['mean_blocking_rate']
    plt.axhline(y=mean_blocking, color='green', linestyle='--', linewidth=2,
               label=f'Mean = {mean_blocking:.4f}')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Blocking Rate (B)', fontsize=12)
    plt.title(f'Evaluation Results - Capacity {capacity}\n'
              f'Mean Blocking Rate: {mean_blocking:.4f} ± {eval_metrics["std_blocking_rate"]:.4f}',
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plot saved to {save_path}")
    
    plt.show()


def main():
    """
    Main evaluation workflow: load models, evaluate, generate reports.
    
    WORKFLOW:
    1. Load all evaluation request files (held-out test set)
    2. For each capacity configuration (20 and 10):
       a. Load trained model from disk
       b. Run evaluation on all test files (deterministic policy)
       c. Compute statistics (mean, std, min, max blocking)
       d. Save detailed results to JSON
       e. Generate visualization plots
    3. Produce comprehensive evaluation report
    
    This provides complete assessment of how well the agent generalizes
    to unseen traffic patterns.
    """
    
    # Create output directories if they don't exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load all evaluation request files from data/eval/
    # These are DIFFERENT from training files to test generalization
    eval_files = load_eval_requests()
    print(f"Loaded {len(eval_files)} evaluation files")
    
    # Evaluate both models
    capacities = [20, 10]
    
    for capacity in capacities:
        print(f"\n{'='*70}")
        print(f"Evaluating Model with Capacity = {capacity}")
        print(f"{'='*70}")
        
        # Load trained model from disk
        # Models are saved as .zip files containing:
        # - Neural network weights
        # - Hyperparameters
        # - Normalization statistics
        model_path = f'models/dqn_capacity_{capacity}.zip'
        
        # Check if model exists (must train first)
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using dqn_runner.py")
            continue  # Skip this capacity, try next one
        
        print(f"Loading model from {model_path}")
        # DQN.load() reconstructs the full model including policy network
        model = DQN.load(model_path)
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            capacity=capacity,
            eval_files=eval_files,
            deterministic=True
        )
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Evaluation Summary - Capacity {capacity}")
        print(f"{'='*70}")
        print(f"Episodes evaluated: {len(eval_metrics['episode_blocking_rates'])}")
        print(f"Mean Blocking Rate: {eval_metrics['mean_blocking_rate']:.4f}")
        print(f"Std Blocking Rate: {eval_metrics['std_blocking_rate']:.4f}")
        print(f"Min Blocking Rate: {min(eval_metrics['episode_blocking_rates']):.4f}")
        print(f"Max Blocking Rate: {max(eval_metrics['episode_blocking_rates']):.4f}")
        
        # Save results
        results_file = f'results/eval_capacity_{capacity}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'mean_blocking_rate': float(eval_metrics['mean_blocking_rate']),
                'std_blocking_rate': float(eval_metrics['std_blocking_rate']),
                'episode_blocking_rates': [float(b) for b in eval_metrics['episode_blocking_rates']],
                'episode_rewards': [float(r) for r in eval_metrics['episode_rewards']]
            }, f, indent=2)
        print(f"Results saved to {results_file}")
        
        # Plot results
        plot_evaluation_results(
            eval_metrics,
            capacity=capacity,
            save_path=f'plots/evaluation_capacity_{capacity}.png'
        )
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")
    print("All plots saved in 'plots/' directory")
    print("All results saved in 'results/' directory")


if __name__ == '__main__':
    main()
