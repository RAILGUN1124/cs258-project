"""
Evaluation Script for Trained DQN Models

This script evaluates trained DQN agents on the evaluation dataset
and generates plots of blocking rates over episodes.
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
    Evaluate a trained model on evaluation dataset.
    
    Args:
        model: Trained DQN model
        capacity: Link capacity
        eval_files: List of evaluation request files
        deterministic: Whether to use deterministic policy
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_blocking_rates = []
    episode_rewards = []
    
    print(f"\nEvaluating on {len(eval_files)} episodes...")
    
    for i, req_file in enumerate(eval_files):
        # Create environment for this episode
        env = RSAEnv(capacity=capacity, request_file=req_file)
        
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # Record metrics
        episode_blocking_rates.append(info['blocking_rate'])
        episode_rewards.append(episode_reward)
        
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
    Plot evaluation results.
    
    Args:
        eval_metrics: Dictionary with evaluation metrics
        capacity: Link capacity used
        save_path: Path to save the plot
    """
    episodes = range(1, len(eval_metrics['episode_blocking_rates']) + 1)
    blocking_rates = eval_metrics['episode_blocking_rates']
    
    plt.figure(figsize=(10, 6))
    
    # Plot blocking rates
    plt.plot(episodes, blocking_rates, alpha=0.5, label='Episode Blocking Rate')
    
    # Plot moving average
    window = 10
    if len(blocking_rates) >= window:
        moving_avg = np.convolve(blocking_rates, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(episodes) + 1), moving_avg,
                label=f'Moving Avg (window={window})', linewidth=2, color='red')
    
    # Add mean line
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
    """Main evaluation function"""
    
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load evaluation files
    eval_files = load_eval_requests()
    print(f"Loaded {len(eval_files)} evaluation files")
    
    # Evaluate both models
    capacities = [20, 10]
    
    for capacity in capacities:
        print(f"\n{'='*70}")
        print(f"Evaluating Model with Capacity = {capacity}")
        print(f"{'='*70}")
        
        # Load trained model
        model_path = f'models/dqn_capacity_{capacity}.zip'
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using dqn_runner.py")
            continue
        
        print(f"Loading model from {model_path}")
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
