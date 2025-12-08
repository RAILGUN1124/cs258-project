#!/usr/bin/env zsh
# Run training and evaluation for PPO. This script runs from inside the `ppo/` folder
# and keeps all outputs (models, plots, results, tensorboard logs) inside `ppo/`.

set -euo pipefail

# Move to the script directory (ppo/)
cd "$(dirname "$0")"

echo "Starting PPO training and evaluation (working dir: $(pwd))"

# Run training (this can take a long time). Adjust python invocation as needed.
python3 -u ppo_runner.py

echo "Training finished — now running evaluation"
python3 -u ppo_evaluate.py

echo "PPO run complete. Models: ppo/models/ , Plots: ppo/plots/ , Results: ppo/results/"
