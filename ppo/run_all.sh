#!/usr/bin/env zsh
# Run training and evaluation for PPO. This script runs from inside the `ppo/` folder
# and keeps all outputs (models, plots, results, tensorboard logs) inside `ppo/`.

set -euo pipefail

# Move to the script directory (ppo/)
cd "$(dirname "$0")"

### Configuration and defaults
CAPACITIES="20,10"
TRIALS=${TRIALS:-50}
N_JOBS=${N_JOBS:--1}
N_TRAIN_EPISODES=${N_TRAIN_EPISODES:-200}
N_FINAL_EPISODES=${N_FINAL_EPISODES:-1000}
SKIP_TUNING=${SKIP_TUNING:-false}
TRAIN_FINAL=${TRAIN_FINAL:-true}
DATA_DIR=${DATA_DIR:-}

# Choose Python executable (prefer project venv if present)
PROJECT_ROOT="$(cd .. && pwd)"
VENV_PY="$PROJECT_ROOT/venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
	PYTHON_CMD="$VENV_PY"
else
	PYTHON_CMD="$(command -v python3 || command -v python)"
	if [[ -z "$PYTHON_CMD" ]]; then
		echo "No Python found on PATH. Please install python3 and re-run."
		exit 1
	fi
fi

echo "Starting PPO pipeline (working dir: $(pwd))"
echo "Using Python: $PYTHON_CMD"
echo "Capacities: $CAPACITIES  Trials: $TRIALS  Optuna jobs: $N_JOBS"

if [[ "$SKIP_TUNING" != "true" ]]; then
	echo "Running Optuna hyperparameter tuning (this can take a long time)"
	OPTUNA_CMD=(
		"$PYTHON_CMD"
		-u ppo_optuna_tuning.py
		--capacities "$CAPACITIES"
		--n-trials "$TRIALS"
		--n-training-episodes-per-trial "$N_TRAIN_EPISODES"
		--n-final-training-episodes "$N_FINAL_EPISODES"
		--n-jobs "$N_JOBS"
		--no-prompt
	)
	if [[ "$TRAIN_FINAL" == "true" ]]; then
		OPTUNA_CMD+=(--train-final)
	fi
	# add optional data directory if passed
	if [[ -n "$DATA_DIR" ]]; then
		OPTUNA_CMD+=(--data-dir "$DATA_DIR")
	fi

	# Run optuna (non-fatal) — portable across POSIX shells
	"${OPTUNA_CMD[@]}" || echo "Optuna tuning failed or was skipped"
else
	echo "Skipping Optuna hyperparameter tuning (SKIP_TUNING=$SKIP_TUNING)"
fi

echo "Now training standard PPO policies (ppo_runner)"
"$PYTHON_CMD" -u ppo_runner.py || echo "ppo_runner failed"

echo "Training finished — now running evaluation"
"$PYTHON_CMD" -u ppo_evaluate.py || echo "ppo_evaluate failed"

echo "PPO run complete. Models: ppo/models/ , Plots: ppo/plots/ , Results: ppo/results/"
