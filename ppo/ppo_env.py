"""
Utilities to create a Gym-compatible environment for PPO training.

This module reuses the project's existing RSAEnv (defined at the project
root in rsaenv.py) and provides a MultiFileEnv wrapper that cycles
through CSV request files. Each CSV file represents one episode (about 100
requests). The wrapper is Monitor-wrapped so Stable-Baselines3 logging and
callbacks work as expected.

The wrapper intentionally keeps behavior identical to the DQN training
pipeline so PPO experiments are comparable to previously run DQN agents.

Usage:
    from ppo_env import create_env
    env = create_env(capacity=20, request_files=list_of_csv_paths)

Note: When running scripts from inside ppo/, the parent project root is
added to Python's import path so we can import rsaenv without changing
working directories.
"""

from typing import List
import os
import sys

# Add the project root (parent of this file) to sys.path so imports like
# from rsaenv import RSAEnv work even when the current working directory
# is .../cs258-project/ppo (this matches how run_all.sh executes).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Monitor wrapper from stable-baselines3 to capture episode info
from stable_baselines3.common.monitor import Monitor

# Import the RSA environment defined at project root
from rsaenv import RSAEnv


def create_env(capacity: int, request_files: List[str]):
    """Create a Monitor-wrapped environment that cycles through request files.

    Args:
        capacity: Number of wavelengths per link for the RSA environment.
        request_files: List of CSV file paths; each file is one episode.

    Returns:
        A Monitor-wrapped environment instance suitable for Stable-Baselines3.
    """

    # Define a small wrapper class that inherits RSAEnv and overrides reset()
    # so each new episode loads the next CSV file from the provided list.
    class MultiFileEnv(RSAEnv):
        """Environment wrapper that cycles through multiple request files.

        This wrapper preserves the public Gym API of RSAEnv but ensures that
        training sees diverse episodes by loading a different request file
        each time reset() is called.
        """

        def __init__(self, capacity, files):
            # Store file list and initialize file index
            self.files = files
            self.file_idx = 0
            # Initialize parent RSAEnv using the first file
            super().__init__(capacity=capacity, request_file=self.files[self.file_idx])

        def reset(self, seed=None, options=None):
            # Advance to next file (circular indexing) before resetting
            self.file_idx = (self.file_idx + 1) % len(self.files)
            # Load requests from the selected CSV file into the env
            self.requests = self._load_requests(self.files[self.file_idx])
            # Call parent reset to reinitialize link states and counters
            return super().reset(seed=seed, options=options)

    # Instantiate the wrapper and wrap with Monitor for logging
    env = MultiFileEnv(capacity, request_files)
    env = Monitor(env)
    return env



def create_env(capacity: int, request_files: List[str]):
    """Create a Monitor-wrapped environment that cycles through request files.

    Args:
        capacity: Link capacity for the RSAEnv
        request_files: List of CSV request file paths (each file = one episode)

    Returns:
        Monitor-wrapped environment instance
    """
    class MultiFileEnv(RSAEnv):
        def __init__(self, capacity, files):
            self.files = files
            self.file_idx = 0
            super().__init__(capacity=capacity, request_file=self.files[self.file_idx])

        def reset(self, seed=None, options=None):
            # Advance to next file (circular)
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.requests = self._load_requests(self.files[self.file_idx])
            return super().reset(seed=seed, options=options)

    env = MultiFileEnv(capacity, request_files)
    env = Monitor(env)
    return env
