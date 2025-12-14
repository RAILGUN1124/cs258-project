"""
Helpers to build a Gym-compatible env for PPO training.
This reuses the existing RSAEnv and adds a small wrapper that cycles through
multiple CSV request files. Each CSV corresponds to one episode (~100 requests).
Behavior is intentionally the same as DQN setup
Results for PPO and DQN are directly comparable.

Example:
    from ppo_env import create_env
    env = create_env(capacity=20, request_files=csv_files)

Note:
    When running code from inside ppo/, we add the project root to sys.path
    so rsaenv can be imported without changing directories.
"""

from typing import List
import os
import sys

# Make sure the project root is on the path.
# This matches how run scripts are typically executed from ppo/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3.common.monitor import Monitor
from rsaenv import RSAEnv


def create_env(capacity: int, request_files: List[str]):
    #Create an environment that cycles through request files.
    # One CSV file == one episode. Each reset() loads the next file.
    
    class MultiFileEnv(RSAEnv):
        # Thin wrapper around RSAEnv that swaps the request file on reset()
        def __init__(self, capacity, files):
            self.files = files
            self.file_idx = 0
            # Initialize using the first request file
            super().__init__(capacity=capacity, request_file=self.files[self.file_idx])

        def reset(self, seed=None, options=None):
            # Move to the next file (wrap around at the end)
            self.file_idx = (self.file_idx + 1) % len(self.files)
            # Load requests for this episode
            self.requests = self._load_requests(self.files[self.file_idx])
            # Let RSAEnv handle the actual reset logic
            return super().reset(seed=seed, options=options)

    # Build env and wrap with Monitor (SB3 can log episode stats)
    env = MultiFileEnv(capacity, request_files)
    env = Monitor(env)

    return env
