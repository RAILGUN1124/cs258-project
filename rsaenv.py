"""
Routing and Spectrum Allocation (RSA) Environment for Reinforcement Learning

This custom Gym environment simulates an optical network where requests must be
allocated paths and wavelengths while satisfying wavelength continuity, capacity,
and conflict constraints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import networkx as nx
from nwutil import generate_sample_graph, Request
from typing import List, Tuple, Optional
import csv


class RSAEnv(gym.Env):
    """
    Custom Gym environment for Routing and Spectrum Allocation problem.
    
    State: Network state including link utilizations and current request
    Action: Select one of 8 predefined paths
    Reward: -1 for blocked requests, 0 for successful allocation
    """
    
    metadata = {'render.modes': ['human']}
    
    # Predefined paths for each source-destination pair
    PATHS = {
        (0, 3): [
            [0, 1, 2, 3],          # P1
            [0, 8, 7, 6, 3]        # P2
        ],
        (0, 4): [
            [0, 1, 5, 4],          # P3
            [0, 8, 7, 6, 3, 4]     # P4
        ],
        (7, 3): [
            [7, 1, 2, 3],          # P5
            [7, 6, 3]              # P6
        ],
        (7, 4): [
            [7, 1, 5, 4],          # P7
            [7, 6, 3, 4]           # P8
        ]
    }
    
    def __init__(self, capacity: int = 20, request_file: Optional[str] = None, 
                 requests: Optional[List[Request]] = None):
        """
        Initialize the RSA environment.
        
        Args:
            capacity: Number of wavelengths per link
            request_file: Path to CSV file with requests
            requests: List of Request objects (alternative to request_file)
        """
        super(RSAEnv, self).__init__()
        
        self.capacity = capacity
        self.graph = generate_sample_graph()
        
        # Update link capacities
        for u, v, data in self.graph.edges(data=True):
            data['state'].capacity = capacity
            data['state'].wavelengths = [False] * capacity
            data['state'].lightpaths = {}
            data['state'].utilization = 0.0
        
        # Load requests
        if request_file:
            self.requests = self._load_requests(request_file)
        elif requests:
            self.requests = requests
        else:
            raise ValueError("Must provide either request_file or requests")
        
        # Action space: 8 possible paths (indexed 0-7)
        self.action_space = spaces.Discrete(8)
        
        # Observation space: 
        # - Link utilizations (12 links)
        # - Available wavelengths per link (12 links)
        # - Current request (source, destination, holding_time normalized)
        # - Path availability for each of 8 paths (binary)
        obs_dim = 12 + 12 + 3 + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.current_request = None
        self.blocked_count = 0
        self.successful_count = 0
        self.request_id_counter = 0
        
        # Track active lightpaths for expiration
        self.active_lightpaths = {}  # request_id -> (path, wavelength, expiration_time)
        
    def _load_requests(self, filepath: str) -> List[Request]:
        """Load requests from CSV file"""
        requests = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                req = Request(
                    source=int(row['source']),
                    destination=int(row['destination']),
                    holding_time=int(row['holding_time'])
                )
                requests.append(req)
        return requests
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset graph state
        for u, v, data in self.graph.edges(data=True):
            state = data['state']
            state.wavelengths = [False] * self.capacity
            state.lightpaths = {}
            state.utilization = 0.0
        
        # Reset counters
        self.current_step = 0
        self.blocked_count = 0
        self.successful_count = 0
        self.request_id_counter = 0
        self.active_lightpaths = {}
        
        # Load first request
        if len(self.requests) > 0:
            self.current_request = self.requests[0]
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int):
        """
        Execute one time step.
        
        Args:
            action: Index of path to use (0-7)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Process expired lightpaths first
        self._process_expirations()
        
        # Try to allocate the request on the selected path
        blocked = not self._allocate_request(action)
        
        # Calculate reward
        if blocked:
            reward = -1.0
            self.blocked_count += 1
        else:
            reward = 0.0
            self.successful_count += 1
        
        # Move to next request
        self.current_step += 1
        terminated = self.current_step >= len(self.requests)
        
        if not terminated:
            self.current_request = self.requests[self.current_step]
        
        obs = self._get_observation()
        info = self._get_info()
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def _process_expirations(self):
        """Release lightpaths that have expired"""
        current_time = self.current_step
        expired_ids = []
        
        for req_id, (path, wavelength, expiration_time) in self.active_lightpaths.items():
            if current_time >= expiration_time:
                expired_ids.append(req_id)
                # Release wavelength on all links in the path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    link_state = self._get_link_state(u, v)
                    link_state.release_wavelength(wavelength)
        
        # Remove expired lightpaths
        for req_id in expired_ids:
            del self.active_lightpaths[req_id]
    
    def _allocate_request(self, action: int) -> bool:
        """
        Try to allocate current request using the selected path.
        
        Args:
            action: Path index (0-7)
            
        Returns:
            True if allocation successful, False if blocked
        """
        if self.current_request is None:
            return False
        
        # Map action to path
        path = self._get_path_for_action(action)
        if path is None:
            return False  # Invalid path for this request
        
        # Check wavelength continuity constraint
        # Find a wavelength available on ALL links in the path
        available_wavelength = self._find_available_wavelength(path)
        
        if available_wavelength is None:
            return False  # No wavelength available - BLOCKED
        
        # Allocate the wavelength on all links
        request_id = self.request_id_counter
        self.request_id_counter += 1
        expiration_time = self.current_step + self.current_request.holding_time
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = self._get_link_state(u, v)
            link_state.allocate_wavelength(available_wavelength, request_id, expiration_time)
        
        # Track the lightpath
        self.active_lightpaths[request_id] = (path, available_wavelength, expiration_time)
        
        return True
    
    def _find_available_wavelength(self, path: List[int]) -> Optional[int]:
        """
        Find the smallest wavelength index available on all links in path.
        
        Args:
            path: List of node indices forming the path
            
        Returns:
            Wavelength index or None if no wavelength available
        """
        # Get available wavelengths for each link
        available_sets = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = self._get_link_state(u, v)
            available = set(link_state.get_free_wavelengths())
            available_sets.append(available)
        
        # Find intersection (wavelengths available on ALL links)
        if not available_sets:
            return None
        
        common_available = available_sets[0]
        for avail_set in available_sets[1:]:
            common_available = common_available.intersection(avail_set)
        
        if not common_available:
            return None
        
        # Return smallest index (first-fit)
        return min(common_available)
    
    def _get_path_for_action(self, action: int) -> Optional[List[int]]:
        """
        Map action index to actual path based on current request.
        
        Args:
            action: Action index (0-7)
            
        Returns:
            Path as list of nodes, or None if invalid
        """
        if self.current_request is None:
            return None
        
        src = self.current_request.source
        dst = self.current_request.destination
        
        # Map global action index to path
        if (src, dst) == (0, 3):
            if action < 2:
                return self.PATHS[(0, 3)][action]
        elif (src, dst) == (0, 4):
            if 2 <= action < 4:
                return self.PATHS[(0, 4)][action - 2]
        elif (src, dst) == (7, 3):
            if 4 <= action < 6:
                return self.PATHS[(7, 3)][action - 4]
        elif (src, dst) == (7, 4):
            if 6 <= action < 8:
                return self.PATHS[(7, 4)][action - 6]
        
        return None
    
    def _get_link_state(self, u: int, v: int):
        """Get link state, handling undirected edges"""
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['state']
        elif self.graph.has_edge(v, u):
            return self.graph[v][u]['state']
        else:
            raise ValueError(f"No edge between {u} and {v}")
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            Numpy array of shape (obs_dim,)
        """
        obs = []
        
        # Link utilizations (12 links)
        for u, v, data in sorted(self.graph.edges(data=True)):
            obs.append(data['state'].utilization)
        
        # Available wavelengths per link (normalized, 12 links)
        for u, v, data in sorted(self.graph.edges(data=True)):
            available = data['state'].get_available_count()
            obs.append(available / self.capacity)
        
        # Current request features (source, destination, holding_time normalized)
        if self.current_request:
            obs.append(self.current_request.source / 8.0)  # Normalize by max node id
            obs.append(self.current_request.destination / 8.0)
            obs.append(min(self.current_request.holding_time / 100.0, 1.0))  # Normalize
        else:
            obs.extend([0.0, 0.0, 0.0])
        
        # Path availability (8 paths, binary)
        for action in range(8):
            path = self._get_path_for_action(action)
            if path is not None:
                # Check if at least one wavelength is available
                available_wl = self._find_available_wavelength(path)
                obs.append(1.0 if available_wl is not None else 0.0)
            else:
                obs.append(0.0)  # Invalid path for current request
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> dict:
        """Return additional information"""
        total = self.blocked_count + self.successful_count
        blocking_rate = self.blocked_count / total if total > 0 else 0.0
        
        return {
            'step': self.current_step,
            'blocked': self.blocked_count,
            'successful': self.successful_count,
            'blocking_rate': blocking_rate,
            'active_lightpaths': len(self.active_lightpaths)
        }
    
    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            if self.current_request:
                print(f"Current Request: {self.current_request}")
            print(f"Blocked: {self.blocked_count}, Successful: {self.successful_count}")
            print(f"Active lightpaths: {len(self.active_lightpaths)}")
            print(f"Blocking rate: {self._get_info()['blocking_rate']:.3f}")
