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
    Custom Gym environment for Routing and Spectrum Allocation (RSA) problem.
    
    This environment models an optical communication network where an RL agent
    must learn to route requests and allocate wavelengths to minimize blocking.
    
    RL FORMULATION:
    ---------------
    State Space (35-dim):
        - Link utilizations (12 values): How busy each link is (0.0-1.0)
        - Available wavelengths per link (12 values): Normalized free capacity
        - Current request features (3 values): source, destination, holding_time
        - Path availability (8 binary values): Which paths have free wavelengths
    
    Action Space (8 discrete actions):
        - Each action corresponds to selecting one of 8 predefined paths
        - Actions 0-1: Paths for (0→3)
        - Actions 2-3: Paths for (0→4)
        - Actions 4-5: Paths for (7→3)
        - Actions 6-7: Paths for (7→4)
    
    Reward Function:
        - Sparse rewards: -1 for blocking (failure), 0 for success
        - Encourages agent to minimize blocking rate
        - Episode reward = -(number of blocked requests)
    
    CONSTRAINTS ENFORCED:
    --------------------
    1. Wavelength Continuity: Same wavelength must be used on ALL links in path
    2. Capacity Constraint: Each link has limited wavelengths (10 or 20)
    3. Conflict Constraint: No two lightpaths can share same wavelength on same link
    4. Time-based expiration: Lightpaths automatically release after holding_time
    
    WHY THIS MATTERS:
    ----------------
    RSA is NP-hard and critical for real optical networks. Traditional heuristics
    (shortest path, first-fit) often perform poorly. DQN can learn sophisticated
    strategies that balance path length, current utilization, and future demand.
    """
    
    metadata = {'render.modes': ['human']}
    
    # Predefined paths for each source-destination pair
    # 
    # Using predefined paths (vs. on-the-fly computation) is common in optical
    # networks because:
    # 1. Reduces computational overhead during routing
    # 2. Paths can be pre-engineered for quality (e.g., avoid congestion-prone links)
    # 3. Simplifies the action space for RL (8 discrete actions vs. complex path search)
    # 4. Mirrors real-world practice: ISPs often use k-shortest paths
    #
    # For each (source, destination) pair, we provide 2 alternate paths:
    # - Usually one shorter path (lower resource usage)
    # - One longer path (provides alternate route, better load balancing)
    PATHS = {
        (0, 3): [
            [0, 1, 2, 3],          # P1: 3 hops (shorter)
            [0, 8, 7, 6, 3]        # P2: 4 hops (longer, alternate)
        ],
        (0, 4): [
            [0, 1, 5, 4],          # P3: 3 hops (shorter)
            [0, 8, 7, 6, 3, 4]     # P4: 5 hops (longer, diverse route)
        ],
        (7, 3): [
            [7, 1, 2, 3],          # P5: 3 hops
            [7, 6, 3]              # P6: 2 hops (shortest path for this pair!)
        ],
        (7, 4): [
            [7, 1, 5, 4],          # P7: 3 hops
            [7, 6, 3, 4]           # P8: 3 hops (via different intermediate nodes)
        ]
    }
    
    def __init__(self, capacity: int = 20, request_file: Optional[str] = None, 
                 requests: Optional[List[Request]] = None):
        """
        Initialize the RSA environment with network topology and request sequence.
        
        Args:
            capacity: Number of wavelengths per link (typically 10 or 20)
                     Lower capacity = harder problem with more blocking
            request_file: Path to CSV file with request sequence
                         CSV format: source,destination,holding_time
            requests: List of Request objects (alternative to request_file)
                     Used for programmatic environment creation
        
        The initialization:
        1. Creates network topology (9 nodes, 12 links)
        2. Configures wavelength capacity for each link
        3. Loads the sequence of requests to be served
        4. Defines observation and action spaces for RL
        """
        super(RSAEnv, self).__init__()
        
        self.capacity = capacity
        self.graph = generate_sample_graph()
        
        # Initialize all links with specified capacity and empty wavelength state
        # This ensures consistent starting state across episodes
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
        # Discrete action space is ideal for DQN, which learns Q(s,a) for each action
        self.action_space = spaces.Discrete(8)
        
        # Observation space (35 dimensions): Carefully designed to give agent
        # all information needed to make intelligent routing decisions
        # 
        # Component breakdown:
        # 1. Link utilizations (12 values): Shows current network load
        #    - Helps agent avoid congested paths
        #    - Values in [0.0, 1.0] where 1.0 = fully utilized
        # 
        # 2. Available wavelengths per link (12 values): Remaining capacity
        #    - Normalized by capacity: available/capacity
        #    - Indicates how "tight" resources are on each link
        # 
        # 3. Current request features (3 values): What needs to be routed
        #    - source/8, destination/8 (normalized node IDs)
        #    - holding_time/100 (normalized duration)
        #    - Tells agent the demand characteristics
        # 
        # 4. Path availability (8 binary values): Feasibility indicators
        #    - 1.0 if path has at least one free wavelength, 0.0 if blocked
        #    - Prevents agent from selecting infeasible paths
        #    - Speeds up learning by filtering invalid actions
        obs_dim = 12 + 12 + 3 + 8  # Total: 35 dimensions
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
        Execute one time step in the environment (core RL training loop).
        
        This is called once per request. The agent observes state, chooses an
        action (path), and receives reward based on success/failure.
        
        Execution order is critical:
        1. Release expired lightpaths (time progresses)
        2. Attempt to allocate current request on selected path
        3. Compute reward (-1 if blocked, 0 if successful)
        4. Advance to next request
        5. Return new observation and metadata
        
        Args:
            action: Index of path to use (0-7)
                   Agent's decision about which route to take
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: New state after action (35-dim array)
                - reward: -1.0 if blocked, 0.0 if successful
                - terminated: True if all requests processed
                - truncated: Always False (no time limits)
                - info: Dict with episode statistics
        """
        # Process expired lightpaths first (free up resources before new allocation)
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
        """
        Release lightpaths whose holding time has expired.
        
        This method is critical for dynamic resource management. Unlike static
        allocation, optical networks have time-varying demand:
        - When holding_time expires, wavelengths are automatically released
        - Released wavelengths become available for new requests
        - Simulates realistic network behavior (calls end, bandwidth is freed)
        
        Called at the START of each step (before allocating new request) to
        ensure state is up-to-date. This temporal ordering matters:
        1. Release expired resources
        2. Update network state
        3. Attempt to allocate new request
        
        Why this is important for learning:
        - Agent must consider holding_time when making decisions
        - Short-lived requests free resources quickly (lower risk)
        - Long-lived requests block resources longer (higher opportunity cost)
        - DQN learns to balance these trade-offs
        """
        current_time = self.current_step
        expired_ids = []
        
        # Find all lightpaths that have reached their expiration time
        for req_id, (path, wavelength, expiration_time) in self.active_lightpaths.items():
            if current_time >= expiration_time:
                expired_ids.append(req_id)
                # Release wavelength on ALL links in the path (maintains consistency)
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    link_state = self._get_link_state(u, v)
                    link_state.release_wavelength(wavelength)
        
        # Remove from tracking dictionary
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
        Find a wavelength available on ALL links in the path (continuity constraint).
        
        This is THE CORE of the RSA problem! Due to wavelength continuity constraint,
        we cannot convert wavelengths at intermediate nodes. Therefore, the SAME
        wavelength must be free on EVERY link in the path.
        
        Algorithm (First-Fit):
        1. For each link in path, get set of free wavelengths
        2. Compute intersection: wavelengths free on ALL links
        3. Return smallest index (first-fit heuristic)
        4. Return None if intersection is empty (→ BLOCKING)
        
        Example:
        Path: [0, 1, 2, 3]
        Link (0,1) free: {0, 2, 5, 7}
        Link (1,2) free: {0, 3, 5, 9}
        Link (2,3) free: {0, 1, 5, 8}
        Intersection: {0, 5} → Returns 0 (first-fit)
        
        If no wavelength is common to all links → blocking occurs!
        
        Args:
            path: List of node indices forming the path [n1, n2, n3, ...]
            
        Returns:
            int: Wavelength index if available, None if request must be blocked
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
        Construct the observation vector that DQN uses to make decisions.
        
        This is a carefully designed state representation that balances:
        - Completeness: Contains all info needed for optimal decisions
        - Compactness: Small enough for neural network to learn efficiently
        - Normalization: All values in [0, 1] for stable learning
        
        The observation gives the agent both GLOBAL information (network state)
        and LOCAL information (current request), enabling it to learn policies
        that consider both immediate and future impacts.
        
        Returns:
            np.ndarray: Observation vector of shape (35,), dtype float32
        """
        obs = []
        
        # PART 1: Link utilizations (12 values)
        # Shows which links are congested. Helps agent learn to avoid overloaded paths.
        # Sorted order ensures consistent observation structure across episodes.
        for u, v, data in sorted(self.graph.edges(data=True)):
            obs.append(data['state'].utilization)
        
        # PART 2: Available wavelengths per link (12 values, normalized)
        # Indicates remaining capacity on each link. Crucial for predicting whether
        # future requests can be served. Normalized by capacity for scale-invariance.
        for u, v, data in sorted(self.graph.edges(data=True)):
            available = data['state'].get_available_count()
            obs.append(available / self.capacity)  # Normalize to [0, 1]
        
        # PART 3: Current request features (3 values)
        # Tells agent WHAT needs to be routed. The holding_time is particularly
        # important: long-lived requests lock up resources, so agent may prefer
        # longer paths for them (to preserve shorter paths for future requests).
        if self.current_request:
            obs.append(self.current_request.source / 8.0)  # Normalize node ID
            obs.append(self.current_request.destination / 8.0)
            obs.append(min(self.current_request.holding_time / 100.0, 1.0))  # Cap at 1.0
        else:
            obs.extend([0.0, 0.0, 0.0])  # Episode ended, use zeros
        
        # PART 4: Path availability mask (8 binary values)
        # CRITICAL for learning efficiency! Tells agent which actions are valid.
        # - 1.0 = path has free wavelength (action will succeed if chosen)
        # - 0.0 = path has no free wavelength (action will cause blocking)
        # This dramatically speeds up learning by reducing exploration of bad actions.
        for action in range(8):
            path = self._get_path_for_action(action)
            if path is not None:
                # Check if at least one wavelength is available on entire path
                available_wl = self._find_available_wavelength(path)
                obs.append(1.0 if available_wl is not None else 0.0)
            else:
                obs.append(0.0)  # Invalid path for current request's src-dst pair
        
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
