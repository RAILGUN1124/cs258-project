"""Network Utilities for Routing and Spectrum Allocation

This module provides core data structures for modeling optical network requests,
link states, and topology generation. These classes form the foundation of the
RSA simulation environment.
"""

import networkx as nx


class Request:
    """
    Represents a lightpath request in the optical network.
    
    Each request specifies:
    - source: Origin node where traffic enters the network
    - destination: Target node where traffic exits the network
    - holding_time: Duration (in time slots) the connection will remain active
    
    This is the fundamental unit of demand in the RSA problem. The agent must
    decide how to route and allocate wavelengths for each request.
    """
    def __init__(self, source: int, destination: int, holding_time: int):
        self.source = source
        self.destination = destination
        self.holding_time = holding_time

    def __repr__(self):
        return f"Request(src={self.source}, dst={self.destination}, hold={self.holding_time})"


class BaseLinkState:
    """
    Base class for link state information (DO NOT MODIFY).
    
    Maintains basic link properties:
    - endpoints: Tuple of (u, v) nodes, always sorted to ensure consistency
    - capacity: Total number of wavelength slots available on this link
    - utilization: Fraction of wavelengths currently in use (0.0 to 1.0)
    
    This base class should not be modified to maintain compatibility with
    the assignment requirements.
    """
    def __init__(self, u, v, capacity=20, utilization=0.0):
        if u > v: # Sort by node ID to ensure undirected edge consistency
            u, v = v, u
        self.endpoints = (u, v)
        self.capacity = capacity
        self.utilization = utilization

    def __repr__(self):
        return f"LinkState(capacity={self.capacity}, util={self.utilization})"

class LinkState(BaseLinkState):
    """ 
    Extended link state tracking wavelength-level allocation details.
    
    This class extends BaseLinkState to support fine-grained wavelength management,
    which is critical for the RSA problem. It tracks:
    
    Additional attributes:
    - wavelengths: Boolean array [False, True, False, ...] where:
        * False = wavelength is FREE and can be allocated
        * True = wavelength is OCCUPIED by an active lightpath
        * Index represents wavelength slot number (0 to capacity-1)
    
    - lightpaths: Dictionary mapping wavelength_idx -> (request_id, expiration_time)
        * Stores metadata about active connections on each wavelength
        * Used to release wavelengths when lightpaths expire
        * Key: wavelength index (int)
        * Value: tuple of (request_id, expiration_time in time slots)
    
    This detailed tracking enables:
    1. Wavelength continuity constraint checking (same wavelength on all links)
    2. Capacity constraint enforcement (max wavelengths per link)
    3. Conflict constraint prevention (no two lightpaths on same wavelength)
    4. Time-based automatic resource release
    """
    def __init__(self, u, v, capacity=20, utilization=0.0):
        super().__init__(u, v, capacity, utilization)
        # Track wavelength usage: True = occupied, False = free
        self.wavelengths = [False] * capacity
        # Store active lightpaths: wavelength_idx -> (request_id, expiration_time)
        self.lightpaths = {}
    
    def get_free_wavelengths(self):
        """
        Return indices of all free (unoccupied) wavelengths on this link.
        
        Critical for wavelength continuity constraint: the environment must
        find a wavelength that is free on ALL links in the selected path.
        
        Returns:
            List[int]: Indices of available wavelengths [0, 2, 5, ...]
        """
        return [i for i, occupied in enumerate(self.wavelengths) if not occupied]
    
    def allocate_wavelength(self, wavelength_idx, request_id, expiration_time):
        """
        Allocate a specific wavelength slot for a new lightpath.
        
        This method enforces the conflict constraint: a wavelength can only
        be allocated if it's currently free. Updates utilization metrics.
        
        Args:
            wavelength_idx: Which wavelength slot to allocate (0 to capacity-1)
            request_id: Unique identifier for this request (for tracking)
            expiration_time: Time step when this lightpath should be released
        
        Returns:
            bool: True if allocation succeeded, False if wavelength already occupied
        """
        if self.wavelengths[wavelength_idx]:
            return False  # Already occupied - conflict constraint violated
        self.wavelengths[wavelength_idx] = True
        self.lightpaths[wavelength_idx] = (request_id, expiration_time)
        self.utilization = sum(self.wavelengths) / self.capacity
        return True
    
    def release_wavelength(self, wavelength_idx):
        """
        Release a wavelength slot when its lightpath expires or is torn down.
        
        This is called automatically when a request's holding time expires,
        freeing up the wavelength for future requests. Critical for dynamic
        resource management.
        
        Args:
            wavelength_idx: Which wavelength slot to release
        """
        if wavelength_idx in self.lightpaths:
            del self.lightpaths[wavelength_idx]
        self.wavelengths[wavelength_idx] = False
        self.utilization = sum(self.wavelengths) / self.capacity
    
    def get_available_count(self):
        """
        Count how many wavelengths are currently available.
        
        Used in the observation space to inform the agent about resource
        availability. Helps the agent make informed routing decisions.
        
        Returns:
            int: Number of free wavelengths (0 to capacity)
        """
        return sum(1 for w in self.wavelengths if not w) 


def generate_sample_graph():
    """
    Generate the optical network topology for the RSA problem.
    
    Creates a 9-node network with specific topology:
    - Ring topology: Nodes 0-8 connected in a circle (9 links)
    - Additional cross-links: (1,7), (1,5), (3,6) for redundancy (3 links)
    - Total: 12 bidirectional links
    
    Each edge is annotated with a LinkState object that tracks wavelength
    allocation state. This topology provides multiple path options between
    source-destination pairs, enabling the RL agent to learn intelligent
    routing strategies.
    
    The ring + cross-links structure is common in optical networks as it
    provides:
    1. Multiple diverse paths for redundancy
    2. Load balancing opportunities
    3. Realistic constraints (limited path choices)
    
    Returns:
        networkx.Graph: Undirected graph with 9 nodes and 12 edges,
                        each edge has 'state' attribute (LinkState object)
    """
    # Create the sample graph
    G = nx.Graph()

    # Add 9 nodes numbered 0 through 8
    G.add_nodes_from(range(9))

    # Define links: ring links + extra links for path diversity
    # Ring: 0-1-2-3-4-5-6-7-8-0 (9 links)
    # Cross: (1,7), (1,5), (3,6) (3 links)
    links = [(n, (n + 1) % 9) for n in range(9)] + [(1, 7), (1, 5), (3, 6)]

    # Add edges with link state objects for tracking wavelength usage
    for u, v in links:
        G.add_edge(u, v, state=LinkState(u, v))
    return G