import networkx as nx

class Request:
    def __init__(self, source: int, destination: int, holding_time: int):
        self.source = source
        self.destination = destination
        self.holding_time = holding_time

    def __repr__(self):
        return f"Request(src={self.source}, dst={self.destination}, hold={self.holding_time})"


class BaseLinkState:
    def __init__(self, u, v, capacity=20, utilization=0.0):
        if u > v: # sort by the node ID
            u, v = v, u
        self.endpoints = (u, v)
        self.capacity = capacity
        self.utilization = utilization

    def __repr__(self):
        return f"LinkState(capacity={self.capacity}, util={self.utilization})"

class LinkState(BaseLinkState):
    """ 
    Data structure to store the link state.
    You can extend this class to add more attributes if needed.
    Do not change the BaseLinkState class.
    
    Additional attributes:
    - wavelengths: list of booleans indicating whether each wavelength slot is occupied
    - lightpaths: dict mapping (wavelength_idx) to (request_id, expiration_time)
    """
    def __init__(self, u, v, capacity=20, utilization=0.0):
        super().__init__(u, v, capacity, utilization)
        # Track wavelength usage: True = occupied, False = free
        self.wavelengths = [False] * capacity
        # Store active lightpaths: wavelength_idx -> (request_id, expiration_time)
        self.lightpaths = {}
    
    def get_free_wavelengths(self):
        """Return list of free wavelength indices"""
        return [i for i, occupied in enumerate(self.wavelengths) if not occupied]
    
    def allocate_wavelength(self, wavelength_idx, request_id, expiration_time):
        """Allocate a wavelength for a lightpath"""
        if self.wavelengths[wavelength_idx]:
            return False  # Already occupied
        self.wavelengths[wavelength_idx] = True
        self.lightpaths[wavelength_idx] = (request_id, expiration_time)
        self.utilization = sum(self.wavelengths) / self.capacity
        return True
    
    def release_wavelength(self, wavelength_idx):
        """Release a wavelength"""
        if wavelength_idx in self.lightpaths:
            del self.lightpaths[wavelength_idx]
        self.wavelengths[wavelength_idx] = False
        self.utilization = sum(self.wavelengths) / self.capacity
    
    def get_available_count(self):
        """Get number of available wavelengths"""
        return sum(1 for w in self.wavelengths if not w) 


def generate_sample_graph():
    # Create the sample graph
    G = nx.Graph()

    G.add_nodes_from(range(9))

    # Define links: ring links + extra links
    links = [(n, (n + 1) % 9) for n in range(9)] + [(1, 7), (1, 5), (3, 6)]

    # Add edges with link state objects
    for u, v in links:
        G.add_edge(u, v, state=LinkState(u, v))
    return G