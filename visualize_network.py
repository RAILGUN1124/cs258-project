"""
Visualize the network topology used in the RSA problem.
"""

import matplotlib.pyplot as plt
import networkx as nx
from nwutil import generate_sample_graph


def visualize_network():
    """Create a visualization of the network topology"""
    
    # Generate the graph
    G = generate_sample_graph()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Layout 1: Circular layout
    pos1 = nx.circular_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos1, node_color='lightblue', 
                          node_size=800, ax=ax1)
    nx.draw_networkx_labels(G, pos1, font_size=14, font_weight='bold', ax=ax1)
    nx.draw_networkx_edges(G, pos1, width=2, alpha=0.6, ax=ax1)
    
    ax1.set_title('Network Topology - Circular Layout', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Layout 2: Spring layout for better visualization of connections
    pos2 = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G, pos2, node_color='lightcoral', 
                          node_size=800, ax=ax2)
    nx.draw_networkx_labels(G, pos2, font_size=14, font_weight='bold', ax=ax2)
    nx.draw_networkx_edges(G, pos2, width=2, alpha=0.6, ax=ax2)
    
    ax2.set_title('Network Topology - Spring Layout', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/network_topology.png', dpi=300, bbox_inches='tight')
    print("Network topology visualization saved to 'plots/network_topology.png'")
    
    plt.show()


def print_network_info():
    """Print detailed network information"""
    
    G = generate_sample_graph()
    
    print("\n" + "="*70)
    print("NETWORK TOPOLOGY INFORMATION")
    print("="*70)
    
    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    print("\n" + "-"*70)
    print("EDGE LIST:")
    print("-"*70)
    
    for i, (u, v, data) in enumerate(sorted(G.edges(data=True)), 1):
        print(f"{i:2d}. Node {u} ↔ Node {v}")
    
    print("\n" + "-"*70)
    print("SOURCE-DESTINATION PAIRS AND PATHS:")
    print("-"*70)
    
    from rsaenv import RSAEnv
    
    for (src, dst), paths in sorted(RSAEnv.PATHS.items()):
        print(f"\n{src} → {dst}:")
        for i, path in enumerate(paths, 1):
            path_str = " → ".join(map(str, path))
            print(f"  Path {i}: {path_str} ({len(path)-1} hops)")
    
    print("\n" + "="*70)


def print_path_statistics():
    """Print statistics about the predefined paths"""
    
    from rsaenv import RSAEnv
    
    print("\n" + "="*70)
    print("PATH STATISTICS")
    print("="*70)
    
    all_path_lengths = []
    
    for (src, dst), paths in sorted(RSAEnv.PATHS.items()):
        print(f"\n{src} → {dst}:")
        for i, path in enumerate(paths, 1):
            hops = len(path) - 1
            all_path_lengths.append(hops)
            print(f"  Path {i}: {hops} hops")
    
    print(f"\nOverall Statistics:")
    print(f"  Total paths: {len(all_path_lengths)}")
    print(f"  Min hops: {min(all_path_lengths)}")
    print(f"  Max hops: {max(all_path_lengths)}")
    print(f"  Avg hops: {sum(all_path_lengths)/len(all_path_lengths):.2f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)
    
    print_network_info()
    print_path_statistics()
    
    print("\nGenerating network topology visualization...")
    visualize_network()
