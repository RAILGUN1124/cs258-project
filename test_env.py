"""
Test script to verify RSA environment is working correctly
"""

import os
import glob
from rsaenv import RSAEnv
from nwutil import generate_sample_graph


def test_environment():
    """Test basic environment functionality"""
    print("Testing RSA Environment...")
    print("=" * 60)
    
    # Test 1: Environment creation
    print("\n1. Testing environment creation...")
    eval_files = sorted(glob.glob('data/eval/requests-*.csv'))
    if not eval_files:
        print("ERROR: No evaluation files found in data/eval/")
        return False
    
    request_file = eval_files[0]
    print(f"   Using file: {request_file}")
    
    try:
        env = RSAEnv(capacity=20, request_file=request_file)
        print("   ✓ Environment created successfully")
    except Exception as e:
        print(f"   ✗ Error creating environment: {e}")
        return False
    
    # Test 2: Reset
    print("\n2. Testing environment reset...")
    try:
        obs, info = env.reset()
        print(f"   ✓ Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Info: {info}")
    except Exception as e:
        print(f"   ✗ Error resetting: {e}")
        return False
    
    # Test 3: Step execution
    print("\n3. Testing environment step...")
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Step successful")
        print(f"   Action: {action}")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Info: {info}")
    except Exception as e:
        print(f"   ✗ Error stepping: {e}")
        return False
    
    # Test 4: Full episode
    print("\n4. Testing full episode...")
    try:
        env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"   ✓ Episode completed")
        print(f"   Steps: {steps}")
        print(f"   Total reward: {total_reward}")
        print(f"   Blocking rate: {info['blocking_rate']:.3f}")
        print(f"   Blocked: {info['blocked']}, Successful: {info['successful']}")
    except Exception as e:
        print(f"   ✗ Error during episode: {e}")
        return False
    
    # Test 5: Network graph
    print("\n5. Testing network graph...")
    try:
        graph = generate_sample_graph()
        print(f"   ✓ Graph created")
        print(f"   Nodes: {graph.number_of_nodes()}")
        print(f"   Edges: {graph.number_of_edges()}")
    except Exception as e:
        print(f"   ✗ Error with graph: {e}")
        return False
    
    # Test 6: Different capacities
    print("\n6. Testing different capacities...")
    for cap in [10, 20]:
        try:
            env_test = RSAEnv(capacity=cap, request_file=request_file)
            env_test.reset()
            print(f"   ✓ Capacity {cap} works")
        except Exception as e:
            print(f"   ✗ Error with capacity {cap}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


def test_path_definitions():
    """Test that all path definitions are correct"""
    print("\nTesting path definitions...")
    print("=" * 60)
    
    from rsaenv import RSAEnv
    
    paths = RSAEnv.PATHS
    print(f"\nDefined paths:")
    for (src, dst), path_list in paths.items():
        print(f"\n  Source {src} → Destination {dst}:")
        for i, path in enumerate(path_list, 1):
            print(f"    P{i}: {' → '.join(map(str, path))}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("RSA Environment Test Suite")
    print("=" * 60)
    
    # Run tests
    success = test_environment()
    test_path_definitions()
    
    if success:
        print("\n✓ Environment is ready for training!")
        print("\nNext steps:")
        print("  1. Run 'python dqn_runner.py' to train the models")
        print("  2. Run 'python evaluate.py' to evaluate trained models")
    else:
        print("\n✗ Environment tests failed. Please fix the errors above.")
