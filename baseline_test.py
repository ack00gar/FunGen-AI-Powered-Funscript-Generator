#!/usr/bin/env python
"""
Baseline performance test with original plugin code.
"""

import sys
import os
import time

# Add the project root to the path so we can import the plugins
sys.path.insert(0, '/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator')

from funscript.plugins.amplify_plugin import AmplifyPlugin
from funscript.dual_axis_funscript import DualAxisFunscript

def create_large_test_funscript(num_points=100000):
    """Create a large test funscript with sample data."""
    print(f"Creating test funscript with {num_points} points...")
    funscript = DualAxisFunscript()
    
    # Create sample data for primary axis
    for i in range(num_points):
        timestamp = i * 10  # 10ms intervals
        position = 50 + 30 * (i % 100) / 100  # Simple pattern
        funscript.add_action(timestamp, int(position), None)
    
    print(f"Created funscript with {len(funscript.primary_actions)} actions")
    return funscript

def benchmark_original():
    """Benchmark original plugin performance."""
    print("Benchmarking ORIGINAL (unoptimized) plugin performance...")
    
    # Create large test data
    funscript = create_large_test_funscript(100000)
    
    # Create plugin instance
    plugin = AmplifyPlugin()
    
    # Time the operation
    start_time = time.time()
    
    try:
        # Apply amplification
        result = plugin.transform(funscript, axis='primary', scale_factor=1.5, center_value=50)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"ORIGINAL plugin completed in {elapsed:.4f} seconds")
        print(f"Actions processed: {len(funscript.primary_actions)}")
        print(f"Result: {'Success' if result is None else 'Failed'}")
        return elapsed
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"ORIGINAL plugin failed after {elapsed:.4f} seconds with error: {e}")
        return None

def main():
    """Run baseline performance test."""
    print("Running baseline performance test with 100,000 points...\n")
    
    original_time = benchmark_original()
    
    if original_time is not None:
        print(f"\nBASELINE (ORIGINAL): {original_time:.4f} seconds")
    else:
        print("\nBASELINE test failed!")

if __name__ == "__main__":
    main()