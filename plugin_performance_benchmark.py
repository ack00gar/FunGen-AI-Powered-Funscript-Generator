#!/usr/bin/env python3
"""
Performance benchmark for funscript plugins with large datasets (300k+ points).

This script measures the performance of plugins on large funscript files to identify
optimization opportunities and measure improvements after vectorization.
"""

import time
import numpy as np
import json
import sys
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.base_plugin import plugin_registry
from funscript.plugins.plugin_loader import plugin_loader


class LargeFunscriptGenerator:
    """Generate large funscript files for performance testing."""
    
    @staticmethod
    def generate_large_funscript(num_points: int = 300000, duration_ms: int = 3600000) -> DualAxisFunscript:
        """
        Generate a large funscript with realistic data patterns.
        
        Args:
            num_points: Number of data points to generate
            duration_ms: Duration in milliseconds (default: 1 hour)
        """
        print(f"Generating funscript with {num_points:,} points over {duration_ms/1000:.0f} seconds...")
        
        # Generate realistic timestamps (non-uniform distribution)
        timestamps = np.sort(np.random.exponential(scale=duration_ms/num_points, size=num_points).cumsum())
        timestamps = (timestamps * duration_ms / timestamps[-1]).astype(int)
        
        # Generate realistic position data with various patterns
        t_normalized = np.linspace(0, 10*np.pi, num_points)
        
        # Mix of patterns: slow waves, fast oscillations, random noise
        slow_wave = 50 + 30 * np.sin(t_normalized * 0.1)
        fast_oscillation = 10 * np.sin(t_normalized * 2)
        random_noise = np.random.normal(0, 5, num_points)
        
        positions = slow_wave + fast_oscillation + random_noise
        positions = np.clip(positions, 0, 100).astype(int)
        
        # Create actions
        primary_actions = [
            {'at': int(timestamps[i]), 'pos': int(positions[i])}
            for i in range(num_points)
        ]
        
        # Generate secondary axis with different pattern
        secondary_positions = np.clip(50 + 20 * np.cos(t_normalized * 0.3) + np.random.normal(0, 3, num_points), 0, 100).astype(int)
        secondary_actions = [
            {'at': int(timestamps[i]), 'pos': int(secondary_positions[i])}
            for i in range(num_points)
        ]
        
        funscript = DualAxisFunscript()
        funscript.primary_actions = primary_actions
        funscript.secondary_actions = secondary_actions
        
        print(f"Generated funscript: {len(primary_actions):,} primary actions, {len(secondary_actions):,} secondary actions")
        return funscript


class PluginBenchmark:
    """Benchmark funscript plugins for performance analysis."""
    
    def __init__(self):
        self.results = {}
        
        # Load all plugins
        plugin_loader.load_builtin_plugins()
        
        # Get list of plugins to test
        self.plugins_to_test = [
            'Amplify',
            'Threshold Clamp', 
            'Clamp',
            'Invert',
            'Simplify (RDP)',
            'Smooth (SG)',
            'Speed Limiter',
            'Keyframes'
        ]
        
        print(f"Available plugins: {[p['name'] for p in plugin_registry.list_plugins()]}")
        print(f"Will benchmark: {self.plugins_to_test}")
    
    def create_test_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Create test parameters for each plugin."""
        return {
            'Amplify': {
                'scale_factor': 1.5,
                'center_value': 50
            },
            'Threshold Clamp': {
                'lower_threshold': 20,
                'upper_threshold': 80
            },
            'Clamp': {
                'clamp_value': 50
            },
            'Invert': {},  # No parameters needed
            'Simplify (RDP)': {
                'epsilon': 5.0
            },
            'Smooth (SG)': {
                'window_length': 7,
                'polyorder': 3
            },
            'Speed Limiter': {
                'min_interval_ms': 50,
                'speed_threshold': 300.0,
                'vibe_amount': 0
            },
            'Keyframes': {
                'position_tolerance': 10,
                'time_tolerance_ms': 50
            }
        }
    
    def benchmark_plugin(self, plugin_name: str, funscript: DualAxisFunscript, 
                        parameters: Dict[str, Any], num_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark a single plugin with multiple runs.
        
        Args:
            plugin_name: Name of the plugin to benchmark
            funscript: Funscript to test on
            parameters: Plugin parameters
            num_runs: Number of runs for averaging
        """
        plugin = plugin_registry.get_plugin(plugin_name)
        if not plugin:
            return {'error': f'Plugin {plugin_name} not found'}
        
        print(f"\nBenchmarking {plugin_name}...")
        
        # Check dependencies
        if not plugin.check_dependencies():
            return {'error': f'Plugin {plugin_name} dependencies not available'}
        
        times = []
        memory_usage = []
        
        for run in range(num_runs):
            # Create a deep copy for each run
            test_funscript = DualAxisFunscript()
            test_funscript.primary_actions = [action.copy() for action in funscript.primary_actions]
            test_funscript.secondary_actions = [action.copy() for action in funscript.secondary_actions]
            
            # Measure memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time the plugin execution
            start_time = time.perf_counter()
            
            try:
                plugin.transform(test_funscript, axis='both', **parameters)
                success = True
                error_msg = None
            except Exception as e:
                success = False
                error_msg = str(e)
                break
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before
            
            times.append(execution_time)
            memory_usage.append(memory_delta)
            
            print(f"  Run {run + 1}: {execution_time:.3f}s, Memory: {memory_delta:+.1f}MB")
        
        if not success:
            return {'error': error_msg}
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        avg_memory = np.mean(memory_usage)
        
        result = {
            'plugin_name': plugin_name,
            'success': True,
            'average_time_seconds': avg_time,
            'std_time_seconds': std_time,
            'min_time_seconds': min_time,
            'max_time_seconds': max_time,
            'average_memory_mb': avg_memory,
            'num_runs': num_runs,
            'points_processed': len(funscript.primary_actions),
            'points_per_second': len(funscript.primary_actions) / avg_time,
            'parameters': parameters
        }
        
        print(f"  Result: {avg_time:.3f}±{std_time:.3f}s, {result['points_per_second']:,.0f} points/sec")
        
        return result
    
    def run_full_benchmark(self, num_points: int = 300000) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print(f"\n{'='*60}")
        print(f"FUNSCRIPT PLUGIN PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        print(f"Testing with {num_points:,} data points")
        
        # Generate large funscript
        large_funscript = LargeFunscriptGenerator.generate_large_funscript(num_points)
        
        # Get test parameters
        test_parameters = self.create_test_parameters()
        
        # Benchmark each plugin
        benchmark_results = {}
        
        for plugin_name in self.plugins_to_test:
            if plugin_name not in test_parameters:
                print(f"Skipping {plugin_name} - no test parameters defined")
                continue
            
            parameters = test_parameters[plugin_name]
            result = self.benchmark_plugin(plugin_name, large_funscript, parameters)
            benchmark_results[plugin_name] = result
        
        # Create summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_points': num_points,
            'test_environment': {
                'python_version': sys.version,
                'numpy_version': np.__version__,
            },
            'results': benchmark_results
        }
        
        # Print summary
        self.print_benchmark_summary(summary)
        
        return summary
    
    def print_benchmark_summary(self, summary: Dict[str, Any]):
        """Print formatted benchmark summary."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Test date: {summary['timestamp']}")
        print(f"Data points: {summary['num_points']:,}")
        print(f"Python: {summary['test_environment']['python_version'].split()[0]}")
        print(f"NumPy: {summary['test_environment']['numpy_version']}")
        
        print(f"\n{'Plugin':<20} {'Time (s)':<12} {'Points/sec':<15} {'Memory (MB)':<12} {'Status'}")
        print(f"{'-'*70}")
        
        successful_results = []
        
        for plugin_name, result in summary['results'].items():
            if result.get('success', False):
                time_str = f"{result['average_time_seconds']:.3f}±{result['std_time_seconds']:.3f}"
                points_per_sec = f"{result['points_per_second']:,.0f}"
                memory_str = f"{result['average_memory_mb']:+.1f}"
                status = "✓"
                successful_results.append((plugin_name, result['average_time_seconds'], result['points_per_second']))
            else:
                time_str = "FAILED"
                points_per_sec = "-"
                memory_str = "-"
                status = "✗"
            
            print(f"{plugin_name:<20} {time_str:<12} {points_per_sec:<15} {memory_str:<12} {status}")
        
        # Performance ranking
        if successful_results:
            print(f"\n📊 PERFORMANCE RANKING (by speed):")
            successful_results.sort(key=lambda x: x[1])  # Sort by time (ascending)
            for i, (name, time_s, points_per_sec) in enumerate(successful_results, 1):
                print(f"  {i}. {name:<20} ({points_per_sec:,.0f} points/sec)")
    
    def save_results(self, summary: Dict[str, Any], filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"plugin_benchmark_{int(time.time())}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main benchmark execution."""
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during benchmarking
    
    # Create benchmark instance
    benchmark = PluginBenchmark()
    
    # Run benchmark with different data sizes
    test_sizes = [50000, 100000, 300000]
    
    for size in test_sizes:
        print(f"\n🚀 Running benchmark with {size:,} points...")
        summary = benchmark.run_full_benchmark(size)
        
        # Save results
        filename = f"benchmark_results_{size}_points.json"
        benchmark.save_results(summary, filename)
    
    print(f"\n✅ All benchmarks completed!")


if __name__ == "__main__":
    main()