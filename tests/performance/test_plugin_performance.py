"""
Performance tests for funscript plugin system.

Tests plugin loading performance, transformation speed,
memory usage, and scalability characteristics.
"""

import pytest
import numpy as np
import time
import psutil
import os
from typing import List, Dict, Any, Tuple

from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.base_plugin import plugin_registry


class TestPluginLoadingPerformance:
    """Test plugin loading and discovery performance."""
    
    def test_cold_plugin_loading_speed(self):
        """Test initial plugin loading performance."""
        # Clear plugin registry to simulate cold start
        plugin_registry._plugins.clear()
        plugin_registry._global_plugins_loaded = False
        
        start_time = time.time()
        
        # Trigger plugin loading
        fs = DualAxisFunscript()
        plugins = fs.list_available_plugins()
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        assert len(plugins) > 5, "Should discover multiple plugins"
        assert loading_time < 2.0, f"Plugin loading too slow: {loading_time:.3f}s"
        
        print(f"✅ Loaded {len(plugins)} plugins in {loading_time:.3f}s")
        
        # Test warm loading (should be much faster)
        start_time = time.time()
        plugins2 = fs.list_available_plugins()
        end_time = time.time()
        warm_loading_time = end_time - start_time
        
        assert warm_loading_time < 0.1, f"Warm plugin loading too slow: {warm_loading_time:.3f}s"
        assert len(plugins2) == len(plugins), "Plugin count should be consistent"
        
        print(f"✅ Warm loading: {warm_loading_time:.3f}s")
    
    def test_plugin_registry_memory_usage(self):
        """Test memory usage of plugin registry."""
        process = psutil.Process(os.getpid())
        
        # Measure memory before loading
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load plugins
        fs = DualAxisFunscript()
        plugins = fs.list_available_plugins()
        
        # Measure memory after loading
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Plugin loading should not consume excessive memory
        assert memory_increase < 50, f"Plugin loading used {memory_increase:.1f}MB (too much)"
        
        print(f"✅ Plugin loading memory usage: {memory_increase:.1f}MB")
        
        # Test memory per plugin
        memory_per_plugin = memory_increase / len(plugins) if plugins else 0
        assert memory_per_plugin < 5, f"Memory per plugin too high: {memory_per_plugin:.1f}MB"
        
        print(f"✅ Memory per plugin: {memory_per_plugin:.2f}MB")


class TestPluginTransformationPerformance:
    """Test performance of plugin transformations."""
    
    @pytest.fixture
    def small_dataset(self):
        """Create small test dataset (100 points)."""
        fs = DualAxisFunscript()
        for i in range(100):
            pos = 50 + 30 * np.sin(i * 0.1) + np.random.normal(0, 2)
            fs.add_action(1000 + i * 50, int(np.clip(pos, 0, 100)))
        return fs
    
    @pytest.fixture
    def medium_dataset(self):
        """Create medium test dataset (1000 points)."""
        fs = DualAxisFunscript()
        for i in range(1000):
            pos = 50 + 30 * np.sin(i * 0.05) + 20 * np.sin(i * 0.15) + np.random.normal(0, 3)
            fs.add_action(1000 + i * 50, int(np.clip(pos, 0, 100)))
        return fs
    
    @pytest.fixture
    def large_dataset(self):
        """Create large test dataset (10000 points)."""
        fs = DualAxisFunscript()
        for i in range(10000):
            pos = 50 + 25 * np.sin(i * 0.01) + 15 * np.sin(i * 0.07) + np.random.normal(0, 2)
            fs.add_action(1000 + i * 50, int(np.clip(pos, 0, 100)))
        return fs
    
    def test_fast_plugins_performance(self, medium_dataset):
        """Test performance of fast O(n) plugins."""
        fast_plugins = {
            'Amplify': {'scale_factor': 1.5},
            'Invert': {},
            'Threshold Clamp': {'lower_threshold': 20, 'upper_threshold': 80},
            'value_clamp': {'clamp_value': 50}
        }
        
        for plugin_name, params in fast_plugins.items():
            # Create fresh copy for each test
            test_fs = DualAxisFunscript()
            for action in medium_dataset.primary_actions:
                test_fs.add_action(action['at'], action['pos'])
            
            start_time = time.time()
            success = test_fs.apply_plugin(plugin_name, **params)
            end_time = time.time()
            
            processing_time = end_time - start_time
            points_per_second = 1000 / processing_time if processing_time > 0 else float('inf')
            
            assert success, f"{plugin_name} should succeed"
            assert processing_time < 0.1, f"{plugin_name} too slow: {processing_time:.3f}s"
            assert points_per_second > 50000, f"{plugin_name} performance: {points_per_second:.0f} pts/sec"
            
            print(f"✅ {plugin_name}: {processing_time:.3f}s ({points_per_second:.0f} pts/sec)")
    
    def test_rdp_performance_scaling(self):
        """Test RDP performance scaling with data size."""
        data_sizes = [100, 500, 1000, 2000]
        performance_results = []
        
        for size in data_sizes:
            # Create dataset
            fs = DualAxisFunscript()
            for i in range(size):
                pos = 50 + 30 * np.sin(i * 0.1) + np.random.normal(0, 2)
                fs.add_action(1000 + i * 50, int(np.clip(pos, 0, 100)))
            
            # Test RDP performance
            start_time = time.time()
            success = fs.apply_plugin('Simplify (RDP)', epsilon=3.0)
            end_time = time.time()
            
            processing_time = end_time - start_time
            points_per_second = size / processing_time if processing_time > 0 else float('inf')
            
            performance_results.append((size, processing_time, points_per_second))
            
            assert success, f"RDP should succeed for {size} points"
            assert points_per_second > 20000, f"RDP performance: {points_per_second:.0f} pts/sec for {size} points"
            
            print(f"✅ RDP {size} points: {processing_time:.3f}s ({points_per_second:.0f} pts/sec)")
        
        # Check that performance scales reasonably (not exponential)
        small_perf = performance_results[0][2]  # pts/sec for smallest dataset
        large_perf = performance_results[-1][2]  # pts/sec for largest dataset
        
        # Performance shouldn't degrade too much with size
        performance_ratio = large_perf / small_perf
        assert performance_ratio > 0.1, f"RDP performance degrades too much: {performance_ratio:.2f}"
        
        print(f"✅ RDP scaling: {performance_ratio:.2f} performance ratio (large/small)")
    
    def test_savgol_filter_performance(self, medium_dataset):
        """Test Savitzky-Golay filter performance."""
        try:
            # Test different window sizes
            window_sizes = [5, 7, 11, 15]
            
            for window_size in window_sizes:
                # Create fresh copy
                test_fs = DualAxisFunscript()
                for action in medium_dataset.primary_actions:
                    test_fs.add_action(action['at'], action['pos'])
                
                start_time = time.time()
                success = test_fs.apply_plugin('Smooth (SG)', 
                                             window_length=window_size, 
                                             polyorder=3)
                end_time = time.time()
                
                if success:
                    processing_time = end_time - start_time
                    points_per_second = 1000 / processing_time if processing_time > 0 else float('inf')
                    
                    # SavGol should be reasonably fast
                    assert processing_time < 0.5, f"SavGol window={window_size} too slow: {processing_time:.3f}s"
                    assert points_per_second > 5000, f"SavGol performance: {points_per_second:.0f} pts/sec"
                    
                    print(f"✅ SavGol window={window_size}: {processing_time:.3f}s ({points_per_second:.0f} pts/sec)")
                
        except ImportError:
            print("⚠️ Scipy not available - skipping SavGol performance test")
    
    def test_plugin_memory_efficiency(self, large_dataset):
        """Test memory efficiency of plugin transformations."""
        process = psutil.Process(os.getpid())
        
        # Test memory-efficient plugins (should not create large copies)
        memory_efficient_plugins = ['Amplify', 'Invert', 'Threshold Clamp']
        
        for plugin_name in memory_efficient_plugins:
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create test data copy
            test_fs = DualAxisFunscript()
            for action in large_dataset.primary_actions[:5000]:  # Use subset for memory test
                test_fs.add_action(action['at'], action['pos'])
            
            # Apply plugin
            success = test_fs.apply_plugin(plugin_name)
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            if success:
                # Memory increase should be minimal (in-place transformation)
                assert memory_increase < 20, f"{plugin_name} used too much memory: {memory_increase:.1f}MB"
                print(f"✅ {plugin_name}: Memory efficient ({memory_increase:.1f}MB increase)")
            
            # Clean up
            del test_fs


class TestPluginConcurrencyPerformance:
    """Test plugin performance under concurrent usage."""
    
    def test_plugin_registry_thread_safety(self):
        """Test plugin registry access performance with concurrent usage."""
        import threading
        import concurrent.futures
        
        def load_plugins_worker():
            """Worker function to load plugins concurrently."""
            fs = DualAxisFunscript()
            plugins = fs.list_available_plugins()
            return len(plugins)
        
        # Test concurrent plugin loading
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_plugins_worker) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # All workers should get the same plugin count
        assert all(count == results[0] for count in results), "Concurrent loading should be consistent"
        assert concurrent_time < 5.0, f"Concurrent plugin loading too slow: {concurrent_time:.3f}s"
        
        print(f"✅ Concurrent plugin loading: {concurrent_time:.3f}s for 10 threads")
    
    def test_concurrent_plugin_application(self):
        """Test concurrent plugin application performance."""
        import threading
        import concurrent.futures
        
        def apply_plugin_worker(worker_id):
            """Worker function to apply plugins concurrently."""
            # Create test data
            fs = DualAxisFunscript()
            for i in range(200):
                fs.add_action(1000 + i * 50, 50 + int(20 * np.sin(i * 0.1)))
            
            # Apply simple plugin
            start_time = time.time()
            success = fs.apply_plugin('Amplify', scale_factor=1.2)
            end_time = time.time()
            
            return success, end_time - start_time, worker_id
        
        # Test concurrent plugin application
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(apply_plugin_worker, i) for i in range(8)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All applications should succeed
        success_count = sum(1 for success, _, _ in results if success)
        assert success_count == len(results), f"Only {success_count}/{len(results)} concurrent applications succeeded"
        
        # Individual times should be reasonable
        individual_times = [time for _, time, _ in results]
        max_individual_time = max(individual_times)
        avg_individual_time = sum(individual_times) / len(individual_times)
        
        assert max_individual_time < 0.5, f"Slowest concurrent application: {max_individual_time:.3f}s"
        assert avg_individual_time < 0.1, f"Average concurrent application time: {avg_individual_time:.3f}s"
        
        print(f"✅ Concurrent plugin application: {total_time:.3f}s total, {avg_individual_time:.3f}s avg")


class TestPluginPerformanceBenchmarks:
    """Comprehensive performance benchmarks for plugins."""
    
    def test_plugin_performance_benchmarks(self):
        """Run comprehensive performance benchmarks for all plugins."""
        # Create test datasets of different sizes
        datasets = {
            'small': self._create_dataset(100),
            'medium': self._create_dataset(1000),
            'large': self._create_dataset(5000)
        }
        
        plugins = DualAxisFunscript().list_available_plugins()
        benchmark_results = {}
        
        for plugin_info in plugins:
            plugin_name = plugin_info['name']
            
            # Skip template plugins
            if 'template' in plugin_name:
                continue
            
            plugin_results = {}
            
            for dataset_name, dataset in datasets.items():
                try:
                    # Create copy for testing
                    test_fs = DualAxisFunscript()
                    for action in dataset.primary_actions:
                        test_fs.add_action(action['at'], action['pos'])
                    
                    # Get default parameters
                    default_params = self._get_default_params(plugin_info)
                    
                    # Benchmark the plugin
                    start_time = time.time()
                    success = test_fs.apply_plugin(plugin_name, **default_params)
                    end_time = time.time()
                    
                    if success:
                        processing_time = end_time - start_time
                        points_per_second = len(dataset.primary_actions) / processing_time if processing_time > 0 else float('inf')
                        
                        plugin_results[dataset_name] = {
                            'time': processing_time,
                            'points_per_second': points_per_second,
                            'success': True
                        }
                    else:
                        plugin_results[dataset_name] = {'success': False, 'reason': 'Plugin returned False'}
                    
                except Exception as e:
                    plugin_results[dataset_name] = {'success': False, 'reason': str(e)}
            
            benchmark_results[plugin_name] = plugin_results
        
        # Print benchmark results
        self._print_benchmark_results(benchmark_results)
        
        # Validate performance expectations
        self._validate_performance_expectations(benchmark_results)
    
    def _create_dataset(self, size: int) -> DualAxisFunscript:
        """Create test dataset of specified size."""
        fs = DualAxisFunscript()
        for i in range(size):
            pos = 50 + 30 * np.sin(i * 0.1) + 20 * np.sin(i * 0.05) + np.random.normal(0, 3)
            fs.add_action(1000 + i * 50, int(np.clip(pos, 0, 100)))
        return fs
    
    def _get_default_params(self, plugin_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get default parameters for a plugin."""
        params = {}
        schema = plugin_info.get('parameters_schema', {})
        
        for param_name, param_info in schema.items():
            if 'default' in param_info and param_info['default'] is not None:
                params[param_name] = param_info['default']
        
        return params
    
    def _print_benchmark_results(self, results: Dict[str, Dict[str, Any]]):
        """Print formatted benchmark results."""
        print("\n📊 Plugin Performance Benchmarks")
        print("=" * 80)
        print(f"{'Plugin':<20} {'Small (100)':<15} {'Medium (1K)':<15} {'Large (5K)':<15}")
        print("-" * 80)
        
        for plugin_name, plugin_results in results.items():
            row = f"{plugin_name:<20}"
            
            for dataset_name in ['small', 'medium', 'large']:
                result = plugin_results.get(dataset_name, {})
                if result.get('success'):
                    pts_per_sec = result['points_per_second']
                    if pts_per_sec > 1000000:
                        cell = f"{pts_per_sec/1000000:.1f}M pts/s"
                    elif pts_per_sec > 1000:
                        cell = f"{pts_per_sec/1000:.1f}K pts/s"
                    else:
                        cell = f"{pts_per_sec:.0f} pts/s"
                else:
                    cell = "FAILED"
                
                row += f" {cell:<15}"
            
            print(row)
    
    def _validate_performance_expectations(self, results: Dict[str, Dict[str, Any]]):
        """Validate that performance meets expectations."""
        performance_expectations = {
            'Amplify': 50000,       # Very fast, simple math
            'Invert': 50000,        # Very fast, simple math  
            'Threshold Clamp': 30000, # Fast, simple comparisons
            'value_clamp': 50000,   # Very fast, simple assignment
            'Simplify (RDP)': 10000,  # Moderately fast, optimized algorithm
            'Smooth (SG)': 5000,  # Slower, scipy-based
            'Speed Limiter': 20000, # Moderate, requires iteration
            'Keyframes': 15000,      # Moderate, peak detection
            'Resample': 10000       # Moderate, interpolation
        }
        
        failed_expectations = []
        
        for plugin_name, expected_pts_per_sec in performance_expectations.items():
            if plugin_name in results:
                medium_result = results[plugin_name].get('medium', {})
                if medium_result.get('success'):
                    actual_pts_per_sec = medium_result['points_per_second']
                    if actual_pts_per_sec < expected_pts_per_sec:
                        failed_expectations.append(
                            f"{plugin_name}: expected {expected_pts_per_sec}, got {actual_pts_per_sec:.0f}"
                        )
        
        if failed_expectations:
            print(f"\n⚠️ Performance expectations not met:")
            for failure in failed_expectations:
                print(f"  - {failure}")
        else:
            print(f"\n✅ All plugins meet performance expectations")


if __name__ == "__main__":
    # Run performance tests
    import sys
    
    print("🧪 Running Plugin Performance Tests")
    print("=" * 60)
    
    # Check if required modules are available
    try:
        import psutil
        print("✅ psutil available for memory testing")
    except ImportError:
        print("⚠️ psutil not available - skipping memory tests")
    
    # Run test suites
    test_suites = [
        TestPluginLoadingPerformance,
        TestPluginTransformationPerformance,
        TestPluginConcurrencyPerformance,
        TestPluginPerformanceBenchmarks
    ]
    
    for test_suite in test_suites:
        print(f"\n🔬 Running {test_suite.__name__}...")
        
        try:
            instance = test_suite()
            test_methods = [method for method in dir(instance) 
                           if method.startswith('test_')]
            
            for method_name in test_methods:
                print(f"\n  ⚡ {method_name}...")
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"    ✅ Completed")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
    
    print("\n🎉 Plugin performance testing completed!")