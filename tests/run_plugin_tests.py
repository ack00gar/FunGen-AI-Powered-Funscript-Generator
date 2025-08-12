#!/usr/bin/env python3
"""
Dedicated test runner for funscript plugin system.

Usage:
    python tests/run_plugin_tests.py [--category CATEGORY] [--plugin PLUGIN] [--benchmark]

Categories:
    all            - Run all plugin tests (default)
    functionality  - Test individual plugin functionality
    integration    - Test plugin system integration
    validation     - Test parameter validation
    preview        - Test preview system
    performance    - Run performance benchmarks
    quick          - Run fast tests only

Options:
    --plugin PLUGIN  - Test specific plugin only
    --benchmark      - Include detailed performance benchmarks
    --verbose        - Verbose output
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_plugin_tests(category: str = 'all', specific_plugin: str = None, 
                    benchmark: bool = False, verbose: bool = False):
    """Run plugin tests for specified category."""
    
    base_cmd = ['python', '-m', 'pytest']
    if verbose:
        base_cmd.append('-v')
    else:
        base_cmd.append('-q')
    
    # Set test environment
    import os
    env = os.environ.copy()
    env['FUNGEN_TESTING'] = '1'
    
    tests_passed = 0
    tests_failed = 0
    
    if category == 'all':
        print("🔌 Running Complete Plugin Test Suite")
        print("=" * 60)
        
        # Run all categories in order
        categories = ['functionality', 'integration', 'validation', 'preview']
        if benchmark:
            categories.append('performance')
        
        for cat in categories:
            print(f"\n📋 Running {cat} tests...")
            passed, failed = run_plugin_tests(cat, specific_plugin, False, verbose)
            tests_passed += passed
            tests_failed += failed
            
        if benchmark:
            print(f"\n⚡ Running performance benchmarks...")
            passed, failed = run_plugin_tests('performance', specific_plugin, True, verbose)
            tests_passed += passed
            tests_failed += failed
    
    elif category == 'functionality':
        print("🔧 Testing Individual Plugin Functionality")
        cmd = base_cmd + ['tests/unit/test_plugin_functionality.py', '--tb=short']
        
        if specific_plugin:
            cmd.extend(['-k', specific_plugin])
            print(f"🎯 Testing plugin: {specific_plugin}")
        
        result = run_test_command(cmd, env)
        tests_passed = 1 if result == 0 else 0
        tests_failed = 0 if result == 0 else 1
    
    elif category == 'integration':
        print("🔗 Testing Plugin System Integration")
        cmd = base_cmd + ['tests/integration/test_funscript_plugin_system.py', '--tb=short']
        
        if specific_plugin:
            cmd.extend(['-k', specific_plugin])
            print(f"🎯 Testing plugin: {specific_plugin}")
        
        result = run_test_command(cmd, env)
        tests_passed = 1 if result == 0 else 0
        tests_failed = 0 if result == 0 else 1
    
    elif category == 'validation':
        print("✅ Testing Parameter Validation")
        cmd = base_cmd + ['tests/unit/test_plugin_parameter_validation.py', '--tb=short']
        
        if specific_plugin:
            cmd.extend(['-k', specific_plugin])
            print(f"🎯 Testing plugin: {specific_plugin}")
        
        result = run_test_command(cmd, env)
        tests_passed = 1 if result == 0 else 0
        tests_failed = 0 if result == 0 else 1
    
    elif category == 'preview':
        print("👁️ Testing Preview System")
        cmd = base_cmd + ['tests/unit/test_plugin_preview_system.py', '--tb=short']
        
        if specific_plugin:
            cmd.extend(['-k', specific_plugin])
            print(f"🎯 Testing plugin: {specific_plugin}")
        
        result = run_test_command(cmd, env)
        tests_passed = 1 if result == 0 else 0
        tests_failed = 0 if result == 0 else 1
    
    elif category == 'performance':
        print("⚡ Testing Plugin Performance")
        cmd = base_cmd + ['tests/performance/test_plugin_performance.py', '--tb=short']
        
        if specific_plugin:
            cmd.extend(['-k', specific_plugin])
            print(f"🎯 Benchmarking plugin: {specific_plugin}")
        
        if benchmark:
            # Run detailed benchmarks
            cmd.extend(['-m', 'benchmark'])
        
        result = run_test_command(cmd, env)
        tests_passed = 1 if result == 0 else 0
        tests_failed = 0 if result == 0 else 1
    
    elif category == 'quick':
        print("🚀 Running Quick Plugin Tests")
        print("=" * 40)
        
        # Run fast tests only
        quick_tests = [
            ('functionality', 'tests/unit/test_plugin_functionality.py'),
            ('validation', 'tests/unit/test_plugin_parameter_validation.py')
        ]
        
        for test_name, test_file in quick_tests:
            print(f"\n📋 {test_name}...")
            cmd = base_cmd + [test_file, '--tb=line', '-x']  # Stop on first failure
            
            if specific_plugin:
                cmd.extend(['-k', specific_plugin])
            
            result = run_test_command(cmd, env)
            if result == 0:
                tests_passed += 1
                print(f"✅ {test_name} passed")
            else:
                tests_failed += 1
                print(f"❌ {test_name} failed")
    
    else:
        print(f"❌ Unknown category: {category}")
        return 0, 1
    
    return tests_passed, tests_failed


def run_test_command(cmd, env):
    """Run a test command and return the exit code."""
    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def list_available_plugins():
    """List available plugins for testing."""
    try:
        # Import here to avoid import issues during test setup
        from funscript.dual_axis_funscript import DualAxisFunscript
        
        fs = DualAxisFunscript()
        plugins = fs.list_available_plugins()
        
        print("🔌 Available Plugins:")
        print("-" * 30)
        
        for plugin in plugins:
            name = plugin['name']
            description = plugin.get('description', 'No description')
            print(f"  {name:<20} - {description}")
        
        print(f"\nTotal: {len(plugins)} plugins")
        
    except Exception as e:
        print(f"❌ Error listing plugins: {e}")


def run_plugin_discovery():
    """Run basic plugin discovery test."""
    print("🔍 Testing Plugin Discovery...")
    
    try:
        from funscript.dual_axis_funscript import DualAxisFunscript
        
        start_time = time.time()
        fs = DualAxisFunscript()
        plugins = fs.list_available_plugins()
        end_time = time.time()
        
        discovery_time = end_time - start_time
        
        print(f"✅ Discovered {len(plugins)} plugins in {discovery_time:.3f}s")
        
        # Test basic functionality
        if plugins:
            test_plugin = plugins[0]['name']
            print(f"🧪 Testing basic functionality with '{test_plugin}'...")
            
            # Create simple test data
            fs.add_action(1000, 50)
            fs.add_action(1100, 60)
            
            try:
                success = fs.apply_plugin(test_plugin)
                if success:
                    print(f"✅ {test_plugin} applied successfully")
                else:
                    print(f"⚠️ {test_plugin} returned False (may need more data)")
            except Exception as e:
                print(f"⚠️ {test_plugin} error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Plugin discovery failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run funscript plugin tests")
    parser.add_argument('--category', 
                       choices=['all', 'functionality', 'integration', 'validation', 'preview', 'performance', 'quick'],
                       default='all', help='Category of tests to run')
    parser.add_argument('--plugin', help='Test specific plugin only')
    parser.add_argument('--benchmark', action='store_true', help='Include detailed performance benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list-plugins', action='store_true', help='List available plugins and exit')
    parser.add_argument('--discovery-test', action='store_true', help='Run basic plugin discovery test')
    
    args = parser.parse_args()
    
    print("🔌 VR Funscript AI Generator - Plugin Test Runner")
    print("=" * 60)
    
    if args.list_plugins:
        list_available_plugins()
        return 0
    
    if args.discovery_test:
        success = run_plugin_discovery()
        return 0 if success else 1
    
    if args.plugin:
        print(f"🎯 Testing specific plugin: {args.plugin}")
    if args.benchmark:
        print("📊 Including performance benchmarks")
    if args.verbose:
        print("🔍 Verbose output enabled")
    print()
    
    start_time = time.time()
    
    tests_passed, tests_failed = run_plugin_tests(
        category=args.category,
        specific_plugin=args.plugin,
        benchmark=args.benchmark,
        verbose=args.verbose
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("PLUGIN TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print(f"⏱️ Total time: {total_time:.1f}s")
    
    if tests_failed == 0:
        print("\n🎉 All plugin tests passed!")
        return 0
    else:
        print(f"\n💥 {tests_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())