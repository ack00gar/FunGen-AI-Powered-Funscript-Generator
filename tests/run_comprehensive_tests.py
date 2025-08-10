#!/usr/bin/env python3
"""
Comprehensive test runner script for VR Funscript AI Generator.

Usage:
    python tests/run_comprehensive_tests.py [--mode MODE] [--quick] [--report]

Modes:
    smoke          - Run smoke tests only (fast)
    unit           - Run unit tests only (fast)  
    integration    - Run integration tests (medium)
    e2e            - Run end-to-end GUI tests (slow)
    perf           - Run performance benchmarks (slow)
    comprehensive  - Run all test types (very slow)
    all            - Legacy alias for comprehensive

Options:
    --quick  - Skip slow tests
    --report - Generate performance report after tests
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_tests(test_type: str, quick: bool = False, report: bool = False):
    """Run tests of specified type."""
    
    base_cmd = ['python', '-m', 'pytest', '-v']
    
    if test_type == 'smoke':
        cmd = base_cmd + [
            'tests/smoke/',
            '-m', 'smoke',
            '--tb=short'
        ]
        print("🔥 Running smoke tests...")
        
    elif test_type == 'unit':
        cmd = base_cmd + [
            'tests/unit/',
            '--tb=short'
        ]
        print("🔧 Running unit tests...")
    
    elif test_type == 'integration':
        cmd = base_cmd + [
            'tests/integration/',
            '--tb=short'
        ]
        if quick:
            cmd.extend(['-m', 'not slow'])
        print("🔗 Running integration tests...")
        
    elif test_type == 'e2e':
        cmd = base_cmd + [
            'tests/e2e/',
            '--tb=short'
        ]
        if quick:
            cmd.extend(['-m', 'not slow'])
        print("🎭 Running end-to-end tests...")
        
    elif test_type == 'comprehensive':
        print("🚀 Running comprehensive test suite...")
        
        # Run all test types in sequence
        test_types = ['smoke', 'unit', 'integration', 'e2e', 'perf']
        
        for sub_type in test_types:
            print(f"\\n" + "="*50)
            print(f"RUNNING {sub_type.upper()} TESTS")
            print("="*50)
            
            result = run_tests(sub_type, quick, False)  # Don't generate report for each sub-type
            if result != 0:
                print(f"❌ {sub_type} tests failed, continuing with other tests...")
        
        if report:
            generate_performance_report()
        
        return 0
        
    elif test_type == 'perf':
        cmd = base_cmd + [
            'tests/performance/',
            '-m', 'performance',
            '--tb=short'
        ]
        if quick:
            cmd.extend(['-m', 'not slow'])
        print("⚡ Running performance tests...")
        
    elif test_type == 'all':
        # Legacy support - redirect to comprehensive
        return run_tests('comprehensive', quick, report)
        
    else:
        print(f"❌ Unknown test type: {test_type}")
        return 1
    
    # Set environment
    import os
    env = os.environ.copy()
    env['FUNGEN_TESTING'] = '1'
    
    # Run the tests
    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent.parent)
        
        if report and test_type in ['perf', 'all']:
            generate_performance_report()
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def generate_performance_report():
    """Generate a performance report from test results."""
    
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE REPORT")
    print("="*60)
    
    results_file = Path(__file__).parent / "performance" / "performance_results.json"
    
    if not results_file.exists():
        print("📊 No performance results found")
        return
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Generate report
        report_lines = []
        report_lines.append("# Performance Test Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for test_name, results in data.items():
            if not results:
                continue
                
            report_lines.append(f"## {test_name.replace('_', ' ').title()}")
            report_lines.append("")
            
            # Group by mode
            by_mode = {}
            for result in results:
                mode = result['mode']
                if mode not in by_mode:
                    by_mode[mode] = []
                by_mode[mode].append(result)
            
            for mode, mode_results in by_mode.items():
                durations = [r['duration_seconds'] for r in mode_results]
                latest = durations[-1]
                avg = sum(durations) / len(durations)
                
                report_lines.append(f"### {mode}")
                report_lines.append(f"- Latest: {latest:.1f}s")
                report_lines.append(f"- Average: {avg:.1f}s ({len(durations)} runs)")
                report_lines.append(f"- Best: {min(durations):.1f}s")
                report_lines.append(f"- Worst: {max(durations):.1f}s")
                
                if len(durations) > 1:
                    trend = "⬇️ Improving" if latest < avg else "⬆️ Degrading" if latest > avg * 1.1 else "➡️ Stable"
                    change_pct = ((latest - avg) / avg) * 100
                    report_lines.append(f"- Trend: {trend} ({change_pct:+.1f}%)")
                
                report_lines.append("")
        
        # Save report
        report_file = Path(__file__).parent / "performance" / "performance_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📊 Performance report saved to: {report_file}")
        
        # Print summary
        print("\nPERFORMANCE SUMMARY:")
        for test_name, results in data.items():
            if results:
                latest_result = results[-1]
                print(f"  {test_name}: {latest_result['duration_seconds']:.1f}s ({latest_result['mode']})")
        
    except Exception as e:
        print(f"❌ Error generating performance report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive test pipeline")
    parser.add_argument('--mode', choices=['smoke', 'unit', 'integration', 'e2e', 'perf', 'comprehensive', 'all'], default='smoke',
                       help='Type of tests to run')
    parser.add_argument('--quick', action='store_true',
                       help='Skip slow tests')
    parser.add_argument('--report', action='store_true',
                       help='Generate performance report after tests')
    
    args = parser.parse_args()
    
    print(f"🎯 VR Funscript AI Generator - Comprehensive Test Runner")
    print(f"Mode: {args.mode}")
    if args.quick:
        print("Quick mode: Skipping slow tests")
    if args.report:
        print("Will generate performance report")
    print()
    
    exit_code = run_tests(args.mode, args.quick, args.report)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()