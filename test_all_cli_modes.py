#!/usr/bin/env python
"""
Comprehensive CLI mode testing script.
Tests all available modes with all option combinations.
"""

import subprocess
import json
import os
import sys
from datetime import datetime

# Test configuration
TEST_VIDEO = "/Users/k00gar/Downloads/test/test_koogar_extra_short_A.mp4"
OUTPUT_DIR = "/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator/output/test_koogar_extra_short_A"

# All available modes from --help
MODES = [
    "axis_projection",
    "axis_projection_enhanced", 
    "beat_marker",
    "oscillation_experimental_2",
    "oscillation",
    "osc",
    "live-osc",
    "oscillation-experimental",
    "oscillation_experimental",
    "oscillation_legacy",
    "oscillation-legacy",
    "user_roi",
    "yolo_roi",
    "yolo",
    "live-yolo",
    "auto-roi",
    "axis_projection_working",
    "OFFLINE_2_STAGE",
    "OFFLINE_3_STAGE_MIXED",
    "OFFLINE_3_STAGE",
    "community_example",
    "dot_marker"
]

# Modes that support od-mode option (offline modes with stage 3)
MODES_WITH_OD = ["OFFLINE_3_STAGE", "OFFLINE_3_STAGE_MIXED"]

def run_test(mode, autotune=True, od_mode=None):
    """Run a single test with specified options."""
    cmd = [
        "python", "main.py",
        TEST_VIDEO,
        "--mode", mode,
        "--overwrite",
        "--no-copy"
    ]
    
    if not autotune:
        cmd.append("--no-autotune")
    
    if od_mode and mode in MODES_WITH_OD:
        cmd.extend(["--od-mode", od_mode])
    
    # Build test name
    test_name = f"{mode}"
    if not autotune:
        test_name += "_no-autotune"
    if od_mode and mode in MODES_WITH_OD:
        test_name += f"_od-{od_mode}"
    
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            env={**os.environ, "FUNGEN_TESTING": "1"}
        )
        
        # Extract key metrics from output
        fps_match = None
        points_generated = None
        error_occurred = False
        
        # Look for FPS in progress bar
        for line in result.stdout.split('\n'):
            if 'FPS' in line and '100.00%' in line:
                # Extract FPS from final progress line
                parts = line.split('|')
                for part in parts:
                    if 'FPS' in part:
                        fps_str = part.strip().split()[0]
                        try:
                            fps_match = int(fps_str)
                        except:
                            pass
            
            # Check for errors
            if 'ERROR' in line or 'Exception' in line or 'Traceback' in line:
                error_occurred = True
        
        # Check the generated funscript
        funscript_path = os.path.join(OUTPUT_DIR, "test_koogar_extra_short_A.funscript")
        raw_funscript_path = os.path.join(OUTPUT_DIR, "test_koogar_extra_short_A_t1_raw.funscript")
        
        final_points = None
        raw_points = None
        
        if os.path.exists(funscript_path):
            try:
                with open(funscript_path, 'r') as f:
                    data = json.load(f)
                    final_points = len(data.get('actions', []))
            except:
                pass
        
        if os.path.exists(raw_funscript_path):
            try:
                with open(raw_funscript_path, 'r') as f:
                    data = json.load(f)
                    raw_points = len(data.get('actions', []))
            except:
                pass
        
        # Determine success
        success = result.returncode == 0 and not error_occurred and final_points is not None
        
        return {
            'test_name': test_name,
            'mode': mode,
            'autotune': autotune,
            'od_mode': od_mode,
            'success': success,
            'return_code': result.returncode,
            'fps': fps_match,
            'raw_points': raw_points,
            'final_points': final_points,
            'error': error_occurred,
            'stderr': result.stderr[:500] if result.stderr else None
        }
        
    except subprocess.TimeoutExpired:
        return {
            'test_name': test_name,
            'mode': mode,
            'autotune': autotune,
            'od_mode': od_mode,
            'success': False,
            'error': True,
            'stderr': 'TIMEOUT after 60 seconds'
        }
    except Exception as e:
        return {
            'test_name': test_name,
            'mode': mode,
            'autotune': autotune,
            'od_mode': od_mode,
            'success': False,
            'error': True,
            'stderr': str(e)
        }

def main():
    """Run all tests and generate report."""
    results = []
    
    print(f"Starting comprehensive CLI testing at {datetime.now()}")
    print(f"Test video: {TEST_VIDEO}")
    print(f"Total modes to test: {len(set(MODES))}")
    
    # Test each unique mode
    tested_modes = set()
    for mode in MODES:
        if mode in tested_modes:
            continue
        tested_modes.add(mode)
        
        # Test with autotune
        results.append(run_test(mode, autotune=True))
        
        # Test without autotune
        results.append(run_test(mode, autotune=False))
        
        # If mode supports od-mode, test both options
        if mode in MODES_WITH_OD:
            results.append(run_test(mode, autotune=True, od_mode="legacy"))
            results.append(run_test(mode, autotune=False, od_mode="legacy"))
    
    # Generate report
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    # Summary statistics
    total_tests = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_tests - successful
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Detailed results table
    print("\n" + "-"*80)
    print(f"{'Mode':<30} {'Autotune':<10} {'OD':<8} {'Success':<8} {'FPS':<8} {'Raw Pts':<10} {'Final Pts':<10}")
    print("-"*80)
    
    for r in results:
        mode = r['mode'][:28]
        autotune = "Yes" if r['autotune'] else "No"
        od = r['od_mode'] or "-"
        success = "✓" if r['success'] else "✗"
        fps = str(r.get('fps', '-'))
        raw_pts = str(r.get('raw_points', '-'))
        final_pts = str(r.get('final_points', '-'))
        
        print(f"{mode:<30} {autotune:<10} {od:<8} {success:<8} {fps:<8} {raw_pts:<10} {final_pts:<10}")
        
        # Print error details for failed tests
        if not r['success'] and r.get('stderr'):
            print(f"  ERROR: {r['stderr'][:100]}")
    
    # Identify issues
    print("\n" + "="*80)
    print("ISSUES IDENTIFIED")
    print("="*80)
    
    # Check for modes that fail
    failed_modes = [r['mode'] for r in results if not r['success']]
    if failed_modes:
        print(f"\nModes that failed: {', '.join(set(failed_modes))}")
    
    # Check for slow modes (< 100 FPS)
    slow_modes = []
    for r in results:
        fps = r.get('fps')
        if fps is not None and fps < 100 and fps > 0:
            slow_modes.append((r['mode'], fps))
    if slow_modes:
        print(f"\nModes with slow performance (<100 FPS):")
        for mode, fps in slow_modes:
            print(f"  - {mode}: {fps} FPS")
    
    # Check for autotune issues
    autotune_issues = []
    for r in results:
        if not r['autotune'] and r.get('final_points'):
            # When no-autotune, final should equal raw (or be close)
            if r.get('raw_points') and abs(r['final_points'] - r['raw_points']) > 10:
                autotune_issues.append(r)
    
    if autotune_issues:
        print(f"\n--no-autotune not working properly for:")
        for r in autotune_issues:
            print(f"  - {r['mode']}: Raw={r.get('raw_points')}, Final={r.get('final_points')}")
    
    # Save results to JSON
    with open('cli_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to cli_test_results.json")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())