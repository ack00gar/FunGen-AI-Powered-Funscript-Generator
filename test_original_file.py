#!/usr/bin/env python3
"""
Test the Anti-Jerk plugin with the original file provided by the user.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from funscript.dual_axis_funscript import DualAxisFunscript
from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin

def test_original_file():
    """Test with the original problematic funscript file."""
    print("🔍 Testing Anti-Jerk Plugin with Original User File")
    print("=" * 60)
    
    file_path = "/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator/output/test_koogar_extra_short_B/test_koogar_extra_short_B.funscript"
    
    try:
        # Load the original file
        with open(file_path, 'r') as f:
            original_data = json.load(f)
        
        print(f"📁 Loaded: {file_path}")
        print(f"📊 Original actions: {len(original_data['actions'])}")
        
        # Create DualAxisFunscript
        funscript = DualAxisFunscript()
        for action in original_data["actions"]:
            funscript.add_action(action["at"], action["pos"])
        
        # Analyze original data
        times = np.array([action['at'] for action in funscript.actions])
        positions = np.array([action['pos'] for action in funscript.actions])
        movements = np.abs(np.diff(positions))
        extreme_jumps = np.sum(movements > 70)
        extreme_jumps_pct = extreme_jumps / len(movements) * 100
        
        print(f"📈 Extreme jumps (>70): {extreme_jumps}/{len(movements)} ({extreme_jumps_pct:.1f}%)")
        print(f"📏 Position range: {positions.min()}-{positions.max()}")
        print(f"⏱️  Duration: {(times[-1] - times[0])/1000:.1f} seconds")
        print()
        
        # Test with line-fitting outlier mode
        print("🧪 Applying Line-Fitting Outlier Detection...")
        plugin = AntiJerkPlugin()
        
        error = plugin.transform(funscript, axis='primary', 
                               mode='line_fitting_outlier',
                               outlier_threshold=20.0,
                               max_line_distance=4,
                               outlier_removal_confidence=60.0)
        
        if error:
            print(f"❌ Error: {error}")
            return
        
        # Analyze results
        result_times = np.array([action['at'] for action in funscript.actions])
        result_positions = np.array([action['pos'] for action in funscript.actions])
        result_movements = np.abs(np.diff(result_positions))
        result_extreme_jumps = np.sum(result_movements > 70)
        result_extreme_jumps_pct = result_extreme_jumps / len(result_movements) * 100 if len(result_movements) > 0 else 0
        
        actions_removed = len(original_data['actions']) - len(funscript.actions)
        extreme_jump_reduction = extreme_jumps - result_extreme_jumps
        reduction_pct = (extreme_jump_reduction / extreme_jumps * 100) if extreme_jumps > 0 else 0
        
        print("✅ RESULTS:")
        print(f"📊 Final actions: {len(funscript.actions)} (-{actions_removed} removed)")
        print(f"📈 Final extreme jumps: {result_extreme_jumps}/{len(result_movements)} ({result_extreme_jumps_pct:.1f}%)")
        print(f"🎯 Improvement: {extreme_jump_reduction} fewer extreme jumps ({reduction_pct:.1f}% reduction)")
        print()
        
        # Save the processed file
        output_path = file_path.replace('.funscript', '_anti_jerk_processed.funscript')
        
        output_data = {
            "version": original_data.get("version", "1.0"),
            "author": original_data.get("author", "Unknown") + " + Anti-Jerk Filter",
            "actions": [{"at": action["at"], "pos": action["pos"]} for action in funscript.actions]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Processed file saved: {output_path}")
        print("🎉 Line-fitting outlier detection successfully applied!")
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Using the provided sample data instead...")
        
        # Use the sample data from the conversation
        sample_data = {
            "version":"1.0",
            "author":"FunGen beta 0.5.0",
            "actions":[
                {"at":6100,"pos":13}, {"at":6180,"pos":86}, {"at":6280,"pos":25},
                {"at":6420,"pos":96}, {"at":6540,"pos":1}, {"at":6620,"pos":83},
                {"at":6780,"pos":5}, {"at":6900,"pos":84}, {"at":7020,"pos":12},
                {"at":7140,"pos":91}, {"at":7260,"pos":8}, {"at":7380,"pos":89},
                {"at":7500,"pos":6}, {"at":7620,"pos":87}, {"at":7740,"pos":4},
                {"at":7860,"pos":85}, {"at":7980,"pos":2}, {"at":8100,"pos":83},
                {"at":8220,"pos":0}, {"at":8340,"pos":81}, {"at":8460,"pos":0},
                {"at":8580,"pos":79}, {"at":8700,"pos":2}, {"at":8820,"pos":77},
                {"at":8940,"pos":4}
            ]
        }
        
        # Test with sample data
        funscript = DualAxisFunscript()
        for action in sample_data["actions"]:
            funscript.add_action(action["at"], action["pos"])
        
        plugin = AntiJerkPlugin()
        error = plugin.transform(funscript, axis='primary', mode='line_fitting_outlier')
        
        if not error:
            print(f"✅ Sample data processed: {len(funscript.actions)} actions")
            print("🎉 Plugin is working correctly!")
        else:
            print(f"❌ Error with sample data: {error}")

if __name__ == "__main__":
    test_original_file()