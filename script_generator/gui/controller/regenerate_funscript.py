from tkinter import messagebox

from utils.lib_FunscriptHandler import FunscriptGenerator


def regenerate_funscript(state):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    print("Regenerating Funscript with tweaked settings...")

    # Apply tweaks to funscript_data
    if state.boost_enabled:
        print(f"Applying Boost: Up {state.boost_up_percent}%, Down {state.boost_down_percent}%")
        # Add boost logic here

    if state.threshold_enabled:
        print(f"Applying Threshold: Low {state.threshold_low}, High {state.threshold_high}")
        # Add threshold logic here

    if state.vw_simplification_enabled:
        print(f"Applying VW Simplification with Factor: {state.vw_factor} then rounding to {state.rounding}")
        # Add VW simplification logic here

    # Save and regenerate funscript
    funscript_handler = FunscriptGenerator()

    # Simplifying the funscript data and generating the file
    funscript_handler.generate(state)

    print("Funscript re-generation complete.")

    # Optional, compare generated funscript with reference funscript if specified, or a simple generic report
    funscript_handler.create_report_funscripts(state)

    print("Report generation complete.")