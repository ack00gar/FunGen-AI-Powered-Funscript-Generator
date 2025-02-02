import json
import os

import numpy as np
from scipy.signal import savgol_filter
from simplification.cutil import simplify_coords

from script_generator.debug.logger import log_fun
from script_generator.funscript.util.adjust_peaks_and_lows import adjust_peaks_and_lows
from script_generator.funscript.util.util import write_funscript
from script_generator.funscript.util.export_funscript import export_funscript
from script_generator.utils.file import get_output_file_path


def create_funscript(state):
    output_path, _ = get_output_file_path(state.video_path, ".funscript")

    raw_funscript_path, _ = get_output_file_path(state.video_path, ".json", "rawfunscript")
    if os.path.exists(raw_funscript_path) and (state.funscript_data is None or len(state.funscript_data) == 0):
        log_fun.info("len funscript data is 0, trying to load file")
        # Read the funscript data from the JSON file
        with open(raw_funscript_path, 'r') as f:
            log_fun.info(f"Loading funscript from {raw_funscript_path}")
            try:
                data = json.load(f)
            except Exception as e:
                log_fun.error(f"Error loading funscript from {raw_funscript_path}: {e}")
                return
    else:
        data = state.funscript_data

    try:
        log_fun.info(f"Generating funscript based on {len(data)} points...")

        # Extract timestamps and positions
        ats = [p[0] for p in data]
        positions = [p[1] for p in data]

        log_fun.info(f"Positions adjustment - step 1 (noise removal)")
        # Run the Savitzky-Golay filter
        positions = savgol_filter(positions, int(state.video_info.fps // 4), 3)

        # zip adjusted positions
        zip_positions = list(zip(ats, positions))

        # Apply VW simplification if enabled
        if state.vw_simplification_enabled:
            log_fun.info("Positions adjustment - step 2 (VW algorithm simplification)")
            zip_positions = simplify_coords(zip_positions, state.vw_factor)
            log_fun.info(f"Length of VW filtered positions: {len(zip_positions)}")
        else:
            log_fun.info("Skipping positions adjustment - step 2 (VW algorithm simplification)")

        # Extract timestamps and positions
        ats = [p[0] for p in zip_positions]
        positions = [p[1] for p in zip_positions]

        # Remap positions to 0-100 range
        log_fun.info("Positions adjustment - step 3 (remapping)")
        adjusted_positions = np.interp(positions, (min(positions), max(positions)), (0, 100))

        # Apply thresholding
        if state.threshold_enabled:
            log_fun.info(f"Positions adjustment - step 4 (thresholding)")
            adjusted_positions = adjusted_positions.tolist()  # Convert to list
            adjusted_positions = [
                0 if p < state.threshold_low else 100 if p > state.threshold_high else p for p in
                adjusted_positions]
        else:
            log_fun.info("Skipping positions adjustment - step 4 (thresholding)")

        # Apply amplitude boosting
        if state.boost_enabled:
            log_fun.info("Positions adjustment - step 5 (amplitude boosting)")
            # self.boost_amplitude(adjusted_positions, boost_factor=1.2, min_value=0, max_value=100)
            adjusted_positions = adjust_peaks_and_lows(
                adjusted_positions,
                peak_boost=state.boost_up_percent,
                low_reduction=state.boost_down_percent
            )
        else:
            log_fun.info("Skipping positions adjustment - step 5 (amplitude boosting)")

        # Round position values to the closest multiple of 5, still between 0 and 100
        if state.vw_simplification_enabled:
            log_fun.info(
                f"Positions adjustment - step 6 (rounding to the closest multiple of {state.rounding})")
            adjusted_positions = [round(p / state.rounding) * state.rounding for p in
                                  adjusted_positions]

        else:
            log_fun.info(
                f"Skipping positions adjustment - step 6 (rounding to the closest multiple of {state.rounding})")

        # Recombine timestamps and adjusted positions
        log_fun.info("Re-assembling ats and positions")
        zip_adjusted_positions = list(zip(ats, adjusted_positions))

        # Write the final funscript
        write_funscript(zip_adjusted_positions, output_path, state.video_info.fps)

        # copy funscript if specified
        export_funscript(state)

        log_fun.info(f"Funscript generation complete")
    except Exception as e:
        log_fun.error(f"Error generating funscript: {e}")
        raise
