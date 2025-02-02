import datetime

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from script_generator.constants import HEATMAP_COLORS, STEP_SIZE
from script_generator.debug.logger import log_fun
from script_generator.funscript.util.util import load_funscript
from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path


def generate_heatmap(state: AppState):
    try:
        # Load funscript data
        funscript_path, _ = get_output_file_path(state.video_path, ".funscript")
        output_image_path, _ = get_output_file_path(state.video_path, ".png")
        times, positions, _, _ = load_funscript(funscript_path)
        if not times or not positions:
            log_fun.info("Failed to load funscript data.")
            return

        # add a timing: 0, position: 100 at the beginning if no value for 0
        if times[0] != 0:
            times.insert(0, 0)
            positions.insert(0, 100)

        # Print loaded data for debugging
        # logger.debug(f"Times: {times}")
        # logger.debug(f"Positions: {positions}")
        log_fun.debug(f"Total Actions: {len(times)}")
        log_fun.debug(f"Time Range: {times[0]} to {datetime.timedelta(seconds=int(times[-1] / 1000))}")

        # Calculate speed (position change per time interval)
        times = np.array(times)
        positions = np.array(positions)
        valid_indices = ~np.isnan(times) & ~np.isnan(positions)
        filtered_positions = positions[valid_indices]
        filtered_times = times[valid_indices]

        # Calculate speed (position change per time interval)
        # We add 1e-10 to prevent dividing by zero
        speeds = np.abs(np.diff(filtered_positions) / (np.diff(filtered_times) + 1e-10)) * 1000  # Positions per second

        log_fun.debug(f"Speeds: {speeds}")

        def get_color(intensity):
            if intensity <= 0:
                return HEATMAP_COLORS[0]
            if intensity > 5 * STEP_SIZE:
                return HEATMAP_COLORS[6]
            intensity += STEP_SIZE / 2.0
            index = int(intensity // STEP_SIZE)
            t = (intensity - index * STEP_SIZE) / STEP_SIZE
            return [
                HEATMAP_COLORS[index][0] + (HEATMAP_COLORS[index + 1][0] - HEATMAP_COLORS[index][0]) * t,
                HEATMAP_COLORS[index][1] + (HEATMAP_COLORS[index + 1][1] - HEATMAP_COLORS[index][1]) * t,
                HEATMAP_COLORS[index][2] + (HEATMAP_COLORS[index + 1][2] - HEATMAP_COLORS[index][2]) * t
            ]

        # Create figure and plot
        plt.figure(figsize=(30, 2))
        ax = plt.gca()

        # Draw lines between points with colors based on speed
        for i in range(len(times) - 1):
            x_start = times[i] / 1000  # Convert ms to seconds
            x_end = times[i + 1] / 1000
            y_start = positions[i]
            y_end = positions[i + 1]
            speed = speeds[i]

            # Get color based on speed
            color = get_color(speed)
            line_color = (color[0] / 255, color[1] / 255, color[2] / 255)  # Normalize to [0, 1]

            # Plot the line
            ax.plot([x_start, x_end], [y_start, y_end], color=line_color, linewidth=2)

        # Customize plot
        ax.set_title(
            f'Funscript Heatmap\nDuration: {datetime.timedelta(seconds=int(times[-1] / 1000))} - Avg. Speed {int(np.mean(speeds))} - Actions: {len(times)}')
        ax.set_xlabel('Time (s)')
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlim(times[0] / 1000, times[-1] / 1000)
        ax.set_ylim(0, 100)

        # Remove borders (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_heatmap", [
            (HEATMAP_COLORS[i][0] / 255, HEATMAP_COLORS[i][1] / 255, HEATMAP_COLORS[i][2] / 255) for i in
            range(len(HEATMAP_COLORS))
        ])
        norm = mcolors.Normalize(vmin=0, vmax=5 * STEP_SIZE)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2,
        #                    ticks=np.arange(0, 5 * step_size + 1, step_size))
        # cbar.set_label('Speed (positions/s)')

        # Save the figure
        plt.savefig(output_image_path, bbox_inches='tight', dpi=200)  # Increase resolution
        plt.close()
        log_fun.info(f"Funscript heatmap saved to {output_image_path}")
    except Exception as e:
        log_fun.error(f"Error generating heatmap for funscript: {e}")

def generate_heatmap_inline(ax, times, positions):
    """
    Generates a heatmap on the given axes using the existing `generate_heatmap` logic.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the heatmap on.
        times (list): List of times from the funscript.
        positions (list): List of positions from the funscript.
    """
    if not times or not positions:
        return

    # Bug fix, happened on some reference scripts with 2 identical times values : keep only the first one
    for i in range(1, len(times)):
        if times[i] == times[i - 1]:
            times.pop(i)
            positions.pop(i)
            log_fun.info(f"Removed duplicate time value {times[i]}")
            break

    # Calculate speed (position change per time interval)
    # prevent division by zero by adding 1e-10
    speeds = np.abs(np.diff(positions) / (np.diff(times) + 1e-10)) * 1000  # Positions per second

    def get_color(intensity):
        if np.isnan(intensity):
            return [128, 128, 128]
        if intensity <= 0:
            return HEATMAP_COLORS[0]
        if intensity > 5 * STEP_SIZE:
            return HEATMAP_COLORS[-1]
        intensity += STEP_SIZE / 2.0
        index = int(intensity // STEP_SIZE)
        t = (intensity - index * STEP_SIZE) / STEP_SIZE
        return [
            HEATMAP_COLORS[index][0] + (HEATMAP_COLORS[index + 1][0] - HEATMAP_COLORS[index][0]) * t,
            HEATMAP_COLORS[index][1] + (HEATMAP_COLORS[index + 1][1] - HEATMAP_COLORS[index][1]) * t,
            HEATMAP_COLORS[index][2] + (HEATMAP_COLORS[index + 1][2] - HEATMAP_COLORS[index][2]) * t
        ]

    # Draw lines between points with colors based on speed
    for i in range(len(times) - 1):
        x_start = times[i] / 1000  # Convert ms to seconds
        x_end = times[i + 1] / 1000
        y_start = positions[i]
        y_end = positions[i + 1]
        speed = speeds[i]

        # Get color based on speed
        color = get_color(speed)
        line_color = (color[0] / 255, color[1] / 255, color[2] / 255)  # Normalize to [0, 1]

        # Plot the line
        ax.plot([x_start, x_end], [y_start, y_end], color=line_color, linewidth=2)

    # Customize plot
    avg_speed = int(np.nanmean(speeds))

    # Set title with safe average speed
    ax.set_title(
        f'Funscript Heatmap\nDuration: {datetime.timedelta(seconds=int(times[-1] / 1000))} - Avg. Speed {avg_speed} - Actions: {len(times)}'
    )
    ax.set_xlabel('Time (s)')
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlim(times[0] / 1000, times[-1] / 1000)
    ax.set_ylim(0, 100)

    # Remove borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add colorbar
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_heatmap", [
        (HEATMAP_COLORS[i][0] / 255, HEATMAP_COLORS[i][1] / 255, HEATMAP_COLORS[i][2] / 255) for i in
        range(len(HEATMAP_COLORS))
    ])
    norm = mcolors.Normalize(vmin=0, vmax=5 * STEP_SIZE)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])