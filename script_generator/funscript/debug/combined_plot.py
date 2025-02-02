import datetime
import os

import numpy as np
from matplotlib import pyplot as plt, gridspec

from script_generator.debug.logger import log_fun
from script_generator.funscript.debug.heatmap import generate_heatmap_inline
from script_generator.state.app_state import AppState


def create_combined_plot(state: AppState, ref_sections, gen_sections, screenshots, sections, output_image_path, ref_times, ref_positions, gen_times, gen_positions):
    """
    Creates a combined plot with heatmaps as a header, comparative information, and section comparisons below.

    Args:
        state: AppState
        ref_sections (list): List of reference sections (times, positions).
        gen_sections (list): List of generated sections (times, positions).
        screenshots (list): List of screenshots for each section.
        sections (list): List of tuples (start, end) for each section.
        output_image_path (str): Path to save the combined plot.
        ref_times (list): Times from the reference funscript.
        ref_positions (list): Positions from the reference funscript.
        gen_times (list): Times from the generated funscript.
        gen_positions (list): Positions from the generated funscript.
    """

    # TODO why is this empty sometimes? and result in errors
    if not gen_times or not gen_positions:
        log_fun.error("Could not created combined plot")
        return

    # Create a flexible grid layout
    fig = plt.figure(figsize=(28, 24))
    gs = gridspec.GridSpec(5, 4, height_ratios=[1, .5, 2, 2, 2], width_ratios=[1, 2, 1, 2])

    # Heatmaps (First row: 2 columns spanning the entire width)
    if ref_sections:
        ax_ref_heatmap = fig.add_subplot(gs[0, :2])
        generate_heatmap_inline(ax_ref_heatmap, ref_times, ref_positions)
        ax_ref_heatmap.set_title('Reference Funscript Heatmap', fontsize=14)

    ax_gen_heatmap = fig.add_subplot(gs[0, 2:])
    generate_heatmap_inline(ax_gen_heatmap, gen_times, gen_positions)
    ax_gen_heatmap.set_title('Generated Funscript Heatmap', fontsize=14)

    if ref_sections:
        # Comparative information (Second row: 2 columns)
        ax_comparative_left = fig.add_subplot(gs[1, :2])
        ref_metrics = calculate_metrics(ref_times, ref_positions)
        ref_comparative_text = (
            f"Reference:\n"
            f"Number of Strokes: {ref_metrics['num_strokes']}\n"
            f"Avg Stroke Duration: {ref_metrics['avg_stroke_duration']:.2f}s\n"
            f"Avg Speed: {int(ref_metrics['avg_speed'])} positions/s\n"
            f"Avg Depth of Stroke: {int(ref_metrics['avg_depth'])}\n"
            f"Avg Max: {int(ref_metrics['avg_max'])}\n"
            f"Avg Min: {int(ref_metrics['avg_min'])}"
        )
        ax_comparative_left.text(0.5, 0.5, ref_comparative_text, fontsize=12, va='center', ha='center')
        ax_comparative_left.axis('off')

    ax_comparative_right = fig.add_subplot(gs[1, 2:])
    gen_metrics = calculate_metrics(gen_times, gen_positions)

    gen_comparative_text = (
        f"Generated:\n"
        f"Number of Strokes: {gen_metrics['num_strokes']}\n"
        f"Avg Stroke Duration: {gen_metrics['avg_stroke_duration']:.2f}s\n"
        f"Avg Speed: {int(gen_metrics['avg_speed'])} positions/s\n"
        f"Avg Depth of Stroke: {int(gen_metrics['avg_depth'])}\n"
        f"Avg Max: {int(gen_metrics['avg_max'])}\n"
        f"Avg Min: {int(gen_metrics['avg_min'])}"
    )
    ax_comparative_right.text(0.5, 0.5, gen_comparative_text, fontsize=12, va='center', ha='center')
    ax_comparative_right.axis('off')

    # Sections (Rows 3-8: Each row has 2 subplots for screenshot and data plot)
    for i in range(3, 6):  # Section rows
        for j in range(2):  # Two columns per row
            idx = (i - 3) * 2 + j  # Section index
            if idx >= len(sections):
                break

            # Screenshot (first column)
            ax_screenshot = fig.add_subplot(gs[i - 1, j * 2])
            ax_screenshot.imshow(screenshots[idx])
            ax_screenshot.axis('off')

            # Funscript comparison (second column)
            ax_plot = fig.add_subplot(gs[i - 1, j * 2 + 1])
            # Scale the y axis 0 to 100
            ax_plot.set_ylim(0, 100)
            gen_times_sec = [t / 1000 for t in gen_sections[idx][0]]
            ax_plot.plot(gen_times_sec, gen_sections[idx][1], label='Generated', color='blue')

            if ref_sections:
                ref_times_sec = [t / 1000 for t in ref_sections[idx][0]]
                ax_plot.plot(ref_times_sec, ref_sections[idx][1], label='Reference', color='red')

            start_time = datetime.timedelta(seconds=int(sections[idx][0]))  # datetime.datetime.fromtimestamp(sections[idx][0]).strftime('%H:%M:%S')
            end_time = datetime.timedelta(seconds=int(sections[idx][1]))  # datetime.datetime.fromtimestamp(sections[idx][1]).strftime('%H:%M:%S')
            ax_plot.set_title(f'Section {idx + 1}: {start_time} - {end_time}', fontsize=10)
            ax_plot.set_xlabel('Time (s)')
            ax_plot.set_ylabel('Position')
            ax_plot.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    # Add "SPOILER_" to the filename
    directory, filename = os.path.split(output_image_path)
    new_filename = f"SPOILER_{filename}"
    output_image_path = os.path.join(directory, new_filename)
    plt.savefig(output_image_path[:-4] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=100)


def calculate_metrics(times, positions):
    """
    Calculates metrics for a funscript.

    Args:
        times (list): List of times from the funscript.
        positions (list): List of positions from the funscript.

    Returns:
        dict: Dictionary containing the calculated metrics.
    """
    if not times or not positions:
        return {}

    times = np.array(times)
    positions = np.array(positions)

    # Calculate number of strokes
    num_strokes = len(times) - 1

    # Calculate average stroke duration
    stroke_durations = np.diff(times) / 1000  # Convert to seconds
    avg_stroke_duration = np.mean(stroke_durations)

    # Calculate average speed
    # prevent division by zero by adding 1e-10
    speeds = np.abs(np.diff(positions) / (np.diff(times) + 1e-10)) * 1000  # Positions per second
    avg_speed = np.mean(speeds)

    # Calculate average depth of stroke
    depths = np.abs(np.diff(positions))
    avg_depth = np.mean(depths)

    # Calculate average max and min
    avg_max = np.mean([max(positions[i], positions[i + 1]) for i in range(len(positions) - 1)])
    avg_min = np.mean([min(positions[i], positions[i + 1]) for i in range(len(positions) - 1)])

    return {
        'num_strokes': num_strokes,
        'avg_stroke_duration': avg_stroke_duration,
        'avg_speed': avg_speed,
        'avg_depth': avg_depth,
        'avg_max': avg_max,
        'avg_min': avg_min
    }